"""Новостные признаки на уровне МО: всплески статей о ЧС и об отключениях по городу — центру МО.

Запуск (после src.news_infini --collect и src.external):
  .venv/bin/python -m src.news_features
Читает data/news/infini_ru.parquet (заголовок + лид), собирает:
  data/news/city_gazetteer.parquet — словарь «город → МО»;
  data/news/city_month.parquet     — МО × месяц: число статей по темам и всплеск к обычному уровню.

Привязка. Город — центр МО из справочника СберИндекса (municipal_district_center вида «г Орск»).
Если у нескольких МО один центр (город и одноимённый район), статьи относятся к самому населённому.
Названия, встречающиеся в разных регионах, и названия из AMBIGUOUS не используются, как и похожие
на фамилии (-ов/-ев/-ин) у городов меньше SURNAME_POP жителей. У остальных МО признаков нет (NaN).

Темы — по ключевым словам (RE2 через pyarrow: \\b и \\w там только ASCII, поэтому границы слова заданы явно):
  emerg  — ЧС и стихия: паводок, подтопление, эвакуация, природные пожары, ураган, землетрясение;
  outage — отключения света, тепла, воды, газа и аварии на сетях.

Всплеск месяца m: log((n_m + 1) / (медиана n за месяцы до m + 1)), где n — число статей, приведённое
к одному объёму корпуса (покрытие изданий в корпусе неровное). Медиана — только по прошлому, без утечки.
"""
from __future__ import annotations

import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

NEWS = Path("data/news/infini_ru.parquet")
OUT = Path("data/news")
DICT = Path("data/external/raw/t_dict_municipal_districts.xlsx")
STATIC = Path("data/external/territory_static.parquet")
CONSUMPTION = Path("consumption.parquet")

B, E, W = "(?:^|[^а-я])", "(?:$|[^а-я])", "[а-я]*"
THEMES = {
    "emerg": "режим{W} чс|чрезвычайн{W} ситуац|паводк|наводнен|подтопл|затопл|прорыв{W} дамб|эвакуир|эвакуац"
             "|лесн{W} пожар|ландшафтн{W} пожар|землетрясен|ураган|{B}смерч|оползен|сход{W} сел",
    "outage": "отключ{W} (?:свет|электр|тепл|вод|отоплен|газ)|без (?:света|тепла|воды|электричества|отопления|газа)"
              "|авари{W} на (?:тэц|теплотрасс|подстанц|котельн|водовод)|блэкаут|прорыв{W} труб|замерза",
}
THEMES = {k: v.replace("{B}", B).replace("{W}", W) for k, v in THEMES.items()}
NO_LEAD = ["URA.RU", "Коммерсантъ"]  # в лиде заглушка («Читайте на URA.RU»)
SURNAME_POP = 150_000
REF_ARTICLES = 100_000  # число статей приводится к этому объёму корпуса за месяц
# Фамилии, обычные слова, одноимённые города в других странах и регионах, города федерального значения
AMBIGUOUS = set("""
владимир киров орел зима ключи луга остров артем королев чехов пушкино лермонтов строитель починок гагарин
куйбышев жуков пугачев маркс ленинск октябрьск первомайск комсомольск клин бор гай тара буй нея сим куса реж аша
мир дно холм кола сокол суворов покров шахты тайга белово топки дубна руза истра видное троицк щербинка калач
лиски бобров павловск александров болгар шали аргун москва санкт-петербург севастополь
ростов ливны волчанск бородино валдай покровск красноармейск любим галич горняк озерск калининск петровск
приволжск спасск александровск дмитровск мантурово домодедово лобня суровикино котово фролово павлово ломоносов
""".split())


def _word(w: str, inflect: bool) -> str:
    """Шаблон слова с падежными окончаниями; inflect=False — прилагательное в составном названии."""
    if not inflect:
        return w[:-2] + "[а-я]{1,3}"
    e = w[-1]
    if e == "а":
        return w[:-1] + "(?:а|ы|и|е|у|ой|ою)"
    if e == "я":
        return w[:-1] + "(?:я|и|е|ю|ей)"
    if e == "ь":
        return w[:-1] + "(?:ь|и|я|ю|ью|ем|е)"
    if e == "й":
        return w[:-1] + "(?:й|я|ю|е|ем)"
    if e in "ое":
        return w[:-1] + "(?:о|е|а|у|ом)"
    if e in "ыи":
        return w[:-1] + "(?:ы|и|ов|ах|ам|ами|ей|ях)"
    return w + "(?:а|у|е|ом)?"


def city_pattern(city: str) -> str:
    toks = re.split(r"([ -])", city)
    last = max(i for i, t in enumerate(toks) if t not in (" ", "-"))
    hyphen = "-" in toks  # «Ростов-на-Дону»: склоняются все части
    return B + "".join(t if t in (" ", "-") else _word(t, i == last or hyphen) for i, t in enumerate(toks)) + E


def build_gazetteer() -> pd.DataFrame:
    d = pd.read_excel(DICT)
    d = d[d["year_to"] >= 2024].drop_duplicates("territory_id")
    ids = pd.read_parquet(CONSUMPTION, columns=["territory_id"])["territory_id"].unique()
    d = d[d["territory_id"].isin(ids) & d["municipal_district_center"].fillna("").str.startswith("г ")].copy()
    d["city"] = d["municipal_district_center"].str[2:].str.strip().str.lower().str.replace("ё", "е")
    pop = np.exp(pd.read_parquet(STATIC).set_index("territory_id")["log_pop"])
    d["pop"] = pop.reindex(d["territory_id"]).values
    d = d[d.groupby("city")["region_code"].transform("nunique") == 1]
    d = d.sort_values("pop", ascending=False).drop_duplicates("city")
    surname = d["city"].str.contains(r"(?:ов|ев|ин|ын)$") & ~(d["pop"] >= SURNAME_POP)
    adjective = d["city"].str.contains(r"(?:ый|ий|ой|ая|ое)$")
    d = d[~d["city"].isin(AMBIGUOUS) & ~surname & ~adjective & (d["city"].str.len() >= 4)]
    d["pattern"] = d["city"].map(city_pattern)
    cols = ["territory_id", "city", "region_code", "region_name", "pop", "pattern"]
    return d[cols].sort_values("territory_id").reset_index(drop=True)


def _tag(args: tuple[int, list[int], list[str]]) -> pd.DataFrame:
    """Одна row group: статьи нужных тем → (месяц, МО, тема) с числом статей."""
    i, tids, patterns = args
    t = pq.ParquetFile(NEWS).read_row_group(i, columns=["title", "description", "outlet", "pub_date"])
    lead = pc.if_else(pc.is_in(t["outlet"], value_set=pa.array(NO_LEAD)), "", pc.coalesce(t["description"], ""))
    text = pc.binary_join_element_wise(pc.coalesce(t["title"], ""), lead, " ")
    text = pc.replace_substring(pc.utf8_lower(text), "ё", "е")
    theme = {k: pc.match_substring_regex(text, p).to_numpy(zero_copy_only=False) for k, p in THEMES.items()}
    month = t["pub_date"].to_pandas().dt.strftime("%Y-%m").values
    total = pd.Series(month).value_counts()
    keep = np.flatnonzero(np.logical_or.reduce(list(theme.values())))
    sub = text.take(pa.array(keep))
    rows = []
    for tid, p in zip(tids, patterns):
        hit = keep[pc.match_substring_regex(sub, p).to_numpy(zero_copy_only=False)]
        if len(hit):
            rows.append(pd.DataFrame({"territory_id": tid, "month": month[hit],
                                      **{k: v[hit] for k, v in theme.items()}}))
    cols = ["territory_id", "month", *THEMES]
    out = pd.concat(rows) if rows else pd.DataFrame(columns=cols)
    out = out.groupby(["territory_id", "month"], as_index=False)[list(THEMES)].sum()
    # общее число статей за месяц — строкой с territory_id = -1
    tot = pd.DataFrame({"territory_id": -1, "month": total.index, "emerg": total.values, "outage": 0})
    return pd.concat([out, tot], ignore_index=True)


def spike(n: pd.DataFrame) -> pd.DataFrame:
    """log((n_m + 1) / (медиана по месяцам до m + 1)); первый месяц — NaN (нет истории)."""
    past = n.T.expanding().median().shift(1).T
    return np.log((n + 1) / (past + 1))


def main(workers: int = 4) -> None:
    gaz = build_gazetteer()
    OUT.mkdir(parents=True, exist_ok=True)
    gaz.to_parquet(OUT / "city_gazetteer.parquet", index=False)
    print(f"словарь: {len(gaz)} городов-центров МО")
    n_rg = pq.ParquetFile(NEWS).metadata.num_row_groups
    jobs = [(i, gaz["territory_id"].tolist(), gaz["pattern"].tolist()) for i in range(n_rg)]
    with Pool(workers) as pool:
        res = pd.concat(pool.map(_tag, jobs, chunksize=1), ignore_index=True)
    res = res.groupby(["territory_id", "month"], as_index=False)[list(THEMES)].sum()
    total = res[res["territory_id"] == -1].set_index("month")["emerg"].sort_index()
    res = res[res["territory_id"] != -1]
    months = total.index.tolist()
    scale = REF_ARTICLES / total

    parts = []
    for k in THEMES:
        raw = res.pivot(index="territory_id", columns="month", values=k)
        raw = raw.reindex(index=gaz["territory_id"], columns=months).fillna(0)
        sp = spike(raw * scale)
        long = raw.stack().rename(f"n_{k}").to_frame().join(sp.stack(future_stack=True).rename(f"spike_{k}"))
        parts.append(long)
    cm = pd.concat(parts, axis=1).reset_index()
    cm.to_parquet(OUT / "city_month.parquet", index=False)
    print(f"city_month: {cm['territory_id'].nunique()} МО × {len(months)} мес. ({months[0]}…{months[-1]})")
    for k in THEMES:
        top = cm.nlargest(5, f"n_{k}").merge(gaz[["territory_id", "city"]])
        print(f"  {k}: статей {int(cm[f'n_{k}'].sum()):,}; крупнейшие — "
              + ", ".join(f"{r.city} {r.month} ({int(getattr(r, 'n_' + k))})" for r in top.itertuples()))


if __name__ == "__main__":
    main()
