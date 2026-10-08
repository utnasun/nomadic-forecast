"""Детектор шоков потребления в МО: два способа и их сравнение, новости к найденным шокам.

Запуск (после src.run_forecast и src.evaluate; новости — src.news_infini и src.news_features):
  .venv/bin/python -m src.shocks → results/shocks/

Шок — месяц, когда расходы МО разошлись с ожидаемыми сильнее, чем обычно расходятся у похожих МО.
Способы отличаются только ожидаемым значением:
  resid — прогноз на 1 месяц вперёд лучшей по MAE модели для категории (results/forecasts): учитывает
          сезонность, динамику региона и соседей. Только тест — 2024 г.;
  peer  — прирост за месяц как у МО своего региона (медиана; в регионах, где МО панели меньше
          MIN_REGION, — по стране). Работает с февраля 2023 г.; региональный шок целиком не видит.
Отклонение d = log(факт / ожидание). Внутри (категория, месяц, группа размера МО по уровню расходов)
d центрируется медианой и делится на робастный разброс (1,4826·MAD) → z. Затем z делится на
собственную волатильность МО — робастный разброс его z способом peer за прошлые месяцы (не меньше
MIN_HISTORY, не меньше 1): у удалённых МО расходы скачут постоянно, и это не шок. Будущее не
используется, кроме «разогрева»: для первых MIN_HISTORY месяцев peer (2023-02…2023-07) истории нет,
и волатильность оценивается по этим же месяцам — они помечены в столбце warmup. Сводная оценка МО за месяц — среднеквадратичное z по TOP_K
категориям с наибольшим |z| (z обрезаны ±Z_CLIP): ловит и сдвиг структуры, когда сумма почти не
меняется, и не разбавляется спокойными категориями. Шок — оценка ≥ THRESHOLD (≈1% МО-месяцев).

Новости: статьи, где упомянут город — центр МО (словарь src.news_features) или сам район/округ
(«в Долинском районе»; названия, повторяющиеся в разных регионах, пропускаются). Для шока — число
таких статей за месяц к обычному уровню, тип события по ключевым словам и заголовки.
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from src.data import build_panels, complete_territories, load_consumption
from src.models import resolve_config
from src.news_features import NEWS, NO_LEAD, B, W

MIN_REGION = 5
N_SIZE_BINS = 5
Z_CLIP = 8.0
TOP_K = 3
THRESHOLD = 3.0
MIN_HISTORY = 6  # месяцев истории для собственной волатильности МО
REGION_WIDE = 0.3  # шок региональный, если в тот же месяц шок у ≥30% МО региона
NEWS_SPIKE = (10, 2.5)  # новостной всплеск: ≥10 статей о городе и ≥2,5× обычного
EVENT_TYPES = {  # тип события по заголовку и лиду; порядок — приоритет при равенстве
    "ЧС и стихия": "паводк|наводнен|подтопл|затопл|прорыв{W} дамб|эвакуа|эвакуир|лесн{W} пожар|ураган|смерч"
                   "|землетрясен|режим{W} чс|чрезвычайн{W} ситуац",
    "отключения и аварии на сетях": "отключ{W} (?:свет|электр|тепл|вод|отоплен|газ)|без (?:света|тепла|воды|отопления)"
                                    "|авари{W} на (?:тэц|теплотрасс|подстанц|котельн|водовод)|прорыв{W} труб",
    "атаки и теракты": "беспилотник|{B}бпла|{B}дрон|обстрел|теракт|ракетн{W} опасност",
    "погода": "мороз|снегопад|метел|аномальн{W} (?:жар|холод)|{B}жар[аеуы]{W}|ливн|гололед",
    "предприятия и работа": "завод|предприят|комбинат|сокращен|увольн|зарплат|банкрот|забастов",
    "торговля и цены": "магазин|торгов{W} центр|{B}тц{W}|рынк|маркетплейс|цен{W} на|подорожа",
    "транспорт": "дорог|трасс|мост|автобус|аэропорт|рейс|электричк|перекры",
    "праздники и события": "фестивал|праздник|концерт|форум|сабантуй|ярмарк|турист",
}
EVENT_TYPES = {k: v.replace("{B}", B).replace("{W}", W) for k, v in EVENT_TYPES.items()}


# ---------- отклонения ----------

def standardize(long: pd.DataFrame) -> pd.DataFrame:
    """z внутри (категория, месяц, группа размера): робастно центрированное и нормированное отклонение."""
    long = long.dropna(subset=["d"]).copy()
    long["size_bin"] = long.groupby(["category", "month"])["level_prev"].transform(
        lambda s: pd.qcut(s.rank(method="first"), N_SIZE_BINS, labels=False))
    g = long.groupby(["category", "month", "size_bin"])["d"]
    med = g.transform("median")
    mad = (long["d"] - med).abs().groupby([long["category"], long["month"], long["size_bin"]]).transform("median")
    long["z"] = (long["d"] - med) / (1.4826 * mad).clip(lower=1e-4)
    return long


def own_volatility(peer_long: pd.DataFrame) -> pd.DataFrame:
    """Робастный разброс z МО по месяцам строго до текущего; меньше MIN_HISTORY месяцев — 1."""
    rows = []
    for cat, g in peer_long.groupby("category"):
        z = g.pivot(index="territory_id", columns="month", values="z").sort_index(axis=1)
        vals = z.values
        out = np.ones_like(vals)
        for j in range(vals.shape[1]):
            past = vals[:, :max(j, MIN_HISTORY)]  # разогрев: первые MIN_HISTORY месяцев
            med = np.nanmedian(past, 1, keepdims=True)
            out[:, j] = np.maximum(1.4826 * np.nanmedian(np.abs(past - med), 1), 1.0)
        rows.append(pd.DataFrame(out, index=z.index, columns=z.columns).stack().rename("vol")
                    .reset_index().assign(category=cat))
    return pd.concat(rows, ignore_index=True)


def rescale(long: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    long = long.merge(vol, on=["territory_id", "month", "category"], how="left")
    long["z"] = long["z"] / long["vol"].fillna(1.0)
    return long


def peer_deviations(panels: dict, region: pd.Series) -> pd.DataFrame:
    rows = []
    for cat, p in panels.items():
        L = np.log(np.clip(p.values, 1, None))
        dlog = pd.DataFrame(np.diff(L, axis=1), index=p.territory_ids, columns=p.months[1:])
        reg = region.reindex(p.territory_ids).values
        size = pd.Series(reg).map(pd.Series(reg).value_counts()).values
        peer = dlog.groupby(reg).transform("median")
        peer[size < MIN_REGION] = np.nan
        peer = peer.fillna(dlog.median())  # маленький регион — медиана по стране
        d = (dlog - peer).stack().rename("d")
        lvl = pd.DataFrame(p.values[:, :-1], index=p.territory_ids, columns=p.months[1:]).stack().rename("level_prev")
        rows.append(pd.concat([d, lvl], axis=1).rename_axis(["territory_id", "month"]).reset_index()
                    .assign(category=cat, expected="медиана региона"))
    return pd.concat(rows, ignore_index=True)


def best_models(cfg: dict, out: Path) -> dict[str, str]:
    """Лучшая по MAE модель на горизонте 1 для каждой категории, без моделей с новостями."""
    m = pd.read_csv(out / "metrics_overall.csv")
    m = m[(m["horizon"] == 1) & m["model"].isin(cfg["models"])]
    m = m[[("news" not in resolve_config(cfg["models"], n).get("features", ())) for n in m["model"]]]
    return m.loc[m.groupby("category")["MAE"].idxmin()].set_index("category")["model"].to_dict()


def resid_deviations(cfg: dict, out: Path, categories: list[str]) -> pd.DataFrame:
    best = best_models(cfg, out)
    rows = []
    for cat in categories:
        fc = pd.read_parquet(out / "forecasts" / best[cat] / f"{cat}.parquet")
        fc = fc[fc["step"] == 1]
        rows.append(pd.DataFrame({
            "territory_id": fc["territory_id"], "month": fc["month"], "category": cat,
            "d": np.log(fc["y_true"].clip(lower=1) / fc["y_pred"].clip(lower=1)),
            "level_prev": fc["y_last"], "expected": best[cat]}))
    return pd.concat(rows, ignore_index=True)


def combine(long: pd.DataFrame, region: pd.Series) -> pd.DataFrame:
    """МО × месяц: z по категориям, сводная оценка, флаг шока, главные категории."""
    zc = long.pivot_table(index=["territory_id", "month"], columns="category", values="z")
    dd = long.pivot_table(index=["territory_id", "month"], columns="category", values="d")
    top = -np.sort(-np.abs(zc.clip(-Z_CLIP, Z_CLIP).values), axis=1)[:, :TOP_K]  # NaN уходят в конец
    score = pd.Series(np.sqrt(np.nanmean(top ** 2, axis=1)), index=zc.index)
    res = pd.DataFrame({"score": score, "shock": score >= THRESHOLD})
    res = res.join(zc.add_prefix("z: ")).join((dd * 100).add_prefix("откл., %: ")).reset_index()
    res["region"] = res["territory_id"].map(region)
    first = sorted(long["month"].unique())
    res["warmup"] = res["month"].isin(first[:MIN_HISTORY]) if first[0] < "2024" else False

    def drivers(row):
        z = row[[c for c in row.index if c.startswith("z: ")]].dropna()
        top = z.abs().sort_values(ascending=False).index[:3]
        return "; ".join(f"{c[3:]} {row['откл., %: ' + c[3:]]:+.0f}%" for c in top if abs(z[c]) >= 2)

    res["главное"] = res.apply(drivers, axis=1)
    share = res.groupby(["region", "month"])["shock"].transform("mean")
    size = res.groupby(["region", "month"])["shock"].transform("size")
    wide = (share >= REGION_WIDE) & (size >= MIN_REGION)
    res["масштаб"] = np.where(res["shock"] & wide, "регион", np.where(res["shock"], "МО", ""))
    return res


# ---------- новости ----------

def _news_rg(args):
    """Одна row group: статьи о городах из списка → (МО, месяц, заголовок, типы)."""
    i, tids, patterns = args
    t = pq.ParquetFile(NEWS).read_row_group(i, columns=["title", "description", "outlet", "pub_date"])
    lead = pc.if_else(pc.is_in(t["outlet"], value_set=pa.array(NO_LEAD)), "", pc.coalesce(t["description"], ""))
    text = pc.replace_substring(pc.utf8_lower(pc.binary_join_element_wise(pc.coalesce(t["title"], ""), lead, " ")),
                                "ё", "е")
    title_l = pc.replace_substring(pc.utf8_lower(pc.coalesce(t["title"], "")), "ё", "е")
    month = t["pub_date"].to_pandas().dt.strftime("%Y-%m").values
    titles = t["title"].to_pylist()
    rows = []
    for tid, p in zip(tids, patterns):
        hit = np.flatnonzero(pc.match_substring_regex(text, p).to_numpy(zero_copy_only=False))
        if not len(hit):
            continue
        sub = text.take(pa.array(hit))
        in_title = pc.match_substring_regex(title_l.take(pa.array(hit)), p).to_numpy(zero_copy_only=False)
        types = {k: pc.match_substring_regex(sub, rx).to_numpy(zero_copy_only=False) for k, rx in EVENT_TYPES.items()}
        rows.append(pd.DataFrame({"territory_id": tid, "month": month[hit], "title": [titles[j] for j in hit],
                                  "in_title": in_title, **types}))
    return pd.concat(rows) if rows else None


def mo_patterns(gaz: pd.DataFrame, dct: pd.DataFrame, ids: set) -> pd.DataFrame:
    """МО → шаблон упоминания: город-центр из словаря и/или «<название> район|округ»."""
    d = dct[dct["territory_id"].isin(ids)
            & dct["municipal_district_type"].isin(["муниципальный район", "муниципальный округ"])]
    name = d["municipal_district_name_short"].str.lower().str.replace("ё", "е")
    ok = name.str.fullmatch(r"[а-я-]+(?:ий|ый|ой)") & ~name.duplicated(keep=False)
    dist = pd.Series((B + name[ok].str[:-2] + "[а-я]{1,3} (?:район|округ|муниципальн)").values,
                     index=d.loc[ok, "territory_id"].values)
    city = gaz.set_index("territory_id")["pattern"]
    both = pd.concat([city, dist]).groupby(level=0).agg(lambda p: "|".join(f"(?:{x})" for x in p))
    assert both.str.contains("nan").sum() == 0, "пустой шаблон"
    return both.rename("pattern").rename_axis("territory_id").reset_index()


def city_news(tids: list[int], pats: pd.DataFrame, workers: int = 4) -> pd.DataFrame:
    g = pats[pats["territory_id"].isin(tids)]
    n_rg = pq.ParquetFile(NEWS).metadata.num_row_groups
    with Pool(workers) as pool:
        parts = pool.map(_news_rg, [(i, g["territory_id"].tolist(), g["pattern"].tolist()) for i in range(n_rg)])
    return pd.concat([p for p in parts if p is not None], ignore_index=True)


def news_spikes(art: pd.DataFrame, total: pd.Series) -> pd.DataFrame:
    """Город × месяц: число статей и отношение к медиане остальных месяцев (объём корпуса выровнен)."""
    cnt = art.groupby(["territory_id", "month"]).size().unstack(fill_value=0).reindex(columns=total.index, fill_value=0)
    adj = cnt * (total.mean() / total)
    rows = []
    for tid, row in adj.iterrows():
        for m in row.index:
            rows.append((tid, m, int(cnt.at[tid, m]), (row[m] + 1) / (row.drop(m).median() + 1)))
    sp = pd.DataFrame(rows, columns=["territory_id", "month", "n", "ratio"])
    sp["spike"] = (sp["n"] >= NEWS_SPIKE[0]) & (sp["ratio"] >= NEWS_SPIKE[1])
    return sp


def event_type(a: pd.DataFrame) -> str:
    """Тип события по статьям МО за месяц: самый частый, если статей с ним не меньше 3; иначе ""."""
    a = a.drop_duplicates("title")
    if len(a) < 3:
        return ""
    s = a[list(EVENT_TYPES)].sum()
    return s.idxmax() if s.max() >= 3 else ""


def attach_news(shocks: pd.DataFrame, art: pd.DataFrame, covered: set, sp: pd.DataFrame) -> pd.DataFrame:
    """Всплеск статей о городе в месяц шока, тип события и заголовки."""
    spi = sp.set_index(["territory_id", "month"])
    out = []
    for r in shocks.itertuples():
        key = (r.territory_id, r.month)
        if r.territory_id not in covered:
            out.append(("нет в словаре упоминаний", np.nan, np.nan, "", ""))
            continue
        n, x = (int(spi.at[key, "n"]), spi.at[key, "ratio"]) if key in spi.index else (0, np.nan)
        a = art[(art["territory_id"] == r.territory_id) & (art["month"] == r.month)]
        typ = event_type(a)
        a = a.drop_duplicates("title")
        a = a.assign(pri=a["in_title"].astype(int) * 2 + a[list(EVENT_TYPES)[:3]].any(axis=1).astype(int) * 3
                     + a[list(EVENT_TYPES)].any(axis=1).astype(int))
        heads = " | ".join(a.sort_values("pri", ascending=False)["title"].head(4))
        spike = n >= NEWS_SPIKE[0] and x >= NEWS_SPIKE[1]
        verdict = "новостной всплеск" if spike else ("есть тематические новости" if typ else "объяснения не найдено")
        out.append((verdict, n, x, typ, heads))
    cols = ["новости", "статей", "к обычному", "тип события", "заголовки"]
    return shocks.reset_index(drop=True).join(pd.DataFrame(out, columns=cols))


# ---------- сравнение ----------

def compare(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """a — resid, b — peer; только общие месяцы."""
    key = ["territory_id", "month"]
    m = a[key + ["score", "shock", "главное", "масштаб"]].merge(
        b[key + ["score", "shock", "главное", "масштаб"]], on=key, suffixes=("_resid", "_peer"))
    m["группа"] = np.select([m["shock_resid"] & m["shock_peer"], m["shock_resid"], m["shock_peer"]],
                            ["оба", "только resid", "только peer"], "нет")
    both, ra, pa_ = (m["группа"] == g for g in ("оба", "только resid", "только peer"))
    lines = [f"Общих МО-месяцев: {len(m):,}. Шоков: resid — {int(m['shock_resid'].sum())}, "
             f"peer — {int(m['shock_peer'].sum())}, оба — {int(both.sum())}, "
             f"пересечение / объединение = {both.sum() / max((m['shock_resid'] | m['shock_peer']).sum(), 1):.0%}.",
             f"Ранговая корреляция оценок (Спирмен): {m['score_resid'].corr(m['score_peer'], method='spearman'):.2f}."]
    return m, lines


def disagreement(m: pd.DataFrame, res: dict, region: pd.Series, pop: pd.Series) -> pd.DataFrame:
    """Чем отличаются группы: сезонный повтор (peer), сдвиг региона от прогноза (resid), размер МО."""
    dev = res["peer"].set_index(["territory_id", "month"])
    dev = dev[[c for c in dev.columns if c.startswith("откл., %: ")]]
    ly = dev.copy()
    ly.index = pd.MultiIndex.from_arrays([ly.index.get_level_values(0),
                                          (pd.PeriodIndex(ly.index.get_level_values(1), freq="M") + 12).astype(str)])
    cur, prev = dev.align(ly, join="inner")
    a, b = cur.fillna(0).values, prev.fillna(0).values
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    corr = pd.Series((a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1) + 1e-12), index=cur.index)
    r = res["resid"].assign(reg=res["resid"]["territory_id"].map(region))
    rc = [c for c in r.columns if c.startswith("откл., %: ")]
    shift = r.groupby(["reg", "month"])[rc].median().abs().max(1)
    x = m.assign(corr=pd.MultiIndex.from_frame(m[["territory_id", "month"]]).map(corr.to_dict().get),
                 shift=[shift.get((region.get(t), mm)) for t, mm in zip(m["territory_id"], m["month"])],
                 pop=m["territory_id"].map(pop) / 1000)
    return x.groupby("группа").agg(n=("territory_id", "size"),
                                   **{"похожесть на тот же месяц год назад (peer), медиана": ("corr", "median"),
                                      "сдвиг региона от прогноза, п.п., медиана": ("shift", "median"),
                                      "население, тыс., медиана": ("pop", "median")}).reindex(
        ["оба", "только resid", "только peer", "нет"])


def md_table(df: pd.DataFrame, cols: list[str], n: int) -> str:
    return df[cols].head(n).to_markdown(index=False, floatfmt=".1f")


def main(cfg: dict) -> None:
    out = Path(cfg["output_dir"])
    dst = out / "shocks"
    dst.mkdir(parents=True, exist_ok=True)
    df = load_consumption(cfg["data"]["consumption"])
    ids = complete_territories(df) if cfg["data"]["only_complete"] else None
    cats = cfg["data"]["categories"]
    panels = build_panels(df, cats, ids)
    static = pd.read_parquet(cfg["data"]["external"]["static"]).set_index("territory_id")
    region = static["region"]
    gaz = pd.read_parquet(Path(cfg["data"]["external"]["news"]).with_name("city_gazetteer.parquet"))
    d = pd.read_excel("data/external/raw/t_dict_municipal_districts.xlsx")
    d = d[d["year_to"] >= 2024].drop_duplicates("territory_id")
    pats = mo_patterns(gaz, d, set(panels[cats[0]].territory_ids))
    mo_name = d.set_index("territory_id")["municipal_district_name_short"]
    reg_name = d.drop_duplicates("region_code").set_index("region_code")["region_name"]
    reg_name.index = reg_name.index.astype(str)

    peer_long = standardize(peer_deviations(panels, region))
    vol = own_volatility(peer_long)
    res = {"peer": combine(rescale(peer_long, vol), region),
           "resid": combine(rescale(standardize(resid_deviations(cfg, out, cats)), vol), region)}
    for k, r in res.items():
        print(f"{k}: доля МО-месяцев с оценкой ≥3 / 3,5 / 4 / 5: "
              + " / ".join(f"{(r['score'] >= x).mean():.2%}" for x in (3, 3.5, 4, 5)))
    print("лучшие модели для resid:", best_models(cfg, out))

    covered = set(pats["territory_id"])
    shocked = sorted(set().union(*[set(r.loc[r["shock"], "territory_id"]) for r in res.values()]) & covered)
    print(f"шоков: peer {int(res['peer']['shock'].sum())}, resid {int(res['resid']['shock'].sum())}; "
          f"МО с шаблоном упоминания: {len(covered)} из {len(panels[cats[0]].territory_ids)}, "
          f"среди МО с шоками — {len(shocked)}; ищу новости…")
    art = city_news(shocked, pats)
    total = pq.read_table(NEWS, columns=["pub_date"]).to_pandas()["pub_date"].dt.strftime("%Y-%m").value_counts().sort_index()
    sp = news_spikes(art, total)

    cards = {}
    for k, r in res.items():
        s = r[r["shock"]].sort_values("score", ascending=False)
        s = attach_news(s, art, covered, sp)
        s.insert(1, "МО", s["territory_id"].map(mo_name))
        s.insert(2, "регион", s["region"].map(reg_name))
        r.to_parquet(dst / f"scores_{k}.parquet", index=False)
        s.to_csv(dst / f"shocks_{k}.csv", index=False)
        cards[k] = s

    m, head = compare(res["resid"], res["peer"])
    m.to_csv(dst / "compare.csv", index=False)
    lines = ["# Шоки потребления: два детектора и где они расходятся\n",
             f"Шок — сводная оценка ≥ {THRESHOLD} (среднеквадратичное робастное z по {TOP_K} категориям "
             "с наибольшим отклонением). "
             "resid — отклонение от прогноза лучшей модели на 1 месяц (только 2024), peer — от медианы "
             "прироста МО своего региона (2023-02…2024-12). Подробности — в docstring src/shocks.py.\n",
             *head, ""]

    lines.append("\n## Сколько шоков по месяцам\n")
    per = pd.DataFrame({k: r.groupby("month")["shock"].sum() for k, r in res.items()}).fillna(0).astype(int)
    lines.append(per.T.to_markdown() + "\n")

    lines.append("\n## Объясняются ли шоки новостями\n")
    lines.append("Только МО, у которых есть шаблон упоминания (город-центр или название района). «Базовая частота» — доля всплесков новостей "
                 "среди МО-месяцев без шока (по тем же городам и месяцам, что и шоки).\n")
    t = []
    for k, s in cards.items():
        c = s[s["новости"] != "нет в словаре упоминаний"]
        r = res[k]
        calm = r[~r["shock"] & r["territory_id"].isin(c["territory_id"]) & r["month"].isin(c["month"])]
        base = calm[["territory_id", "month"]].merge(sp, how="left")["spike"].astype(float).fillna(0).mean() * 100
        typed = art.groupby(["territory_id", "month"]).apply(event_type).rename("typ").reset_index()
        base_t = calm[["territory_id", "month"]].merge(typed, how="left")["typ"].fillna("").ne("").mean() * 100
        t.append({"способ": k, "шоков": len(s), "из них с упоминаниями": len(c),
                  "новостной всплеск, %": (c["новости"] == "новостной всплеск").mean() * 100,
                  "базовая частота всплеска, %": base,
                  "есть тип события, %": (c["тип события"] != "").mean() * 100,
                  "базовая частота типа, %": base_t,
                  "региональных шоков, %": (s["масштаб"] == "регион").mean() * 100})
    lines.append(pd.DataFrame(t).round(1).to_markdown(index=False) + "\n")

    cols = ["month", "МО", "регион", "score", "главное", "масштаб", "новости", "статей", "к обычному",
            "тип события", "заголовки"]
    for k, title in (("resid", "resid (2024)"), ("peer", "peer (2023–2024)")):
        lines.append(f"\n## Крупнейшие шоки: {title}\n")
        lines.append(md_table(cards[k], cols, 15) + "\n")

    lines.append("\n## Контрольные случаи\n")
    t = []
    for tid, mm in [(1673, "2024-04"), (1673, "2024-05"), (1665, "2024-04"), (1333, "2024-04")]:
        row = {"МО": mo_name.get(tid), "месяц": mm}
        for k, r in res.items():
            q = r[(r["territory_id"] == tid) & (r["month"] == mm)]
            row[f"оценка {k}"] = q["score"].iloc[0] if len(q) else np.nan
            row[f"главное {k}"] = q["главное"].iloc[0] if len(q) else ""
        t.append(row)
    lines.append("Паводок в Оренбургской и Курганской областях, апрель 2024.\n\n"
                 + pd.DataFrame(t).to_markdown(index=False, floatfmt=".1f") + "\n")

    lines.append("\n## Где способы не совпали (2024)\n")
    pop = np.exp(static["log_pop"])
    lines.append("«Похожесть» — корреляция профиля отклонений МО по категориям (peer) с тем же месяцем 2023 г.: "
                 "высокая — это повторяющаяся сезонность МО, которую модель ожидает. «Сдвиг региона» — "
                 "наибольшее по категориям |медианное| отклонение факта от прогноза по МО региона: высокий — "
                 "весь регион разошёлся с прогнозом, а peer такое поглощает медианой соседей.\n\n"
                 + disagreement(m, res, region, pop).round(2).to_markdown() + "\n")
    for g in ("только resid", "только peer"):
        x = m[m["группа"] == g]
        lines.append(f"\n**{g}: {len(x)}**. По месяцам: "
                     + ", ".join(f"{mm} — {n}" for mm, n in x.groupby("month").size().items()) + ".\n")
        other = "peer" if g == "только resid" else "resid"
        x = x.sort_values(f"score_{g.split()[-1]}", ascending=False).head(10)
        x = x.assign(МО=x["territory_id"].map(mo_name), регион=x["territory_id"].map(region).map(reg_name))
        lines.append(x[["month", "МО", "регион", f"score_{g.split()[-1]}", f"score_{other}",
                        f"главное_{g.split()[-1]}", f"главное_{other}"]].to_markdown(index=False, floatfmt=".1f") + "\n")
    (dst / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/forecast.yaml")
    main(yaml.safe_load(open(p.parse_args().config)))
