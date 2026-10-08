"""Внешние данные: справочник МО СберИндекса и муниципальная статистика Росстата (БДПМО).

Запуск: .venv/bin/python -m src.external
Скачивает источники в data/external/raw/ и собирает:
  data/external/territory_static.parquet  — статические признаки МО (справочник + население);
  data/external/rosstat_quarterly.parquet — квартальные показатели БДПМО нарастающим итогом и их прирост г/г;
  data/external/national_monthly.parquet  — национальные потребительские расходы СберИндекса по типам, помесячно
                                            (для сезонного профиля);
  data/external/region_monthly.parquet    — ИПЦ к прошлому месяцу (всего, продовольствие, непрод., услуги)
                                            и среднемесячная зарплата по субъектам, помесячно.
Отдельный этап: .venv/bin/python -m src.external --only region_monthly

Источники (см. reports/external_data.md):
  справочник МО  — https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities (CC BY-SA 4.0);
  БДПМО Росстата — https://tochno.st/datasets/bdmo, обработка «Если быть точным» (CC BY 4.0);
  национальный ряд — https://sberindex.ru/ru/dashboards/consumer-spending;
  ИПЦ по субъектам — https://rosstat.gov.ru/statistics/price («Индексы потребительских цен ... по субъектам»);
  зарплата по субъектам — ЕМИСС, показатель 57824 (https://www.fedstat.ru/indicator/57824).
"""
from __future__ import annotations

import argparse
import http.cookiejar
import json
import re
import shutil
import ssl
import subprocess
import urllib.request
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/external")
RAW = OUT / "raw"

DICT_URL = "https://www.sberbank.com/common/files/t_dict_municipal.rar"
DICT_XLSX = "t_dict_municipal_districts.xlsx"
BDMO_URL = ("https://storage.yandexcloud.net/tochno-st-catalog/Rosstat/data_bdmo_118_v20250918/"
            "indicators/section{section}/data_Y4{code}_112_v20250918.zip")

TOURISM = ("60", "7000004")      # число ночёвок в коллективных средствах размещения за год
POPULATION = ("41", "8112013")  # среднегодовая численность постоянного населения
# Квартальные показатели (нарастающим итогом, без субъектов МСП): имя -> (раздел, код)
QUARTERLY = {
    "retail": ("2", "8401003"),      # оборот розничной торговли, тыс. руб.
    "catering": ("2", "8401006"),    # оборот общественного питания, тыс. руб.
    "wage": ("32", "8423007"),       # среднемесячная зарплата работников организаций, руб.
    "employment": ("32", "8423005"), # среднесписочная численность работников, чел.
}
NATIONAL_URL = "https://sberindex.ru/api/dataset/v1/consumer-spending"
TOTAL_OKVED = "Всего по обследуемым видам экономической деятельности"
PERIOD_Q = {"Январь-март": 1, "Январь-июнь": 2, "Январь-сентябрь": 3, "Январь-декабрь": 4}
# Файл Росстата обновляется ежемесячно, имя содержит последний месяц; лист на месяц, с 01(2022).
CPI_URL = "https://rosstat.gov.ru/storage/mediabank/ipc_RF_fo_sub_08-2026.xlsx"
CPI_COLS = ["cpi", "food", "nonfood", "services"]  # ИПЦ всего, продовольственные, непродовольственные, услуги
WAGE_INDICATOR = 57824  # ЕМИСС: среднемесячная номинальная начисленная зарплата работающих в экономике с 2017 г.
WAGE_YEARS = range(2019, 2027)
# ЕМИСС: оборот розничной торговли, в т.ч. пищевыми продуктами и непродовольственными товарами, общественного питания
TURNOVER_INDICATORS = {"retail": 31260, "food": 33531, "nonfood": 31261, "catering": 31258}
TURNOVER_YEARS = range(2014, 2025)
MONTHS_RU = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь",
             "ноябрь", "декабрь"]
BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36"
# Регионы справочника, у которых в Росстате другое название или нужен ряд без автономных округов.
OKATO_MANUAL = {29: "11001000", 42: "32000000", 72: "71001000", 83: "11100000", 87: "77000000", 89: "71140000"}
# У МО городов федерального значения в справочнике нет координат центра — берём центр города.
CITY_CENTERS = {"Москва": (55.7558, 37.6173), "Санкт-Петербург": (59.9386, 30.3141), "Севастополь": (44.6167, 33.5254)}


def download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"скачиваю {url}")
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(path)
    return path


def load_dictionary() -> pd.DataFrame:
    """Версии МО: territory_id, oktmo8, year_from, year_to, ... (year_to = 9999 — действует)."""
    xlsx = RAW / DICT_XLSX
    if not xlsx.exists():
        rar = download(DICT_URL, RAW / "t_dict_municipal.rar")
        # bsdtar (macOS) распаковывает RAR5; иначе нужен unar / 7z
        for cmd in (["tar", "-xf", str(rar), "-C", str(RAW), DICT_XLSX], ["unar", "-f", "-o", str(RAW), str(rar)],
                    ["7z", "x", "-y", f"-o{RAW}", str(rar), DICT_XLSX]):
            if shutil.which(cmd[0]) and subprocess.run(cmd, capture_output=True).returncode == 0 and xlsx.exists():
                break
        else:
            raise RuntimeError(f"Не удалось распаковать {rar}: нужен bsdtar, unar или 7z")
    d = pd.read_excel(xlsx)
    d["oktmo8"] = d["oktmo"].str.replace("-", "").str[:8]
    return d


def read_bdmo(section: str, code: str) -> pd.DataFrame:
    """Строки показателя БДПМО по МО верхнего уровня."""
    path = download(BDMO_URL.format(section=section, code=code), RAW / f"data_Y4{code}_112_v20250918.zip")
    stem = f"data_Y4{code}_112_v20250918"
    cols = ["okved2", "mun_level", "oktmo", "oktmo_stable", "year", "indicator_period", "indicator_value"]
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        if f"{stem}.parquet" in names:
            with z.open(f"{stem}.parquet") as f:
                df = pd.read_parquet(f)
        else:
            with z.open(f"{stem}.csv") as f:
                df = pd.read_csv(f, sep=";", usecols=lambda c: c in cols, dtype=str)
    df = df[[c for c in cols if c in df.columns]]  # у населения нет okved2
    df = df[df["mun_level"].str.contains("верхнего", na=False)].drop(columns="mun_level")
    df["year"] = df["year"].astype(int)
    df["indicator_value"] = pd.to_numeric(df["indicator_value"], errors="coerce")
    return df


def attach_territory(df: pd.DataFrame, dct: pd.DataFrame) -> pd.DataFrame:
    """Добавляет territory_id по ОКТМО.

    Сначала — версия справочника, действующая в год наблюдения (справочник покрывает годы до 2024,
    для 2025 берём версии 2024). Для остальных строк — по oktmo_stable, если этот код однозначно
    соответствует одной территории во всех версиях справочника.
    """
    last_year = int(dct["year_from"].max())
    parts = []
    for y, g in df.groupby("year"):
        yy = min(y, last_year)
        v = dct[(dct["year_from"] <= yy) & (dct["year_to"] > yy)][["oktmo8", "territory_id"]]
        parts.append(g.merge(v, left_on="oktmo", right_on="oktmo8", how="left").drop(columns="oktmo8"))
    out = pd.concat(parts, ignore_index=True)

    uniq = dct.groupby("oktmo8")["territory_id"].nunique()
    fallback = dct[dct["oktmo8"].isin(uniq[uniq == 1].index)].drop_duplicates("oktmo8").set_index("oktmo8")["territory_id"]
    miss = out["territory_id"].isna()
    out.loc[miss, "territory_id"] = out.loc[miss, "oktmo_stable"].map(fallback)
    out = out.dropna(subset=["territory_id"])
    out["territory_id"] = out["territory_id"].astype(int)
    return out


def build_static(dct: pd.DataFrame) -> pd.DataFrame:
    """G1 (справочник) + G3 (население) для каждого territory_id, по последней версии МО."""
    last = dct.sort_values("year_from").groupby("territory_id").tail(1).set_index("territory_id")
    st = pd.DataFrame(index=last.index)
    st["region"] = last["region_code"].astype(str)
    st["mo_type"] = last["municipal_district_type"]
    status = last["municipal_district_status"].fillna("")
    st["is_capital"] = (status == "административный_центр_субъекта").astype(int).astype(str)
    st["is_zato"] = (status == "ЗАТО").astype(int).astype(str)
    st["shape"] = last["shape"].astype(str)
    lat, lon = last["municipal_district_center_lat"].copy(), last["municipal_district_center_lon"].copy()
    for city, (la, lo) in CITY_CENTERS.items():
        m = (last["region_name"] == city) & lat.isna()
        lat[m], lon[m] = la, lo
    st["lat"], st["lon"] = lat, lon

    # Население: берём 2022 г. (опубликовано к концу 2023-го — без утечки для теста 2024 г.)
    pop = attach_territory(read_bdmo(*POPULATION), dct)
    pop = pop[pop["indicator_period"] == "Значение показателя за год"]
    pop = pop.groupby(["territory_id", "year"])["indicator_value"].first().unstack()
    st["log_pop"] = np.log(pop.get(2022)).reindex(st.index)
    st["pop_growth"] = np.log(pop.get(2022) / pop.get(2021)).reindex(st.index)
    return st.reset_index()


def build_quarterly(dct: pd.DataFrame) -> pd.DataFrame:
    """G4: показатели нарастающим итогом с начала года по кварталам и их прирост г/г (в логарифмах).

    Прирост считается к тому же периоду прошлого года (янв–мар к янв–мар и т.д.) — так корректно
    и для сумм (оборот), и для средних (зарплата, численность).
    """
    rows = []
    for name, (section, code) in QUARTERLY.items():
        df = read_bdmo(section, code)
        df = df[(df["okved2"] == TOTAL_OKVED) & df["indicator_period"].isin(PERIOD_Q) & (df["indicator_value"] > 0)]
        df = attach_territory(df, dct)
        df["quarter"] = df["indicator_period"].map(PERIOD_Q)
        g = df.groupby(["territory_id", "year", "quarter"])["indicator_value"].first().rename("value").reset_index()
        prev = g.assign(year=g["year"] + 1).rename(columns={"value": "value_prev"})
        g = g.merge(prev, on=["territory_id", "year", "quarter"], how="left")
        g["yoy_log"] = np.log(g["value"] / g["value_prev"])
        rows.append(g.drop(columns="value_prev").assign(indicator=name))
        print(f"  {name}: {g['territory_id'].nunique()} МО, годы {g['year'].min()}–{g['year'].max()}")
    return pd.concat(rows, ignore_index=True)


def fetch_national() -> pd.DataFrame:
    """Национальные потребительские расходы (млрд руб.) по типам трат, помесячно, с 2018-12."""
    # сертификат sberindex.ru выпущен российским УЦ, которого нет в стандартных хранилищах
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(NATIONAL_URL, headers={
        "User-Agent": "Mozilla/5.0", "X-Language": "ru", "RqUID": uuid.uuid4().hex,
        "Accept": "application/json"})
    with urllib.request.urlopen(req, context=ctx) as r:
        d = json.load(r)
    assert d["pagination"]["total_pages"] == 1, "ряд не поместился в одну страницу"
    df = pd.DataFrame(d["data"], columns=d["fields"])
    # период — начало месяца по Москве, записанное в UTC: 2018-11-30T21:00Z = декабрь 2018
    month = (pd.to_datetime(df["period"]) + pd.Timedelta(hours=3)).dt.strftime("%Y-%m")
    return pd.DataFrame({"type": df["type"], "month": month, "value": df["value"].astype(float)})


def _unverified_opener(cookies: bool = False) -> urllib.request.OpenerDirector:
    """Сайты Росстата и ЕМИСС — с сертификатом российского УЦ; ЕМИСС отдаёт выгрузку только в своей сессии."""
    handlers = [urllib.request.HTTPSHandler(context=ssl._create_unverified_context())]
    if cookies:
        handlers.append(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    opener = urllib.request.build_opener(*handlers)
    opener.addheaders = [("User-Agent", BROWSER_UA)]
    return opener


def fetch_regional_cpi() -> pd.DataFrame:
    """ИПЦ по субъектам к предыдущему месяцу, %: okato, name, month, cpi, food, nonfood, services."""
    path = RAW / CPI_URL.rsplit("/", 1)[1]
    if not path.exists():
        print(f"скачиваю {CPI_URL}")
        path.write_bytes(_unverified_opener().open(CPI_URL).read())
    x = pd.ExcelFile(path)
    rows = []
    for sheet in x.sheet_names:
        m = re.fullmatch(r"(\d\d)\((\d{4})\)", sheet)
        if not m:
            continue
        df = x.parse(sheet, header=None).iloc[4:, :6]
        df.columns = ["okato", "name"] + CPI_COLS
        rows.append(df.dropna(subset=["okato"]).assign(month=f"{m.group(2)}-{m.group(1)}"))
    df = pd.concat(rows, ignore_index=True)
    df["okato"] = df["okato"].astype(str).str.strip().str.zfill(8)  # Excel теряет ведущий ноль (Алтайский край — 01…)
    df["name"] = df["name"].astype(str).str.strip()
    df[CPI_COLS] = df[CPI_COLS].apply(pd.to_numeric, errors="coerce")
    return df


def _emiss_sdmx(indicator: int, keep: dict[str, str], path: Path) -> Path:
    """Выгрузка показателя ЕМИСС в SDMX (кэш — path). keep: подстрока заголовка фильтра → regex значений
    (остальные фильтры — все значения). ЕМИСС отдаёт выгрузку только в своей сессии и с token страницы."""
    if path.exists():
        return path
    print(f"выгружаю ЕМИСС {indicator}")
    opener = _unverified_opener(cookies=True)
    page = opener.open(f"https://www.fedstat.ru/indicator/{indicator}").read().decode("utf-8", "ignore")
    token = re.search(r'name="token" value="([^"]+)"', page).group(1)
    # Конфигурация таблицы — JS-объект FGrid: фильтры со значениями и раскладка строк/столбцов
    seg = page[page.find("filters: {"):]
    end = seg.find("left_columns")
    text = re.sub(r"\\u([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), re.sub(r"\s+", " ", seg[:end]))
    heads = list(re.finditer(r"(\d+): \{ title: '([^']*)', all: \w+, values: \{", text))
    selected = []
    for k, h in enumerate(heads):
        body = text[h.end(): heads[k + 1].start() if k + 1 < len(heads) else len(text)]
        rule = next((r for key, r in keep.items() if key in h.group(2)), ".*")
        selected += [f"{h.group(1)}_{v}" for v, title in re.findall(r"(\w+): \{ title: '([^']*)'", body)
                     if re.fullmatch(rule, title.strip())]
    layout = seg[end:end + 400]
    left = re.findall(r"\d+", re.search(r"left_columns: \[([^\]]*)\]", layout).group(1))
    top = re.findall(r"\d+", re.search(r"top_columns: \[([^\]]*)\]", layout).group(1))
    other = [h.group(1) for h in heads if h.group(1) not in left + top]
    data = "&".join([f"id={indicator}", "struts.token.name=token", f"token={token}"]
                    + [f"lineObjectIds={x}" for x in left] + [f"columnObjectIds={x}" for x in top]
                    + [f"selectedFilterIds={x}" for x in selected] + [f"filterObjectIds={x}" for x in other])
    body = opener.open("https://www.fedstat.ru/indicator/downloadData.do?format=sdmx", data=data.encode()).read()
    assert body.lstrip().startswith(b"<?xml"), "ЕМИСС вернула не SDMX"
    path.write_bytes(body)
    return path


def _sdmx_rows(path: Path, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Наблюдения SDMX ЕМИСС: okato (8 знаков), year, period, value и концепты extra."""
    ns = "{http://www.SDMX.org/resources/SDMXML/schemas/v1_0/generic}"
    rows = []
    for _, el in ET.iterparse(path):
        if el.tag == f"{ns}Series":
            key = {v.get("concept"): v.get("value") for v in el.iter(f"{ns}Value")}
            for o in el.iter(f"{ns}Obs"):
                rows.append((key["s_OKATO"][:8], o.find(f"{ns}Time").text, key["PERIOD"],
                             float(o.find(f"{ns}ObsValue").get("value").replace(",", ".")),
                             *[key.get(c) for c in extra]))
            el.clear()
    return pd.DataFrame(rows, columns=["okato", "year", "period", "value", *extra])


def fetch_regional_wages() -> pd.DataFrame:
    """Среднемесячная зарплата по субъектам, руб., всего по экономике: okato, month, wage (выгрузка ЕМИСС в SDMX)."""
    keep = {"ОКВЭД": "Всего по обследуемым.*", "Год": "|".join(map(str, WAGE_YEARS)), "Период": "|".join(MONTHS_RU)}
    df = _sdmx_rows(_emiss_sdmx(WAGE_INDICATOR, keep, RAW / f"emiss_{WAGE_INDICATOR}.xml")).rename(columns={"value": "wage"})
    df["month"] = df["year"] + "-" + df["period"].map({m: f"{i + 1:02d}" for i, m in enumerate(MONTHS_RU)})
    return df[["okato", "month", "wage"]]


def fetch_regional_turnover() -> pd.DataFrame:
    """Оборот розничной торговли (всего, пищевые, непродовольственные) и общественного питания по субъектам,
    млн руб. в месяц: okato, month, retail, food, nonfood, catering (ЕМИСС, TURNOVER_YEARS)."""
    keep = {"Год": "|".join(map(str, TURNOVER_YEARS)), "Период": "|".join(MONTHS_RU), "Хозяйствующие": "Всего"}
    out = None
    for name, ind in TURNOVER_INDICATORS.items():
        df = _sdmx_rows(_emiss_sdmx(ind, keep, RAW / f"emiss_{ind}.xml"))
        df = df[df["period"].isin(MONTHS_RU)]
        df["month"] = df["year"] + "-" + df["period"].map({m: f"{i + 1:02d}" for i, m in enumerate(MONTHS_RU)})
        df = df.groupby(["okato", "month"], as_index=False)["value"].sum().rename(columns={"value": name})
        out = df if out is None else out.merge(df, on=["okato", "month"], how="outer")
    return out


def build_region_monthly(dct: pd.DataFrame) -> pd.DataFrame:
    """ИПЦ и зарплата по регионам справочника (region — region_code как строка, как в territory_static)."""
    cpi, wage = fetch_regional_cpi(), fetch_regional_wages()

    def norm(s):  # «Республика Адыгея (Адыгея)» и «Республика Адыгея» → один ключ
        s = str(s).lower().replace("ё", "е")
        for w in ("республика", "область", "край", "автономный округ", "авт.округ", "город", "г."):
            s = s.replace(w, "")
        return re.sub(r"[^а-я]", "", s)[:12]

    names = cpi.drop_duplicates("okato").assign(key=lambda d: d["name"].map(norm)).drop_duplicates("key")
    reg = dct[["region_code", "region_name"]].drop_duplicates("region_code").assign(key=lambda d: d["region_name"].map(norm))
    reg = reg.merge(names[["key", "okato"]], on="key", how="left")
    reg["okato"] = [OKATO_MANUAL.get(int(c), o) for c, o in zip(reg["region_code"], reg["okato"])]
    assert reg["okato"].notna().all(), f"не сопоставлены: {reg.loc[reg['okato'].isna(), 'region_name'].tolist()}"
    out = (reg[["region_code", "okato"]]
           .merge(cpi[["okato", "month"] + CPI_COLS], on="okato", how="left")
           .merge(wage, on=["okato", "month"], how="outer"))
    out = out[out["okato"].isin(reg["okato"])]
    out["region"] = out["okato"].map(dict(zip(reg["okato"], reg["region_code"].astype(str))))
    return out[["region", "month"] + CPI_COLS + ["wage"]].sort_values(["region", "month"]).reset_index(drop=True)


def build_tourism(dct: pd.DataFrame) -> pd.DataFrame:
    """Ночёвки в гостиницах и других коллективных средствах размещения по МО за год: territory_id, year, nights."""
    t = attach_territory(read_bdmo(*TOURISM), dct)
    return (t.groupby(["territory_id", "year"], as_index=False)["indicator_value"].sum()
            .rename(columns={"indicator_value": "nights"}))


def main(only: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dct = load_dictionary()
    if only == "tourism":
        t = build_tourism(dct)
        t.to_parquet(OUT / "tourism.parquet", index=False)
        print(f"tourism: {t['territory_id'].nunique()} МО, годы {sorted(t['year'].unique())}")
        return
    if only == "region_monthly":
        rm = build_region_monthly(dct)
        rm.to_parquet(OUT / "region_monthly.parquet", index=False)
        print(f"region_monthly: {rm['region'].nunique()} регионов, ИПЦ {rm.dropna(subset=['cpi'])['month'].min()}…"
              f"{rm.dropna(subset=['cpi'])['month'].max()}, зарплата {rm.dropna(subset=['wage'])['month'].min()}…"
              f"{rm.dropna(subset=['wage'])['month'].max()}")
        return
    st = build_static(dct)
    st.to_parquet(OUT / "territory_static.parquet", index=False)
    print(f"territory_static: {len(st)} МО, население есть у {st['log_pop'].notna().sum()}")
    q = build_quarterly(dct)
    q.to_parquet(OUT / "rosstat_quarterly.parquet", index=False)
    print(f"rosstat_quarterly: {len(q)} строк")
    nat = fetch_national()
    nat.to_parquet(OUT / "national_monthly.parquet", index=False)
    print(f"national_monthly: {nat['type'].nunique()} типов, {nat['month'].min()}…{nat['month'].max()}")
    rm = build_region_monthly(dct)
    rm.to_parquet(OUT / "region_monthly.parquet", index=False)
    print(f"region_monthly: {rm['region'].nunique()} регионов")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["region_monthly", "tourism"], help="собрать только этот набор")
    main(p.parse_args().only)
