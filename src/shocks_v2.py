"""Детектор шоков v2: неожиданность прогноза, разложение «регион / МО», три оценки, проверка на искусственных шоках.

Запуск (после src.run_forecast, src.evaluate): .venv/bin/python -m src.shocks_v2 → results/shocks_v2/

Шок — месяц, когда расходы МО разошлись с прогнозом на 1 месяц (модель final, данные строго до этого месяца) сильнее, чем обычно расходятся у этого МО. Шаги:
  1. d = log(факт / прогноз модели final на 1 месяц) по 6 категориям; z — робастно центрированное и нормированное d внутри
     (категория, месяц, группа размера МО), делённое на собственную волатильность МО (src.shocks: own_volatility
     по способу peer, только прошлые месяцы). Медиана по месяцу убирает общий для страны промах прогноза.
  2. Разложение: z_reg — медиана z по МО региона в этот месяц (регион ≥ MIN_REGION МО), z_loc = z − z_reg.
     Сдвиг всего региона — один региональный шок, а не десятки «шоков МО». Затем z_loc делится на разброс прошлых
     z_loc того же МО (adapt): у хронически нестабильных МО тревога не повторяется каждый месяц.
  3. Оценки по вектору z_loc (6 категорий): topk — среднеквадратичное по 3 категориям с наибольшим |z| (как в v1);
     mahal — робастное расстояние Махаланобиса (MinCovDet); iforest — Isolation Forest; combo — максимум
     процентилей topk и mahal. Региональная оценка — topk по z_reg.
  4. Порог — доля ложных тревог ALARM_RATE на месяцах без искусственных шоков; доля найденных искусственных
     шоков разных форм и величин при этом пороге — главный критерий сравнения оценок (calibration.md).
  5. Тип шока — по уровню в следующем месяце: какая доля отклонения осталась (shock_type). Для декабря 2024
     типа нет. Это подтверждение через месяц; сам сигнал — в месяц шока.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest

from src.data import build_panels, complete_territories, load_consumption
from src.shocks import MIN_REGION, own_volatility, peer_deviations, rescale, standardize

Z_CLIP = 8.0
TOP_K = 3
ALARM_RATE = 0.01      # доля МО-месяцев без шока, которые детектор всё равно отметит
REGION_ALARM = 0.02    # то же для региональной оценки (регионов-месяцев мало)
EXPECT = "final"       # модель ожидания (src.final); раньше — лучшая по h=1 на тесте, что менялось от экспериментов
N_INJECT = 400         # искусственных шоков за один прогон проверки (≈1,7% МО-месяцев)
ADAPTIVE = True        # делить z_loc на разброс прошлых z_loc МО (adapt)
ADAPT_MIN = 3          # месяцев истории z_loc для адаптивной нормировки
SEED = 0
OUT = Path("results/shocks_v2")


# ---------- z, разложение, оценки ----------

def deviations(cfg: dict, panels: dict, region: pd.Series, early: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """d по прогнозу (2024; с early — и месяцы 2023 г. из прогнозов с коротким обучением) и волатильность МО
    по способу peer (с 2023 г.)."""
    out = Path(cfg["output_dir"])
    rows = []
    for cat in panels:  # ожидание — итоговая модель final (выбрана по обучающим данным, не меняется от экспериментов)
        fc = pd.read_parquet(out / "forecasts" / EXPECT / f"{cat}.parquet")
        fc = fc[fc["step"] == 1]
        rows.append(pd.DataFrame({
            "territory_id": fc["territory_id"], "month": fc["month"], "category": cat,
            "d": np.log(fc["y_true"].clip(lower=1) / fc["y_pred"].clip(lower=1)),
            "level_prev": fc["y_last"], "expected": EXPECT}))
    resid = pd.concat(rows, ignore_index=True)
    if early is not None:
        from src.final import HETEROGENEOUS, HOMOGENEOUS, THRESHOLD, common_month_r2
        rows = []
        for cat, p in panels.items():
            # модель — по правилу final, R² — по тем же первым 6 месяцам, что видит самый ранний прогноз
            model = HOMOGENEOUS if common_month_r2(p.values[:, :6]) >= THRESHOLD else HETEROGENEOUS
            fc = pd.read_parquet(early / "forecasts" / model / f"{cat}.parquet")
            fc = fc[(fc["step"] == 1) & (fc["month"] < resid["month"].min())]
            rows.append(pd.DataFrame({
                "territory_id": fc["territory_id"], "month": fc["month"], "category": cat,
                "d": np.log(fc["y_true"].clip(lower=1) / fc["y_pred"].clip(lower=1)),
                "level_prev": fc["y_last"], "expected": model}))
        resid = pd.concat(rows + [resid], ignore_index=True)
    vol = own_volatility(standardize(peer_deviations(panels, region)))
    return resid, vol


def adapt(zl: pd.DataFrame) -> pd.DataFrame:
    """z_loc / max(1, робастный разброс прошлых z_loc того же МО и категории); меньше ADAPT_MIN месяцев — без
    изменений. У хронически нестабильных МО тревога не повторяется каждый месяц."""
    out = []
    for c in zl.columns:
        w = zl[c].unstack("month").sort_index(axis=1)
        v, sc = w.values, np.ones(w.shape)
        for j in range(ADAPT_MIN, v.shape[1]):
            past = v[:, :j]
            med = np.nanmedian(past, 1, keepdims=True)
            sc[:, j] = np.maximum(1.4826 * np.nanmedian(np.abs(past - med), 1), 1.0)
        out.append(pd.DataFrame(v / sc, index=w.index, columns=w.columns).stack().rename(c))
    return pd.concat(out, axis=1).reindex(zl.index)


def zscores(resid: pd.DataFrame, vol: pd.DataFrame, region: pd.Series) -> pd.DataFrame:
    """МО-месяц × категория: z, z_reg, z_loc (широкие таблицы с MultiIndex столбцов)."""
    long = rescale(standardize(resid), vol)
    z = long.pivot_table(index=["territory_id", "month"], columns="category", values="z").clip(-Z_CLIP, Z_CLIP)
    reg = z.index.get_level_values("territory_id").map(region)
    month = z.index.get_level_values("month")
    size = pd.Series(1, index=z.index).groupby([reg, month]).transform("size").values
    z_reg = z.groupby([reg, month]).transform("median")
    z_reg[size < MIN_REGION] = 0.0
    z_loc = z - z_reg
    return pd.concat({"z": z, "z_reg": z_reg, "z_loc": adapt(z_loc) if ADAPTIVE else z_loc}, axis=1)


def window(zl: pd.DataFrame) -> pd.DataFrame:
    """Местные z за два месяца подряд: (z_t + z_{t−1}) / √2 по каждой категории (первый месяц — только z_t)."""
    months = sorted(zl.index.get_level_values("month").unique())
    nxt = dict(zip(months[:-1], months[1:]))
    lag = zl.copy()
    lag.index = lag.index.set_levels([nxt.get(m, "—") for m in lag.index.levels[1]], level="month", verify_integrity=False)
    return (zl + lag.groupby(level=[0, 1]).first().reindex(zl.index).fillna(0.0)) / np.sqrt(2)


def topk(Z: np.ndarray) -> np.ndarray:
    top = -np.sort(-np.abs(np.nan_to_num(Z)), axis=1)[:, :TOP_K]
    return np.sqrt((top ** 2).mean(1))


def scores(Z: np.ndarray) -> dict[str, np.ndarray]:
    """Три оценки по матрице z_loc (строки — МО-месяцы, столбцы — категории); NaN → 0 («как ожидалось»)."""
    X = np.nan_to_num(Z)
    mcd = MinCovDet(random_state=SEED, support_fraction=0.9).fit(X)
    iso = IsolationForest(n_estimators=300, max_samples=4096, random_state=SEED).fit(X)
    out = {"topk": topk(Z), "mahal": np.sqrt(mcd.mahalanobis(X)), "iforest": -iso.score_samples(X)}
    # combo — максимум из процентилей topk и mahal: topk сильнее на общих сдвигах, mahal — на смене структуры
    out["combo"] = np.maximum(pd.Series(out["topk"]).rank(pct=True).values, pd.Series(out["mahal"]).rank(pct=True).values)
    return out


# ---------- проверка на искусственных шоках ----------

SCENARIOS = {  # имя -> {категория: множитель величины x}; величина x — падение (или рост для «+»)
    "все категории −x": {c: -1.0 for c in ["Все категории", "Продовольствие", "Здоровье", "Маркетплейсы",
                                           "Общественное питание", "Транспорт"]},
    "одна категория −x": None,  # случайная категория
    "ЧС: продукты +x/2, маркетплейсы, общепит −x, всего −x/3": {
        "Продовольствие": 0.5, "Маркетплейсы": -1.0, "Общественное питание": -1.0, "Транспорт": -0.5,
        "Все категории": -1 / 3},
}
MAGNITUDES = [0.05, 0.10, 0.20, 0.30]


def inject(resid: pd.DataFrame, keys: pd.DataFrame, scenario: str, x: float, rng) -> pd.DataFrame:
    """Сдвиг d в выбранных МО-месяцах: d += log(1 + множитель·x). Прогноз на этот месяц не меняется — он сделан
    по данным до месяца, поэтому для оценки в месяц шока подмена факта точна."""
    shift = []
    cats = sorted(resid["category"].unique())
    for tid, month in keys.itertuples(index=False):
        mult = SCENARIOS[scenario] or {cats[rng.integers(len(cats))]: -1.0}
        shift += [(tid, month, c, np.log1p(m * x)) for c, m in mult.items()]
    s = pd.DataFrame(shift, columns=["territory_id", "month", "category", "shift"])
    r = resid.merge(s, on=["territory_id", "month", "category"], how="left")
    r["d"] = r["d"] + r["shift"].fillna(0.0)
    return r.drop(columns="shift")


def calibrate(resid: pd.DataFrame, vol: pd.DataFrame, region: pd.Series) -> pd.DataFrame:
    """Вставки и пороги — только в месяцах теста (с FIRST_TEST); более ранние месяцы — история для adapt."""
    rng = np.random.default_rng(SEED)
    base = resid.loc[resid["month"] >= FIRST_TEST, ["territory_id", "month"]].drop_duplicates()
    rows = []
    for scenario in SCENARIOS:
        for x in MAGNITUDES:
            keys = base.sample(N_INJECT, random_state=int(rng.integers(1e9)))
            zz = zscores(inject(resid, keys, scenario, x, rng), vol, region)
            sc = scores(zz["z_loc"].values)
            hit = zz.index.isin(pd.MultiIndex.from_frame(keys))
            clean = ~hit & (zz.index.get_level_values("month") >= FIRST_TEST)
            for name, s in sc.items():
                thr = np.quantile(s[clean], 1 - ALARM_RATE)
                rows.append({"сценарий": scenario, "x": x, "оценка": name, "найдено, %": 100 * (s[hit] >= thr).mean()})
            # без разложения на регион: та же topk по полному z
            s = topk(zz["z"].values)
            thr = np.quantile(s[clean], 1 - ALARM_RATE)
            rows.append({"сценарий": scenario, "x": x, "оценка": "topk без разложения",
                         "найдено, %": 100 * (s[hit] >= thr).mean()})
    # региональный шок: все МО случайного региона −x по всем категориям
    regs = region[region.index.isin(base["territory_id"])].value_counts()
    regs = regs[regs >= MIN_REGION].index
    for x in MAGNITUDES:
        hits = []
        for _ in range(20):
            r, month = regs[rng.integers(len(regs))], sorted(base["month"].unique())[rng.integers(12)]
            keys = base[(base["month"] == month) & base["territory_id"].map(region).eq(r)]
            zz = zscores(inject(resid, keys, "все категории −x", x, rng), vol, region)
            reg_score = pd.Series(topk(zz["z_reg"].values), index=zz.index)
            per_region = reg_score.groupby([zz.index.get_level_values("territory_id").map(region),
                                            zz.index.get_level_values("month")]).first()
            pr = per_region[per_region.index.get_level_values(1) >= FIRST_TEST]
            thr = np.quantile(pr.drop((r, month)), 1 - REGION_ALARM)
            loc = topk(zz["z_loc"].values)
            hit = zz.index.isin(pd.MultiIndex.from_frame(keys))
            clean = ~hit & (zz.index.get_level_values("month") >= FIRST_TEST)
            full = topk(zz["z"].values)
            hits.append((per_region[(r, month)] >= thr,
                         (loc[hit] >= np.quantile(loc[clean], 1 - ALARM_RATE)).mean(),
                         (full[hit] >= np.quantile(full[clean], 1 - ALARM_RATE)).mean()))
        h = np.array(hits, dtype=float)
        rows += [{"сценарий": "весь регион −x", "x": x, "оценка": "региональная", "найдено, %": 100 * h[:, 0].mean()},
                 {"сценарий": "весь регион −x", "x": x, "оценка": "topk по z_loc (МО региона)", "найдено, %": 100 * h[:, 1].mean()},
                 {"сценарий": "весь регион −x", "x": x, "оценка": "topk без разложения (МО региона)",
                  "найдено, %": 100 * h[:, 2].mean()}]
    return pd.DataFrame(rows)


def recall_by_size(resid: pd.DataFrame, vol: pd.DataFrame, region: pd.Series) -> pd.DataFrame:
    """Доля найденных искусственных шоков (оценка SCORE, 1% ложных тревог) по квинтилям расходов на жителя МО."""
    rng = np.random.default_rng(SEED + 1)
    base = resid.loc[resid["month"] >= FIRST_TEST, ["territory_id", "month"]].drop_duplicates()
    lvl = resid[resid["category"] == "Все категории"].groupby("territory_id")["level_prev"].median()
    size = pd.qcut(lvl.rank(method="first"), 5, labels=["1 (низкие)", "2", "3", "4", "5 (высокие)"])
    rows = []
    for scenario in ["все категории −x", "одна категория −x"]:
        for x in (0.10, 0.20):
            keys = base.sample(2 * N_INJECT, random_state=int(rng.integers(1e9)))
            zz = zscores(inject(resid, keys, scenario, x, rng), vol, region)
            s = scores(zz["z_loc"].values)[SCORE]
            hit = zz.index.isin(pd.MultiIndex.from_frame(keys))
            in_test = zz.index.get_level_values("month") >= FIRST_TEST
            thr = np.quantile(s[~hit & in_test], 1 - ALARM_RATE)
            q = zz.index.get_level_values("territory_id").map(size)
            for b in size.cat.categories:
                m = hit & (q == b)
                rows.append({"сценарий": scenario, "x": x, "расходы на жителя": b, "найдено, %": 100 * (s[m] >= thr).mean(),
                             "ложных тревог, %": 100 * (s[~hit & in_test & (q == b)] >= thr).mean()})
    return pd.DataFrame(rows)


# ---------- тип шока ----------

def shock_type(flagged: pd.DataFrame, d: pd.DataFrame, panels: dict) -> pd.Series:
    """Тип по главной категории: сколько отклонения осталось в следующем месяце.

    Без шока в t+1 было бы log ŷ_t + g, где ŷ_t = y_t / exp(d_t) — прогноз на t, g — медианный прирост
    t → t+1 по МО этой категории. Доля p = (log y_{t+1} − log ŷ_t − g) / d_t: p ≥ 1,5 — «нарастающий»,
    p ≥ 0,5 — «сдвиг уровня», |p| < 0,5 — «разовый», p ≤ −0,5 — «провал с отскоком» (или всплеск с провалом).
    """
    logy = {c: pd.DataFrame(np.log(np.clip(p.values, 1, None)), index=p.territory_ids, columns=p.months)
            for c, p in panels.items()}
    growth = {c: L.diff(axis=1).median() for c, L in logy.items()}
    out = []
    for (tid, month), row in flagged.iterrows():
        c = row["главная категория"]
        L = logy[c]
        j = L.columns.get_loc(month)
        if j + 1 >= L.shape[1]:
            out.append("")
            continue
        d0 = d.at[(tid, month), c]
        p_ = (L.at[tid, L.columns[j + 1]] - (L.at[tid, month] - d0) - growth[c].iloc[j + 1]) / d0
        out.append("нарастающий" if p_ >= 1.5 else "сдвиг уровня" if p_ >= 0.5 else
                   "разовый" if p_ > -0.5 else "провал с отскоком" if d0 < 0 else "всплеск с провалом")
    return pd.Series(out, index=flagged.index)


def main(cfg: dict, recalibrate: bool = False, early: Path | None = None, n_news: int = 0) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_consumption(cfg["data"]["consumption"])
    ids = complete_territories(df)
    panels = build_panels(df, cfg["data"]["categories"], ids)
    st = pd.read_parquet(cfg["data"]["external"]["static"]).set_index("territory_id")
    region = st["region"].reindex(ids)
    print("модель ожидания:", EXPECT)
    resid, vol = deviations(cfg, panels, region, early)

    if not (OUT / "calibration.csv").exists() or recalibrate:
        cal = calibrate(resid, vol, region)
        cal.to_csv(OUT / "calibration.csv", index=False)
    cal = pd.read_csv(OUT / "calibration.csv")
    piv = cal.pivot_table(index=["сценарий", "оценка"], columns="x", values="найдено, %").round(1)
    piv.columns = [f"x={c:.0%}" for c in piv.columns]
    print(piv.to_string())

    if not (OUT / "recall_by_size.csv").exists() or recalibrate:
        recall_by_size(resid, vol, region).to_csv(OUT / "recall_by_size.csv", index=False)
    rbs = pd.read_csv(OUT / "recall_by_size.csv")
    rbs_t = rbs.pivot_table(index="расходы на жителя", columns=["сценарий", "x"], values="найдено, %").round(1)
    fa_t = rbs.groupby("расходы на жителя")["ложных тревог, %"].mean().round(2)
    rbs_t["ложных тревог, %"] = fa_t
    print(rbs_t.to_string())

    names = load_names(ids)
    zz = zscores(resid, vol, region)
    d = resid.pivot_table(index=["territory_id", "month"], columns="category", values="d").reindex(zz.index)
    sc = scores(zz["z_loc"].values)
    res = pd.DataFrame({k: v for k, v in sc.items()}, index=zz.index)
    res["score"] = res[SCORE]
    # порог — по 2024 г., где доля ложных тревог проверена; к месяцам 2023 г. (прогноз с обучением < 12 мес.)
    # применяется тот же порог, поэтому там отмечается больше МО-месяцев — часть из них лишние тревоги
    in_test = res.index.get_level_values("month") >= FIRST_TEST
    q_shock = res.loc[in_test, "score"].quantile(1 - ALARM_RATE)
    q_watch = res.loc[in_test, "score"].quantile(1 - 3 * ALARM_RATE)
    res["надёжность"] = np.where(in_test, "", "ниже: обучение < 12 мес.")
    res["уровень"] = np.where(res["score"] >= q_shock, "шок", np.where(res["score"] >= q_watch, "внимание", ""))
    # постепенный сдвиг: оценка по окну из двух месяцев в верхнем ALARM_RATE, а по одному месяцу — нет
    # (на искусственных шоках 2 × −5% окно находит 48% против 28%, но резкие шоки −10% — 38% против 78%)
    res["score_window"] = scores(window(zz["z_loc"]).values)[SCORE]
    q_win = res.loc[in_test, "score_window"].quantile(1 - ALARM_RATE)
    res["постепенный"] = (res["score_window"] >= q_win) & (res["уровень"] != "шок")
    zl = zz["z_loc"]
    res["главная категория"] = zl.abs().idxmax(axis=1)
    res["главное"] = [drivers(zl.loc[i], d.loc[i]) for i in zl.index]
    flagged = res[res["уровень"] == "шок"]
    res.loc[flagged.index, "тип"] = shock_type(flagged, d, panels)
    res = res.join(zl.round(2).add_prefix("z: ")).reset_index()
    res["МО"] = res["territory_id"].map(names["name"])
    res["регион"] = res["territory_id"].map(names["region_name"])
    res.to_parquet(OUT / "scores.parquet", index=False)
    cols = (["month", "МО", "регион", "territory_id", "уровень", "постепенный", "score", "главная категория", "главное",
             "тип", "надёжность"]
            + [f"z: {c}" for c in zl.columns])
    (res[(res["уровень"] != "") | res["постепенный"]][cols].assign(score=lambda x: (100 * x["score"]).round(2))
     .rename(columns={"score": "оценка (процентиль)"})
     .sort_values(["уровень", "оценка (процентиль)"], ascending=[False, False])
     .to_csv(OUT / "shocks.csv", index=False))

    reg = regional(zz, region, names)
    reg.to_csv(OUT / "regional.csv", index=False)
    news = news_check(res, cfg)
    heads = None
    if n_news:
        heads = headlines(res, cfg, n_news)
        heads.to_csv(OUT / "top_shocks_news.csv", index=False)
    elif (OUT / "top_shocks_news.csv").exists():
        heads = pd.read_csv(OUT / "top_shocks_news.csv")
    report(piv, res, reg, news, control(res), rosstat_check(res, cfg), rbs_t, heads)
    from src.shocks_v2_plots import main as plots
    plots()


FIRST_TEST = "2024-01"
SCORE = "combo"  # выбрана по calibration.md: не хуже лучшей из topk / mahal во всех сценариях


def load_names(ids: np.ndarray) -> pd.DataFrame:
    d = pd.read_excel("data/external/raw/t_dict_municipal_districts.xlsx",
                      usecols=["territory_id", "municipal_district_name_short", "region_name", "year_from"])
    d = d.sort_values("year_from").groupby("territory_id").tail(1).set_index("territory_id")
    return d.rename(columns={"municipal_district_name_short": "name"}).reindex(ids)


def drivers(z: pd.Series, d: pd.Series) -> str:
    top = z.abs().sort_values(ascending=False).index[:3]
    return "; ".join(f"{c} {np.expm1(d[c]) * 100:+.0f}%" for c in top if abs(z[c]) >= 2)


def regional(zz: pd.DataFrame, region: pd.Series, names: pd.DataFrame) -> pd.DataFrame:
    """Регион-месяц: оценка по z_reg (одинакова у всех МО региона), доля МО с местным z того же знака."""
    tid = zz.index.get_level_values("territory_id")
    key = [tid.map(region), zz.index.get_level_values("month")]
    zr = zz["z_reg"].groupby(key).first()
    size = zz["z"].groupby(key).size()
    out = pd.DataFrame({"score": topk(zr.values), "МО": size.values}, index=zr.index)
    out = out[out["МО"] >= MIN_REGION]
    out["главное"] = [", ".join(f"{c} {v:+.1f}σ" for c, v in r.sort_values(key=abs, ascending=False)[:3].items()
                                if abs(v) >= 1) for _, r in zr.loc[out.index].iterrows()]
    thr = out["score"].quantile(1 - REGION_ALARM)
    out["шок"] = out["score"] >= thr
    rn = names.drop_duplicates("region_name").assign(region=lambda x: x.index.map(region)).set_index("region")
    out = out.reset_index().rename(columns={"level_0": "region", "level_1": "month"})
    out.columns = ["region", "month"] + list(out.columns[2:])
    out["регион"] = out["region"].map(rn["region_name"])
    # прогноз на январь 2024 строился без января в обучении: региональный «шок» в январе — скорее всего своя
    # сезонность региона (глубже новогодний провал; виден и по более сильному февральскому отскоку 2023 г.)
    out["примечание"] = np.where(out["month"] == FIRST_TEST, "модель не видела января — вероятно, сезонность региона", "")
    return out.sort_values("score", ascending=False)


def news_check(res: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Доля МО-месяцев со всплеском новостей о ЧС/отключениях среди шоков и среди остальных (те же МО с новостями)."""
    path = cfg["data"]["external"].get("news")
    if not path or not Path(path).exists():
        return pd.DataFrame()
    cm = pd.read_parquet(path)
    cm["событие"] = ((cm["spike_emerg"] >= 1) & (cm["n_emerg"] >= 3)) | ((cm["spike_outage"] >= 1) & (cm["n_outage"] >= 3))
    m = res.merge(cm[["territory_id", "month", "событие"]], on=["territory_id", "month"], how="inner")
    rows = []
    for lvl, g in [("шок", m[m["уровень"] == "шок"]), ("внимание", m[m["уровень"] == "внимание"]),
                   ("остальные", m[m["уровень"] == ""])]:
        rows.append({"группа": lvl, "МО-месяцев": len(g), "со всплеском новостей, %": round(100 * g["событие"].mean(), 2)})
    return pd.DataFrame(rows)


def rosstat_check(res: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Росстат (БДПМО, крупные и средние организации): квартальный оборот г/г к медиане региона у МО с шоком
    вниз / вверх в соответствующей категории и у остальных МО."""
    q = pd.read_parquet(cfg["data"]["external"]["quarterly"])
    reg = pd.read_parquet(cfg["data"]["external"]["static"]).set_index("territory_id")["region"]
    r = res[res["month"].str[:4] == "2024"]  # кварталы Росстата — 2024 г.; шоки 2023 г. (--early) не сопоставляются
    r = r.assign(quarter=(r["month"].str[5:7].astype(int) - 1) // 3 + 1)
    rows = []
    for ind, cat in [("retail", "Продовольствие"), ("retail", "Все категории"), ("catering", "Общественное питание")]:
        g = q[q["indicator"] == ind].sort_values(["territory_id", "year", "quarter"]).copy()
        g["flow"] = g.groupby(["territory_id", "year"])["value"].diff().fillna(g["value"])  # из нарастающего итога
        prev = g[["territory_id", "year", "quarter", "flow"]].assign(year=lambda x: x["year"] + 1)
        g = g.merge(prev.rename(columns={"flow": "flow_prev"}), on=["territory_id", "year", "quarter"])
        g = g[(g["year"] == 2024) & (g["flow"] > 0) & (g["flow_prev"] > 0)]
        g["yoy"] = np.log(g["flow"] / g["flow_prev"])
        g["yoy_rel"] = g["yoy"] - g.groupby([g["territory_id"].map(reg), "quarter"])["yoy"].transform("median")
        sh = r[(r["уровень"] == "шок") & (r["главная категория"] == cat)]
        sign = np.sign(sh["главное"].str.extract(cat + r" ([+-]\d+)%")[0].astype(float))
        sh = sh.assign(sign=sign)[["territory_id", "quarter", "sign"]].drop_duplicates(["territory_id", "quarter"])
        m = g.merge(sh, on=["territory_id", "quarter"], how="left")
        for lab, x in [("шок вниз", m[m["sign"] < 0]), ("шок вверх", m[m["sign"] > 0]), ("без шока", m[m["sign"].isna()])]:
            rows.append({"Росстат": "розница" if ind == "retail" else "общепит", "категория шока": cat, "группа": lab,
                         "МО-кварталов": len(x), "г/г к региону, медиана, п.п.": round(100 * x["yoy_rel"].median(), 1)})
    return pd.DataFrame(rows)


def headlines(res: pd.DataFrame, cfg: dict, n: int) -> pd.DataFrame:
    """Заголовки новостей о МО в месяц шока для n крупнейших шоков (поиск по корпусу — функции src.shocks)."""
    import pyarrow.parquet as pq

    from src.shocks import NEWS, attach_news, city_news, mo_patterns, news_spikes
    sh = res[res["уровень"] == "шок"].sort_values("score", ascending=False).head(n)
    gaz = pd.read_parquet(Path(cfg["data"]["external"]["news"]).with_name("city_gazetteer.parquet"))
    d = pd.read_excel("data/external/raw/t_dict_municipal_districts.xlsx")
    d = d[d["year_to"] >= 2024].drop_duplicates("territory_id")
    pats = mo_patterns(gaz, d, set(res["territory_id"]))
    covered = set(pats["territory_id"])
    tids = sorted(set(sh["territory_id"]) & covered)
    print(f"новости: {len(tids)} МО из {len(sh)} шоков с шаблоном упоминания; ищу…")
    art = city_news(tids, pats)
    total = (pq.read_table(NEWS, columns=["pub_date"]).to_pandas()["pub_date"].dt.strftime("%Y-%m")
             .value_counts().sort_index())
    return attach_news(sh, art, covered, news_spikes(art, total))


CONTROL = [("Орск", "2024-04"), ("Орск", "2024-05"), ("Оренбург", "2024-04"), ("Курган", "2024-04")]


def control(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, month in CONTROL:
        g = res[(res["МО"] == name) & (res["month"] == month)]
        for _, r in g.iterrows():
            rows.append({"МО": name, "месяц": month, "оценка (процентиль)": round(100 * r["score"], 1),
                         "уровень": r["уровень"], "главное": r["главное"]})
    return pd.DataFrame(rows)


def report(piv, res, reg, news, ctrl, ros, rbs, heads=None) -> None:
    sh = res[res["уровень"] == "шок"]
    cols = ["month", "МО", "регион", "score", "главное", "тип"]
    lines = ["# Шоки v2: неожиданность прогноза, разложение «регион / МО», проверка на искусственных шоках\n",
             "Как устроено — в docstring `src/shocks_v2.py`. Оценка — `combo` (процентиль), «шок» — верхний "
             f"{ALARM_RATE:.0%} МО-месяцев 2024 г., «внимание» — следующие {2 * ALARM_RATE:.0%}.\n",
             "## Проверка на искусственных шоках\n",
             f"Доля найденных шоков, %, при пороге с {ALARM_RATE:.0%} ложных тревог; {N_INJECT} шоков на прогон. "
             "«Весь регион» — 20 случайных регион-месяцев, региональная оценка — с 2% ложных тревог среди "
             "регион-месяцев.\n",
             piv.to_markdown(), "", "![Доля найденных шоков](recall.png)", "",
             "По уровню расходов на жителя (квинтили медианы «Все категории»; оценка итоговая), "
             "доля найденных, %, и доля ложных тревог в квинтиле при общем пороге:\n",
             rbs.to_markdown(), "",
             "## Сколько шоков\n",
             f"Шоков МО: {len(sh)}, по типам: " + ", ".join(f"{k} — {v}" for k, v in sh["тип"].value_counts().items()),
             f"Региональных шоков: {int(reg['шок'].sum())} из {len(reg)} регион-месяцев.",
             f"Постепенных сдвигов (окно из двух месяцев, вне списка шоков), 2024 г.: "
             f"{int(res.loc[res['month'] >= FIRST_TEST, 'постепенный'].sum())}.\n",
             "По месяцам (шоки МО). Порог — по 2024 г.; месяцы 2023 г. (только с --early) — прогноз с обучением "
             "6–11 мес., шоков там больше, часть из них — лишние тревоги:\n", sh["month"].value_counts().sort_index().to_frame("шоков").T.to_markdown(), "",
             "![Шоки по месяцам и типу](by_month.png)", "", "![Карта шоков](map.png)", "",
             "## Контрольные случаи (паводок в Оренбургской и Курганской областях, апрель 2024)\n",
             ctrl.to_markdown(index=False) if len(ctrl) else "нет данных", "",
             "## Новости\n",
             "Всплеск новостей о ЧС или отключениях (≥3 статей и ≥e× обычного) в тот же месяц, только МО с городом "
             "в словаре.\n", news.to_markdown(index=False) if len(news) else "нет данных", "",
             "## Росстат\n",
             "Квартальный оборот (крупные и средние организации) г/г к медиане своего региона в квартале шока. "
             "Выборки малы — это проверка направления, а не точности.\n", ros.to_markdown(index=False), "",
             "## Крупнейшие шоки МО, 2024 г.\n",
             sh[sh["month"] >= FIRST_TEST].sort_values("score", ascending=False)[cols].head(25).round(4).to_markdown(index=False), "",
             "## Новости к крупнейшим шокам\n",
             ("Поиск упоминаний МО (город-центр или «… район/округ») в корпусе новостей за месяц шока; "
              "«к обычному» — число статей к медиане других месяцев.\n\n"
              + heads[["month", "МО", "регион", "главное", "новости", "статей", "тип события", "заголовки"]]
              .assign(заголовки=lambda x: x["заголовки"].fillna("").str.slice(0, 160)).to_markdown(index=False))
             if heads is not None else "не искались (`--news N`)", "",
             "## Крупнейшие региональные шоки\n",
             reg[reg["шок"]][["month", "регион", "score", "МО", "главное", "примечание"]].head(20).round(2)
             .to_markdown(index=False)]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-12:]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/forecast.yaml")
    p.add_argument("--recalibrate", action="store_true", help="пересчитать проверку на искусственных шоках (~6 мин)")
    p.add_argument("--early", type=Path, help="папка прогнозов с коротким обучением (configs/early.yaml): "
                                              "добавить месяцы 2023 г.")
    p.add_argument("--news", type=int, default=0, help="искать заголовки новостей для N крупнейших шоков (~минуты)")
    a = p.parse_args()
    main(yaml.safe_load(open(a.config)), a.recalibrate, a.early, a.news)
