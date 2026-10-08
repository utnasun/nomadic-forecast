"""Данные для страницы проекта (GitHub Pages, папка docs/).

Запуск (после src.final, src.shocks_v2, src.shock_benchmark, src.forecast_2025):
  .venv/bin/python -m tools.site_data
Пишет:
  docs/data/summary.json   — метрики моделей по категориям и горизонтам, стенд детекторов, точки МО для карты;
  docs/data/mo/<регион>.json — по каждому МО региона: факт 2023–2024, прогноз на 2025 г. с 90%-интервалом, шоки.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("docs/data")
CATS = ["Все категории", "Продовольствие", "Маркетплейсы", "Транспорт", "Здоровье", "Общественное питание"]
HORIZONS = [1, 3, 6, 12]

# модели на странице: внутреннее имя → (подпись, группа)
MODELS = {
    "naive": ("«как в прошлом месяце»", "простая"),
    "mean_3": ("среднее за 3 месяца", "простая"),
    "seasonal_naive_growth": ("«как год назад» × рост", "простая"),
    "prophet_default": ("Prophet", "Prophet"),
    "prophet_yearly": ("Prophet с годовой сезонностью", "Prophet"),
    "auto_ets": ("ETS", "статистическая"),
    "auto_theta": ("Theta", "статистическая"),
    "auto_arima": ("ARIMA", "статистическая"),
    "chronos2": ("Chronos-2", "foundation"),
    "chronos2_nat": ("Chronos-2 + национальный ряд", "foundation"),
    "ridge_diff": ("Ridge на признаках", "глобальная"),
    "catboost_diff": ("CatBoost, версия Б", "глобальная"),
    "catboost_logdiff_dsp_g123": ("CatBoost, версия А", "глобальная"),
    "final": ("final (итоговая)", "итоговая"),
}

DETECTORS = {
    "jump": "1. скачок к прошлому месяцу",
    "peer": "2. отклонение от региона",
    "level": "3а. уровень к среднему за 6 мес.",
    "cusum": "3б. CUSUM",
    "resid": "4. отклонение от прогноза",
    "v2_topk": "5. Разложение, topk",
    "v2_mahal": "5. Разложение, Махаланобис",
    "v2_iforest": "5. Разложение, Isolation Forest",
    "v2_combo": "5. Разложение, combo (выбран)",
}


def r(x, nd=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def model_metrics() -> dict:
    m = pd.read_csv("results/metrics_overall.csv")
    m = m[m.model.isin(MODELS) & m.horizon.isin(HORIZONS)]
    rows = []
    for _, x in m.iterrows():
        rows.append({"model": x.model, "category": x.category, "h": int(x.horizon),
                     "MAE": r(x.MAE), "WAPE": r(x.WAPE, 2), "R2": r(x.R2, 4), "R2g": r(x.R2_growth, 4)})
    return {"models": [{"id": k, "label": v[0], "group": v[1]} for k, v in MODELS.items()], "rows": rows}


def detectors() -> dict:
    b = pd.read_csv("results/shock_benchmark.csv")
    b = b[b["детектор"].isin(DETECTORS)]
    mean = b.groupby(["детектор", "x", "за k мес."])["найдено, %"].mean().reset_index()
    rows = [{"det": d, "x": float(x), "k": int(k), "recall": r(v)} for d, x, k, v in mean.itertuples(index=False)]
    by_scn = b[(b.x == 0.1) & (b["за k мес."] == 0)]
    scn = [{"det": d, "scenario": s, "recall": r(v)}
           for d, s, v in by_scn[["детектор", "сценарий", "найдено, %"]].itertuples(index=False)]
    return {"detectors": [{"id": k, "label": v} for k, v in DETECTORS.items()], "rows": rows, "by_scenario": scn}


def main() -> None:
    (OUT / "mo").mkdir(parents=True, exist_ok=True)
    d = pd.read_excel("data/external/raw/t_dict_municipal_districts.xlsx")
    d = d.sort_values("year_to").drop_duplicates("territory_id", keep="last").set_index("territory_id")

    c = pd.read_parquet("consumption.parquet")
    c["date"] = c["date"].astype(str).str[:7]
    full = c.groupby("territory_id").size()
    ids = sorted(full[full == 24 * 6].index)          # 2016 МО с полными рядами
    months = sorted(c["date"].unique())
    hist = c[c.territory_id.isin(ids)].pivot_table(index=["territory_id", "category"], columns="date",
                                                   values="value")[months]

    fc = pd.concat([pd.read_parquet(f"results/forecast_2025/{k}.parquet") for k in CATS])
    fc = fc.sort_values(["territory_id", "category", "step"])

    sh = pd.read_csv("results/shocks_v2/shocks.csv")
    sh = sh[sh.territory_id.isin(ids)]

    # прирост «Все категории» 2025 к 2024 г. по сумме за год
    tot = fc[fc.category == "Все категории"].groupby("territory_id")["y_pred"].sum()
    h24 = hist.xs("Все категории", level="category")[[m for m in months if m.startswith("2024")]].sum(axis=1)
    growth = (tot / h24 - 1) * 100
    n_shock = sh[(sh["уровень"] == "шок") & sh.month.str.startswith("2024")].groupby("territory_id").size()

    points = []
    for t in ids:
        x = d.loc[t]
        points.append([int(t), r(x.municipal_district_center_lat, 3), r(x.municipal_district_center_lon, 3),
                       x.municipal_district_name_short, int(x.region_code), r(growth.get(t), 1),
                       int(n_shock.get(t, 0))])
    regions = {int(k): v for k, v in d.loc[ids].groupby("region_code")["region_name"].first().items()}

    fc_g = {k: g for k, g in fc.groupby("territory_id")}
    sh_g = {k: g for k, g in sh.groupby("territory_id")}
    for reg in sorted(set(int(d.loc[t].region_code) for t in ids)):
        payload = {}
        for t in [t for t in ids if int(d.loc[t].region_code) == reg]:
            f = fc_g[t]
            item = {"name": d.loc[t].municipal_district_name, "hist": {}, "fc": {}, "shocks": []}
            for k in CATS:
                item["hist"][k] = [int(v) for v in hist.loc[(t, k)].values]
                g = f[f.category == k]
                item["fc"][k] = [[int(a), int(b), int(e)] for a, b, e in g[["y_pred", "lo90", "hi90"]].values]
            for _, s in sh_g.get(t, pd.DataFrame()).iterrows():
                txt = lambda v: None if pd.isna(v) else str(v)
                item["shocks"].append({"month": s.month, "level": txt(s["уровень"]), "type": txt(s["тип"]),
                                       "main": txt(s["главное"]), "gradual": bool(s["постепенный"])})
            payload[int(t)] = item
        (OUT / "mo" / f"{reg}.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    summary = {"months": months, "months_2025": [f"2025-{i:02d}" for i in range(1, 13)], "categories": CATS,
               "forecast": model_metrics(), "shocks": detectors(),
               "map": {"fields": ["id", "lat", "lon", "name", "region", "growth2025", "shocks2024"],
                       "points": points, "regions": regions}}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    print(f"{len(points)} МО, {len(regions)} регионов → {OUT}")


if __name__ == "__main__":
    main()
