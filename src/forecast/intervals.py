"""Интервалы прогноза: эмпирические квантили остатков с учётом волатильности МО и недавнего шока.

Запуск (после src.forecast.final и src.shocks.detector): .venv/bin/python -m src.forecast.intervals → results/intervals.md

Остаток r = log(факт / прогноз) модели final. Квантили (LO, HI) оцениваются на прогнозах на январь–июнь 2024 (H1)
отдельно для (категория, горизонт h, квинтиль волатильности МО), проверяются на июле–декабре (H2): доля фактов
внутри интервала (цель — HI − LO) и ширина. Волатильность МО — разброс его месячных log-приростов до точки старта.

После шока (уровень «шок» в src.shocks.detector в последнем известном месяце) остатки шире: множитель интервала
SHOCK_K оценивается на H1 как отношение размаха остатков после шока к обычному в тех же ячейках.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import build_panels, complete_territories, load_consumption
from src.evaluation.evaluate import horizon_slice

LO, HI = 0.05, 0.95
N_VOL = 5
MODEL = "final"


def residuals(cfg: dict) -> pd.DataFrame:
    out = Path(cfg["output_dir"])
    df = load_consumption(cfg["data"]["consumption"])
    ids = complete_territories(df)
    panels = build_panels(df, cfg["data"]["categories"], ids)
    shocks = pd.read_parquet(out / "shocks_v2" / "scores.parquet")
    shocked = set(zip(shocks.loc[shocks["уровень"] == "шок", "territory_id"], shocks.loc[shocks["уровень"] == "шок", "month"]))
    months = panels[cfg["data"]["categories"][0]].months
    rows = []
    for cat, p in panels.items():
        fc = pd.read_parquet(out / "forecasts" / MODEL / f"{cat}.parquet")
        d = np.diff(np.log(np.clip(p.values, 1, None)), axis=1)
        vol = {T: pd.Series(np.nanstd(d[:, :T - 1], axis=1), index=p.territory_ids) for T in fc["origin"].unique()}
        for h in cfg["protocol"]["horizons"]:
            s = horizon_slice(fc, h, cfg["protocol"]["min_train"], cfg["protocol"]["min_train"] + 12).copy()
            s["h"] = h
            s["r"] = np.log(s["y_true"].clip(lower=1) / s["y_pred"].clip(lower=1))
            s["vol"] = [vol[T].get(t) for T, t in zip(s["origin"], s["territory_id"])]
            s["after_shock"] = [(t, months[T - 1]) in shocked for T, t in zip(s["origin"], s["territory_id"])]
            rows.append(s)
    r = pd.concat(rows, ignore_index=True)
    r["vol_q"] = r.groupby(["category", "h"])["vol"].transform(lambda v: pd.qcut(v.rank(method="first"), N_VOL, labels=False))
    r["half"] = np.where(r["month"].str[5:7].astype(int) <= 6, "H1", "H2")
    return r


def evaluate(r: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    key = ["category", "h", "vol_q"]
    h1 = r[(r["half"] == "H1") & ~r["after_shock"]]
    q = h1.groupby(key)["r"].quantile([LO, HI]).unstack()
    q.columns = ["lo", "hi"]
    # множитель после шока: размах (q95 − q05) после шока / обычный, по H1, медиана по категориям и h
    a = r[(r["half"] == "H1") & r["after_shock"]].join(q, on=key)
    k = float(np.quantile(np.abs(a["r"]) / ((a["hi"] - a["lo"]) / 2), HI - LO) / 1.0) if len(a) else 1.0
    k = max(k, 1.0)
    t = r[r["half"] == "H2"].join(q, on=key)
    mid = (t["lo"] + t["hi"]) / 2
    half_w = (t["hi"] - t["lo"]) / 2
    rows = []
    for name, mult in [("без учёта шока", 1.0), (f"после шока ×{k:.1f}", k)]:
        w = np.where(t["after_shock"], half_w * mult, half_w)
        inside = (t["r"] >= mid - w) & (t["r"] <= mid + w)
        for (cat, h), g in t.assign(inside=inside, width=2 * w).groupby(["category", "h"]):
            rows.append({"вариант": name, "категория": cat, "h": h, "покрытие, %": 100 * g["inside"].mean(),
                         "покрытие после шока, %": 100 * g.loc[g["after_shock"], "inside"].mean() if g["after_shock"].any() else np.nan,
                         "МО после шока": int(g["after_shock"].sum()),
                         "ширина, % уровня": 100 * (np.exp(g["width"]) - 1).median()})
    return pd.DataFrame(rows), k


def main(cfg: dict) -> None:
    r = residuals(cfg)
    res, k = evaluate(r)
    summary = res.groupby(["вариант", "h"])[["покрытие, %", "покрытие после шока, %", "ширина, % уровня"]].mean().round(1)
    text = ["# Интервалы прогноза final\n",
            f"Интервал {round((HI - LO) * 100)}%: квантили {LO:.0%}/{HI:.0%} остатков log(факт/прогноз), оценены на H1 "
            "(январь–июнь 2024) по категории × горизонту × квинтилю волатильности МО, проверены на H2 (июль–декабрь). "
            f"После шока интервал расширяется в {k:.1f} раза (множитель оценён на H1). Подробности — в docstring "
            "`src/forecast/intervals.py`.\n",
            "## Среднее по категориям\n", summary.to_markdown(), "",
            "## По категориям\n", res.round(1).to_markdown(index=False), ""]
    Path(cfg["output_dir"], "intervals.md").write_text("\n".join(text))
    print("\n".join(text[:4]))


if __name__ == "__main__":
    main(yaml.safe_load(open("configs/forecast.yaml")))
