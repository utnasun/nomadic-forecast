"""Сбор метрик по сохранённым прогнозам.

Для горизонта h берутся точки старта из src.evaluation.protocol.folds и шаги 1..h.
  results/metrics_overall.csv — метрика, усреднённая по всем шагам горизонта;
  results/metrics_by_step.csv — метрика отдельно для каждого шага (1-й, 2-й, ... месяц вперёд);
  results/summary.md          — сводные таблицы.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation import metrics
from src.evaluation.protocol import folds


def horizon_slice(fc: pd.DataFrame, h: int, min_train: int, n_months: int) -> pd.DataFrame:
    """Точки старта горизонта; у неполных МО — только строки с фактом и последним известным значением."""
    fc = fc[fc["origin"].isin(folds(h, min_train, n_months)) & (fc["step"] <= h)]
    return fc[np.isfinite(fc["y_true"]) & np.isfinite(fc["y_last"])]


def compute_table(fc: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in fc.groupby(by, sort=False):
        m = metrics.compute(g["y_true"].values, g["y_pred"].values, g["y_last"].values)
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), **m})
    return pd.DataFrame(rows)


def main(cfg: dict) -> None:
    out = Path(cfg["output_dir"])
    files = sorted((out / "forecasts").glob("*/*.parquet"))
    fc = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    n_months = cfg["protocol"]["min_train"] + 12  # 24: весь 2024 — тест
    mt = cfg["protocol"]["min_train"]

    overall, by_step = [], []
    for h in cfg["protocol"]["horizons"]:
        s = horizon_slice(fc, h, mt, n_months)
        overall.append(compute_table(s, ["model", "category"]).assign(horizon=h))
        by_step.append(compute_table(s, ["model", "category", "step"]).assign(horizon=h))
    overall = pd.concat(overall, ignore_index=True)
    by_step = pd.concat(by_step, ignore_index=True)
    overall.to_csv(out / "metrics_overall.csv", index=False)
    by_step.to_csv(out / "metrics_by_step.csv", index=False)

    lines = ["# Сравнение моделей прогноза\n",
             "Тест — весь 2024 г. при любом горизонте; метрика усреднена по всем шагам горизонта.\n"]

    # Лучшая модель по MAE для каждой категории и горизонта + выигрыш относительно Prophet
    idx = overall.groupby(["category", "horizon"])["MAE"].idxmin()
    best = overall.loc[idx, ["category", "horizon", "model", "MAE", "WAPE"]]
    ref = overall[overall["model"] == "prophet_default"].set_index(["category", "horizon"])["MAE"]
    if len(ref):
        best["MAE_prophet"] = [ref.get((c, h)) for c, h in zip(best["category"], best["horizon"])]
        best["выигрыш_vs_prophet_%"] = (1 - best["MAE"] / best["MAE_prophet"]) * 100
    lines.append("\n## Лучшая модель по MAE\n\n" + best.round(2).to_markdown(index=False) + "\n")
    for cat in overall["category"].unique():
        lines.append(f"\n## {cat}\n")
        for metric in ["MAE", "WAPE", "R2", "R2_growth"]:
            t = overall[overall["category"] == cat].pivot(index="model", columns="horizon", values=metric)
            t = t.sort_values(t.columns[0], ascending=metric in ("MAE", "WAPE"))
            t.columns = [f"h={c}" for c in t.columns]
            fmt = "{:.0f}" if metric == "MAE" else ("{:.2f}" if metric == "WAPE" else "{:.3f}")
            lines.append(f"\n**{metric}**\n\n" + t.map(fmt.format).to_markdown() + "\n")
    (out / "summary.md").write_text("\n".join(lines))
    print((out / "summary.md").read_text()[:6000])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/forecast.yaml")
    main(yaml.safe_load(open(p.parse_args().config)))
