"""Шум от случайности обучения: одна модель с разными seed (ключи *_s1, *_s2 в конфиге).

Запуск: .venv/bin/python -m src.experiments.noise → results/noise.md
Размах — (max − min) / среднее MAE по трём seed, %. Разница между моделями меньше размаха — не довод.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BASES = ["catboost_logdiff_dsp_g123", "catboost_diff", "catboost_diff_g1234"]
SEEDS = ["", "_s1", "_s2"]


def main() -> None:
    m = pd.read_csv("results/metrics_overall.csv")
    rows = []
    for b in BASES:
        g = m[m["model"].isin([b + s for s in SEEDS])]
        if g["model"].nunique() < len(SEEDS):
            continue
        r = g.groupby(["category", "horizon"])["MAE"].agg(["min", "max", "mean"])
        rows.append(((r["max"] - r["min"]) / r["mean"] * 100).rename(b))
    t = pd.concat(rows, axis=1).round(1)
    by_h = t.groupby(level="horizon").agg(["median", "max"]).round(1)
    text = ["# Шум от seed: размах MAE между тремя seed, %\n",
            "Размах — (max − min) / среднее MAE трёх прогонов одной модели с seed 42, 1, 2.\n",
            "## По горизонтам: медиана и максимум по категориям\n", by_h.to_markdown(), "",
            "## По ячейкам\n", t.to_markdown(), ""]
    Path("results/noise.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
