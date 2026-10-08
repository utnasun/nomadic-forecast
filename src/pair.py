"""Парное сравнение двух моделей: ΔMAE по категории × горизонту, победы по точкам старта, H1 / H2 2024 г.

Запуск: .venv/bin/python -m src.pair catboost_diff_rw1 catboost_diff [--horizons 1 3]
ΔMAE = MAE(A) / MAE(B) − 1, %; «побед» — в скольких точках старта у A меньше MAE (для h=12 старт один).
Шум от seed — 1,6–3% у моделей с профилем по панели и до 5–12% у catboost_diff (results/noise.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.evaluate import horizon_slice


def load(out: Path, model: str, cat: str, h: int, mt: int) -> pd.DataFrame:
    fc = pd.read_parquet(out / "forecasts" / model / f"{cat}.parquet")
    s = horizon_slice(fc, h, mt, mt + 12)
    return s.assign(ae=(s["y_pred"] - s["y_true"]).abs())[["territory_id", "origin", "step", "month", "ae"]]


def compare(cfg: dict, a: str, b: str, horizons: list[int] | None = None,
            categories: list[str] | None = None) -> pd.DataFrame:
    out, mt = Path(cfg["output_dir"]), cfg["protocol"]["min_train"]
    rows = []
    for cat in categories or cfg["data"]["categories"]:
        for h in horizons or cfg["protocol"]["horizons"]:
            m = load(out, a, cat, h, mt).merge(load(out, b, cat, h, mt), on=["territory_id", "origin", "step", "month"],
                                               suffixes=("_a", "_b"))
            by_o = m.groupby("origin")[["ae_a", "ae_b"]].mean()
            half = m.assign(H=m["month"].str[5:7].astype(int).le(6).map({True: "H1", False: "H2"})) \
                .groupby("H")[["ae_a", "ae_b"]].mean()
            rows.append({"категория": cat, "h": h, "MAE A": m["ae_a"].mean(), "MAE B": m["ae_b"].mean(),
                         "Δ, %": 100 * (m["ae_a"].mean() / m["ae_b"].mean() - 1),
                         "побед A": f"{int((by_o['ae_a'] < by_o['ae_b']).sum())}/{len(by_o)}",
                         "Δ H1, %": 100 * (half.loc["H1", "ae_a"] / half.loc["H1", "ae_b"] - 1),
                         "Δ H2, %": 100 * (half.loc["H2", "ae_a"] / half.loc["H2", "ae_b"] - 1)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--horizons", nargs="*", type=int)
    p.add_argument("--categories", nargs="*")
    p.add_argument("--config", default="configs/forecast.yaml")
    x = p.parse_args()
    t = compare(yaml.safe_load(open(x.config)), x.a, x.b, x.horizons, x.categories)
    print(f"A = {x.a}, B = {x.b}\n")
    print(t.round(1).to_markdown(index=False))
    print(f"\nсреднее Δ по категориям, %:\n{t.groupby('h')[['Δ, %', 'Δ H1, %', 'Δ H2, %']].mean().round(1).to_markdown()}")
