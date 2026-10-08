"""Честный выбор модели: выбор по первой половине 2024 г., проверка — по второй.

Запуск (после src.run_forecast):
  .venv/bin/python -m src.halves                       → results/metrics_halves.csv, results/selection.md
  .venv/bin/python -m src.halves --models a b c        только эти модели (кандидаты выбора)

Половина определяется месяцем прогноза: H1 — январь–июнь 2024, H2 — июль–декабрь. Горизонт h и точки старта —
как в src.evaluate. При h=12 точка старта одна, и H1/H2 — это шаги 1–6 и 7–12 одного прогноза.

Правило выбора для ячейки «категория × горизонт»: модель с наименьшей MAE на H1. В selection.md — её MAE на H2
против MAE на H2 моделей-ориентиров (REFERENCE) и против модели, лучшей на H2 (недостижимый «оракул»).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluate import horizon_slice

REFERENCE = ["naive", "seasonal_naive_growth", "catboost_diff"]


def table(out: Path, models: list[str] | None, min_train: int, horizons: list[int]) -> pd.DataFrame:
    rows = []
    dirs = sorted(p for p in (out / "forecasts").iterdir() if p.is_dir())
    for d in dirs:
        if models and d.name not in set(models) | set(REFERENCE):
            continue
        for f in d.glob("*.parquet"):
            fc = pd.read_parquet(f, columns=["category", "origin", "step", "month", "y_true", "y_pred", "y_last"])
            for h in horizons:
                s = horizon_slice(fc, h, min_train, min_train + 12)
                s = s.assign(half=np.where(s["month"].str[5:7].astype(int) <= 6, "H1", "H2"),
                             ae=(s["y_pred"] - s["y_true"]).abs())
                for half, g in s.groupby("half"):
                    rows.append({"model": d.name, "category": fc["category"].iloc[0], "horizon": h, "half": half,
                                 "MAE": g["ae"].mean(), "n": len(g)})
    return pd.DataFrame(rows)


def selection(t: pd.DataFrame, candidates: list[str] | None) -> str:
    w = t.pivot_table(index=["category", "horizon", "model"], columns="half", values="MAE").reset_index()
    cand = w[w["model"].isin(candidates)] if candidates else w
    lines = ["# Выбор модели по H1 (январь–июнь 2024), проверка на H2 (июль–декабрь)\n",
             "ΔH2, % — MAE выбранной модели на H2 к MAE ориентира на H2 (отрицательное — лучше). "
             "«Оракул» — лучшая на H2 модель среди кандидатов: выше этого правило выбора не прыгнет.\n"]
    rows = []
    for (c, h), g in cand.groupby(["category", "horizon"]):
        pick = g.loc[g["H1"].idxmin()]
        oracle = g.loc[g["H2"].idxmin()]
        ref = w[(w["category"] == c) & (w["horizon"] == h)].set_index("model")["H2"]
        row = {"категория": c, "h": h, "выбрана по H1": pick["model"], "MAE H2": round(pick["H2"], 1),
               "оракул H2": f"{oracle['model']} ({oracle['H2']:.1f})"}
        for r in REFERENCE:
            if r in ref:
                row[f"ΔH2 к {r}, %"] = round((pick["H2"] / ref[r] - 1) * 100, 1)
        rows.append(row)
    lines.append(pd.DataFrame(rows).to_markdown(index=False))
    return "\n".join(lines) + "\n"


def h2_mae(out: Path, models: list[str], cat: str, h: int, min_train: int) -> float:
    """MAE на H2 среднего прогноза нескольких моделей."""
    fs = [horizon_slice(pd.read_parquet(out / "forecasts" / m / f"{cat}.parquet"), h, min_train, min_train + 12)
          for m in models]
    f = fs[0].copy()
    f["y_pred"] = np.mean([x["y_pred"].values for x in fs], axis=0)
    f = f[f["month"].str[5:7].astype(int) > 6]
    return float((f["y_pred"] - f["y_true"]).abs().mean())


def rules(t: pd.DataFrame, out: Path, candidates: list[str], min_train: int, fixed: list[str]) -> str:
    """Сравнение правил выбора на H2: лучшая на H1, среднее 3 лучших на H1, фиксированные модели."""
    w = t.pivot_table(index=["category", "horizon", "model"], columns="half", values="MAE").reset_index()
    rows = []
    for (c, h), g in w.groupby(["category", "horizon"]):
        cand = g[g["model"].isin(candidates)].sort_values("H1")
        ref = g.set_index("model")["H2"]
        row = {"категория": c, "h": h, "naive": ref.get("naive"),
               "лучшая на H1": cand["H2"].iloc[0],
               "среднее 3 лучших на H1": h2_mae(out, cand["model"].iloc[:3].tolist(), c, h, min_train)}
        row.update({m: ref.get(m) for m in fixed})
        rows.append(row)
    r = pd.DataFrame(rows)
    cols = [c for c in r.columns if c not in ("категория", "h", "naive")]
    rel = r[cols].div(r["naive"], axis=0).sub(1).mul(100)
    summary = rel.groupby(r["h"]).mean().round(1)
    lines = ["\n## Правила выбора: MAE на H2, % к naive (среднее по 6 категориям; отрицательное — лучше)\n",
             summary.to_markdown(), "\n**По ячейкам, MAE на H2**\n", r.round(1).to_markdown(index=False)]
    return "\n".join(lines) + "\n"


def rule_table(t: pd.DataFrame, out: Path, candidates: list[str], min_train: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Правила, выбирающие только по H1; результат — MAE на H2 к naive, %.

    R1 — одна модель на всё: наименьшее среднее по ячейкам (MAE H1 / MAE naive H1);
    R2 — одна модель на категорию (все горизонты), то же среднее по горизонтам категории;
    R3 — лучшая на H1 в каждой ячейке; R4 — среднее прогнозов трёх лучших на H1 в ячейке.
    """
    w = t.pivot_table(index=["category", "horizon", "model"], columns="half", values="MAE").reset_index()
    nv = w[w["model"] == "naive"].set_index(["category", "horizon"])
    w = w.join(nv[["H1", "H2"]].rename(columns={"H1": "n1", "H2": "n2"}), on=["category", "horizon"])
    w["r1"], w["r2"] = w["H1"] / w["n1"], w["H2"] / w["n2"]
    c = w[w["model"].isin(candidates)]
    complete = c.groupby("model").size() == c.groupby(["category", "horizon"]).ngroups
    c = c[c["model"].isin(complete[complete].index)]
    r1 = c.groupby("model")["r1"].mean().idxmin()
    r2 = c.groupby(["category", "model"])["r1"].mean().reset_index()
    r2 = r2.loc[r2.groupby("category")["r1"].idxmin()].set_index("category")["model"]
    rows = []
    for (cat, h), g in c.groupby(["category", "horizon"]):
        g = g.set_index("model")
        top = g.sort_values("H1").index
        rows.append({"категория": cat, "h": h, "R1": g.at[r1, "r2"], "R2": g.at[r2[cat], "r2"],
                     "R3": g.at[top[0], "r2"], "R4": h2_mae(out, list(top[:3]), cat, h, min_train) / g["n2"].iloc[0],
                     "оракул": g["r2"].min(), "R2 модель": r2[cat], "R3 модель": top[0]})
    r = pd.DataFrame(rows)
    for k in ("R1", "R2", "R3", "R4", "оракул"):
        r[k] = (r[k] - 1) * 100
    summary = r.groupby("h")[["R1", "R2", "R3", "R4", "оракул"]].mean().round(1)
    summary.attrs["R1"] = r1
    return r, summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/forecast.yaml")
    p.add_argument("--models", nargs="*", help="кандидаты выбора (по умолчанию — все модели в results/forecasts)")
    p.add_argument("--name", default="selection", help="имя отчёта: results/<name>.md")
    p.add_argument("--fixed", nargs="*", default=[], help="фиксированные модели для сравнения правил")
    p.add_argument("--rules", action="store_true", help="сравнить правила R1–R4, выбирающие только по H1")
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = Path(cfg["output_dir"])
    t = table(out, a.models, cfg["protocol"]["min_train"], cfg["protocol"]["horizons"])
    if not a.models:
        t.to_csv(out / "metrics_halves.csv", index=False)
    text = selection(t, a.models)
    if a.rules:
        cands = a.models or sorted(t["model"].unique())
        r, summary = rule_table(t, out, cands, cfg["protocol"]["min_train"])
        text += ("\n## Правила, выбирающие только по H1: MAE на H2, % к naive (среднее по категориям)\n\n"
                 f"R1 — одна модель на всё ({summary.attrs['R1']}); R2 — одна модель на категорию; R3 — лучшая в "
                 "ячейке; R4 — среднее трёх лучших в ячейке; «оракул» — лучшая на H2 (недостижимо).\n\n"
                 + summary.to_markdown() + "\n\n" + r.round(1).to_markdown(index=False) + "\n")
    if a.fixed:
        cands = a.models or sorted(t["model"].unique())
        text += rules(t, out, cands, cfg["protocol"]["min_train"], a.fixed)
    (out / f"{a.name}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
