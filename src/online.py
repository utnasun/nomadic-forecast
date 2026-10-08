"""Онлайн-выбор модели на горизонте 1: в каждом месяце — модель, лучшая по уже известным ошибкам прошлых месяцев.

Запуск (после src.run_forecast): .venv/bin/python -m src.online → results/online.md

Так модель выбирали бы на практике: к прогнозу месяца m известны факты до m−1, значит и ошибки прогнозов
на месяцы до m−1. Правила (окно — прошлые месяцы 2024 г.):
  best_all  — наименьшая MAE за все прошлые месяцы;
  best_3    — наименьшая MAE за последние 3 месяца;
  top3_all  — среднее прогнозов трёх лучших за все прошлые месяцы.
В январе прошлых ошибок нет — берётся DEFAULT. Сравнение — с фиксированными моделями на тех же месяцах
(февраль–декабрь, по умолчанию).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT = "catboost_diff"  # лучшая общая модель до ночной работы
FIXED = ["naive", "seasonal_naive_growth", "catboost_diff", "catboost_logdiff_dsp_g123", "catboost_diff_dsp_g123"]


def load(out: Path, models: list[str] | None) -> dict[str, pd.DataFrame]:
    """категория -> (МО·месяц × модель) прогнозов на шаг 1 + факт."""
    by_cat = {}
    for d in sorted(p for p in (out / "forecasts").iterdir() if p.is_dir()):
        if models and d.name not in models:
            continue
        for f in d.glob("*.parquet"):
            fc = pd.read_parquet(f, columns=["category", "territory_id", "month", "step", "y_true", "y_pred"])
            fc = fc[fc["step"] == 1]
            if fc["month"].nunique() < 12:  # нет прогноза на шаг 1 для каждого месяца 2024
                continue
            cat = fc["category"].iloc[0]
            by_cat.setdefault(cat, []).append(fc.set_index(["territory_id", "month"])["y_pred"].rename(d.name))
            by_cat.setdefault(cat + "|y", fc.set_index(["territory_id", "month"])["y_true"])
    out_ = {}
    for cat, parts in by_cat.items():
        if cat.endswith("|y"):
            continue
        P = pd.concat(parts, axis=1)
        out_[cat] = P.assign(y_true=by_cat[cat + "|y"])
    return out_


def run(P: pd.DataFrame, start: str) -> pd.DataFrame:
    months = sorted(P.index.get_level_values("month").unique())
    models = [c for c in P.columns if c != "y_true"]
    ae = P[models].sub(P["y_true"], axis=0).abs()
    mae = ae.groupby(level="month").mean()  # месяц × модель
    rows = []
    for i, m in enumerate(months):
        if m < start:
            continue
        past = mae.loc[months[:i]]
        rows_m = P.xs(m, level="month")
        pick = {}
        if len(past):
            s_all, s_3 = past.mean().sort_values(), past.iloc[-3:].mean().sort_values()
            pick["best_all"], pick["best_3"] = rows_m[s_all.index[0]], rows_m[s_3.index[0]]
            pick["top3_all"] = rows_m[list(s_all.index[:3])].mean(1)
            chosen = s_all.index[0]
        else:
            pick = {k: rows_m[DEFAULT] for k in ("best_all", "best_3", "top3_all")}
            chosen = DEFAULT
        r = {"month": m, "выбрана (best_all)": chosen}
        r.update({k: (v - rows_m["y_true"]).abs().mean() for k, v in pick.items()})
        r.update({k: mae.at[m, k] for k in FIXED if k in mae})
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-02", help="первый оцениваемый месяц")
    p.add_argument("--models", nargs="*", help="кандидаты (по умолчанию — все модели с прогнозом на шаг 1)")
    a = p.parse_args()
    out = Path("results")
    data = load(out, a.models)
    lines = ["# Онлайн-выбор модели, горизонт 1\n",
             f"Месяцы {a.start}…2024-12. Число в ячейке — MAE, руб. В строке «к naive» — среднее по месяцам "
             "отношение MAE к naive, %.\n"]
    summary = []
    for cat, P in data.items():
        r = run(P, a.start)
        cols = ["best_all", "best_3", "top3_all"] + [k for k in FIXED if k in r]
        rel = (r[cols].div(r["naive"], axis=0) - 1).mul(100).mean().round(1)
        summary.append(rel.rename(cat))
        lines += [f"\n## {cat}\n", r.round(1).to_markdown(index=False), "\n**к naive, %:** "
                  + ", ".join(f"{k} {v:+.1f}" for k, v in rel.items())]
    s = pd.concat(summary, axis=1).T
    s.loc["среднее"] = s.mean()
    lines.insert(2, "## Сводка: MAE к naive, % (среднее по месяцам)\n\n" + s.round(1).to_markdown() + "\n")
    (out / "online.md").write_text("\n".join(lines) + "\n")
    print(lines[2])


if __name__ == "__main__":
    main()
