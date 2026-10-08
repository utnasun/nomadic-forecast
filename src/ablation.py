"""Абляция внешних признаков: сравнение вариантов моделей (ключ base в конфиге) с их базовой моделью.

Запуск (после src.evaluate): .venv/bin/python -m src.ablation → results/ablation.md
ΔMAE, % = (MAE варианта / MAE базы − 1) · 100: отрицательное — вариант лучше базы.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def main(cfg: dict) -> None:
    out = Path(cfg["output_dir"])
    m = pd.read_csv(out / "metrics_overall.csv")
    variants = {name: c for name, c in cfg["models"].items() if "base" in c and name in set(m["model"])}
    key = ["category", "horizon"]

    def root(name):  # исходная модель цепочки base
        return root(cfg["models"][name]["base"]) if "base" in cfg["models"][name] else name

    rows = []
    for name, c in variants.items():
        v = m[m["model"] == name].set_index(key)
        b = m[m["model"] == c["base"]].set_index(key)
        r = m[m["model"] == root(name)].set_index(key)
        d = pd.DataFrame({
            "root": root(name), "base": c["base"], "variant": name,
            "dMAE_root_%": (v["MAE"] / r["MAE"] - 1) * 100,
            "dMAE_%": (v["MAE"] / b["MAE"] - 1) * 100,
            "dWAPE_pp": v["WAPE"] - b["WAPE"],
            "dR2_growth": v["R2_growth"] - b["R2_growth"],
        }).reset_index()
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(out / "ablation.csv", index=False)

    lines = ["# Абляция внешних признаков\n",
             "Каждый вариант сравнивается со своей базой (ключ base в конфиге); "
             "тест — весь 2024 г., протокол как в summary.md.",
             "ΔMAE, %: отрицательное — лучше базы. g1 — справочник МО, g2 — регион/соседи, g3 — население, "
             "g4 — квартальный Росстат, n — новости о ЧС и отключениях по городу — центру МО "
             "(оценка на событиях — news_eval.md), ds — десезонирование по национальному профилю, "
             "dsp — по профилю самой панели, m1 — первая цель обучения — февраль (seasonality.md).\n"]

    lines.append("\n## Сводка: ΔMAE, % к исходной модели (без внешних признаков и десезонирования)\n")
    for rt, g in d.groupby("root", sort=False):
        mean = g.pivot_table(index="variant", columns="horizon", values="dMAE_root_%", aggfunc="mean", sort=False)
        mean.columns = [f"h={h}" for h in mean.columns]
        lines.append(f"\n**{rt}, среднее по 6 категориям**\n\n" + mean.round(1).to_markdown() + "\n")

    for base, g in d.groupby("base", sort=False):
        lines.append(f"\n## {base}\n")
        mean = g.pivot_table(index="variant", columns="horizon", values="dMAE_%", aggfunc="mean", sort=False)
        mean.columns = [f"h={h}" for h in mean.columns]
        lines.append("**ΔMAE, %, среднее по 6 категориям**\n\n" + mean.round(1).to_markdown() + "\n")
        for metric, fmt in [("dMAE_%", 1), ("dR2_growth", 3)]:
            t = g.pivot_table(index=["category", "variant"], columns="horizon", values=metric, sort=False)
            t.columns = [f"h={h}" for h in t.columns]
            title = "ΔMAE, % по категориям" if metric == "dMAE_%" else "ΔR²_growth по категориям"
            lines.append(f"\n**{title}**\n\n" + t.round(fmt).to_markdown() + "\n")

    # Лучшая модель в каждой ячейке: среди старых моделей и среди всех, включая варианты
    old = m[~m["model"].isin(variants)]
    best_old = old.loc[old.groupby(key)["MAE"].idxmin()].set_index(key)
    best_all = m.loc[m.groupby(key)["MAE"].idxmin()].set_index(key)
    cmp = pd.DataFrame({
        "лучшая_была": best_old["model"], "MAE_была": best_old["MAE"],
        "лучшая_стала": best_all["model"], "MAE_стала": best_all["MAE"],
    })
    cmp["выигрыш_%"] = (1 - cmp["MAE_стала"] / cmp["MAE_была"]) * 100
    lines.append("\n## Лучшая модель по MAE: до и после внешних признаков\n\n"
                 + cmp.reset_index().round(2).to_markdown(index=False) + "\n")

    (out / "ablation.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/forecast.yaml")
    main(yaml.safe_load(open(p.parse_args().config)))
