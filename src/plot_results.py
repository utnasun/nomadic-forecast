"""Графики по results/metrics_*.csv → results/figures/.

Запуск: .venv/bin/python -m src.plot_results
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eda import INK, INK2, PALETTE  # noqa: F401  (общий стиль графиков)

# Модели на графиках (не все, чтобы линии читались): лучшие представители групп.
KEY = ["naive", "seasonal_naive_growth", "auto_ets", "prophet_default", "prophet_yearly",
       "catboost_logdiff", "ridge_diff"]


def main() -> None:
    res = Path("results")
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    ov = pd.read_csv(res / "metrics_overall.csv")
    cats = list(ov["category"].unique())
    models = [m for m in KEY if m in set(ov["model"])]
    colors = dict(zip(models, PALETTE + ["#4a3aa7", "#e34948"]))

    # 1. WAPE vs горизонт, малые графики по категориям
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for ax, cat in zip(axes.flat, cats):
        d = ov[ov["category"] == cat]
        for m in models:
            s = d[d["model"] == m].sort_values("horizon")
            ax.plot(range(len(s)), s["WAPE"], marker="o", ms=4, color=colors[m], label=m)
        ax.set_xticks(range(4)); ax.set_xticklabels(["1", "3", "6", "12"])
        ax.set_title(cat)
    for ax in axes[1]:
        ax.set_xlabel("горизонт, мес")
    for ax in axes[:, 0]:
        ax.set_ylabel("WAPE, %")
    axes[0, 2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle("Ошибка прогноза по горизонтам (тест — 2024 г.)", color=INK)
    fig.tight_layout(); fig.savefig(out / "wape_by_horizon.png", bbox_inches="tight"); plt.close(fig)

    # 2. Тепловая карта: ранг моделей по MAE (1 = лучшая) для каждой категории × горизонт
    ov["rank"] = ov.groupby(["category", "horizon"])["MAE"].rank()
    piv = ov.pivot_table(index="model", columns=["category", "horizon"], values="rank")
    piv = piv.loc[piv.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(13, 0.38 * len(piv) + 1.8))
    im = ax.imshow(piv.values, cmap="Blues_r", aspect="auto")
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index)
    ax.set_xticks(range(piv.shape[1]))
    short = {"Все категории": "Все", "Продовольствие": "Прод", "Здоровье": "Здор", "Маркетплейсы": "МП",
             "Общественное питание": "Общепит", "Транспорт": "Трансп"}
    ax.set_xticklabels([f"{short.get(c, c)}\nh={h}" for c, h in piv.columns], fontsize=7)
    for j in range(4, piv.shape[1], 4):
        ax.axvline(j - 0.5, color="white", lw=2)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v <= piv.values[np.isfinite(piv.values)].max() / 3 else INK)
    ax.grid(False)
    ax.set_title("Ранг модели по MAE (1 = лучшая) в каждой категории и горизонте; строки по среднему рангу")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout(); fig.savefig(out / "rank_heatmap.png"); plt.close(fig)

    # 3. MAE по шагам для горизонта 12 («Все категории»)
    bs = pd.read_csv(res / "metrics_by_step.csv")
    d = bs[(bs["horizon"] == 12) & (bs["category"] == cats[0])]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for m in models:
        s = d[d["model"] == m].sort_values("step")
        ax.plot(s["step"], s["ME"], marker="o", ms=4, color=colors[m], label=m)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(1, 13)); ax.set_xlabel("месяц 2024 г. (шаг прогноза от декабря 2023)")
    ax.set_ylabel("смещение ME, руб.")
    ax.set_title(f"Горизонт 12, «{cats[0]}»: смещение прогноза по месяцам (обучение только на 2023 г.)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout(); fig.savefig(out / "h12_bias_by_step.png", bbox_inches="tight"); plt.close(fig)
    print("Графики сохранены в", out)


if __name__ == "__main__":
    main()
