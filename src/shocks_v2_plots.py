"""Графики к отчёту о шоках v2 (results/shocks_v2/*.png). Запуск после src.shocks_v2:
  .venv/bin/python -m src.shocks_v2_plots
Цвета — эталонная проверенная палитра (категориальные слоты по порядку, расходящаяся пара синий ↔ красный)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("results/shocks_v2")
SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DOWN, UP = "#2a78d6", "#e34948"  # падение — синий полюс, рост — красный


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def recall(cal: pd.DataFrame) -> None:
    order = ["topk", "mahal", "iforest", "combo"]
    names = {"topk": "topk (как v1)", "mahal": "Махаланобис", "iforest": "Isolation Forest", "combo": "combo (итог)"}
    scen = [s for s in cal["сценарий"].unique() if s != "весь регион −x"]
    fig, axes = plt.subplots(1, len(scen), figsize=(4 * len(scen), 3.4), sharey=True, facecolor=SURFACE)
    for ax, s in zip(axes, scen):
        g = cal[cal["сценарий"] == s]
        for i, k in enumerate(order):
            r = g[g["оценка"] == k].sort_values("x")
            ax.plot(r["x"] * 100, r["найдено, %"], color=SERIES[i], lw=2, marker="o", ms=4, label=names[k])
        style(ax)
        short = {"все категории −x": "Все категории −x", "одна категория −x": "Одна категория −x"}
        ax.set_title(short.get(s, "«ЧС»: продукты +x/2, маркетплейсы и общепит −x"), fontsize=9, color=INK, loc="left")
        ax.set_xlabel("величина шока, %", fontsize=8, color=INK2)
        ax.set_xticks([5, 10, 20, 30])
    axes[0].set_ylabel("найдено при 1% ложных тревог, %", fontsize=8, color=INK2)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "recall.png", dpi=150, facecolor=SURFACE)


def by_month(res: pd.DataFrame) -> None:
    sh = res[res["уровень"] == "шок"].copy()
    sh["тип"] = sh["тип"].replace("", "нет (декабрь)").fillna("нет (декабрь)")
    types = ["сдвиг уровня", "разовый", "нарастающий", "провал с отскоком", "всплеск с провалом"]
    t = sh.pivot_table(index="month", columns="тип", values="score", aggfunc="size").reindex(columns=types).fillna(0)
    t["без типа (декабрь)"] = sh.groupby("month").size() - t.sum(1)
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    bottom = 0
    colors = SERIES[:3] + ["#e87ba4", "#4a3aa7", NEUTRAL]
    for i, c in enumerate(list(t.columns)):
        if t[c].sum() == 0:
            continue
        ax.bar([m[5:] for m in t.index], t[c], bottom=bottom, color=colors[i], label=c, width=0.7,
               edgecolor=SURFACE, linewidth=1)
        bottom = bottom + t[c]
    style(ax)
    ax.set_title("Шоки МО по месяцам 2024 г. и тип (по следующему месяцу)", fontsize=9, color=INK, loc="left")
    ax.set_ylabel("шоков", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=7, ncol=6, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(OUT / "by_month.png", dpi=150, facecolor=SURFACE)


def shock_map(res: pd.DataFrame) -> None:
    st = pd.read_parquet("data/external/territory_static.parquet").set_index("territory_id")[["lat", "lon"]]
    r = res.join(st, on="territory_id")
    allmo = r.drop_duplicates("territory_id")
    sh = r[r["уровень"] == "шок"].copy()
    first = sh["главное"].str.extract(r"([+-]\d+)%")[0].astype(float)
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    ax.scatter(allmo["lon"], allmo["lat"], s=3, color=NEUTRAL, lw=0)
    for sign, color, lab in [(-1, DOWN, "падение"), (1, UP, "рост")]:
        m = sh[(first * sign) > 0]
        ax.scatter(m["lon"], m["lat"], s=16, color=color, edgecolor=SURFACE, linewidth=0.8, label=f"{lab} ({len(m)})")
    ax.set_facecolor(SURFACE)
    ax.set_axis_off()
    ax.set_xlim(19, 180)
    ax.set_title("Шоки МО в 2024 г. (главная категория: падение / рост); серые — все МО панели",
                 fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "map.png", dpi=150, facecolor=SURFACE)


def main() -> None:
    recall(pd.read_csv(OUT / "calibration.csv"))
    res = pd.read_parquet(OUT / "scores.parquet")
    by_month(res)
    shock_map(res)
    print("графики: recall.png, by_month.png, map.png")


if __name__ == "__main__":
    main()
