"""Разведочный анализ полных рядов потребления.

Запуск: .venv/bin/python -m src.viz.eda --config configs/forecast.yaml
Результат: report/eda/*.png и report/eda/eda_stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data import TOTAL, build_panels, complete_territories, load_consumption, load_market_access

# Категориальная палитра, фиксированный порядок слотов.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlesize": 10,
    "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 2,
    "legend.frameon": False,
})


def month_ticks(ax, months, step=3):
    idx = list(range(0, len(months), step))
    ax.set_xticks(idx)
    ax.set_xticklabels([months[i] for i in idx], rotation=45, ha="right")


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    out = Path("report/eda")
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {}

    df = load_consumption(cfg["data"]["consumption"])
    cats = cfg["data"]["categories"]
    all_ids = np.sort(df["territory_id"].unique())
    ids = complete_territories(df)
    panels = build_panels(df, cats, ids)
    months = panels[TOTAL].months
    T = len(months)

    # ---------- 1. Покрытие: полные vs неполные МО ----------
    tot_all = df[df["category"] == TOTAL]
    n_months = tot_all.groupby("territory_id")["date"].nunique()
    mean_lvl = tot_all.groupby("territory_id")["value"].mean()
    is_full = mean_lvl.index.isin(ids)
    stats["coverage"] = {
        "territories_total": int(len(all_ids)),
        "territories_complete": int(len(ids)),
        "territories_incomplete": int(len(all_ids) - len(ids)),
        "median_level_complete": float(mean_lvl[is_full].median()),
        "median_level_incomplete": float(mean_lvl[~is_full].median()),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    n_months[~n_months.index.isin(ids)].value_counts().sort_index().plot.bar(ax=axes[0], color=PALETTE[0], width=0.8)
    axes[0].set_title("Неполные МО: сколько месяцев есть в «Все категории»")
    axes[0].set_xlabel("месяцев с данными"); axes[0].set_ylabel("число МО")
    bins = np.logspace(np.log10(mean_lvl.min()), np.log10(mean_lvl.max()), 40)
    axes[1].hist(mean_lvl[is_full], bins=bins, color=PALETTE[0], alpha=0.8, label=f"полные ({is_full.sum()})", density=True)
    axes[1].hist(mean_lvl[~is_full], bins=bins, color=PALETTE[1], alpha=0.7, label=f"неполные ({(~is_full).sum()})", density=True)
    axes[1].set_xscale("log"); axes[1].legend()
    axes[1].set_title("Средние расходы на жителя, «Все категории»"); axes[1].set_xlabel("руб./мес (лог. шкала)")
    fig.tight_layout(); fig.savefig(out / "01_coverage.png"); plt.close(fig)

    # ---------- 2. Распределение уровней по категориям ----------
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5))
    stats["levels"] = {}
    for ax, cat in zip(axes.flat, cats):
        m = panels[cat].values.mean(axis=1)
        ax.hist(m, bins=np.logspace(np.log10(m.min()), np.log10(m.max()), 40), color=PALETTE[0])
        ax.set_xscale("log"); ax.set_title(cat)
        q = np.percentile(m, [5, 50, 95])
        stats["levels"][cat] = {"p5": q[0], "median": q[1], "p95": q[2], "p95_to_p5": q[2] / q[0]}
    fig.suptitle("Средние расходы МО за 24 мес., руб./мес (лог. шкала)", color=INK)
    fig.tight_layout(); fig.savefig(out / "02_levels.png"); plt.close(fig)

    # ---------- 3. Динамика: индекс к среднему 2023, медиана и 10–90% ----------
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, cat in zip(axes.flat, cats):
        v = panels[cat].values
        idx = v / v[:, :12].mean(axis=1, keepdims=True)
        q10, q50, q90 = np.percentile(idx, [10, 50, 90], axis=0)
        ax.fill_between(range(T), q10, q90, color=PALETTE[0], alpha=0.18, linewidth=0, label="10–90% МО")
        ax.plot(q50, color=PALETTE[0], label="медиана")
        ax.axhline(1, color=INK2, lw=0.8, ls=":")
        ax.set_title(cat); month_ticks(ax, months)
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Динамика расходов, индекс к среднему за 2023 г. = 1", color=INK)
    fig.tight_layout(); fig.savefig(out / "03_dynamics.png"); plt.close(fig)

    # ---------- 4. Сезонность: медианный прирост м/м, 2023 vs 2024 ----------
    stats["seasonality"] = {}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    x = np.arange(1, 13)
    for ax, cat in zip(axes.flat, cats):
        dlog = np.diff(np.log(panels[cat].values), axis=1)  # (n, 23), переход к месяцам 2..24
        med = np.median(dlog, axis=0)
        s23 = np.r_[np.nan, med[:11]]           # фев..дек 2023
        s24 = med[11:]                          # янв..дек 2024
        ax.bar(x - 0.2, s23 * 100, width=0.38, color=PALETTE[0], label="2023")
        ax.bar(x + 0.2, s24 * 100, width=0.38, color=PALETTE[1], label="2024")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(cat); ax.set_xticks(x)
        corr = np.corrcoef(s23[1:], s24[1:])[0, 1]
        stats["seasonality"][cat] = {
            "corr_mom_profile_2023_vs_2024": corr,
            "median_mom_2023_pct": dict(zip(range(2, 13), (s23[1:] * 100).round(2).tolist())),
            "median_mom_2024_pct": dict(zip(range(1, 13), (s24 * 100).round(2).tolist())),
        }
        ax.text(0.02, 0.95, f"корр. профилей = {corr:.2f}", transform=ax.transAxes, va="top", color=INK2)
    axes[0, 0].legend(loc="lower right")
    for ax in axes[1]:
        ax.set_xlabel("месяц года")
    fig.suptitle("Медианный прирост к прошлому месяцу, % (сезонный профиль)", color=INK)
    fig.tight_layout(); fig.savefig(out / "04_seasonality.png"); plt.close(fig)

    # ---------- 5. Рост 2024 к 2023 ----------
    stats["yoy"] = {}
    fig, ax = plt.subplots(figsize=(9, 3.6))
    data = []
    for cat in cats:
        v = panels[cat].values
        g = (v[:, 12:].sum(1) / v[:, :12].sum(1) - 1) * 100
        data.append(g)
        stats["yoy"][cat] = {"median_pct": float(np.median(g)), "p10": float(np.percentile(g, 10)), "p90": float(np.percentile(g, 90))}
    bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True, widths=0.55)
    for b in bp["boxes"]:
        b.set(facecolor=PALETTE[0], alpha=0.35, edgecolor=PALETTE[0])
    for med in bp["medians"]:
        med.set(color=INK, linewidth=1.5)
    ax.set_yticks(range(1, len(cats) + 1)); ax.set_yticklabels(cats)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("рост суммы 2024 к сумме 2023, % (без выбросов)")
    ax.set_title("Годовой рост по МО")
    fig.tight_layout(); fig.savefig(out / "05_yoy.png"); plt.close(fig)

    # ---------- 6. Общая vs индивидуальная компонента приростов ----------
    stats["variance"] = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    shares, idio_sd = [], []
    for cat in cats:
        v = panels[cat].values
        dlog = np.diff(np.log(v), axis=1)
        common = np.median(dlog, axis=0, keepdims=True)
        idio = dlog - common
        share_common = 1 - idio.var() / dlog.var()
        # автокорреляция индивидуальной компоненты (lag 1..3), усреднённая по МО
        ac = {}
        for lag in (1, 2, 3):
            a, b = idio[:, lag:], idio[:, :-lag]
            a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
            ac[lag] = float(np.median((a * b).sum(1) / np.sqrt((a**2).sum(1) * (b**2).sum(1))))
        stats["variance"][cat] = {
            "share_var_common": float(share_common),
            "idio_sd_median": float(np.median(idio.std(1))),
            "idio_abs_gt_10pct_share": float((np.abs(idio) > 0.1).mean()),
            "idio_abs_gt_20pct_share": float((np.abs(idio) > 0.2).mean()),
            "idio_autocorr_median": ac,
        }
        shares.append(share_common)
        idio_sd.append(idio.std(1))
    axes[0].barh(cats, np.array(shares) * 100, color=PALETTE[0], height=0.55)
    for i, s in enumerate(shares):
        axes[0].text(s * 100 + 1, i, f"{s*100:.0f}%", va="center", color=INK2)
    axes[0].set_xlim(0, 100); axes[0].invert_yaxis()
    axes[0].set_title("Доля дисперсии приростов м/м,\nобъяснённая общим для всех МО фактором")
    axes[1].boxplot(idio_sd, vert=False, showfliers=False, widths=0.55, medianprops={"color": INK})
    axes[1].set_yticks(range(1, len(cats) + 1)); axes[1].set_yticklabels(cats); axes[1].invert_yaxis()
    axes[1].set_title("Индивидуальная волатильность МО:\nст. откл. прироста м/м без общего фактора")
    axes[1].set_xlabel("лог-пункты")
    fig.tight_layout(); fig.savefig(out / "06_common_vs_idio.png"); plt.close(fig)

    # ---------- 7. Волатильность vs размер МО и доступность рынков ----------
    ma = load_market_access(cfg["data"]["market_access"], ids)
    v = panels[TOTAL].values
    lvl = v.mean(1)
    dlog = np.diff(np.log(v), axis=1)
    idio_sd_tot = (dlog - np.median(dlog, 0, keepdims=True)).std(1)
    ok = ~np.isnan(ma)
    stats["market_access"] = {
        "missing_for_complete_mo": int((~ok).sum()),
        "spearman_level_vs_ma": float(pd.Series(lvl[ok]).corr(pd.Series(ma[ok]), method="spearman")),
        "spearman_idio_sd_vs_ma": float(pd.Series(idio_sd_tot[ok]).corr(pd.Series(ma[ok]), method="spearman")),
        "spearman_idio_sd_vs_level": float(pd.Series(idio_sd_tot).corr(pd.Series(lvl), method="spearman")),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].scatter(ma[ok], lvl[ok], s=6, alpha=0.4, color=PALETTE[0], linewidths=0)
    axes[0].set_yscale("log"); axes[0].set_xlabel("индекс доступности рынков"); axes[0].set_ylabel("руб./мес")
    axes[0].set_title(f"Уровень расходов vs доступность рынков (Спирмен {stats['market_access']['spearman_level_vs_ma']:.2f})")
    axes[1].scatter(lvl, idio_sd_tot, s=6, alpha=0.4, color=PALETTE[0], linewidths=0)
    axes[1].set_xscale("log"); axes[1].set_xlabel("руб./мес"); axes[1].set_ylabel("инд. волатильность")
    axes[1].set_title(f"Волатильность vs уровень (Спирмен {stats['market_access']['spearman_idio_sd_vs_level']:.2f})")
    fig.suptitle("«Все категории»", color=INK, y=1.02)
    fig.tight_layout(); fig.savefig(out / "07_market_access.png", bbox_inches="tight"); plt.close(fig)

    # ---------- 8. Доли категорий в «Все категории» ----------
    fig, ax = plt.subplots(figsize=(9, 3.8))
    stats["shares"] = {}
    tot = panels[TOTAL].values
    for i, cat in enumerate([c for c in cats if c != TOTAL]):
        sh = np.median(panels[cat].values / tot, axis=0) * 100
        ax.plot(sh, color=PALETTE[i], label=cat)
        stats["shares"][cat] = {"median_share_2023_pct": float(sh[:12].mean()), "median_share_2024_pct": float(sh[12:].mean())}
    ax.set_ylabel("% от «Все категории»"); month_ticks(ax, months)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title("Медианная доля категории в общих расходах МО")
    fig.tight_layout(); fig.savefig(out / "08_shares.png"); plt.close(fig)

    # ---------- 9. Примеры рядов ----------
    rng = np.random.default_rng(cfg["seed"])
    sample = rng.choice(len(ids), 6, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharex=True)
    for ax, i in zip(axes.flat, sample):
        for j, cat in enumerate(cats):
            ax.plot(panels[cat].values[i], color=PALETTE[j], lw=1.5, label=cat)
        ax.set_yscale("log"); ax.set_title(f"territory_id = {ids[i]}"); month_ticks(ax, months, 6)
    axes[0, 2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle("Случайные МО: расходы по категориям, руб./мес (лог. шкала)", color=INK)
    fig.tight_layout(); fig.savefig(out / "09_examples.png"); plt.close(fig)

    # ---------- 10. Крупнейшие индивидуальные скачки ----------
    top = []
    for cat in cats:
        vv = panels[cat].values
        d = np.diff(np.log(vv), axis=1)
        idio = d - np.median(d, 0, keepdims=True)
        flat = np.argsort(-np.abs(idio), axis=None)[:5]
        for f in flat:
            i, t = np.unravel_index(f, idio.shape)
            top.append({"category": cat, "territory_id": int(ids[i]), "month": months[t + 1],
                        "idio_dlog": round(float(idio[i, t]), 3), "value_prev": int(vv[i, t]), "value": int(vv[i, t + 1])})
    stats["top_idio_jumps"] = top
    # сколько МО одновременно имеют |idio| > 10% в каждом месяце — «волны» шоков
    d = np.diff(np.log(tot), axis=1)
    idio = d - np.median(d, 0, keepdims=True)
    stats["idio_gt10_by_month_total"] = dict(zip(months[1:], (np.abs(idio) > 0.1).sum(0).tolist()))

    with open(out / "eda_stats.json", "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=float)
    print(json.dumps(stats, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/forecast.yaml")
    main(p.parse_args().config)
