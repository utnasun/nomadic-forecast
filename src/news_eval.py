"""Оценка новостных признаков на подвыборке событий.

Запуск (после src.run_forecast для вариантов с features: [..., news]):
  .venv/bin/python -m src.news_eval → results/news_eval.md, results/news_eval.csv

В среднем по всем МО новостные признаки почти ничего не меняют: событий мало. Поэтому вариант с news
сравнивается со своей базой (ключ base в конфиге) отдельно на наблюдениях, где модель знала о событии:
  lag1 — прогноз на месяц сразу после события (шаг 1: новости месяца t−1 известны);
  lag2 — прогноз на второй месяц после события (шаги 1 и 2: известны новости месяца t−2).
Событие — всплеск статей о городе — центре МО (data/news/city_month.parquet): не меньше MIN_N статей
и не меньше RATIO раз к обычному уровню.

Ошибка — |log(прогноз / факт)| в п.п. (не зависит от размера МО) и MAE в рублях.
Доверительный интервал разницы — бутстрап по МО (ошибки одного МО в разные месяцы зависимы).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models import resolve_config

EVENTS = {"emerg": ("ЧС и стихия", 10, 5.0), "outage": ("отключения", 5, 4.0)}  # тема: (имя, MIN_N, RATIO)
N_BOOT = 2000


def event_table(path: str | Path) -> pd.DataFrame:
    """Строка на (МО, месяц события, тема)."""
    cm = pd.read_parquet(path)
    rows = []
    for k, (_, min_n, ratio) in EVENTS.items():
        e = cm[(cm[f"n_{k}"] >= min_n) & (cm[f"spike_{k}"] >= np.log(ratio))]
        rows.append(e[["territory_id", "month"]].assign(theme=k))
    return pd.concat(rows, ignore_index=True)


def shift_month(m: pd.Series, k: int) -> pd.Series:
    return (pd.PeriodIndex(m, freq="M") + k).astype(str)


def boot_ci(d: pd.Series, groups: pd.Series, rng: np.random.Generator) -> tuple[float, float]:
    """95% интервал среднего d при пересэмплировании групп (МО)."""
    g = d.groupby(groups.values).agg(["sum", "size"])
    s, n = g["sum"].values, g["size"].values
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    means = s[idx].sum(1) / n[idx].sum(1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def compare(fc: pd.DataFrame, rng: np.random.Generator) -> dict:
    ale_b, ale_v = fc["ale_base"] * 100, fc["ale_news"] * 100
    d = ale_v - ale_b
    lo, hi = boot_ci(d, fc["territory_id"], rng) if fc["territory_id"].nunique() >= 5 else (np.nan, np.nan)
    changed = d.abs() > 1e-9
    return {
        "n": len(fc), "МО": fc["territory_id"].nunique(),
        "ошибка базы, п.п.": ale_b.mean(), "ошибка с news, п.п.": ale_v.mean(),
        "Δ, п.п.": d.mean(), "95% от": lo, "95% до": hi,
        "ΔMAE, %": (fc["ae_news"].mean() / fc["ae_base"].mean() - 1) * 100,
        "доля улучшений, %": (d[changed] < 0).mean() * 100 if changed.any() else np.nan,
    }


def main(cfg: dict) -> None:
    out = Path(cfg["output_dir"])
    ev = event_table(cfg["data"]["external"]["news"])
    covered = set(pd.read_parquet(cfg["data"]["external"]["news"], columns=["territory_id"])["territory_id"])
    variants = [n for n, c in cfg["models"].items()
                if "base" in c and "news" in resolve_config(cfg["models"], n).get("features", ())
                and "news" not in resolve_config(cfg["models"], c["base"]).get("features", ())]
    key = ["category", "territory_id", "origin", "step", "month"]
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for name in variants:
        base = cfg["models"][name]["base"]
        files = sorted((out / "forecasts" / name).glob("*.parquet"))
        if not files:
            print(f"[skip] {name}: нет прогнозов")
            continue
        v = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        b = pd.concat([pd.read_parquet(out / "forecasts" / base / f.name) for f in files], ignore_index=True)
        fc = v.merge(b[key + ["y_pred"]], on=key, suffixes=("_news", "_base"))
        for s in ("news", "base"):
            fc[f"ae_{s}"] = (fc[f"y_pred_{s}"] - fc["y_true"]).abs()
            fc[f"ale_{s}"] = np.log(fc[f"y_pred_{s}"].clip(lower=1) / fc["y_true"]).abs()
        subsets = {"все МО, шаг 1": fc["step"] == 1}
        known = pd.Series(False, index=fc.index)
        for k, (title, _, _) in EVENTS.items():
            e = ev[ev["theme"] == k]
            for lag, steps in ((1, (1,)), (2, (1, 2))):
                hit = pd.MultiIndex.from_frame(e.assign(month=shift_month(e["month"], lag))[["territory_id", "month"]])
                mask = fc["step"].isin(steps) & pd.MultiIndex.from_frame(fc[["territory_id", "month"]]).isin(hit)
                subsets[f"{title}: {lag}-й месяц после события"] = mask
                known |= mask
        subsets["любое событие, известное модели"] = known
        subsets["МО с городом без событий, шаг 1"] = (fc["step"] == 1) & fc["territory_id"].isin(covered) & ~known
        subsets["МО без города, шаг 1"] = (fc["step"] == 1) & ~fc["territory_id"].isin(covered)
        for label, mask in subsets.items():
            for cat, g in [("все категории вместе", fc[mask]), *fc[mask].groupby("category")]:
                if len(g):
                    rows.append({"variant": name, "base": base, "подвыборка": label, "категория": cat,
                                 **compare(g, rng)})
    if not rows:
        return
    res = pd.DataFrame(rows)
    res.to_csv(out / "news_eval.csv", index=False)

    n_ev = ev.groupby("theme").size()
    lines = ["# Новостные признаки: оценка на подвыборке событий\n",
             "Вариант с группой news против своей базы. Ошибка — |log(прогноз / факт)|, п.п.; "
             "Δ < 0 — с новостями лучше. 95% интервал — бутстрап по МО. "
             "«Доля улучшений» — среди наблюдений, где прогноз изменился.\n",
             "События за 2023–2024 (МО × месяц): "
             + ", ".join(f"{EVENTS[k][0]} — {n_ev.get(k, 0)} (≥{EVENTS[k][1]} статей и ≥{EVENTS[k][2]:g}× обычного)"
                         for k in EVENTS) + ".\n"]
    for name, g in res.groupby("variant", sort=False):
        lines.append(f"\n## {name} против {g['base'].iloc[0]}\n")
        t = g[g["категория"] == "все категории вместе"].drop(columns=["variant", "base", "категория"])
        lines.append("**Все категории вместе**\n\n" + t.round(2).to_markdown(index=False) + "\n")
        t = g[(g["подвыборка"] == "любое событие, известное модели") & (g["категория"] != "все категории вместе")]
        lines.append("\n**Любое событие, известное модели, — по категориям**\n\n"
                     + t.drop(columns=["variant", "base", "подвыборка"]).round(2).to_markdown(index=False) + "\n")
    (out / "news_eval.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/forecast.yaml")
    main(yaml.safe_load(open(p.parse_args().config)))
