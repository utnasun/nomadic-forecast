"""Восстановление после шока: какая доля разрыва с регионом остаётся через k месяцев.

Запуск (после src.shocks.detector): .venv/bin/python -m src.shocks.recovery → results/shocks_v2/recovery.md, recovery.csv

Как у Всемирного банка (ураганы по данным Mastercard) и BBVA (время восстановления после стихийных бедствий):
контроль — медиана МО того же региона. g_t = log y_t(МО) − медиана log y_t по МО региона, категория — главная
категория шока. Величина шока s = g_m − g_{m−1}; доля, оставшаяся через k месяцев, — (g_{m+k} − g_{m−1}) / s
(1 — сдвиг сохранился, 0 — вернулся к региону). Восстановление — первый месяц, когда осталось меньше трети.

Контроль на «возврат к среднему»: такие же по величине (≥ 5%) скачки разрыва с регионом без отметки детектора.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import build_panels, complete_territories, load_consumption

OUT = Path("results/shocks_v2")
MIN_JUMP = 0.05
K = 6
SIZES = [0.05, 0.10, 0.20, 0.40, np.inf]
SIZE_LABELS = ["5–10%", "10–20%", "20–40%", ">40%"]
N_PLACEBO = 400  # скачков без отметки на категорию


def gaps(panels: dict, region: pd.Series) -> dict[str, pd.DataFrame]:
    out = {}
    for c, p in panels.items():
        L = pd.DataFrame(np.log(np.clip(p.values, 1, None)), index=p.territory_ids, columns=p.months)
        out[c] = L - L.groupby(region.reindex(L.index).values).transform("median")
    return out


def paths(events: pd.DataFrame, G: dict[str, pd.DataFrame], group: str) -> pd.DataFrame:
    """Доля оставшегося разрыва через 1..K месяцев для событий (territory_id, month, category)."""
    rows = []
    for tid, month, cat in events[["territory_id", "month", "category"]].itertuples(index=False):
        g = G[cat].loc[tid]
        j = g.index.get_loc(month)
        s = g.iloc[j] - g.iloc[j - 1] if j >= 1 else np.nan
        if not abs(s) >= MIN_JUMP:
            continue
        r = {"группа": group, "territory_id": tid, "month": month, "category": cat,
             "направление": "вниз" if s < 0 else "вверх", "величина": s}
        r.update({k: (g.iloc[j + k] - g.iloc[j - 1]) / s if j + k < len(g) else np.nan for k in range(1, K + 1)})
        rows.append(r)
    return pd.DataFrame(rows)


def recovery_month(x: pd.DataFrame) -> pd.Series:
    """Первый k, когда осталось меньше трети; K+1 — не восстановилось за K месяцев; NaN — истории мало."""
    left = x[list(range(1, K + 1))]
    done = left.abs() < 1 / 3
    first = done.idxmax(axis=1).where(done.any(axis=1), K + 1)
    return first.where(done.any(axis=1) | left[K].notna())


def main(cfg: dict) -> None:
    df = load_consumption(cfg["data"]["consumption"])
    ids = complete_territories(df)
    panels = build_panels(df, cfg["data"]["categories"], ids)
    region = pd.read_parquet(cfg["data"]["external"]["static"]).set_index("territory_id")["region"].reindex(ids)
    G = gaps(panels, region)
    s = pd.read_parquet(OUT / "scores.parquet")
    shocks = s[s["уровень"] == "шок"].rename(columns={"главная категория": "category"})
    A = paths(shocks, G, "шок")
    flagged = set(zip(shocks["territory_id"], shocks["month"]))
    first = shocks["month"].min()
    placebo = []
    for c, g in G.items():
        d = g.diff(axis=1).stack().rename("jump").reset_index()
        d.columns = ["territory_id", "month", "jump"]
        d = d[(d["jump"].abs() >= MIN_JUMP) & (d["month"] >= first)]
        d = d[[k not in flagged for k in zip(d["territory_id"], d["month"])]]
        placebo.append(d.sample(min(len(d), N_PLACEBO), random_state=0).assign(category=c))
    B = paths(pd.concat(placebo), G, "скачок без отметки")
    X = pd.concat([A, B], ignore_index=True)
    X["размер"] = pd.cut(X["величина"].abs(), SIZES, labels=SIZE_LABELS, right=False)
    X["восстановление, мес."] = recovery_month(X)
    X.to_csv(OUT / "recovery.csv", index=False)

    cols = [1, 2, 3, 6]
    by_size = X.groupby(["группа", "размер"], observed=True)[cols].median().round(2)
    by_size["n"] = X.groupby(["группа", "размер"], observed=True).size()
    sh = X[X["группа"] == "шок"]
    by_cat = sh.groupby(["category", "направление"])[cols].median().round(2)
    by_cat["n"] = sh.groupby(["category", "направление"]).size()
    full = X[X["восстановление, мес."].notna()]
    rec = full.groupby(["группа", "category"])["восстановление, мес."].agg(
        **{"за 1 мес., %": lambda r: 100 * (r <= 1).mean(), "за 3 мес., %": lambda r: 100 * (r <= 3).mean(),
           f"не за {K} мес., %": lambda r: 100 * (r > K).mean(), "n": "size"}).round(0)
    text = [
        "# Восстановление после шока\n",
        "Метод — в docstring `src/shocks/recovery.py`: разрыв МО с медианой своего региона в главной категории шока, "
        "доля разрыва, оставшаяся через k месяцев (1 — сдвиг сохранился, 0 — вернулся). Контроль — такие же по "
        f"величине (≥ {MIN_JUMP:.0%}) скачки разрыва без отметки детектора. Шоки — `scores.parquet` "
        "(с июля 2023 г.; через 6 месяцев видны шоки по июнь 2024 г.).\n",
        "## Доля оставшегося разрыва, медиана\n", by_size.to_markdown(), "",
        "## Шоки по категории и направлению\n", by_cat.to_markdown(), "",
        "## Время восстановления (осталось меньше трети разрыва)\n", rec.to_markdown(), ""]
    (OUT / "recovery.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main(yaml.safe_load(open("configs/forecast.yaml")))
