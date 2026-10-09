"""Одна модель на все категории: строки обучения шести категорий вместе, категория — признак.

Признаки, цель и сезонная поправка — как у базовой модели из configs/forecast.yaml (base), но CatBoost обучается
один раз на строках всех категорий (в 6 раз больше примеров для общих закономерностей: возврат к среднему,
эффект уровня, реакция на общий сдвиг месяца). Цель — log_diff: прирост в рублях у категорий несопоставим.
Прогноз по категориям рекурсивный, как в GlobalML; ограничение цели — квантили своей категории.

Запуск: .venv/bin/python -m src.experiments.pooled --base catboost_logdiff_dsp_g123 --name pooled_logdiff_dsp_g123
  → results/forecasts/<name>/<категория>.parquet (формат run_forecast, сравнение — src.experiments.pair)
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import build_panels, complete_territories, load_consumption, load_external, load_market_access, national_season
from src.models import Context, build_model, resolve_config
from src.models.global_ml import _invert, _target, panel_season
from src.evaluation.protocol import forecast_plan


def pooled_forecast(models: dict, ys: dict, ctxs: dict, steps: int) -> dict[str, np.ndarray]:
    """models/ys/ctxs — по категориям: GlobalML, история (n, T), контекст с национальным профилем."""
    prep = {}
    for cat, m in models.items():
        y, ctx = ys[cat], ctxs[cat]
        T = y.shape[1]
        f = np.ones(T + steps)
        if m.deseason == "panel":
            ctx = dataclasses.replace(ctx, season=panel_season(y, ctx, m.season_blend, m.season_unseen))
        if m.deseason and ctx.season is not None:
            f = ctx.season_factor(0, T + steps)
        yd = y / f[:T]
        static = m._static(yd, ctx)
        X = pd.concat([m._features(yd, t, ctx, static, T) for t in range(m.min_t, T)], ignore_index=True)
        z = np.concatenate([_target(yd, t, m.target) for t in range(m.min_t, T)])
        ok = np.isfinite(z)
        prep[cat] = dict(y=yd, ctx=ctx, static=static, f=f, X=X[ok].assign(category=cat), z=z[ok],
                         lo=np.quantile(z[ok], m.clip_quantile), hi=np.quantile(z[ok], 1 - m.clip_quantile))
    m0 = next(iter(models.values()))
    m0.cat_features = m0.cat_features + ["category"]
    X = pd.concat([p["X"] for p in prep.values()], ignore_index=True)
    z = np.concatenate([p["z"] for p in prep.values()])
    _, predict = m0._fit(X, z, next(iter(prep.values()))["ctx"])
    out = {}
    for cat, p in prep.items():
        m, ext, T = models[cat], p["y"].copy(), p["y"].shape[1]
        for t in range(T, T + steps):
            pred = np.clip(predict(m._features(ext, t, p["ctx"], p["static"], T).assign(category=cat)), p["lo"], p["hi"])
            ext = np.column_stack([ext, _invert(ext, t, pred, m.target)])
        out[cat] = ext[:, T:] * p["f"][T:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/forecast.yaml")
    ap.add_argument("--base", default="catboost_logdiff_dsp_g123")
    ap.add_argument("--name", default="pooled_logdiff_dsp_g123")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    base = resolve_config(cfg["models"], a.base)
    assert base["target"] == "log_diff", "общая модель — только с целью log_diff"
    df = load_consumption(cfg["data"]["consumption"])
    ids = complete_territories(df)
    cats = cfg["data"]["categories"]
    panels = build_panels(df, cats, ids)
    first = panels[cats[0]]
    ctx = Context(months=first.months, market_access=load_market_access(cfg["data"]["market_access"], ids),
                  seed=cfg["seed"], n_jobs=cfg["n_jobs"],
                  ext=load_external(cfg["data"]["external"], ids, first.months[0]) if base.get("features") else None)
    sc = cfg["data"]["seasonal"]
    seasons = {c: national_season(sc["national"], k, tuple(sc["years"])) if k else None
               for c, k in sc["profiles"].items()}
    plan = forecast_plan(cfg["protocol"]["horizons"], cfg["protocol"]["min_train"], len(first.months))
    rows = {c: [] for c in cats}
    for T, steps in plan.items():
        t0 = time.time()
        models = {c: build_model(a.base, base) for c in cats}
        fcs = pooled_forecast(models, {c: panels[c].values[:, :T] for c in cats},
                              {c: dataclasses.replace(ctx, season=seasons.get(c)) for c in cats}, steps)
        for c in cats:
            p = panels[c]
            for k in range(steps):
                rows[c].append(pd.DataFrame({"territory_id": p.territory_ids, "origin": T, "step": k + 1,
                                             "month": p.months[T + k], "y_true": p.values[:, T + k],
                                             "y_pred": fcs[c][:, k], "y_last": p.values[:, T - 1]}))
        print(f"T={T}: {time.time() - t0:.0f}s", flush=True)
    out = Path(cfg["output_dir"]) / "forecasts" / a.name
    out.mkdir(parents=True, exist_ok=True)
    for c in cats:
        res = pd.concat(rows[c], ignore_index=True)
        res.insert(0, "category", c)
        res.insert(0, "model", a.name)
        res.to_parquet(out / f"{c}.parquet", index=False)
    print(out)


if __name__ == "__main__":
    main()
