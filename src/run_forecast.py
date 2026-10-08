"""Прогон моделей по протоколу расширяющегося окна.

Запуск:
  .venv/bin/python -m src.run_forecast --config configs/forecast.yaml
  .venv/bin/python -m src.run_forecast --models naive catboost_logdiff --categories "Все категории"
Прогнозы кэшируются в results/forecasts/<model>/<category>.parquet; --force пересчитывает.
Метрики: .venv/bin/python -m src.evaluate
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import (build_panels, complete_territories, load_consumption, load_external, load_market_access,
                      national_season)
from src.models import Context, build_model, resolve_config
from src.models.global_ml import QUANTILES
from src.protocol import forecast_plan


def run(cfg: dict, models: list[str] | None, categories: list[str] | None, force: bool) -> None:
    df = load_consumption(cfg["data"]["consumption"])
    ids = complete_territories(df) if cfg["data"]["only_complete"] else None
    cats = categories or cfg["data"]["categories"]
    panels = build_panels(df, cfg["data"]["categories"], ids)
    out_root = Path(cfg["output_dir"]) / "forecasts"

    names = models or [m for m, c in cfg["models"].items() if c.get("enabled", True)]
    model_cfgs = {name: resolve_config(cfg["models"], name) for name in names}

    first = panels[cats[0]]
    need_ext = any(c.get("features") for c in model_cfgs.values())
    ctx = Context(months=first.months,
                  market_access=load_market_access(cfg["data"]["market_access"], first.territory_ids),
                  seed=cfg["seed"], n_jobs=cfg["n_jobs"],
                  ext=load_external(cfg["data"]["external"], first.territory_ids, first.months[0]) if need_ext else None,
                  others={c: p.values for c, p in panels.items()})
    seasons = {}
    if any(c.get("deseason") or "pseason" in c.get("features", ()) for c in model_cfgs.values()):
        sc = cfg["data"]["seasonal"]
        seasons = {cat: national_season(sc["national"], kind, tuple(sc["years"]))
                   for cat, kind in sc["profiles"].items() if kind}
    nationals = {}
    if any(c.get("covariate") == "national" for c in model_cfgs.values()):
        sc = cfg["data"]["seasonal"]
        nat = pd.read_parquet(sc["national"])
        # у Маркетплейсов нет своего профиля — берём общий ряд
        nationals = {cat: nat[nat["type"] == (sc["profiles"].get(cat) or "Всего")].set_index("month")["value"]
                     .sort_index() for cat in cats}
    plan = forecast_plan(cfg["protocol"]["horizons"], cfg["protocol"]["min_train"], len(first.months))
    print(f"МО: {len(first.territory_ids)}, план {{старт: шагов}}: {plan}")

    for name in names:
        for cat in cats:
            path = out_root / name / f"{cat}.parquet"
            if path.exists() and not force:
                print(f"[skip] {name} / {cat}")
                continue
            panel = panels[cat]
            model = build_model(name, model_cfgs[name])
            t0 = time.time()
            rows = []
            for T, steps in plan.items():
                fc = model.forecast(panel.values[:, :T], steps,
                                    dataclasses.replace(ctx, season=seasons.get(cat), national=nationals.get(cat),
                                                        category=cat))
                assert fc.shape == (len(panel.territory_ids), steps), (name, fc.shape)
                q1 = getattr(model, "q1", None)  # квантили прогноза на 1-й шаг (TabPFN)
                for k in range(steps):
                    rows.append(pd.DataFrame({
                        "territory_id": panel.territory_ids,
                        "origin": T,                       # число месяцев в обучении
                        "step": k + 1,
                        "month": panel.months[T + k],
                        "y_true": panel.values[:, T + k],
                        "y_pred": fc[:, k],
                        "y_last": panel.values[:, T - 1],  # последнее известное значение
                        **({f"q{round(q * 100):02d}": q1[:, j] if k == 0 else np.nan for j, q in enumerate(QUANTILES)}
                           if q1 is not None else {}),
                    }))
            res = pd.concat(rows, ignore_index=True)
            res.insert(0, "category", cat)
            res.insert(0, "model", name)
            path.parent.mkdir(parents=True, exist_ok=True)
            res.to_parquet(path, index=False)
            n_bad = int((~np.isfinite(res["y_pred"])).sum())
            print(f"[done] {name:24s} / {cat:22s} {time.time() - t0:7.1f}s" + (f"  НЕ КОНЕЧНЫХ ПРОГНОЗОВ: {n_bad}" if n_bad else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/forecast.yaml")
    p.add_argument("--models", nargs="*")
    p.add_argument("--categories", nargs="*")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    run(yaml.safe_load(open(a.config)), a.models, a.categories, a.force)
