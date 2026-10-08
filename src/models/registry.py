from __future__ import annotations

from src.models.chronos_model import ChronosModel
from src.models.baselines import Drift, MeanWindow, Naive, SeasonalNaive, SeasonalNaiveGrowth
from src.models.global_ml import GlobalML
from src.models.prophet_model import ProphetModel
from src.models.statistical import StatsForecastModel


def resolve_config(models: dict, name: str) -> dict:
    """Конфиг модели; ключ base — унаследовать параметры другой модели и переопределить часть из них."""
    cfg = dict(models[name])
    base = cfg.pop("base", None)
    return {**resolve_config(models, base), **cfg} if base else cfg


def build_model(name: str, cfg: dict):
    if name == "naive":
        return Naive()
    if name.startswith("mean_"):
        return MeanWindow(cfg["window"])
    if name == "drift":
        return Drift()
    if name == "seasonal_naive":
        return SeasonalNaive()
    if name == "seasonal_naive_growth":
        return SeasonalNaiveGrowth(cfg.get("growth_window", 3))
    if name in ("auto_ets", "auto_theta", "auto_arima"):
        return StatsForecastModel(name)
    if name.startswith("prophet"):
        return ProphetModel(cfg.get("params"))
    if cfg.get("kind") in ("catboost", "ridge", "tabpfn", "tabfm"):
        return GlobalML(cfg["kind"], cfg["target"], cfg.get("params", {}),
                        min_t=cfg.get("min_t", 2), min_unique=cfg.get("min_unique", 6),
                        clip_quantile=cfg.get("clip_quantile", 0.005), features=cfg.get("features", ()),
                        deseason=cfg.get("deseason", False), strategy=cfg.get("strategy", "recursive"),
                        seed=cfg.get("seed"), weight=cfg.get("weight"),
                        winsor=cfg.get("winsor"), despike=cfg.get("despike"), per_day=cfg.get("per_day"),
                        season_blend=cfg.get("season_blend"), recency=cfg.get("recency"),
                        season_unseen=cfg.get("season_unseen", "national"), rebase=cfg.get("rebase"),
                        drop_jumps=cfg.get("drop_jumps"))
    if cfg.get("kind") == "chronos":
        return ChronosModel(cfg.get("model_id", "amazon/chronos-2"), covariate=cfg.get("covariate"),
                            cross_learning=cfg.get("cross_learning", False), batch_size=cfg.get("batch_size", 100),
                            deseason=cfg.get("deseason", False))
    raise ValueError(f"Неизвестная модель: {name}")
