"""Модели прогноза. Единый интерфейс:

    model.forecast(y_train, steps, ctx) -> np.ndarray формы (n, steps)

y_train — матрица (n МО, T месяцев) известной истории, ctx — src.models.base.Context.
"""
from src.models.base import Context
from src.models.registry import build_model, resolve_config

__all__ = ["Context", "build_model", "resolve_config"]
