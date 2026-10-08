"""Простые модели: наивная, скользящие средние, дрейф, сезонная наивная."""
from __future__ import annotations

import numpy as np

from src.models.base import Context


class Naive:
    """Последнее известное значение."""

    def forecast(self, y, steps, ctx: Context):
        return np.repeat(y[:, -1:], steps, axis=1)


class MeanWindow:
    """Среднее за последние `window` месяцев."""

    def __init__(self, window: int):
        self.window = window

    def forecast(self, y, steps, ctx: Context):
        with np.errstate(all="ignore"):  # пропуски в окне (неполные МО) не учитываются
            m = np.nanmean(y[:, -self.window:], axis=1, keepdims=True) if np.isnan(y).any() \
                else y[:, -self.window:].mean(axis=1, keepdims=True)
        return np.repeat(m, steps, axis=1)


class Drift:
    """Последнее значение + средний прирост за всю историю × номер шага."""

    def forecast(self, y, steps, ctx: Context):
        slope = (y[:, -1] - y[:, 0]) / (y.shape[1] - 1)
        return y[:, -1:] + slope[:, None] * np.arange(1, steps + 1)


class SeasonalNaive:
    """Значение того же месяца год назад. Для шага > 12 берёт свой же прогноз."""

    def forecast(self, y, steps, ctx: Context):
        ext = y.copy()
        for _ in range(steps):
            ext = np.column_stack([ext, ext[:, -12]])
        return ext[:, y.shape[1]:]


class SeasonalNaiveGrowth:
    """y[t-12] × рост год к году за последние m месяцев, m = min(growth_window, T - 12).

    При T = 12 (обучение только на 2023) рост посчитать нельзя, и модель совпадает
    с сезонной наивной.
    """

    def __init__(self, growth_window: int = 3):
        self.growth_window = growth_window

    def forecast(self, y, steps, ctx: Context):
        m = min(self.growth_window, y.shape[1] - 12)
        g = y[:, -m:].sum(1) / y[:, -12 - m:-12].sum(1) if m > 0 else np.ones(len(y))
        return SeasonalNaive().forecast(y, steps, ctx) * g[:, None]
