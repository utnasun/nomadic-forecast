"""Метрики качества прогноза.

R2        — по уровням (завышен: основную дисперсию даёт разница уровней между МО).
R2_growth — по приросту к последнему известному значению на точке старта:
            log(y / y_last). Показывает, угадывает ли модель изменение, а не уровень.
"""
from __future__ import annotations

import numpy as np


def compute(y: np.ndarray, p: np.ndarray, y_last: np.ndarray) -> dict[str, float]:
    e = p - y
    g_true = np.log(y / y_last)
    g_pred = np.log(np.clip(p, 1e-9, None) / y_last)
    return {
        "MAE": np.mean(np.abs(e)),
        "RMSE": np.sqrt(np.mean(e**2)),
        "WAPE": np.sum(np.abs(e)) / np.sum(np.abs(y)) * 100,
        "MAPE": np.mean(np.abs(e) / np.abs(y)) * 100,
        "sMAPE": np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(p))) * 100,
        "ME": np.mean(e),
        "R2": 1 - np.sum(e**2) / np.sum((y - y.mean()) ** 2),
        "R2_growth": 1 - np.sum((g_pred - g_true) ** 2) / np.sum((g_true - g_true.mean()) ** 2),
        "n": len(y),
    }
