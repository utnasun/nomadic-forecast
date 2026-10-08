"""Календарные признаки с плавающей датой: Рамадан.

Рамадан сдвигается на ~11 дней в год, поэтому «как год назад» его не ловит. В EDA на Северном Кавказе
общепит в месяц поста ниже на 10–27 п.п., после поста — отскок.

Признаки для целевого месяца t (дата известна заранее — доступны на любом горизонте):
  ramadan_t    — доля дней месяца t, пришедшихся на пост, × доля мусульман в регионе;
  ramadan_prev — то же для месяца t−1 (отскок после поста, Ураза-байрам);
  ramadan_next — то же для месяца t+1 (подготовка к посту).
"""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

# Первый и последний день поста (Ураза-байрам — на следующий день); в отдельных регионах даты
# могут отличаться на день — для месячного признака это несущественно
RAMADAN = {
    2022: ("2022-04-02", "2022-05-01"),
    2023: ("2023-03-23", "2023-04-20"),
    2024: ("2024-03-11", "2024-04-09"),
    2025: ("2025-03-01", "2025-03-29"),
    2026: ("2026-02-18", "2026-03-19"),
}
# Доля народов, традиционно исповедующих ислам, по переписи 2010 г., округлённо.
# Код региона — как в справочнике МО СберИндекса (territory_static.region).
MUSLIM_SHARE = {
    "5": 0.95,   # Дагестан
    "6": 0.95,   # Ингушетия
    "20": 0.95,  # Чечня
    "7": 0.7,    # Кабардино-Балкария
    "9": 0.6,    # Карачаево-Черкесия
    "16": 0.5,   # Татарстан
    "2": 0.5,    # Башкортостан
    "1": 0.25,   # Адыгея
}


def ramadan_share(month: str) -> float:
    """Доля дней месяца 'YYYY-MM' внутри Рамадана."""
    y, m = int(month[:4]), int(month[5:7])
    if y not in RAMADAN:
        raise ValueError(f"Нет дат Рамадана для {y} г.")
    days = pd.date_range(f"{month}-01", periods=calendar.monthrange(y, m)[1], freq="D")
    a, b = (pd.Timestamp(x) for x in RAMADAN[y])
    return float(((days >= a) & (days <= b)).mean())


def ramadan_features(ctx, region: np.ndarray, t: int) -> dict[str, np.ndarray]:
    w = pd.Series(region).astype(str).map(MUSLIM_SHARE).fillna(0.0).values
    month = lambda k: ctx.month_start(k)[:7]  # noqa: E731
    return {"ramadan_t": ramadan_share(month(t)) * w,
            "ramadan_prev": ramadan_share(month(t - 1)) * w,
            "ramadan_next": ramadan_share(month(t + 1)) * w}
