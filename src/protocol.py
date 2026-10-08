"""Протокол оценки с расширяющимся окном.

Для горизонта h точки старта T = min_train, min_train + h, ... (T — число месяцев в обучении),
прогноз на шаги 1..h. Каждый прогноз строится рекурсивно из точки T, поэтому прогноз
из точки T на k шагов не зависит от того, для какого горизонта он считается.
Значит, достаточно один раз сделать прогноз из каждой нужной точки старта
на максимальное число нужных из неё шагов и переиспользовать его для всех горизонтов.
"""
from __future__ import annotations


def folds(horizon: int, min_train: int, n_months: int) -> list[int]:
    """Точки старта (длины обучающего окна) для горизонта."""
    return list(range(min_train, n_months - horizon + 1, horizon))


def forecast_plan(horizons: list[int], min_train: int, n_months: int) -> dict[int, int]:
    """{точка старта T: сколько шагов прогнозировать из неё}."""
    plan: dict[int, int] = {}
    for h in horizons:
        for T in folds(h, min_train, n_months):
            plan[T] = max(plan.get(T, 0), h)
    return dict(sorted(plan.items()))
