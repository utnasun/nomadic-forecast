from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import External


@dataclass
class Context:
    months: list[str]           # все месяцы панели, 'YYYY-MM'
    market_access: np.ndarray   # (n,) статический признак МО, может содержать NaN
    seed: int = 42
    n_jobs: int = -1
    ext: External | None = None  # внешние данные; нужны только моделям с features
    season: np.ndarray | None = None  # (12,) сезонный log-фактор категории по месяцу года; None — без поправки
    national: pd.Series | None = None  # национальный ряд расходов категории, 'YYYY-MM' -> млрд руб. (Chronos)
    # все категории МО (n, все месяцы панели) и своя категория — для признаков structure / cross; модель
    # сама обрезает их до точки старта
    others: dict[str, np.ndarray] | None = None
    category: str | None = None

    def season_factor(self, t0: int, t1: int) -> np.ndarray:
        """exp(сезонный log-фактор) для месяцев t0..t1-1."""
        return np.exp([self.season[self.month_of_year(t) - 1] for t in range(t0, t1)])

    def month_of_year(self, t: int) -> int:
        """Номер месяца года (1..12) для индекса t; работает и за пределами панели."""
        first = int(self.months[0][5:7])
        return (first - 1 + t) % 12 + 1

    def month_start(self, t: int) -> str:
        """Дата начала месяца для индекса t, 'YYYY-MM-01'."""
        y0, m0 = int(self.months[0][:4]), int(self.months[0][5:7])
        k = (m0 - 1) + t
        return f"{y0 + k // 12:04d}-{k % 12 + 1:02d}-01"
