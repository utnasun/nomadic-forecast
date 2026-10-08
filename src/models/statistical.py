"""Статистические модели по ряду на МО (statsforecast): AutoETS, AutoTheta, AutoARIMA.

Сезонность выключена (season_length=1): для годового сезона нужно минимум 2 полных года.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.models.base import Context

os.environ.setdefault("NIXTLA_ID_AS_COL", "1")


class StatsForecastModel:
    def __init__(self, kind: str):
        self.kind = kind

    def _model(self):
        from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, Naive

        if self.kind == "auto_ets":
            return AutoETS(season_length=1, alias="m")
        if self.kind == "auto_theta":
            return AutoTheta(season_length=1, alias="m")
        if self.kind == "auto_arima":
            return AutoARIMA(season_length=1, seasonal=False, alias="m")
        raise ValueError(self.kind)

    def forecast(self, y, steps, ctx: Context):
        from statsforecast import StatsForecast
        from statsforecast.models import Naive

        n, T = y.shape
        ds = pd.to_datetime([ctx.month_start(t) for t in range(T)])
        df = pd.DataFrame({
            "unique_id": np.repeat(np.arange(n), T),
            "ds": np.tile(ds, n),
            "y": y.ravel(),
        })
        sf = StatsForecast(models=[self._model()], freq="MS", n_jobs=ctx.n_jobs, fallback_model=Naive())
        fc = sf.forecast(df=df, h=steps)
        return fc.pivot(index="unique_id", columns="ds", values="m").sort_index().values
