"""Prophet по ряду на МО, параллельно пачками по процессам."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.models.base import Context


def _fit_chunk(y_chunk: np.ndarray, ds: pd.DatetimeIndex, future: pd.DatetimeIndex, params: dict) -> np.ndarray:
    from prophet import Prophet

    for name in ("prophet", "cmdstanpy"):
        logging.getLogger(name).setLevel(logging.ERROR)
    out = np.empty((len(y_chunk), len(future)))
    fut = pd.DataFrame({"ds": future})
    for i, row in enumerate(y_chunk):
        m = Prophet(**params)
        m.fit(pd.DataFrame({"ds": ds, "y": row}))
        out[i] = m.predict(fut)["yhat"].values
    return out


class ProphetModel:
    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def forecast(self, y, steps, ctx: Context):
        T = y.shape[1]
        ds = pd.to_datetime([ctx.month_start(t) for t in range(T)])
        future = pd.to_datetime([ctx.month_start(t) for t in range(T, T + steps)])
        n_jobs = ctx.n_jobs if ctx.n_jobs > 0 else __import__("os").cpu_count()
        chunks = np.array_split(y, n_jobs * 4)
        res = Parallel(n_jobs=n_jobs)(delayed(_fit_chunk)(c, ds, future, self.params) for c in chunks if len(c))
        return np.vstack(res)
