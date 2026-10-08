"""Chronos-2 (amazon/chronos-2): предобученная модель рядов, прогноз без обучения на наших данных (zero-shot).

Прогноз — медиана распределения, сразу на все шаги (без рекурсии). Варианты (configs/forecast.yaml):
  covariate: national — к истории МО добавляется национальный ряд расходов СберИндекса той же категории
      (data/external/national_monthly.parquet, с 2018-12) как прошлая ковариата. История МО до начала
      панели — пропуски, поэтому контекст удлиняется на годы национального ряда: модель видит многолетнюю
      сезонность и рост. Ряд берётся только по последний месяц обучения — утечки нет;
  cross_learning — МО прогнозируются совместно пакетами по batch_size (соседние territory_id, т.е. в основном
      один регион), модель переносит закономерности между рядами пакета;
  deseason — как у GlobalML: ряд делится на национальный сезонный профиль, прогноз умножается обратно.
Работает на CPU; веса (~480 МБ) скачиваются с HuggingFace при первом запуске.
"""
from __future__ import annotations

import numpy as np

from src.models.base import Context

_PIPELINES: dict[str, object] = {}


def _pipeline(model_id: str):
    if model_id not in _PIPELINES:
        import torch
        from chronos import Chronos2Pipeline

        torch.set_num_threads(max(1, torch.get_num_threads()))
        _PIPELINES[model_id] = Chronos2Pipeline.from_pretrained(model_id, device_map="cpu", dtype=torch.float32)
    return _PIPELINES[model_id]


class ChronosModel:
    def __init__(self, model_id: str = "amazon/chronos-2", covariate: str | None = None,
                 cross_learning: bool = False, batch_size: int = 100, deseason: bool = False):
        if covariate not in (None, "national"):
            raise ValueError(f"Неизвестная ковариата: {covariate}")
        self.model_id, self.covariate = model_id, covariate
        self.cross_learning, self.batch_size, self.deseason = cross_learning, batch_size, deseason

    def forecast(self, y: np.ndarray, steps: int, ctx: Context) -> np.ndarray:
        if not self.deseason or ctx.season is None:
            return self._forecast(y, steps, ctx)
        T = y.shape[1]
        f = ctx.season_factor(0, T + steps)
        return self._forecast(y / f[:T], steps, ctx) * f[T:]

    def _inputs(self, y: np.ndarray, ctx: Context):
        if self.covariate is None:
            return [row.astype(np.float32) for row in y]
        if ctx.national is None:
            raise ValueError("Нужен национальный ряд: data.seasonal.national в конфиге")
        T = y.shape[1]
        nat = ctx.national
        before = nat[nat.index < ctx.months[0]]
        cov = np.concatenate([before.values, nat.reindex(ctx.months[:T]).values]).astype(np.float32)
        if np.isnan(cov[len(before):]).any():
            raise ValueError("В национальном ряду нет части месяцев панели")
        pad = np.full(len(before), np.nan, dtype=np.float32)
        return [{"target": np.concatenate([pad, row.astype(np.float32)]), "past_covariates": {"national": cov}}
                for row in y]

    def _forecast(self, y: np.ndarray, steps: int, ctx: Context) -> np.ndarray:
        pipe = _pipeline(self.model_id)
        preds = pipe.predict(self._inputs(y, ctx), prediction_length=steps, batch_size=self.batch_size,
                             cross_learning=self.cross_learning)
        med = pipe.quantiles.index(0.5)
        out = np.stack([p[0, med, :steps].numpy() for p in preds])
        return np.clip(out, 1.0, None)
