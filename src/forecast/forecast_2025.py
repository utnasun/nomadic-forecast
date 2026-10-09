"""Прогноз на 2025 г. моделью final: обучение на всех 24 месяцах, январь–декабрь 2025, 90%-интервалы.

Запуск (после src.forecast.final, src.forecast.intervals, src.shocks.detector): .venv/bin/python -m src.forecast.forecast_2025
  → results/forecast_2025/<категория>.parquet, results/forecast_2025.md

Модель для категории и шага — как выбрал бы final_online на конец 2024 г.: из двух кандидатов (src.forecast.final) та, у
которой меньше MAE на всех прогнозах 2024 г. того же горизонта (шаг s → горизонт: наименьший из 1, 3, 6, 12, не
меньше s). Отличие от final — Транспорт (профиль по панели на всех шагах) и первый месяц Маркетплейсов; у
остальных категорий выбор совпадает с правилом. Интервал — квантили 5% / 95% остатков log(факт / прогноз)
модели final за весь 2024 г. по (категория, корзина горизонта, квинтиль волатильности МО); шаг s прогноза берёт
корзину наименьшего горизонта ≥ s из протокола (1, 3, 6, 12). Если в декабре 2024 г. у МО был шок
(src.shocks.detector), интервал шире в SHOCK_K раз (оценено в src.forecast.intervals). Покрытие на шагах 7–12 при проверке
было ниже цели (≈78%), так что дальние интервалы стоит читать как нижнюю оценку неопределённости.

--seen-only: кандидаты сравниваются только на стартах T ≥ 13 — когда январь уже был в обучении, как в 2025 г.
У старта T = 12 модель с профилем берёт неизвестный переход «декабрь → январь» из национального профиля и
сильно ошибается у Здоровья и Общепита; в 2025 г. этой ошибки не будет. Для h = 12 таких стартов нет — берётся
выбор для h = 6. Результат — results/forecast_2025_seen/ и results/forecast_2025_seen.md.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import (build_panels, complete_territories, load_consumption, load_external, load_market_access,
                      national_season)
from src.evaluation.evaluate import horizon_slice
from src.forecast.final import HETEROGENEOUS, HOMOGENEOUS
from src.forecast.intervals import HI, LO, N_VOL, residuals
from src.models import Context, build_model, resolve_config

STEPS = 12
SHOCK_K = 1.3


def bucket(step: int, horizons: list[int]) -> int:
    return min(h for h in horizons if h >= step)


def choose(out: Path, cat: str, hs: list[int], mt: int, seen_only: bool) -> dict[int, str]:
    """Горизонт → модель с меньшей MAE на прогнозах 2024 г. этого горизонта (seen_only — только старты T ≥ mt + 1)."""
    choice = {}
    for h in hs:
        hh = h if not seen_only or h < 12 else max(x for x in hs if x < 12)
        mae = {}
        for m in (HOMOGENEOUS, HETEROGENEOUS):
            x = horizon_slice(pd.read_parquet(out / "forecasts" / m / f"{cat}.parquet"), hh, mt, mt + 12)
            if seen_only:
                x = x[x["origin"] > mt]
            mae[m] = (x["y_pred"] - x["y_true"]).abs().mean()
        choice[h] = min(mae, key=mae.get)
    return choice


def main(cfg: dict, seen_only: bool = False) -> None:
    suffix = "_seen" if seen_only else ""
    OUT = Path(f"results/forecast_2025{suffix}")
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_consumption(cfg["data"]["consumption"])
    ids = complete_territories(df)
    cats = cfg["data"]["categories"]
    panels = build_panels(df, cats, ids)
    first = panels[cats[0]]
    ctx = Context(months=first.months, market_access=load_market_access(cfg["data"]["market_access"], ids),
                  seed=cfg["seed"], n_jobs=cfg["n_jobs"],
                  ext=load_external(cfg["data"]["external"], ids, first.months[0]))
    sc = cfg["data"]["seasonal"]
    mt, hs = cfg["protocol"]["min_train"], cfg["protocol"]["horizons"]
    future = [str(pd.Period(first.months[-1], "M") + k) for k in range(1, STEPS + 1)]

    r = residuals(cfg)  # остатки final за 2024 г. с квинтилем волатильности
    q = r.groupby(["category", "h", "vol_q"])["r"].quantile([LO, HI]).unstack()
    q.columns = ["lo", "hi"]
    shocks = pd.read_parquet(Path(cfg["output_dir"]) / "shocks_v2" / "scores.parquet")
    dec_shock = set(shocks.loc[(shocks["уровень"] == "шок") & (shocks["month"] == first.months[-1]), "territory_id"])

    rows = []
    out = Path(cfg["output_dir"])
    for cat, p in panels.items():
        choice = choose(out, cat, hs, mt, seen_only)
        kind = sc["profiles"].get(cat)
        season = national_season(sc["national"], kind, tuple(sc["years"])) if kind else None
        fcs = {m: build_model(m, resolve_config(cfg["models"], m)).forecast(p.values, STEPS,
                                                                            dataclasses.replace(ctx, season=season))
               for m in set(choice.values())}
        d = np.diff(np.log(np.clip(p.values, 1, None)), axis=1)
        vol = pd.Series(np.nanstd(d, axis=1), index=p.territory_ids)
        vol_q = pd.qcut(vol.rank(method="first"), N_VOL, labels=False)
        for k in range(STEPS):
            h = bucket(k + 1, hs)
            model = choice[h]
            fc = fcs[model]
            qq = q.loc[(cat, h)].reindex(vol_q.values)
            mult = np.where(np.isin(p.territory_ids, list(dec_shock)), SHOCK_K, 1.0)
            mid, half = (qq["lo"].values + qq["hi"].values) / 2, (qq["hi"].values - qq["lo"].values) / 2 * mult
            rows.append(pd.DataFrame({
                "category": cat, "territory_id": p.territory_ids, "month": future[k], "step": k + 1,
                "y_pred": fc[:, k], "lo90": fc[:, k] * np.exp(mid - half), "hi90": fc[:, k] * np.exp(mid + half),
                "y_last": p.values[:, -1], "model": model, "shock_dec_2024": np.isin(p.territory_ids, list(dec_shock))}))
        print(f"{cat}: " + ", ".join(f"h≤{h}: {m}" for h, m in choice.items()))
    res = pd.concat(rows, ignore_index=True)
    for cat, g in res.groupby("category"):
        g.to_parquet(OUT / f"{cat}.parquet", index=False)
    s = res.groupby(["category", "month"]).agg(прогноз=("y_pred", "median"), lo90=("lo90", "median"), hi90=("hi90", "median"))
    g = (res.groupby(["category", "step"]).apply(lambda x: np.median(np.log(x["y_pred"] / x["y_last"])) * 100)
         .unstack().round(1))
    text = ["# Прогноз на 2025 г. (final с выбором на конец 2024 г., обучение на 2023–2024)\n",
            "Как построен — в docstring `src/forecast/forecast_2025.py`" + (" (выбор модели `--seen-only`)" if seen_only else "")
            + f". Данные по МО — `{OUT}/<категория>.parquet`"
            " (прогноз, 90%-интервал, модель, был ли шок в декабре 2024).\n",
            f"МО с шоком в декабре 2024 г. (интервал ×{SHOCK_K}): {len(dec_shock)}.\n",
            "Модель по шагам: " + "; ".join(f"{c} — " + ", ".join(sorted(set(g["model"]))) for c, g in res.groupby("category"))
            + ".\n",
            "## Медианный прогноз по МО к декабрю 2024 г., % (log-разница), по месяцам 2025 г.\n",
            g.rename(columns=lambda k: future[k - 1]).to_markdown(), "",
            "## Медианы по МО: прогноз и 90%-интервал, руб. на жителя\n", s.round(0).reset_index().to_markdown(index=False), ""]
    Path(f"results/forecast_2025{suffix}.md").write_text("\n".join(text))
    print("\n".join(text[:5]))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seen-only", action="store_true")
    main(yaml.safe_load(open("configs/forecast.yaml")), ap.parse_args().seen_only)
