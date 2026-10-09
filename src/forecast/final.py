"""Итоговый прогноз: модель для категории выбирается по обучающим данным, без взгляда на тест.

Запуск (после src.evaluation.run_forecast): .venv/bin/python -m src.forecast.final → results/forecasts/final/, results/final.md

Правило. Доля дисперсии месячных приростов МО, объяснённая общим для всех МО приростом месяца (R² общего
месяца), считается по первым MIN_TRAIN месяцам — тому, что видит самый ранний прогноз. Если сезонность у МО
почти одинакова (R² ≥ THRESHOLD), подходит модель с сезонным профилем по панели (HOMOGENEOUS); если у каждого МО
своя — модель без сезонной поправки (HETEROGENEOUS). По 2023 г.: Все категории 0,76, Продовольствие 0,83 —
против 0,28–0,41 у остальных, порог в этом зазоре ни на что не влияет.

final_online — то же правило для первого прогноза, дальше выбор между двумя кандидатами (HOMOGENEOUS,
HETEROGENEOUS) по уже известным ошибкам: для старта T и горизонта h — MAE прогнозов того же горизонта со стартов
раньше T, у которых факт уже известен (месяц ≤ T−1). Пока таких нет — правило. final_online3 — то же по ошибкам
только последних трёх известных месяцев: январский провал профиля (январь без опыта января) перестаёт
влиять на выбор, когда январь уже в обучении.

Почему не выбор по ошибкам на тесте: на 2024 г. проверено ~90 вариантов, выбор «лучшей в ячейке» по половине
года и онлайн-выбор по прошлым ошибкам проигрывают одной модели (results/selection_rules.md, online.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import build_panels, complete_territories, load_consumption
from src.evaluation.evaluate import horizon_slice

THRESHOLD = 0.5
HOMOGENEOUS = "catboost_logdiff_dsp_g123"
HETEROGENEOUS = "catboost_diff"
NAME = "final"


def common_month_r2(values: np.ndarray) -> float:
    d = np.diff(np.log(np.clip(values, 1, None)), axis=1)
    return float(1 - np.nanvar(d - np.nanmedian(d, axis=0)) / np.nanvar(d))


def online(out: Path, cat: str, rule_model: str, months: list[str], horizons: list[int], mt: int,
           window: int | None = None) -> tuple[pd.DataFrame, list]:
    """Прогноз final_online для категории: для каждой (точка старта, горизонт) — кандидат с меньшей MAE
    на уже реализованных прогнозах того же горизонта; строки с шагом s берутся из среза горизонта h = s-го
    по возрастанию горизонта, покрывающего этот старт (как в src.evaluation.evaluate: для h — старты folds(h), шаги ≤ h)."""
    fcs = {m: pd.read_parquet(out / "forecasts" / m / f"{cat}.parquet") for m in (HOMOGENEOUS, HETEROGENEOUS)}
    parts, log = [], []
    for h in horizons:
        sl = {m: horizon_slice(f, h, mt, mt + 12).assign(ae=lambda x: (x["y_pred"] - x["y_true"]).abs())
              for m, f in fcs.items()}
        for T in sorted(sl[HOMOGENEOUS]["origin"].unique()):
            last_known = months[T - 1]
            first_known = months[max(0, T - window)] if window else months[0]
            past = {m: s.loc[(s["origin"] < T) & (s["month"] <= last_known) & (s["month"] >= first_known), "ae"]
                    for m, s in sl.items()}
            choice = (min(past, key=lambda m: past[m].mean()) if all(len(v) for v in past.values()) else rule_model)
            rows = sl[choice][sl[choice]["origin"] == T].drop(columns="ae").assign(horizon_key=h)
            parts.append(rows)
            log.append((h, months[T], choice))
    f = pd.concat(parts, ignore_index=True)
    # MAE по горизонтам — по выбранным срезам, без смешения горизонтов
    mae = f.assign(ae=(f["y_pred"] - f["y_true"]).abs()).groupby("horizon_key")["ae"].mean()
    log.append(("mae", mae.to_dict(), None))
    # одна строка на (МО, старт, шаг): для пересекающихся горизонтов берём выбор наименьшего горизонта
    f = f.sort_values("horizon_key").drop_duplicates(["territory_id", "origin", "step"]).drop(columns="horizon_key")
    return f.assign(model=NAME + "_online" + (str(window) if window else "")), log


def main(cfg: dict) -> None:
    out = Path(cfg["output_dir"])
    df = load_consumption(cfg["data"]["consumption"])
    ids = complete_territories(df) if cfg["data"]["only_complete"] else None
    panels = build_panels(df, cfg["data"]["categories"], ids)
    mt = cfg["protocol"]["min_train"]
    rows = []
    (out / "forecasts" / NAME).mkdir(parents=True, exist_ok=True)
    for cat, p in panels.items():
        r2 = common_month_r2(p.values[:, :mt])
        model = HOMOGENEOUS if r2 >= THRESHOLD else HETEROGENEOUS
        fc = pd.read_parquet(out / "forecasts" / model / f"{cat}.parquet").assign(model=NAME)
        fc.to_parquet(out / "forecasts" / NAME / f"{cat}.parquet", index=False)
        rows.append({"категория": cat, f"R² общего месяца (первые {mt} мес.)": round(r2, 2), "модель": model})
        fo3, log3 = online(out, cat, model, p.months, cfg["protocol"]["horizons"], mt, window=3)
        (out / "forecasts" / (NAME + "_online3")).mkdir(parents=True, exist_ok=True)
        fo3.to_parquet(out / "forecasts" / (NAME + "_online3") / f"{cat}.parquet", index=False)
        mae3 = next(m for h, m, c in log3 if h == "mae")
        fo, log = online(out, cat, model, p.months, cfg["protocol"]["horizons"], mt)
        (out / "forecasts" / (NAME + "_online")).mkdir(parents=True, exist_ok=True)
        fo.to_parquet(out / "forecasts" / (NAME + "_online") / f"{cat}.parquet", index=False)
        mae_on = next(m for h, m, c in log if h == "mae")
        log = [x for x in log if x[0] != "mae"]
        for h in cfg["protocol"]["horizons"]:
            base = horizon_slice(fc, h, mt, mt + 12)
            rows[-1][f"h={h}: final / online / online3"] = (f"{(base['y_pred'] - base['y_true']).abs().mean():.0f} / "
                                                            f"{mae_on[h]:.0f} / {mae3[h]:.0f}")
        sw = [f"h={h} с {m}" for h, m, c in log if c != model]
        rows[-1]["final_online: старты с другой моделью"] = ", ".join(sw) if sw else "—"
    t = pd.DataFrame(rows)
    text = ("# Итоговый прогноз `final`\n\nПравило и обоснование — в docstring `src/forecast/final.py`.\n\n"
            + t.to_markdown(index=False) + "\n")
    (out / "final.md").write_text(text)
    print(text)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/forecast.yaml")
    main(yaml.safe_load(open(p.parse_args().config)))
