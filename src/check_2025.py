"""Проверка прогноза на 2025 г. по национальному ряду СберИндекса (данных по МО за 2025 г. нет).

Запуск (после src.forecast_2025): .venv/bin/python -m src.check_2025 → results/check_2025.md

Прогноз по МО сводится в агрегат панели: сумма прогноза расходов на жителя × население МО (2022 г.), и
сравнивается рост к декабрю 2024 г. с ростом национального ряда расходов того же типа за те же месяцы. Ряды разные
(вся страна против 2016 МО, расходы всего против расходов на жителя, своя методика оценки), поэтому та же
сверка делается для 2024 г., где известен факт панели: она показывает, насколько точно агрегат панели вообще
следует за национальным рядом. Ориентир — «как 2024 г. × рост»: агрегат панели за тот же месяц 2024 г., умноженный
на годовой рост последних трёх месяцев 2024 г., к декабрю 2024 г.

Соответствие категорий национальным типам: «Все категории» — «Всего», Продовольствие — «Продовольственные
товары», Общественное питание — «Общественное питание» (точные); Здоровье и Маркетплейсы — «Непродовольственные
товары», Транспорт — «Услуги» (грубые: национальный тип шире категории).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import build_panels, complete_territories, load_consumption

NATIONAL = {"Все категории": ("Всего", True), "Продовольствие": ("Продовольственные товары", True),
            "Общественное питание": ("Общественное питание", True),
            "Здоровье": ("Непродовольственные товары", False), "Маркетплейсы": ("Непродовольственные товары", False),
            "Транспорт": ("Услуги", False)}


def growth_table(cfg: dict, forecast_dir: Path) -> pd.DataFrame:
    df = load_consumption(cfg["data"]["consumption"])
    panels = build_panels(df, cfg["data"]["categories"], complete_territories(df))
    st = pd.read_parquet(cfg["data"]["external"]["static"]).set_index("territory_id")
    nat = pd.read_parquet(cfg["data"]["seasonal"]["national"]).pivot(index="month", columns="type", values="value")
    rows = []
    for cat, p in panels.items():
        pop = np.exp(st["log_pop"].reindex(p.territory_ids)).values
        ok = np.isfinite(pop)
        agg = pd.Series((p.values[ok] * pop[ok, None]).sum(0), index=p.months)
        fc = pd.read_parquet(forecast_dir / f"{cat}.parquet")
        w = pd.Series(pop, index=p.territory_ids)
        fagg = (fc.assign(w=fc["territory_id"].map(w)).dropna(subset=["w"])
                .assign(v=lambda x: x["y_pred"] * x["w"]).groupby("month")["v"].sum())
        n = nat[NATIONAL[cat][0]]
        yoy = agg[p.months[-3:]].sum() / agg[p.months[-15:-12]].sum()
        for m in fagg.index:
            m24 = f"2024-{m[5:]}"
            rows.append({"категория": cat, "точное соответствие": NATIONAL[cat][1], "месяц": m,
                         "страна": n[m] / n["2024-12"] - 1, "final": fagg[m] / agg["2024-12"] - 1,
                         "как 2024 × рост": agg[m24] * yoy / agg["2024-12"] - 1,
                         # калибровка на 2024 г.: агрегат панели (факт) против страны, рост к декабрю 2023 г.
                         "2024: страна": n[m24] / n["2023-12"] - 1, "2024: панель": agg[m24] / agg["2023-12"] - 1})
    return pd.DataFrame(rows)


def main(cfg: dict) -> None:
    out = Path(cfg["output_dir"])
    parts = ["# Прогноз на 2025 г. против национального ряда СберИндекса\n",
             "Как устроена проверка — в docstring `src/check_2025.py`. Рост — к декабрю предыдущего года, %. "
             "«Ошибка» — средняя абсолютная разница с ростом национального ряда, п.п.; в строке «2024 г.» — "
             "та же разница у факта панели в 2024 г., то есть насколько панель вообще отличается от страны.\n"]
    summary = []
    for name, d in [("final (forecast_2025)", out / "forecast_2025"), ("seen-only (forecast_2025_seen)",
                                                                       out / "forecast_2025_seen")]:
        if not d.exists():
            continue
        t = growth_table(cfg, d)
        for cat, g in t.groupby("категория", sort=False):
            summary.append({"прогноз": name, "категория": cat, "точное соответствие": g["точное соответствие"].iloc[0],
                            "ошибка final, п.п.": 100 * (g["final"] - g["страна"]).abs().mean(),
                            "ошибка «как 2024 × рост», п.п.": 100 * (g["как 2024 × рост"] - g["страна"]).abs().mean(),
                            "2024 г.: панель против страны, п.п.": 100 * (g["2024: панель"] - g["2024: страна"]).abs().mean()})
        if name.startswith("final"):
            detail = t
    s = pd.DataFrame(summary).round(1)
    parts += ["## Средняя ошибка роста за 12 месяцев\n", s.to_markdown(index=False), ""]
    for cat, g in detail.groupby("категория", sort=False):
        g = g.set_index("месяц")[["страна", "final", "как 2024 × рост", "2024: страна", "2024: панель"]]
        parts += [f"## {cat}\n", (100 * g).round(1).to_markdown(), ""]
    (out / "check_2025.md").write_text("\n".join(parts))
    print("\n".join(parts[:4]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/forecast.yaml")
    main(yaml.safe_load(open(p.parse_args().config)))
