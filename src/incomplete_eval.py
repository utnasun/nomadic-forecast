"""Эксперимент с неполными МО и Рамаданом: сравнение по подвыборкам → results/incomplete/report.md.

Запуск (после src.run_forecast для configs/forecast.yaml и configs/forecast_incomplete.yaml):
  .venv/bin/python -m src.incomplete_eval

1. Полные МО: та же модель, обученная только на полных МО (results/forecasts) и на всех (results/incomplete).
2. Неполные МО: модели на всех МО; оцениваются строки с фактом и последним известным значением.
3. Рамадан: вариант *r (группа calendar) против той же модели без неё — в регионах с долей мусульман
   ≥ MIN_SHARE, в месяцы поста и после него (2024-03…2024-05), и на остальных МО (проверка, что не вредит).
Ошибка — MAE и |log(прогноз / факт)| в п.п.; горизонт 1 месяц, если не сказано иное.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.calendar_features import MUSLIM_SHARE
from src.data import complete_territories, load_consumption
from src.evaluate import horizon_slice

MIN_SHARE = 0.5
RAMADAN_MONTHS = ["2024-03", "2024-04", "2024-05"]


def load_fc(out: Path, models: list[str]) -> pd.DataFrame:
    parts = [pd.read_parquet(f) for m in models for f in sorted((out / "forecasts" / m).glob("*.parquet"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def errors(fc: pd.DataFrame) -> pd.DataFrame:
    return fc.assign(ae=(fc["y_pred"] - fc["y_true"]).abs(),
                     ale=np.log(fc["y_pred"].clip(lower=1) / fc["y_true"]).abs() * 100)


def pair(fc: pd.DataFrame, a: str, b: str, by: list[str]) -> pd.DataFrame:
    """Сравнение модели b с a на одних и тех же строках."""
    key = ["category", "territory_id", "origin", "step", "month"]
    x = fc[fc["model"] == a].merge(fc[fc["model"] == b][key + ["ae", "ale"]], on=key, suffixes=("_a", "_b"))
    g = x.groupby(by)
    return pd.DataFrame({"n": g.size(), "МО": g["territory_id"].nunique(),
                         f"MAE {a}": g["ae_a"].mean(), f"MAE {b}": g["ae_b"].mean(),
                         "ΔMAE, %": (g["ae_b"].mean() / g["ae_a"].mean() - 1) * 100,
                         "Δ|log|, п.п.": g["ale_b"].mean() - g["ale_a"].mean()})


def main(cfg_main: dict, cfg_inc: dict) -> None:
    out_m, out_i = Path(cfg_main["output_dir"]), Path(cfg_inc["output_dir"])
    mt = cfg_inc["protocol"]["min_train"]
    n_months = mt + 12
    df = load_consumption(cfg_inc["data"]["consumption"])
    full = set(complete_territories(df))
    region = pd.read_parquet(cfg_inc["data"]["external"]["static"]).set_index("territory_id")["region"].astype(str)
    d = pd.read_excel("data/external/raw/t_dict_municipal_districts.xlsx").drop_duplicates("region_code")
    reg_name = d.assign(region_code=d["region_code"].astype(str)).set_index("region_code")["region_name"]
    models = [m for m, c in cfg_inc["models"].items() if c.get("enabled", True)]
    inc = errors(load_fc(out_i, models))
    inc["полное"] = inc["territory_id"].isin(full)
    lines = ["# Неполные МО и признак Рамадана\n",
             f"Обучение на всех {df['territory_id'].nunique()} МО (из них неполных "
             f"{df['territory_id'].nunique() - len(full)}), тест — 2024 г. У неполных МО оцениваются строки "
             "с фактом и последним известным значением. Δ < 0 — второй вариант лучше.\n"]

    # 1. Полные МО: обучение только на полных против обучения на всех
    common = [m for m in models if (out_m / "forecasts" / m).exists() and not m.endswith("r")]
    old = errors(load_fc(out_m, common)).assign(model=lambda x: x["model"] + " (полные)")
    new = inc[inc["полное"] & inc["model"].isin(common)]
    rows = []
    for h in cfg_inc["protocol"]["horizons"]:
        o, n = horizon_slice(old, h, mt, n_months), horizon_slice(new, h, mt, n_months)
        for m in common:
            a = o[o["model"] == m + " (полные)"].groupby("category")["ae"].mean()
            b = n[n["model"] == m].groupby("category")["ae"].mean()
            rows.append(pd.DataFrame({"модель": m, "h": h, "обучение на полных": a, "на всех": b,
                                      "ΔMAE, %": (b / a - 1) * 100}))
    t = pd.concat(rows).reset_index()
    lines.append("\n## 1. Полные МО: помогает ли обучение на неполных\n")
    lines.append("**ΔMAE, %, среднее по 6 категориям**\n\n"
                 + t.pivot_table(index="модель", columns="h", values="ΔMAE, %").round(1).to_markdown() + "\n")
    lines.append("\n**По категориям, h=1**\n\n"
                 + t[t["h"] == 1].drop(columns="h").round(1).to_markdown(index=False) + "\n")

    # 2. Неполные МО
    lines.append("\n## 2. Неполные МО: качество прогноза\n")
    for h in cfg_inc["protocol"]["horizons"]:
        s = horizon_slice(inc[~inc["полное"]], h, mt, n_months)
        if s.empty:
            continue
        p = s.groupby(["category", "model"])["ae"].mean().unstack().round(1)
        n = s.groupby("category")["territory_id"].nunique()
        p.insert(0, "МО", n)
        lines.append(f"\n**MAE, h={h}** (строк: {len(s) // len(models):,} на модель)\n\n" + p.to_markdown() + "\n")
        if h == 1:
            s1 = horizon_slice(inc[inc["полное"]], 1, mt, n_months)
            cmp = pd.DataFrame({"неполные": s.groupby("model")["ale"].mean(),
                                "полные": s1.groupby("model")["ale"].mean()}).round(2)
            lines.append("\n**|log(прогноз/факт)|, п.п., h=1, все категории вместе**\n\n" + cmp.to_markdown() + "\n")

    # 3. Рамадан
    w = region.map(MUSLIM_SHARE).fillna(0)
    muslim = set(w.index[w >= MIN_SHARE])
    s1 = horizon_slice(inc, 1, mt, n_months)
    lines.append("\n## 3. Признак Рамадана\n")
    lines.append(f"Регионы с долей мусульман ≥ {MIN_SHARE}: "
                 + ", ".join(sorted(reg_name.get(r, r) for r, v in MUSLIM_SHARE.items() if v >= MIN_SHARE))
                 + f". Месяцы поста и после: {', '.join(RAMADAN_MONTHS)} (пост 11.03–09.04.2024).\n")
    for base in [m[:-1] for m in models if m.endswith("r") and m[:-1] in models]:
        x = s1.assign(группа=np.where(s1["territory_id"].isin(muslim) & s1["month"].isin(RAMADAN_MONTHS),
                                     "мусульм. регионы, март–май 2024",
                                     np.where(s1["territory_id"].isin(muslim), "мусульм. регионы, другие месяцы",
                                              "остальные МО")))
        t = pair(x, base, base + "r", ["группа", "category"]).round(2)
        lines.append(f"\n### {base}r против {base}\n\n" + t.to_markdown() + "\n")
        y = x[x["группа"] == "мусульм. регионы, март–май 2024"].assign(
            регион=lambda z: z["territory_id"].map(region).map(reg_name))
        t = pair(y[y["category"] == "Общественное питание"], base, base + "r", ["регион", "month"]).round(2)
        lines.append("\n**Общественное питание по регионам и месяцам**\n\n" + t.to_markdown() + "\n")

    (out_i / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/forecast.yaml")
    p.add_argument("--config-incomplete", default="configs/forecast_incomplete.yaml")
    a = p.parse_args()
    main(yaml.safe_load(open(a.config)), yaml.safe_load(open(a.config_incomplete)))
