"""Прогноз против факта: страница с прогнозами на 1, 3, 6 и 12 месяцев у 5 лучших, 5 типичных и 5 худших МО.

МО упорядочены по средней MAPE модели за четыре горизонта (протокол расширяющегося окна, тест — 2024 г.).
.venv/bin/python -m src.viz.forecast_viz [--model final] → docs/forecast_vs_fact.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ["Все категории", "Продовольствие", "Здоровье", "Маркетплейсы", "Общественное питание", "Транспорт"]
HS = (1, 3, 6, 12)


def protocol_rows(d: pd.DataFrame) -> pd.DataFrame:
    """Строки прогноза по горизонтам: старты T = 12, 12+h, …, шаги 1..h."""
    parts = []
    for h in HS:
        x = d[d.origin.isin(range(12, 24, h)) & (d.step <= h)].copy()
        x["h"] = h
        parts.append(x)
    return pd.concat(parts)


def oracle_gain(d: pd.DataFrame, region: pd.Series) -> dict:
    """Насколько упала бы MAE, если бы был известен общий для всех МО сдвиг месяца (медиана log-ошибки по МО)
    и, вдобавок, сдвиг своего региона: {h: [Δ% страна, Δ% страна + регион]}."""
    d = d.assign(e=np.log(d.y_pred / d.y_true), region=d.territory_id.map(region))
    g = ["h", "origin", "step"]
    nat = d.groupby(g).e.transform("median")
    reg = d.groupby(g + ["region"]).e.transform("median") - nat
    out = {}
    for h in HS:
        m = d.h == h
        mae = (d.y_pred - d.y_true)[m].abs().mean()
        out[str(h)] = [round(float(((d.y_pred * np.exp(-x)) - d.y_true)[m].abs().mean() / mae - 1) * 100)
                       for x in (nat, nat + reg)]
    return out


def build(model: str) -> dict:
    dic = pd.read_excel("data/external/raw/t_dict_municipal_districts.xlsx")
    dic = dic.sort_values("year_to").groupby("territory_id").last()
    cons = pd.read_parquet("consumption.parquet")
    region = pd.read_parquet("data/external/territory_static.parquet").set_index("territory_id")["region"]
    cats, summary = {}, {}
    for cat in ORDER:
        d = protocol_rows(pd.read_parquet(f"results/forecasts/{model}/{cat}.parquet"))
        d["ape"] = (d.y_pred - d.y_true).abs() / d.y_true
        per_h = d.groupby(["territory_id", "h"]).ape.mean().unstack()
        score = per_h.mean(1).sort_values()
        summary[cat] = {"median_mape": {str(h): round(float(per_h[h].median() * 100), 1) for h in HS}, "n": len(score),
                        "oracle": oracle_gain(d, region)}
        mid = len(score) // 2
        pick = ([("good", t) for t in score.index[:5]] + [("typical", t) for t in score.index[mid - 2:mid + 3]]
                + [("bad", t) for t in score.index[-5:][::-1]])
        fact = cons[cons.category == cat].pivot(index="territory_id", columns="date", values="value")
        mos = []
        for kind, t in pick:
            r = dic.loc[t]
            fc = {str(h): [[int(o), [round(float(v)) for v in g.y_pred]]
                           for o, g in d[(d.territory_id == t) & (d.h == h)].sort_values(["origin", "step"]).groupby("origin")]
                  for h in HS}
            mos.append(dict(kind=kind, id=int(t), name=str(r.municipal_district_name_short),
                            type=str(r.municipal_district_type), region=str(r.region_name),
                            fact=[None if pd.isna(v) else round(float(v)) for v in fact.loc[t]],
                            mape={str(h): round(float(per_h.loc[t, h] * 100), 1) for h in HS}, fc=fc))
        # «Средний МО»: среднее по всем МО факта и прогнозов — на нём виден общий промах месяца
        avg_fact = fact.loc[score.index].mean()
        avg = protocol_rows(pd.read_parquet(f"results/forecasts/{model}/{cat}.parquet"))
        avg = avg.groupby(["h", "origin", "step"]).y_pred.mean().reset_index()
        e_avg = {}
        for h in HS:
            x = avg[avg.h == h].sort_values(["origin", "step"])
            true = avg_fact.values[x.origin.values + x.step.values - 1]
            e_avg[str(h)] = round(float((abs(x.y_pred.values / true - 1)).mean() * 100), 1)
        mos.insert(0, dict(kind="avg", id=0, name="Средний МО", type=f"среднее по {len(score)} МО",
                           region="вся панель", fact=[round(float(v)) for v in avg_fact],
                           mape=e_avg,
                           fc={str(h): [[int(o), [round(float(v)) for v in g.y_pred]]
                                        for o, g in avg[avg.h == h].sort_values(["origin", "step"]).groupby("origin")]
                               for h in HS}))
        cats[cat] = mos
    return dict(months=list(fact.columns), cats=cats, summary=summary, model=model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="final")
    ap.add_argument("--out", default="docs/forecast_vs_fact.html")
    a = ap.parse_args()
    tpl = (Path(__file__).parent / "forecast_viz_template.html").read_text()
    html = tpl.replace("/*DATA*/null", json.dumps(build(a.model), ensure_ascii=False))
    Path(a.out).write_text(html)
    print(a.out)


if __name__ == "__main__":
    main()
