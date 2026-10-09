"""Сравнение детекторов шоков на одном стенде: устойчивый сдвиг уровня, вставленный в сам ряд расходов.

Запуск (после src.evaluation.run_forecast и src.forecast.final): .venv/bin/python -m src.shocks.benchmark → results/shock_benchmark.md

Стенд. В N_INJECT случайных МО-месяцев 2024 г. (январь–октябрь, чтобы были видны два следующих месяца) расходы МО
умножаются на (1 + множитель·x) начиная с этого месяца и до конца ряда — структурный сдвиг уровня. Формы шока — как
в src.shocks.detector (все категории, одна случайная, «ЧС»), x = 5…30%. Каждый детектор выдаёт оценку для каждого
МО-месяца 2024 г. только по данным до этого месяца включительно. Порог детектора — 99-й процентиль оценки на
МО-месяцах без вставки (1% ложных тревог в месяц). «Найдено за k мес.» — оценка выше порога хотя бы в одном из
месяцев m…m+k, где m — месяц сдвига (порог тот же, поэтому ложных тревог на МО за k+1 месяцев тоже больше).

Детекторы (все сводят шесть категорий в одну оценку, кроме Махаланобиса и Isolation Forest — topk, как в v2):
  jump    — прирост к прошлому месяцу, нормированный внутри (категория, месяц, группа МО по уровню расходов);
  peer    — прирост минус медиана прироста МО своего региона, нормированный и делённый на собственную волатильность
            МО (первая версия детектора, src.shocks.baseline);
  level   — уровень МО относительно медианы МО региона (log) минус его среднее за 6 прошлых месяцев, нормированный
            как peer: в отличие от прироста, устойчивый сдвиг виден и в следующих месяцах;
  cusum   — двусторонний CUSUM по нормированным отклонениям level, k = 0,5, с памятью 3 месяца: накапливает
            небольшие устойчивые отклонения (классический CUSUM без ограничения памяти копит плавный рост МО
            относительно региона, и при 1% ложных тревог почти ничего не находит — 0,4% при шоке 10%);
  resid   — отклонение факта от прогноза final на 1 месяц, нормированное, без разложения на регион и МО;
  v2_topk, v2_mahal, v2_iforest, v2_combo — детектор src.shocks.detector: отклонение от прогноза с разложением на регион и
            МО и поправкой на нестабильные МО; combo — итоговая оценка.

Приближение для детекторов по прогнозу: прогноз final на месяцы после сдвига строится от уже сдвинутого
последнего значения, поэтому сдвиг в них не повторяется — отклонения в m+1, m+2 берутся без вставки (переобучать
модель на каждую вставку слишком долго). Модель после шока откатывает прогноз к прежнему уровню лишь на ~1,7% (методологический отчёт, раздел о шоках и прогнозе),
так что приближение занижает их долю найденного за 1–2 месяца незначительно.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import Panel, build_panels, complete_territories, load_consumption
from src.shocks.baseline import own_volatility, peer_deviations, rescale, standardize
from src.shocks.detector import EXPECT, MAGNITUDES, SCENARIOS, SEED, Z_CLIP, scores, topk, zscores

N_INJECT = 400
ALARM_RATE = 0.01
CUSUM_K = 0.5
CUSUM_MEMORY = 3
LEVEL_WINDOW = 6
NAMES = {"jump": "скачок к прошлому месяцу", "peer": "отклонение от региона (peer)",
         "level": "уровень к среднему за 6 мес.", "cusum": "CUSUM по уровню",
         "resid": "отклонение от прогноза", "v2_topk": "v2: прогноз + регион/МО, topk",
         "v2_mahal": "v2: Махаланобис", "v2_iforest": "v2: Isolation Forest", "v2_combo": "v2: combo (итог)"}


def wide(long: pd.DataFrame, col: str = "z") -> pd.DataFrame:
    return long.pivot_table(index=["territory_id", "month"], columns="category", values=col).clip(-Z_CLIP, Z_CLIP)


def jump_z(panels: dict) -> pd.DataFrame:
    rows = []
    for cat, p in panels.items():
        L = np.log(np.clip(p.values, 1, None))
        d = pd.DataFrame(np.diff(L, axis=1), index=p.territory_ids, columns=p.months[1:]).stack().rename("d")
        lvl = pd.DataFrame(p.values[:, :-1], index=p.territory_ids, columns=p.months[1:]).stack().rename("level_prev")
        rows.append(pd.concat([d, lvl], axis=1).rename_axis(["territory_id", "month"]).reset_index().assign(category=cat))
    return wide(standardize(pd.concat(rows, ignore_index=True)))


def level_z(panels: dict, region: pd.Series, vol: pd.DataFrame) -> pd.DataFrame:
    """Отклонение уровня МО (log y минус медиана МО региона) от его среднего за LEVEL_WINDOW прошлых месяцев,
    нормированное как peer: устойчивый сдвиг виден в каждом следующем месяце, пока не войдёт в окно."""
    rows = []
    for cat, p in panels.items():
        L = pd.DataFrame(np.log(np.clip(p.values, 1, None)), index=p.territory_ids, columns=p.months)
        reg = region.reindex(p.territory_ids).values
        size = pd.Series(reg).map(pd.Series(reg).value_counts()).values
        med = L.groupby(reg).transform("median")
        med[size < 5] = np.nan
        R = L - med.fillna(L.median())
        e = R - R.T.rolling(LEVEL_WINDOW, min_periods=3).mean().shift(1).T
        lvl = pd.DataFrame(p.values, index=p.territory_ids, columns=p.months).shift(1, axis=1)
        rows.append(pd.concat([e.stack().rename("d"), lvl.stack().rename("level_prev")], axis=1)
                    .rename_axis(["territory_id", "month"]).reset_index().assign(category=cat))
    return wide(rescale(standardize(pd.concat(rows, ignore_index=True).dropna()), vol))


def cusum(zw: pd.DataFrame) -> pd.DataFrame:
    """CUSUM с памятью CUSUM_MEMORY месяцев для каждой категории: max по началу j ∈ [t − M + 1, t] суммы (u_i − k)
    и суммы (−u_i − k), не меньше 0. Без ограничения памяти CUSUM копит медленный рост МО относительно региона
    (у многих МО расходы по безналу плавно растут), и порог при 1% ложных тревог становится недостижимым."""
    out = {}
    for c in zw.columns:
        w = zw[c].unstack("month").sort_index(axis=1).fillna(0.0)
        v = w.values
        res = np.zeros(v.shape)
        for j in range(v.shape[1]):
            best = np.zeros(len(w))
            sp = sm = np.zeros(len(w))
            for i in range(j, max(-1, j - CUSUM_MEMORY), -1):
                sp, sm = sp + v[:, i] - CUSUM_K, sm - v[:, i] - CUSUM_K
                best = np.maximum(best, np.maximum(sp, sm))
            res[:, j] = best
        out[c] = pd.DataFrame(res, index=w.index, columns=w.columns).stack()
    return pd.DataFrame(out)


def inject_panels(panels: dict, keys: pd.DataFrame, mults: list[dict], x: float) -> dict:
    out = {}
    for cat, p in panels.items():
        v = p.values.astype(float).copy()
        pos = {t: i for i, t in enumerate(p.territory_ids)}
        mpos = {m: j for j, m in enumerate(p.months)}
        for (tid, month), mult in zip(keys.itertuples(index=False), mults):
            if cat in mult:
                v[pos[tid], mpos[month]:] *= 1 + mult[cat] * x
        out[cat] = Panel(category=cat, territory_ids=p.territory_ids, months=p.months, values=v)
    return out


def detector_scores(panels: dict, resid: pd.DataFrame, region: pd.Series, test: list[str]) -> pd.DataFrame:
    peer_long = standardize(peer_deviations(panels, region))
    vol = own_volatility(peer_long)
    peer_z = wide(rescale(peer_long.drop(columns="vol", errors="ignore"), vol))
    jz = jump_z(panels)
    lz = level_z(panels, region, vol)
    cs = cusum(lz)
    zz = zscores(resid, vol, region)
    idx = zz.index[zz.index.get_level_values("month").isin(test)]
    zz = zz.loc[idx]
    sc = scores(zz["z_loc"].values)
    return pd.DataFrame({
        "jump": topk(jz.reindex(idx).values), "peer": topk(peer_z.reindex(idx).values),
        "level": topk(lz.reindex(idx).values), "cusum": topk(cs.reindex(idx).values), "resid": topk(zz["z"].values),
        "v2_topk": sc["topk"], "v2_mahal": sc["mahal"], "v2_iforest": sc["iforest"], "v2_combo": sc["combo"]},
        index=idx)


def run(cfg: dict) -> pd.DataFrame:
    df = load_consumption(cfg["data"]["consumption"])
    panels = build_panels(df, cfg["data"]["categories"], complete_territories(df))
    p0 = next(iter(panels.values()))
    region = pd.read_parquet(cfg["data"]["external"]["static"]).set_index("territory_id")["region"]
    out = Path(cfg["output_dir"])
    resid = pd.concat([pd.read_parquet(out / "forecasts" / EXPECT / f"{c}.parquet").query("step == 1")
                       .assign(category=c) for c in panels], ignore_index=True)
    resid = pd.DataFrame({"territory_id": resid["territory_id"], "month": resid["month"], "category": resid["category"],
                          "d": np.log(resid["y_true"].clip(lower=1) / resid["y_pred"].clip(lower=1)),
                          "level_prev": resid["y_last"]})
    test = sorted(resid["month"].unique())
    starts = test[:-2]
    rng = np.random.default_rng(SEED)
    cats = sorted(panels)
    rows = []
    for scenario in SCENARIOS:
        for x in MAGNITUDES:
            keys = pd.DataFrame({"territory_id": rng.choice(p0.territory_ids, N_INJECT),
                                 "month": rng.choice(starts, N_INJECT)}).drop_duplicates("territory_id")
            mults = [SCENARIOS[scenario] or {cats[rng.integers(len(cats))]: -1.0} for _ in range(len(keys))]
            inj = inject_panels(panels, keys, mults, x)
            # отклонение от прогноза: сдвиг только в месяце m (прогноз на m сделан до сдвига, дальше — от сдвинутого уровня)
            sh = pd.DataFrame([(t, m, c, np.log1p(v * x)) for (t, m), mult in zip(keys.itertuples(index=False), mults)
                               for c, v in mult.items()], columns=["territory_id", "month", "category", "shift"])
            r = resid.merge(sh, on=["territory_id", "month", "category"], how="left")
            r["d"] = r["d"] + r["shift"].fillna(0.0)
            s = detector_scores(inj, r.drop(columns="shift"), region, test)
            tid, mon = s.index.get_level_values("territory_id"), s.index.get_level_values("month")
            start = tid.map(keys.set_index("territory_id")["month"])
            clean = start.isna() | (mon < start.fillna("9999"))
            mi = {m: i for i, m in enumerate(test)}
            for name in NAMES:
                thr = np.quantile(s.loc[clean, name], 1 - ALARM_RATE)
                hit = set(s.index[s[name] >= thr])
                for k in (0, 1, 2):
                    found = [any((t, mm) in hit for mm in test[mi[m]:mi[m] + k + 1])
                             for t, m in keys.itertuples(index=False)]
                    rows.append({"сценарий": scenario, "x": x, "детектор": name, "за k мес.": k,
                                 "найдено, %": 100 * np.mean(found)})
            print(f"{scenario}, x={x}: готово")
    return pd.DataFrame(rows)


def main(cfg: dict) -> None:
    out = Path(cfg["output_dir"])
    path = out / "shock_benchmark.csv"
    res = run(cfg)
    res.to_csv(path, index=False)
    t0 = res[res["за k мес."] == 0].pivot_table(index=["сценарий", "детектор"], columns="x", values="найдено, %",
                                                 sort=False).round(1)
    t0.columns = [f"шок {c:.0%}" for c in t0.columns]
    t0 = t0.rename(index=NAMES, level=1)
    tk = res[res["x"] == 0.10].pivot_table(index=["сценарий", "детектор"], columns="за k мес.", values="найдено, %",
                                            sort=False).round(1)
    tk.columns = ["в месяц шока", "за 1 мес.", "за 2 мес."]
    tk = tk.rename(index=NAMES, level=1)
    avg = res[res["за k мес."] == 0].groupby(["детектор", "x"])["найдено, %"].mean().unstack().round(1)
    avg.columns = [f"шок {c:.0%}" for c in avg.columns]
    avg = avg.reindex(list(NAMES)).rename(index=NAMES)
    text = ["# Сравнение детекторов шоков на одном стенде\n",
            "Как устроено — docstring `src/shocks/benchmark.py`. Доля найденных сдвигов, %, при 1% ложных тревог в месяц.\n",
            "## Среднее по трём формам шока, в месяц шока\n", avg.to_markdown(), "",
            "## По формам шока, в месяц шока\n", t0.to_markdown(), "",
            "## Шок 10%: в месяц шока и с задержкой\n", tk.to_markdown(), ""]
    (out / "shock_benchmark.md").write_text("\n".join(text))
    print("\n".join(text[:4]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/forecast.yaml")
    main(yaml.safe_load(open(p.parse_args().config)))
