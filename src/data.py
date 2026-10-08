"""Загрузка данных СберИндекса и приведение к матрицам «МО × месяц»."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TOTAL = "Все категории"


@dataclass
class Panel:
    """Панель одной категории: values[i, t] — расходы МО i в месяц t."""

    category: str
    territory_ids: np.ndarray  # (n,)
    months: list[str]          # (T,) 'YYYY-MM'
    values: np.ndarray         # (n, T), float

    @property
    def month_of_year(self) -> np.ndarray:
        return np.array([int(m[5:7]) for m in self.months])


def load_consumption(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path).reset_index(drop=True)
    return df.sort_values(["territory_id", "category", "date"]).reset_index(drop=True)


def complete_territories(df: pd.DataFrame) -> np.ndarray:
    """МО, у которых есть все месяцы во всех категориях."""
    n_months = df["date"].nunique()
    n_cats = df["category"].nunique()
    counts = df.groupby("territory_id").size()
    return np.sort(counts[counts == n_months * n_cats].index.values)


def build_panels(df: pd.DataFrame, categories: list[str], territory_ids: np.ndarray | None = None) -> dict[str, Panel]:
    if territory_ids is not None:
        df = df[df["territory_id"].isin(territory_ids)]
    months = sorted(df["date"].unique())
    panels = {}
    for cat in categories:
        wide = df[df["category"] == cat].pivot(index="territory_id", columns="date", values="value")
        wide = wide.reindex(columns=months).sort_index()
        panels[cat] = Panel(cat, wide.index.values, months, wide.values.astype(float))
    return panels


def load_market_access(path: str | Path, territory_ids: np.ndarray) -> np.ndarray:
    """Индекс доступности рынков в порядке territory_ids; NaN, если МО нет в таблице."""
    ma = pd.read_parquet(path).set_index("territory_id")["market_access"]
    return ma.reindex(territory_ids).values.astype(float)


def national_season(path: str | Path, kind: str, years: tuple[int, int]) -> np.ndarray:
    """Сезонный профиль национального ряда: log-фактор по месяцу года, (12,), среднее по году = 0.

    Для каждого календарного месяца берётся медиана log-прироста к предыдущему месяцу за годы years
    (включительно), приросты центрируются (сумма за год = 0 — тренд убран) и накапливаются в уровни.
    Переход «декабрь → январь» равен приросту января.
    """
    nat = pd.read_parquet(path)
    s = nat[nat["type"] == kind].set_index("month")["value"].sort_index()
    d = np.log(s).diff().dropna()
    year, month = d.index.str[:4].astype(int), d.index.str[5:7].astype(int)
    keep = (year >= years[0]) & (year <= years[1])
    dm = d[keep].groupby(month[keep]).median().reindex(range(1, 13)).values
    assert not np.isnan(dm).any(), f"нет данных для сезонного профиля {kind} {years}"
    level = np.cumsum(dm - dm.mean())
    return level - level.mean()


@dataclass
class External:
    """Внешние данные в порядке territory_ids панели (собираются src.external)."""

    static: pd.DataFrame      # справочник МО + население, по строке на МО
    neighbors: np.ndarray     # (n, k) позиции ближайших МО панели по автодороге; -1 — соседа нет
    # имя показателя -> (прирост г/г нарастающим итогом (n, n_quarters),
    #                    индекс последнего месяца квартала относительно первого месяца панели (n_quarters,))
    quarterly: dict[str, tuple[np.ndarray, np.ndarray]]
    rosstat_lag: int          # через сколько месяцев после конца квартала данные считаются опубликованными
    # тема -> всплеск числа статей о городе — центре МО, (n, n_months) по месяцам панели; NaN — у МО нет
    # города в словаре или месяц без истории (собирается src.news_features); None — файла нет
    news: dict[str, np.ndarray] | None = None
    # региональные ИПЦ и зарплата (src.external, region_monthly): имя -> (n, MACRO_OFFSET + n_months);
    # столбец MACRO_OFFSET + t — месяц t панели, раньше — история до панели; None — файла нет
    macro: dict[str, np.ndarray] | None = None
    macro_lag: dict[str, int] | None = None  # 'cpi' / 'wage' -> через сколько месяцев значение известно


def road_neighbors(path: str | Path, territory_ids: np.ndarray, k: int) -> np.ndarray:
    """k ближайших по автодороге МО среди territory_ids (позиции в territory_ids), -1 — если соседей меньше k."""
    c = pd.read_parquet(path, filters=[("type", "==", "highway")],
                        columns=["territory_id_x", "territory_id_y", "distance"])
    ids = set(territory_ids.tolist())
    c = c[c["territory_id_x"].isin(ids) & c["territory_id_y"].isin(ids) & (c["territory_id_x"] != c["territory_id_y"])]
    # в таблице пары записаны в одну сторону — добавляем обратные
    both = pd.concat([c, c.rename(columns={"territory_id_x": "territory_id_y", "territory_id_y": "territory_id_x"})])
    near = both.sort_values("distance").groupby("territory_id_x").head(k)
    pos = pd.Series(np.arange(len(territory_ids)), index=territory_ids)
    near = near.assign(rank=near.groupby("territory_id_x").cumcount())
    out = np.full((len(territory_ids), k), -1)
    out[pos[near["territory_id_x"]].values, near["rank"].values] = pos[near["territory_id_y"]].values
    return out


def load_external(cfg: dict, territory_ids: np.ndarray, first_month: str) -> External:
    static = pd.read_parquet(cfg["static"]).set_index("territory_id").reindex(territory_ids)
    if cfg.get("tourism") and Path(cfg["tourism"]).exists():
        # ночёвки на жителя за год, опубликованный до начала панели (tourism_year), log(1 + x); нет данных — NaN
        t = pd.read_parquet(cfg["tourism"])
        nights = t[t["year"] == cfg.get("tourism_year", 2022)].set_index("territory_id")["nights"]
        static["log_nights_pc"] = np.log1p(nights.reindex(territory_ids).values / np.exp(static["log_pop"].values))
    q = pd.read_parquet(cfg["quarterly"])
    y0, m0 = int(first_month[:4]), int(first_month[5:7])
    quarterly = {}
    for name, g in q.groupby("indicator"):
        wide = g.pivot_table(index="territory_id", columns=["year", "quarter"], values="yoy_log").sort_index(axis=1)
        wide = wide.reindex(territory_ids).ffill(axis=1)  # последнее известное значение на каждый квартал
        end = np.array([(y - y0) * 12 + 3 * qq - m0 for y, qq in wide.columns])
        quarterly[name] = (wide.values, end)
    return External(static=static, neighbors=road_neighbors(cfg["connection"], territory_ids, cfg["n_neighbors"]),
                    quarterly=quarterly, rosstat_lag=cfg["rosstat_lag_months"],
                    news=load_news(cfg.get("news"), territory_ids, first_month),
                    macro=load_macro(cfg.get("region_monthly"), static["region"].values, first_month),
                    macro_lag=cfg.get("macro_lag_months"))


def load_news(path: str | Path | None, territory_ids: np.ndarray, first_month: str) -> dict[str, np.ndarray] | None:
    """Всплески новостей по темам, столбец j — месяц first_month + j; месяцы без новостей в файле — NaN."""
    if not path or not Path(path).exists():
        return None
    cm = pd.read_parquet(path)
    months = [str(m) for m in pd.period_range(first_month, cm["month"].max(), freq="M")]
    return {c[len("spike_"):]: cm.pivot(index="territory_id", columns="month", values=c)
            .reindex(index=territory_ids, columns=months).values
            for c in cm.columns if c.startswith("spike_")}


MACRO_OFFSET = 24  # месяцев истории до начала панели в матрицах External.macro


def load_macro(path: str | Path | None, region: np.ndarray, first_month: str) -> dict[str, np.ndarray] | None:
    """Региональные признаки роста по месяцам, развёрнутые на МО через регион.

    cpi_yoy_<группа> — рост цен за 12 месяцев по месяц включительно (сумма log ИПЦ к прошлому месяцу);
    wage_yoy3 / wage_yoy12 — log среднего за 3 / 12 месяцев к тем же месяцам год назад.
    """
    if not path or not Path(path).exists():
        return None
    rm = pd.read_parquet(path)
    start = pd.Period(first_month, "M") - MACRO_OFFSET
    months = [str(m) for m in pd.period_range(start, rm["month"].max(), freq="M")]
    out = {}
    for col in ("cpi", "food", "nonfood", "services"):
        w = rm.pivot(index="region", columns="month", values=col).reindex(columns=months)
        out[f"cpi_yoy_{col}"] = np.log(w / 100).T.rolling(12).sum().T
    w = rm.pivot(index="region", columns="month", values="wage").reindex(columns=months)
    for k in (3, 12):
        avg = w.T.rolling(k).mean().T
        out[f"wage_yoy{k}"] = np.log(avg / avg.shift(12, axis=1))
    return {name: m.reindex(region).values for name, m in out.items()}
