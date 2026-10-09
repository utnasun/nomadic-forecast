"""Глобальные ML-модели: одна модель на все МО категории.

strategy="recursive" (по умолчанию) — прогноз на 1 шаг, дальше рекурсивно; strategy="direct" — одна модель на
все шаги, цель — средний месячный прирост за k месяцев (_forecast_direct).

Цель (target):
  log_diff — log y_t − log y_{t−1}: относительный прирост, одинаково работает для малых и больших МО;
  diff     — y_t − y_{t−1}: прирост в рублях.
Признаки есть в обеих формах — логарифмической и абсолютной (рубли).

Внешние признаки включаются списком features (данные — src.data.external, ctx.ext):
  geo        — справочник МО: регион, тип, статус, shape, координаты центра;
  regional   — медианный прирост по МО своего региона и средний по k ближайшим МО (по автодороге) за t−1;
  population — log численности населения и её прирост (2022 г.);
  rosstat_q  — прирост г/г квартальных показателей БДПМО (розница, общепит, зарплата, занятость),
               доступных к последнему известному месяцу с учётом лага публикации;
  news       — всплески числа статей о ЧС и об отключениях по городу — центру МО за t−1 и t−2
               (src.data.news_features); есть только у МО с городом в словаре;
  calendar   — Рамадан в месяцы t−1, t, t+1 × доля мусульман в регионе (src.data.calendar_features);
  relative   — МО относительно своего региона: прирост за t−1, рост г/г и уровень (log) минус медиана по региону;
               деревья плохо строят разность двух признаков сами;
  macro      — рост цен за 12 мес. в регионе (ИПЦ всего, продовольствие, непрод., услуги) и рост зарплаты г/г
               (за 3 и 12 мес.) — последние значения, опубликованные к месяцу t (ИПЦ — с лагом 1 мес., зарплата —
               с лагом 3; после точки старта T новые не появляются).
  structure  — структура трат МО: log доли каждой категории в «Все категории» (среднее за 12 мес. до t);
  scluster   — тип сезонности МО (категориальный): k-means по отклонению log y от медианы МО за последние
               12 мес. до точки старта; в обучении на одном годе даёт модели сезонность МО, которой нет в лагах;
  trend      — собственный тренд МО: наклон log y минус медиана МО того же месяца за последние 12 мес. до t
               (МНК, в log за месяц) и тот же наклон за 6 мес.;
  moid       — сам МО как категориальный признак (CatBoost кодирует его средним по прошлым целям МО);
  tourism    — ночёвки в гостиницах на жителя за 2022 г. (БДПМО, log(1 + x)): туристические МО летом;
  pseason    — сезонный профиль по панели (panel_season) как признак, ряд не очищается: ожидаемый сезонный
               прирост месяца t и уровень профиля; месяцы, которых нет в обучении, — season_unseen;
  shift      — устойчивый сдвиг уровня МО в истории до t (level_shifts): величина в log и сколько месяцев назад;
               нет сдвига — 0 и NaN;
  cross      — прирост за последний месяц и отклонение от среднего за 3 мес. у других категорий того же МО
               (после точки старта — последние известные).

Пропуски в рядах (неполные МО, data.only_complete: false) допустимы: признаки считаются по известным
месяцам, строки обучения без цели отбрасываются; прогноз строится, если известно последнее значение.

deseason=True: ряды делятся на национальный сезонный профиль категории (ctx.season), модель учится и
прогнозирует очищенный ряд, прогноз умножается на профиль обратно. Нужен при коротком обучении: по одному
году модель не видит, например, переход «декабрь → январь».
Устойчивость к шокам в обучении (прогноз строится от фактического последнего значения, обучение — по очищенному):
  winsor  — цель обучения обрезается квантилями [q, 1−q] по МО внутри каждого месяца-цели: артефакты (новые
            продавцы на нулевой базе, +1000…+9600%) и шоки не тянут RMSE-модель;
  despike — одиночные выбросы внутри истории (не последний месяц): если log y_t отклоняется от среднего соседей
            t−1, t+1 больше чем на k робастных σ месяца и в обе стороны с разным знаком, y_t заменяется средним
            геометрическим соседей. Очищаются и цели, и лаговые признаки.

per_day=a: модель прогнозирует y / d^a, d — число дней в месяце (a=1 — расходы на день): профиль по 2023 г. не
знает 29 февраля 2024 г. Не помогает: февраль исправляется, март портится на столько же.

deseason="panel": профиль оценивается по самой панели обучения (panel_season), национальный — только для
месяцев, которых в обучении нет. Национальный профиль 2019–2022 плохо подходит к данным СберИндекса
(например, в феврале в очищенном ряду остаётся +6…+10%).
"""
from __future__ import annotations

import calendar
import dataclasses
import warnings

import numpy as np
import pandas as pd

from src.data.calendar_features import ramadan_features
from src.data import MACRO_OFFSET
from src.models.base import Context

CAT_FEATURES = ["month"]
GEO_CAT = ["region", "mo_type", "is_capital", "is_zato", "shape"]
FEATURE_GROUPS = ("geo", "regional", "population", "rosstat_q", "news", "calendar", "macro", "relative",
                  "structure", "scluster", "cross", "shift", "pseason", "tourism", "trend", "moid")
EXT_GROUPS = {"geo", "regional", "population", "rosstat_q", "news", "calendar", "macro", "relative", "tourism"}  # нужен ctx.ext
TOTAL = "Все категории"
N_SEASON_CLUSTERS = 8
NEWS_LAGS = (1, 2)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def make_features(Y: np.ndarray, t: int, ctx: Context, static: dict[str, np.ndarray]) -> pd.DataFrame:
    """Признаки для прогноза столбца t по известным столбцам Y[:, :t]. Недоступное — NaN."""
    n = Y.shape[0]
    L = np.log(Y)
    nan = np.full(n, np.nan)

    def col(M, k):  # M[:, t-k] или NaN, если такого месяца нет
        return M[:, t - k] if t - k >= 0 else nan

    f = {}
    for k in (1, 2, 3):
        f[f"dlog_{k}"] = col(L, k) - col(L, k + 1)
        f[f"diff_{k}"] = col(Y, k) - col(Y, k + 1)
    f["level_1"] = col(Y, 1)
    f["log_level_1"] = col(L, 1)
    with warnings.catch_warnings():  # у неполных МО окно может быть целиком пустым — NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        for w in (3, 6):
            lo = max(0, t - w)
            f[f"dev_log_mean{w}"] = L[:, t - 1] - np.nanmean(L[:, lo:t], 1)
            f[f"dev_abs_mean{w}"] = Y[:, t - 1] - np.nanmean(Y[:, lo:t], 1)
        lo = max(0, t - 7)
        d = np.diff(L[:, lo:t], axis=1)
        f["vol_dlog_6"] = np.nanstd(d, 1) if d.shape[1] >= 2 else nan
    f["month"] = np.full(n, str(ctx.month_of_year(t)))
    # сезонность прошлого года
    f["dlog_lag12"] = col(L, 12) - col(L, 13)
    f["diff_lag12"] = col(Y, 12) - col(Y, 13)
    f["sn_gap_log"] = col(L, 12) - col(L, 1)   # насколько «год назад» выше последнего значения
    f["sn_gap_abs"] = col(Y, 12) - col(Y, 1)
    f["yoy_log_1"] = col(L, 1) - col(L, 13)
    # общий для всех МО фактор
    f["common_dlog_1"] = np.full(n, np.nanmedian(f["dlog_1"])) if t >= 2 else nan
    f["common_dlog_lag12"] = np.full(n, np.nanmedian(f["dlog_lag12"])) if t >= 13 else nan
    # статика
    f.update(static)
    return pd.DataFrame(f)


def regional_features(X: pd.DataFrame, ext) -> dict[str, np.ndarray]:
    """Прирост по своему региону (медиана) и по ближайшим соседям (среднее) — из уже посчитанных признаков МО."""
    region = ext.static["region"].values
    nb = ext.neighbors
    f = {}
    for c in ("dlog_1", "dlog_lag12", "yoy_log_1"):
        f[f"region_{c}"] = pd.Series(X[c].values).groupby(region).transform("median").values
    for c in ("dlog_1", "yoy_log_1"):
        v = X[c].values
        with warnings.catch_warnings():  # у МО без соседей или без истории — NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            f[f"nbr_{c}"] = np.nanmean(np.where(nb >= 0, v[nb], np.nan), axis=1)
    return f


def relative_features(X: pd.DataFrame, ext) -> dict[str, np.ndarray]:
    """МО минус медиана по МО своего региона за тот же месяц."""
    region = ext.static["region"].values
    f = {}
    for c in ("dlog_1", "yoy_log_1", "log_level_1", "dev_log_mean6"):
        v = pd.Series(X[c].values)
        f[f"rel_{c}"] = (v - v.groupby(region).transform("median")).values
    return f


def structure_features(others: dict[str, np.ndarray], last: int) -> dict[str, np.ndarray]:
    """log доли категории в «Все категории» по месяцам до last (не включая), среднее за 12 мес."""
    lo = max(0, last - 12)
    tot = np.log(np.clip(others[TOTAL][:, lo:last], 1, None))
    f = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for c, M in others.items():
            if c != TOTAL:
                f[f"share_{c}"] = np.nanmean(np.log(np.clip(M[:, lo:last], 1, None)) - tot, 1)
    return f


def cross_features(others: dict[str, np.ndarray], own: str | None, last: int) -> dict[str, np.ndarray]:
    """Прирост за последний известный месяц и отклонение от среднего за 3 мес. у других категорий."""
    f = {}
    n = next(iter(others.values())).shape[0]
    for c, M in others.items():
        if c == own:
            continue
        L = np.log(np.clip(M[:, :last], 1, None))
        f[f"x_dlog_{c}"] = L[:, -1] - L[:, -2] if last >= 2 else np.full(n, np.nan)
        f[f"x_dev3_{c}"] = L[:, -1] - L[:, max(0, last - 3):].mean(1) if last >= 1 else np.full(n, np.nan)
    return f


def season_clusters(y: np.ndarray, k: int = N_SEASON_CLUSTERS, seed: int = 0) -> np.ndarray:
    """Тип сезонности МО: k-means по log y минус медиана МО того же месяца за последние 12 мес., центрировано по МО."""
    from sklearn.cluster import KMeans

    L = np.log(np.clip(y[:, -12:], 1, None))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        R = L - np.nanmedian(L, 0)
        R = np.nan_to_num(R - np.nanmean(R, 1, keepdims=True))
    return KMeans(n_clusters=k, random_state=seed, n_init=4).fit_predict(R).astype(str)


def rosstat_features(ext, last_known: int) -> dict[str, np.ndarray]:
    """Последний опубликованный к месяцу last_known прирост г/г квартальных показателей Росстата."""
    f = {}
    for name, (M, end) in ext.quarterly.items():
        avail = np.flatnonzero(end + ext.rosstat_lag <= last_known)
        f[f"ros_{name}_yoy"] = M[:, avail[-1]] if len(avail) else np.full(M.shape[0], np.nan)
    return f


def news_features(ext, t: int, T: int) -> dict[str, np.ndarray]:
    """Всплески новостей за t−lag. Месяц с точки старта T и позже ещё не наступил: всплеск неизвестен,
    ставим 0 («как обычно»); у МО без города в словаре — NaN на всех шагах."""
    f = {}
    for name, M in ext.news.items():
        covered = ~np.isnan(M).all(1)
        for lag in NEWS_LAGS:
            m = t - lag
            if m < 0:
                v = np.full(M.shape[0], np.nan)
            elif m >= T or m >= M.shape[1]:
                v = np.where(covered, 0.0, np.nan)
            else:
                v = M[:, m]
            f[f"news_{name}_lag{lag}"] = v
    return f


def panel_season(y: np.ndarray, ctx: Context, blend: float | None = None, unseen: str = "national") -> np.ndarray:
    """Сезонный log-фактор (12,) по самой панели обучения, среднее по году = 0.

    Сезонный прирост месяца m — медиана по МО log-прироста в переход «m−1 → m», усреднённая по годам
    обучения, минус средний месячный тренд μ. Месяцы, перехода в которые в обучении нет (январь при
    обучении на одном 2023 г.), берутся из национального профиля (ctx.season; нет его — 0). μ выбирается
    так, чтобы сезонные приросты за год давали 0: μ = (Σ_наблюд p_m + Σ_прочие nat_m) / число наблюдённых.
    unseen="zero": для месяцев, которых нет в обучении, сезонный прирост 0 вместо национального (национальный
    профиль плохо совпадает с панелью: январь Здоровья и Общепита −26% против −2…−7% в панели).
    blend=w: приросты профиля — w · панель + (1 − w) · национальный профиль (сжатие оценки по одному году к
    многолетней национальной).
    """
    nat = np.diff(np.r_[ctx.season[-1], ctx.season]) if ctx.season is not None and unseen == "national" else np.zeros(12)
    L = np.log(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.diff(L, axis=1), axis=0)  # переход в месяц t = 1..T-1
    moy = np.array([ctx.month_of_year(t) for t in range(1, y.shape[1])]) - 1
    p = pd.Series(med).groupby(moy).mean().reindex(range(12)).values
    seen = np.isfinite(p)
    mu = (p[seen].sum() + nat[~seen].sum()) / seen.sum()
    ds = np.where(seen, p - mu, nat)
    if blend is not None:
        ds = blend * ds + (1 - blend) * nat
    level = np.cumsum(ds)
    return level - level.mean()


def days_in_month(ctx: Context, n: int) -> np.ndarray:
    """Число дней в месяцах t = 0..n-1."""
    return np.array([calendar.monthrange(*map(int, ctx.month_start(t)[:7].split("-")))[1] for t in range(n)], float)


def macro_features(ext, t: int, T: int) -> dict[str, np.ndarray]:
    """Региональные ИПЦ и зарплата, известные к прогнозу месяца t при точке старта T."""
    f = {}
    for name, M in ext.macro.items():
        lag = ext.macro_lag["cpi" if name.startswith("cpi") else "wage"]
        f[name] = M[:, MACRO_OFFSET + min(t, T) - lag]
    return f


def despike(y: np.ndarray, k: float) -> np.ndarray:
    """Одиночные выбросы внутри истории → среднее геометрическое соседей (последний месяц не трогается)."""
    L = np.log(np.clip(y, 1, None))
    out = L.copy()
    for t in range(1, L.shape[1] - 1):
        a, b = L[:, t] - L[:, t - 1], L[:, t + 1] - L[:, t]
        dev = L[:, t] - (L[:, t - 1] + L[:, t + 1]) / 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mad = 1.4826 * np.nanmedian(np.abs(dev - np.nanmedian(dev)))
            spike = (np.abs(dev - np.nanmedian(dev)) > k * mad) & (np.sign(a) != np.sign(b))
        out[spike, t] = (L[spike, t - 1] + L[spike, t + 1]) / 2
    return np.where(np.isfinite(y), np.exp(out), y)


def level_shifts(y: np.ndarray, k: float = 4.0, min_step: float = 0.15, min_post: int = 3,
                 pre_window: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Устойчивый сдвиг уровня МО внутри истории: (месяц сдвига s или −1, величина в log).

    R — log y минус медиана по МО того же месяца (общая для страны динамика убрана). Для каждого s сравниваются
    медиана R после s (до конца истории, не меньше min_post месяцев) и медиана за pre_window месяцев до s; сдвиг —
    если разница ≥ min_step и в k раз больше разброса R внутри обоих отрезков. Берётся s с наибольшим отношением.
    Если тот же скачок был годом раньше (s − 12) — это сезонность МО, а не сдвиг.
    """
    L = np.log(np.clip(y, 1, None))
    T = L.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        R = L - np.nanmedian(L, 0)
        best, at, size = np.zeros(len(L)), np.full(len(L), -1), np.zeros(len(L))
        for s in range(3, T - min_post + 1):
            pre, post = R[:, max(0, s - pre_window):s], R[:, s:T]
            step = np.nanmedian(post, 1) - np.nanmedian(pre, 1)
            score = np.abs(step) / (np.sqrt((np.nanvar(pre, 1) + np.nanvar(post, 1)) / 2) + 0.01)
            ok = (np.abs(step) >= min_step) & (score >= k)
            if s - 12 >= 3:
                s12 = s - 12
                step12 = (np.nanmedian(R[:, s12:s12 + min_post], 1)
                          - np.nanmedian(R[:, max(0, s12 - pre_window):s12], 1))
                ok &= ~((np.sign(step12) == np.sign(step)) & (np.abs(step12) > 0.5 * np.abs(step)))
            upd = ok & (score > best)
            best[upd], at[upd], size[upd] = score[upd], s, step[upd]
    return at, size


def rebase(y: np.ndarray, **kw) -> np.ndarray:
    """История до устойчивого сдвига уровня переносится на новый уровень (последний отрезок не меняется)."""
    at, size = level_shifts(y, **kw)
    out = y.astype(float).copy()
    for i in np.flatnonzero(at >= 0):
        out[i, :at[i]] *= np.exp(size[i])
    return out


def _winsor_by_month(z: np.ndarray, tt: np.ndarray, q: float) -> np.ndarray:
    z = z.copy()
    for t in np.unique(tt):
        m = tt == t
        lo, hi = np.nanquantile(z[m], [q, 1 - q])
        z[m] = np.clip(z[m], lo, hi)
    return z


def _target(Y, t, kind):
    return np.log(Y[:, t]) - np.log(Y[:, t - 1]) if kind == "log_diff" else Y[:, t] - Y[:, t - 1]


def _target_k(Y, s, k, kind):
    """Средний месячный прирост за k месяцев от последнего известного y[s-1] до y[s+k-1]."""
    if kind == "log_diff":
        return (np.log(Y[:, s + k - 1]) - np.log(Y[:, s - 1])) / k
    return (Y[:, s + k - 1] - Y[:, s - 1]) / k


def _invert_k(Y, T, k, pred, kind):
    y = np.exp(np.log(Y[:, T - 1]) + k * pred) if kind == "log_diff" else Y[:, T - 1] + k * pred
    return np.clip(y, 1.0, None)


def _invert(Y, t, pred, kind):
    y = np.exp(np.log(Y[:, t - 1]) + pred) if kind == "log_diff" else Y[:, t - 1] + pred
    return np.clip(y, 1.0, None)


class GlobalML:
    def __init__(self, kind: str, target: str, params: dict, min_t: int = 2, min_unique: int = 6,
                 clip_quantile: float | None = 0.005, features: list[str] | tuple = (), deseason: bool | str = False,
                 strategy: str = "recursive", seed: int | None = None, weight: str | None = None,
                 winsor: float | None = None, despike: float | None = None, per_day: float | None = None,
                 season_blend: float | None = None, recency: float | None = None, season_unseen: str = "national",
                 rebase: dict | bool | None = None, drop_jumps: float | None = None):
        self.kind, self.target, self.params = kind, target, params
        self.min_t, self.min_unique, self.clip_quantile = min_t, min_unique, clip_quantile
        unknown = set(features) - set(FEATURE_GROUPS)
        if unknown:
            raise ValueError(f"Неизвестные группы признаков: {unknown}")
        self.features = list(features)
        self.deseason = deseason
        if strategy not in ("recursive", "direct"):
            raise ValueError(f"Неизвестная стратегия: {strategy}")
        self.strategy = strategy
        # weight="level": вес строки — уровень МО перед целью; с целью log_diff ошибка × уровень ≈ ошибка в рублях
        if weight not in (None, "level"):
            raise ValueError(f"Неизвестный вес: {weight}")
        self.weight = weight
        self.winsor, self.despike = winsor, despike
        self.rebase = {} if rebase is True else rebase  # параметры level_shifts; None — без переноса уровня
        # drop_jumps=k: строки обучения, где цель — скачок больше чем в k раз за месяц (артефакты на почти нулевой
        # базе), не участвуют в обучении; признаки следующих месяцев не меняются
        self.drop_jumps = drop_jumps
        self.per_day = per_day
        self.season_blend = season_blend
        self.recency = recency
        if season_unseen not in ("national", "zero"):
            raise ValueError(f"Неизвестный season_unseen: {season_unseen}")
        self.season_unseen = season_unseen
        # TabPFN: квантили прогноза на первый шаг (уровни), q1 — (n, len(QUANTILES)); для детектора шоков и интервалов
        self.q1 = None
        self.seed = seed  # None — общий seed эксперимента (ctx.seed); свой — для оценки шума от случайности
        self.cat_features = (CAT_FEATURES + (["month_k"] if strategy == "direct" else [])
                             + (GEO_CAT if "geo" in self.features else [])
                             + (["season_cluster"] if "scluster" in self.features else [])
                             + (["mo_id"] if "moid" in self.features else []))

    def _static(self, y, ctx):
        with warnings.catch_warnings():  # МО, которых до точки старта ещё нет в данных, — NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            s = {
                "log_mean_level": np.nanmean(np.log(y), 1),
                "market_access": ctx.market_access,
            }
        if set(self.features) & EXT_GROUPS and ctx.ext is None:
            raise ValueError("Нужны внешние данные: .venv/bin/python -m src.data.external и data.external в конфиге")
        st = ctx.ext.static if ctx.ext is not None else None
        if "geo" in self.features:
            s.update({c: st[c].values for c in GEO_CAT + ["lat", "lon"]})
        if "population" in self.features:
            s.update({c: st[c].values for c in ("log_pop", "pop_growth")})
        if "moid" in self.features:
            s["mo_id"] = np.arange(y.shape[0]).astype(str)
        if "tourism" in self.features:
            s["log_nights_pc"] = st["log_nights_pc"].values
        if "scluster" in self.features:
            s["season_cluster"] = season_clusters(y, seed=ctx.seed)
        return s

    def _features(self, Y, t, ctx, static, T):
        """Признаки для цели t; T — точка старта прогноза: после неё новые данные Росстата не появляются."""
        X = make_features(Y, t, ctx, static)
        extra = {}
        if "regional" in self.features:
            extra.update(regional_features(X, ctx.ext))
        if "relative" in self.features:
            extra.update(relative_features(X, ctx.ext))
        if "rosstat_q" in self.features:
            extra.update(rosstat_features(ctx.ext, min(t, T) - 1))
        if "news" in self.features:
            if ctx.ext.news is None:
                raise ValueError("Нужны новостные признаки: .venv/bin/python -m src.data.news_features и data.external.news")
            extra.update(news_features(ctx.ext, t, T))
        if "calendar" in self.features:
            extra.update(ramadan_features(ctx, ctx.ext.static["region"].values, t))
        if "structure" in self.features:
            extra.update(structure_features(ctx.others, min(t, T)))
        if "cross" in self.features:
            extra.update(cross_features(ctx.others, ctx.category, min(t, T)))
        if "trend" in self.features:
            last = min(t, T)
            for w in (12, 6):
                lo = max(0, last - w)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    L = np.log(np.clip(Y[:, lo:last], 1, None))
                    R = L - np.nanmedian(L, 0)
                    x = np.arange(R.shape[1]) - (R.shape[1] - 1) / 2
                    slope = (R * x).sum(1) / (x ** 2).sum() if R.shape[1] >= 3 else np.full(len(Y), np.nan)
                extra[f"trend_{w}"] = slope
        if "pseason" in self.features:
            m, mp = ctx.month_of_year(t) - 1, ctx.month_of_year(t - 1) - 1
            extra.update(ps_dlog=np.full(len(Y), ctx.season[m] - ctx.season[mp]), ps_level=np.full(len(Y), ctx.season[m]))
        if "shift" in self.features:
            last = min(t, T)
            at, size = level_shifts(Y[:, :last]) if last >= 6 else (np.full(len(Y), -1), np.zeros(len(Y)))
            extra.update(shift_size=size, shift_age=np.where(at >= 0, last - at, np.nan))
        if "macro" in self.features:
            if ctx.ext.macro is None:
                raise ValueError("Нужны региональные ИПЦ и зарплата: .venv/bin/python -m src.data.external --only region_monthly")
            extra.update(macro_features(ctx.ext, t, T))
        return X.assign(**extra) if extra else X

    def _fit(self, X: pd.DataFrame, z: np.ndarray, ctx: Context, w: np.ndarray | None = None):
        if self.kind == "catboost":
            from catboost import CatBoostRegressor

            m = CatBoostRegressor(**self.params, random_seed=ctx.seed if self.seed is None else self.seed, verbose=0,
                                  thread_count=ctx.n_jobs, allow_writing_files=False)
            m.fit(X, z, cat_features=self.cat_features, sample_weight=w)
            return m, lambda X_: m.predict(X_)

        if self.kind == "tabpfn":
            # Табличная foundation-модель (TabPFN-3, чекпойнт для рядов): обучения нет, обучающие строки — контекст.
            # Контекст больше ~10 тыс. строк на M2 слишком медленный — случайная подвыборка max_context строк.
            # Нужен Python ≥ 3.11 (окружение .venv-tabfm); лицензия весов — только исследование и оценка.
            import os

            from tabpfn import TabPFNRegressor

            p = dict(self.params)
            max_ctx, out = p.pop("max_context", 8000), p.pop("output_type", "mean")
            seed = ctx.seed if self.seed is None else self.seed
            cats = {c: pd.Index(X[c].astype(str).unique()) for c in self.cat_features}

            def design(X_):
                A = X_.copy()
                for c, idx in cats.items():  # коды категорий; незнакомая категория — NaN
                    A[c] = idx.get_indexer(A[c].astype(str)).astype(float)
                    A.loc[A[c] < 0, c] = np.nan
                return A.astype(float).values

            m = TabPFNRegressor(model_path=os.path.expanduser(p.pop("model_path")), random_state=seed,
                                ignore_pretraining_limits=True,
                                categorical_features_indices=[X.columns.get_loc(c) for c in cats], **p)
            rows = np.random.default_rng(seed).choice(len(X), size=min(max_ctx, len(X)), replace=False)
            m.fit(design(X.iloc[rows]), z[rows])

            def predict(X_, quantiles=None):
                if quantiles:
                    return np.column_stack(m.predict(design(X_), output_type="quantiles", quantiles=list(quantiles)))
                return m.predict(design(X_), output_type=out)
            return m, predict

        if self.kind == "tabfm":
            # TabFM (Google Research, 2026): то же, что tabpfn, но подвыборку строк (max_num_rows) делает сама модель —
            # своя у каждого члена ансамбля. Веса — google/tabfm-1.0.0-pytorch (6,6 ГБ, считается в bfloat16),
            # лицензия некоммерческая. Категориальные признаки — строки, модель кодирует их сама.
            import os

            import torch
            from tabfm import TabFMRegressor
            from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1

            p = dict(self.params)
            fm = tabfm_v1.load(model_type="regression", checkpoint_path=os.path.expanduser(p.pop("checkpoint_path")),
                               device=p.pop("device", "cpu"), dtype=torch.bfloat16)
            cats = [c for c in self.cat_features]

            def design(X_):
                return X_.astype({c: str for c in cats})

            chunk = p.pop("predict_chunk", None)  # прогноз по частям: меньше пиковая память (M2, 16 ГБ)
            m = TabFMRegressor(model=fm, random_state=ctx.seed if self.seed is None else self.seed, **p)
            m.fit(design(X), z)

            def predict(X_, quantiles=None):
                step = chunk or len(X_)
                return np.concatenate([m.predict(design(X_.iloc[i:i + step])) for i in range(0, len(X_), step)])
            return m, predict

        if self.kind == "ridge":
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import OneHotEncoder, StandardScaler

            # Признаки с малым числом уникальных значений (напр., общий фактор при 1–2 месяцах истории)
            # коллинеарны месяцу и дают взрывную экстраполяцию в линейной модели — исключаем.
            cat = self.cat_features
            num = [c for c in X.columns if c not in cat and X[c].nunique() >= self.min_unique]
            partial = [c for c in num if X[c].isna().any()]
            med = X[num].median()
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X[cat])
            sc = StandardScaler()

            def design(X_, fit=False):
                A = X_[num].fillna(med)
                flags = X_[partial].isna().astype(float).add_suffix("_isna").values
                Z = sc.fit_transform(A.values) if fit else sc.transform(A.values)
                return np.hstack([Z, flags, ohe.transform(X_[cat])])

            m = Ridge(**self.params).fit(design(X, fit=True), z, sample_weight=w)
            return m, lambda X_: m.predict(design(X_))

        raise ValueError(self.kind)

    def forecast(self, y, steps, ctx: Context):
        if not self.per_day:
            return self._forecast_seasonal(y, steps, ctx)
        T = y.shape[1]
        d = days_in_month(ctx, T + steps) ** self.per_day
        fc = self._forecast_seasonal(y / d[:T], steps, ctx) * d[T:]
        if self.q1 is not None:
            self.q1 = self.q1 * d[T]
        return fc

    def _forecast_seasonal(self, y, steps, ctx: Context):
        if "pseason" in self.features and not self.deseason:
            return self._forecast(y, steps, dataclasses.replace(
                ctx, season=panel_season(y, ctx, self.season_blend, self.season_unseen)))
        if self.deseason == "panel":
            ctx = dataclasses.replace(ctx, season=panel_season(y, ctx, self.season_blend, self.season_unseen))
        elif not self.deseason or ctx.season is None:
            return self._forecast(y, steps, ctx)
        T = y.shape[1]
        f = ctx.season_factor(0, T + steps)
        fc = self._forecast(y / f[:T], steps, ctx) * f[T:]
        if self.q1 is not None:
            self.q1 = self.q1 * f[T]
        return fc

    def _forecast_direct(self, y, steps, ctx: Context):
        """Прямой прогноз: одна модель на шаги k = 1..steps, цель — средний месячный прирост за k месяцев
        (_target_k), k и месяц года конца отрезка (month_k) — признаки. Пар «старт s, шаг k» с s + k − 1 ≥ T
        в обучении нет; для k больше виденного деревья дают темп как у наибольшего виденного k."""
        T = y.shape[1]
        static = self._static(y, ctx)
        Xs, zs, ws = [], [], []
        for s in range(self.min_t, T):
            X = self._features(y, s, ctx, static, T)
            for k in range(1, min(steps, T - s) + 1):
                Xs.append(X.assign(k=k, month_k=str(ctx.month_of_year(s + k - 1))))
                zs.append(_target_k(y, s, k, self.target))
                ws.append(y[:, s - 1])
        X, z, w = pd.concat(Xs, ignore_index=True), np.concatenate(zs), np.concatenate(ws)
        known = np.isfinite(z)
        X, z, w = X[known].reset_index(drop=True), z[known], w[known]
        _, predict = self._fit(X, z, ctx, w if self.weight == "level" else None)
        q = self.clip_quantile
        lo, hi = (np.quantile(z, q), np.quantile(z, 1 - q)) if q is not None else (-np.inf, np.inf)
        XT = self._features(y, T, ctx, static, T)
        return np.column_stack([
            _invert_k(y, T, k, np.clip(predict(XT.assign(k=k, month_k=str(ctx.month_of_year(T + k - 1)))), lo, hi),
                      self.target)
            for k in range(1, steps + 1)])

    def _forecast(self, y, steps, ctx: Context):
        if self.strategy == "direct":
            return self._forecast_direct(y, steps, ctx)
        if self.despike:
            y = despike(y, self.despike)
        if self.rebase is not None:
            y = rebase(y, **self.rebase)
        T = y.shape[1]
        static = self._static(y, ctx)
        X = pd.concat([self._features(y, t, ctx, static, T) for t in range(self.min_t, T)], ignore_index=True)
        z = np.concatenate([_target(y, t, self.target) for t in range(self.min_t, T)])
        w = np.concatenate([y[:, t - 1] for t in range(self.min_t, T)])  # уровень МО перед целью
        tt = np.repeat(np.arange(self.min_t, T), y.shape[0])
        known = np.isfinite(z)  # у неполных МО часть месяцев пропущена
        if self.drop_jumps:
            jump = np.concatenate([np.abs(_target(y, t, "log_diff")) for t in range(self.min_t, T)])
            known &= ~(jump > np.log(self.drop_jumps))
        X, z, w, tt = X[known].reset_index(drop=True), z[known], w[known], tt[known]
        if self.winsor:
            z = _winsor_by_month(z, tt, self.winsor)
        w = w if self.weight == "level" else None
        if self.recency:  # вес строки — recency^(T − 1 − t): свежие месяцы важнее (дрейф темпа роста)
            r = self.recency ** (T - 1 - tt)
            w = r if w is None else w * r
        _, predict = self._fit(X, z, ctx, w)
        # Защита рекурсии от «разгона»: прогноз цели не выходит за диапазон, виденный в обучении.
        q = self.clip_quantile
        lo, hi = (np.quantile(z, q), np.quantile(z, 1 - q)) if q is not None else (-np.inf, np.inf)

        ext = y.copy()
        if self.kind == "tabpfn":
            qz = predict(self._features(ext, T, ctx, static, T), quantiles=QUANTILES)
            self.q1 = np.column_stack([_invert(ext, T, qz[:, j], self.target) for j in range(len(QUANTILES))])
        for t in range(T, T + steps):
            pred = np.clip(predict(self._features(ext, t, ctx, static, T)), lo, hi)
            ext = np.column_stack([ext, _invert(ext, t, pred, self.target)])
        return ext[:, T:]
