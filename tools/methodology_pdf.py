"""Методология и результаты проекта в PDF (для команды): reports/methodology_v2.pdf.

Запуск из корня (после src.final, src.shocks_v2, src.intervals, src.forecast_2025):
  PYTHONPATH=. .venv/bin/python tools/methodology_pdf.py
Таблицы берутся из актуальных результатов (results/…), HTML печатается в PDF Chrome без интерфейса.
"""
from __future__ import annotations

import glob
import html
import string
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.data import build_panels, complete_territories, load_consumption  # noqa: E402
from src.intervals import evaluate, residuals  # noqa: E402

OUT = Path("reports")
FIG = OUT / "methodology_files"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
H = [1, 3, 6, 12]
CATS = ["Все категории", "Продовольствие", "Здоровье", "Маркетплейсы", "Общественное питание", "Транспорт"]
BASE = ["naive", "mean_3", "mean_6", "mean_12", "drift", "seasonal_naive", "seasonal_naive_growth", "auto_ets", "auto_theta",
        "auto_arima", "prophet_default", "prophet_yearly", "catboost_diff", "catboost_logdiff", "ridge_diff", "ridge_logdiff"]
INK, INK2, GRID, SURF = "#15201c", "#4f5b57", "#e2e6e3", "#ffffff"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#0e6b62"]


def pct(v: float, digits: int = 1) -> str:
    s = f"{abs(v):.{digits}f}".replace(".", ",")
    return ("−" if v < 0 else "+" if v > 0 else "") + s + "%"


def num(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ")


def table(df: pd.DataFrame, cls: str = "", num_cols: tuple = ()) -> str:
    ncls = ' class="n"'
    head = "".join(f"<th{ncls if c in num_cols else ''}>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            klass = ' class="n"' if c in num_cols else ""
            tds.append(f"<td{klass}>{v if isinstance(v, str) and v.startswith('<') else html.escape(str(v))}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def approaches(m: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    nv = m[m.model == "naive"].set_index(["category", "horizon"]).MAE
    pre = m[m.model.isin(BASE) & (m.model != "naive")]
    pre_best = pre.loc[pre.groupby(["category", "horizon"]).MAE.idxmin()].set_index(["category", "horizon"])
    others = m[~m.model.isin(["naive"]) & ~m.model.str.startswith("final")]
    get = lambda name: m[m.model == name].set_index(["category", "horizon"]).MAE  # noqa: E731
    series = {"лучшая базовая модель (выбрана по тесту)": pre_best.MAE, "catboost_diff везде": get("catboost_diff"),
              "профиль по панели везде": get("catboost_logdiff_dsp_g123"), "final": get("final"),
              "лучшая по тесту в каждой ячейке": others.groupby(["category", "horizon"]).MAE.min()}
    rel = {k: ((v / nv.reindex(v.index) - 1) * 100).groupby(level="horizon").mean() for k, v in series.items()}
    t = pd.DataFrame({"подход": list(rel)} | {f"h = {h}": [pct(rel[k][h]) for k in rel] for h in H})
    return t, rel


MODELS = [  # (группа, модель, как устроена)
    ("Простые правила", "naive", "последнее известное значение"),
    ("Простые правила", "mean_3", "среднее за 3 последних месяца"),
    ("Простые правила", "seasonal_naive", "значение того же месяца год назад"),
    ("Простые правила", "seasonal_naive_growth", "год назад × рост последних 3 месяцев к тем же месяцам год назад"),
    ("Статистические, ряд на МО", "auto_ets", "экспоненциальное сглаживание, подбор формы (statsforecast)"),
    ("Статистические, ряд на МО", "auto_theta", "метод Theta"),
    ("Статистические, ряд на МО", "auto_arima", "ARIMA с подбором порядка, без сезонности"),
    ("Prophet, ряд на МО", "prophet_default", "Prophet по умолчанию: при истории < 2 лет годовая сезонность отключается, остаётся тренд"),
    ("Prophet, ряд на МО", "prophet_yearly", "Prophet с принудительной годовой сезонностью (Фурье, порядок 3)"),
    ("Предобученная модель", "chronos2", "Chronos-2 (Amazon) без дообучения, по ряду"),
    ("Предобученная модель", "chronos2_nat_cl", "Chronos-2 + национальный ряд СберИндекса с 2018 г. как ковариата, совместно по МО"),
    ("Глобальные, одна модель на все МО", "ridge_diff", "Ridge на тех же признаках, что CatBoost, цель — прирост в рублях"),
    ("Глобальные, одна модель на все МО", "catboost_diff", "CatBoost, цель — прирост в рублях, без сезонной поправки"),
    ("Глобальные, одна модель на все МО", "catboost_logdiff_dsp_g123", "CatBoost, прирост в логарифмах, профиль по панели, внешние признаки"),
    ("Итоговая", "final", "правило: одна из двух CatBoost-моделей выше для каждой категории"),
]
EVOLUTION = [
    ("catboost_diff", "CatBoost, прирост в рублях, только признаки ряда"),
    ("catboost_logdiff", "то же, прирост в логарифмах"),
    ("catboost_logdiff_ds_g123", "+ справочник МО, регион и соседи, население; ряд очищен национальным профилем"),
    ("catboost_logdiff_ds_g123_m1", "+ февраль в обучении (первая цель — февраль, а не март)"),
    ("catboost_logdiff_dsp_g123", "сезонный профиль по самой панели вместо национального"),
    ("final", "правило: профиль по панели для однородных категорий, catboost_diff — для остальных"),
]


def rel_table(m: pd.DataFrame, rows: list, cols: list) -> pd.DataFrame:
    nv = m[m.model == "naive"].set_index(["category", "horizon"]).MAE
    out = []
    for r in rows:
        name = r[1] if len(r) == 3 else r[0]
        sm = m[m.model == name].set_index(["category", "horizon"]).MAE
        if sm.empty:
            continue
        rel = ((sm / nv.reindex(sm.index) - 1) * 100).groupby(level="horizon").mean()
        out.append(dict(zip(cols, r)) | {f"h = {h}": pct(rel[h]) if h in rel else "—" for h in H})
    return pd.DataFrame(out)


def prophet_table(m: pd.DataFrame) -> pd.DataFrame:
    g = lambda x: m[m.model == x].set_index(["category", "horizon"]).MAE  # noqa: E731
    p, f = g("prophet_default"), g("final")
    rows = []
    for c in CATS:
        rows.append({"категория": c} | {f"h = {h}": f"{num(f[(c, h)])} / {num(p[(c, h)])} ({pct((f[(c, h)] / p[(c, h)] - 1) * 100, 0)})"
                                        for h in H})
    return pd.DataFrame(rows)


def chart_approaches(rel: dict) -> None:
    keys = ["лучшая базовая модель (выбрана по тесту)", "catboost_diff везде", "профиль по панели везде", "final"]
    fig, ax = plt.subplots(figsize=(7.2, 2.9), facecolor=SURF)
    w = 0.19
    for i, k in enumerate(keys):
        xs = np.arange(len(H)) + (i - 1.5) * w
        ax.bar(xs, [rel[k][h] for h in H], width=w * 0.92, color=SERIES[i], label=k, edgecolor=SURF, linewidth=0.8)
    orc = [rel["лучшая по тесту в каждой ячейке"][h] for h in H]
    ax.scatter(np.arange(len(H)) + 2.5 * w - 0.05, orc, marker="_", s=180, color=INK, lw=2,
               label="лучшая по тесту в каждой ячейке (недостижимо)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(np.arange(len(H)), [f"h = {h} мес." for h in H])
    ax.set_ylabel("MAE к naive, %", fontsize=8, color=INK2)
    style(ax)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIG / "approaches.png", dpi=200, facecolor=SURF)


def chart_2025() -> None:
    df = load_consumption("consumption.parquet")
    pan = build_panels(df, CATS, complete_territories(df))
    f25 = pd.concat(pd.read_parquet(f) for f in glob.glob("results/forecast_2025/*.parquet"))
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.0), facecolor=SURF)
    for ax, c in zip(axes.ravel(), CATS):
        hist = np.median(pan[c].values, axis=0)
        g = f25[f25.category == c].groupby("month")[["y_pred", "lo90", "hi90"]].median()
        n0 = len(hist)
        xf = np.arange(n0, n0 + len(g))
        ax.fill_between(xf, g.lo90, g.hi90, color="#d8ebe7", lw=0)
        ax.plot(np.arange(n0), hist, color=INK2, lw=1.6)
        ax.plot(np.r_[n0 - 1, xf], np.r_[hist[-1], g.y_pred.values], color="#0e6b62", lw=1.6, ls="--")
        ax.axvline(n0 - 0.5, color=GRID, lw=0.8, ls=":")
        ax.set_xticks([0, 12, 24], ["2023", "2024", "2025"])
        ax.set_title(c, fontsize=8.5, color=INK, loc="left")
        style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "forecast_2025.png", dpi=200, facecolor=SURF)


def shocks_stats() -> dict:
    res = pd.read_parquet("results/shocks_v2/scores.parquet")
    r24 = res[res.month >= "2024"]
    sh = r24[r24["уровень"] == "шок"]
    reg = pd.read_csv("results/shocks_v2/regional.csv")
    k = sh.territory_id.value_counts()
    return {"n": len(sh), "watch": int((r24["уровень"] == "внимание").sum()), "grad": int(r24["постепенный"].sum()),
            "mo": len(k), "mo2": int((k >= 2).sum()), "mo3": int((k >= 3).sum()),
            "reg": int((reg["шок"] & (reg.month >= "2024")).sum()),
            "reg_jan": int((reg["шок"] & (reg.month == "2024-01")).sum()),
            "n23": int(((res.month < "2024") & (res["уровень"] == "шок")).sum()),
            "types": sh["тип"].replace("", "без типа (декабрь)").fillna("без типа (декабрь)").value_counts()}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv("results/metrics_overall.csv")
    t_appr, rel = approaches(m)
    chart_approaches(rel)
    chart_2025()

    fin = m[m.model == "final"].set_index(["category", "horizon"]).MAE
    pre = m[m.model.isin(BASE) & (m.model != "naive")]
    pre_best = pre.loc[pre.groupby(["category", "horizon"]).MAE.idxmin()].set_index(["category", "horizon"])
    rows = []
    for c in CATS:
        r = {"категория": c}
        for h in H:
            d = (fin[(c, h)] / pre_best.MAE[(c, h)] - 1) * 100
            r[f"h = {h}"] = f"{num(fin[(c, h)])} / {num(pre_best.MAE[(c, h)])} ({pct(d, 0) if abs(d) >= 0.5 else '='})"
        rows.append(r)
    t_cat = pd.DataFrame(rows)

    cal = pd.read_csv("results/shocks_v2/calibration.csv")
    names = {"topk": "topk (среднеквадр. по 3 худшим категориям)", "mahal": "Махаланобис (MinCovDet)",
             "iforest": "Isolation Forest", "combo": "combo (итог)"}
    scen = {"все категории −x": "все категории −x", "одна категория −x": "одна случайная категория −x",
            "ЧС: продукты +x/2, маркетплейсы, общепит −x, всего −x/3": "«ЧС»: продукты +x/2, маркетплейсы и общепит −x"}
    rows = []
    for s_key, s_name in scen.items():
        for k, kn in names.items():
            g = cal[(cal["сценарий"] == s_key) & (cal["оценка"] == k)].set_index("x")["найдено, %"]
            rows.append({"сценарий": s_name, "оценка": kn} | {f"x = {int(x * 100)}%": f"{g[x]:.0f}" for x in g.index})
    t_cal = pd.DataFrame(rows)
    rb = pd.read_csv("results/shocks_v2/recall_by_size.csv")
    rbp = rb.pivot_table(index="расходы на жителя", columns=["сценарий", "x"], values="найдено, %")
    t_rb = pd.DataFrame({"квинтиль расходов на жителя": rbp.index,
                         "все категории −10%": [f"{v:.0f}" for v in rbp[("все категории −x", 0.1)]],
                         "все категории −20%": [f"{v:.0f}" for v in rbp[("все категории −x", 0.2)]],
                         "одна категория −10%": [f"{v:.0f}" for v in rbp[("одна категория −x", 0.1)]],
                         "одна категория −20%": [f"{v:.0f}" for v in rbp[("одна категория −x", 0.2)]]})
    st = shocks_stats()
    t_types = pd.DataFrame({"тип": st["types"].index, "шоков": st["types"].values})

    iv, k_shock = evaluate(residuals(yaml.safe_load(open("configs/forecast.yaml"))))
    s = iv.groupby(["вариант", "h"])[["покрытие, %", "покрытие после шока, %", "ширина, % уровня"]].mean()
    adj = f"после шока ×{k_shock:.1f}"
    t_iv = pd.DataFrame([{"h": h, "покрытие": f"{s.loc[('без учёта шока', h), 'покрытие, %']:.0f}%",
                          "после шока, без поправки": f"{s.loc[('без учёта шока', h), 'покрытие после шока, %']:.0f}%",
                          f"после шока, интервал ×{k_shock:.1f}".replace(".", ","):
                              f"{s.loc[(adj, h), 'покрытие после шока, %']:.0f}%",
                          "медианная ширина": f"{s.loc[('без учёта шока', h), 'ширина, % уровня']:.0f}% уровня"}
                         for h in H])

    fin_rel = rel["final"]
    pre_rel = rel["лучшая базовая модель (выбрана по тесту)"]
    orc_rel = rel["лучшая по тесту в каждой ячейке"]
    tiles = "".join(f'<div class="tile"><span class="h">на {h} мес.</span><span class="v">{pct(fin_rel[h])}</span>'
                    f'<span class="s">лучшая базовая: {pct(pre_rel[h])}</span></div>' for h in H)

    t_models = rel_table(m, MODELS, ["группа", "модель", "как устроена"])
    t_models["группа"] = t_models["группа"].where(t_models["группа"] != t_models["группа"].shift(), "")
    t_evo = rel_table(m, EVOLUTION, ["модель", "что добавлено"])
    t_prophet = prophet_table(m)
    hcols = tuple(f"h = {h}" for h in H)
    body = string.Template(TEMPLATE).safe_substitute(
        t_models=table(t_models, "models", hcols), t_evo=table(t_evo, "models", hcols),
        t_prophet=table(t_prophet, "wide", hcols),
        tiles=tiles, t_appr=table(t_appr, "wide", tuple(f"h = {h}" for h in H)), t_cat=table(t_cat, "wide", tuple(f"h = {h}" for h in H)),
        t_cal=table(t_cal, "wide", tuple(c for c in t_cal.columns if c.startswith("x ="))),
        t_rb=table(t_rb, "", tuple(c for c in t_rb.columns if c != "квинтиль расходов на жителя")),
        t_types=table(t_types, "", ("шоков",)), t_iv=table(t_iv, "", tuple(t_iv.columns[1:])),
        f1=pct(fin_rel[1]), f3=pct(fin_rel[3]), f6=pct(fin_rel[6]), f12=pct(fin_rel[12]),
        o1=pct(orc_rel[1]), o12=pct(orc_rel[12]), **{k: v for k, v in st.items() if k != "types"})
    path_html = OUT / "methodology_v2.html"
    path_html.write_text(body, encoding="utf-8")
    path_pdf = (OUT / "methodology_v2.pdf").resolve()
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={path_pdf}",
                    "--virtual-time-budget=5000", path_html.resolve().as_uri()], check=True, capture_output=True)
    print(f"готово: {path_pdf}")


TEMPLATE = open(Path(__file__).with_name("methodology_template.html"), encoding="utf-8").read()

if __name__ == "__main__":
    main()
