import json, numpy as np, pandas as pd
"""Данные и сборка страницы итогов (results/night_dashboard.html).
Запуск из корня: PYTHONPATH=. .venv/bin/python tools/dashboard_data.py (после src.final, src.shocks_v2 --news)."""
SP = 'results'
m = pd.read_csv('results/metrics_overall.csv')
BASE = ['naive', 'mean_3', 'mean_6', 'mean_12', 'drift', 'seasonal_naive', 'seasonal_naive_growth', 'auto_ets', 'auto_theta',
        'auto_arima', 'prophet_default', 'prophet_yearly', 'catboost_diff', 'catboost_logdiff', 'ridge_diff', 'ridge_logdiff']
nv = m[m.model == 'naive'].set_index(['category', 'horizon']).MAE
def rel(sub):  # Series (cat,h) -> % к naive
    return (sub / nv.reindex(sub.index) - 1) * 100
fin = m[m.model == 'final'].set_index(['category', 'horizon']).MAE
pre = m[m.model.isin(BASE) & (m.model != 'naive')]
pre_best = pre.loc[pre.groupby(['category', 'horizon']).MAE.idxmin()].set_index(['category', 'horizon'])
oracle = m[~m.model.isin(['final', 'naive'])].groupby(['category', 'horizon']).MAE.min()
dsp = m[m.model == 'catboost_logdiff_dsp_g123'].set_index(['category', 'horizon']).MAE
cbd = m[m.model == 'catboost_diff'].set_index(['category', 'horizon']).MAE
approaches = []
for name, s in [('naive', nv), ('лучшая базовая модель (выбрана по тесту)', pre_best.MAE), ('catboost_diff везде', cbd),
                ('профиль по панели везде', dsp), ('final', fin), ('лучшая по тесту в каждой ячейке', oracle)]:
    r = rel(s).groupby(level='horizon').mean().round(1)
    approaches.append({'name': name, 'values': [float(r[h]) for h in (1, 3, 6, 12)]})
cats = []
for c in ['Все категории', 'Продовольствие', 'Здоровье', 'Маркетплейсы', 'Общественное питание', 'Транспорт']:
    cats.append({'cat': c,
                 'final': [round(float(fin[(c, h)])) for h in (1, 3, 6, 12)],
                 'pre': [round(float(pre_best.MAE[(c, h)])) for h in (1, 3, 6, 12)],
                 'pre_model': [pre_best.model[(c, h)] for h in (1, 3, 6, 12)],
                 'naive': [round(float(nv[(c, h)])) for h in (1, 3, 6, 12)]})
res = pd.read_parquet('results/shocks_v2/scores.parquet')
st = pd.read_parquet('data/external/territory_static.parquet').set_index('territory_id')
res = res.join(st[['lat', 'lon']], on='territory_id')
heads = pd.read_csv('results/shocks_v2/top_shocks_news.csv')[['territory_id', 'month', 'новости', 'заголовки']]
sh = res[res['уровень'] == 'шок'].merge(heads, on=['territory_id', 'month'], how='left')
zc = [c for c in res.columns if c.startswith('z: ')]
shocks = []
for rr in sh.to_dict("records"):

    shocks.append({'id': int(rr['territory_id']), 'mo': rr['МО'], 'reg': rr['регион'], 'm': rr['month'],
                   'p': round(float(rr['score']) * 100, 2), 'main': rr['главная категория'], 'drv': rr['главное'] or '',
                   'type': rr['тип'] if isinstance(rr['тип'], str) else '', 'low': bool(rr['надёжность']),
                   'lat': round(float(rr['lat']), 2) if pd.notna(rr['lat']) else None,
                   'lon': round(float(rr['lon']), 2) if pd.notna(rr['lon']) else None,
                   'news': rr['новости'] if isinstance(rr['новости'], str) else '',
                   'heads': (rr['заголовки'] or '')[:300] if isinstance(rr['заголовки'], str) else ''})
bg = st.reindex(res.territory_id.unique())[['lat', 'lon']].dropna().round(2).values.tolist()
reg = pd.read_csv('results/shocks_v2/regional.csv')
reg = reg[reg['шок']][['month', 'регион', 'score', 'МО', 'главное', 'примечание']].fillna('')
cal = pd.read_csv('results/shocks_v2/calibration.csv')
cal = cal[cal['оценка'].isin(['topk', 'mahal', 'iforest', 'combo'])]
calib = {s: {k: g[g['оценка'] == k].sort_values('x')['найдено, %'].round(1).tolist() for k in ['topk', 'mahal', 'iforest', 'combo']}
         for s, g in cal.groupby('сценарий')}
import glob
from src.data import build_panels, complete_territories, load_consumption
dfc = load_consumption('consumption.parquet'); pan = build_panels(dfc, list(nv.index.get_level_values(0).unique()), complete_territories(dfc))
f25 = pd.concat(pd.read_parquet(f) for f in glob.glob('results/forecast_2025/*.parquet'))
fut = []
for c, p in pan.items():
    g = f25[f25.category == c].groupby('month')[['y_pred', 'lo90', 'hi90']].median()
    fut.append({'cat': c, 'hist': [round(float(v)) for v in np.median(p.values, axis=0)], 'hm': p.months,
                'fm': g.index.tolist(), 'f': g.y_pred.round(0).tolist(), 'lo': g.lo90.round(0).tolist(), 'hi': g.hi90.round(0).tolist(),
                'model': ', '.join(sorted(set(f25[f25.category == c].model)))})
data = {'future': fut, 'approaches': approaches, 'cats': cats, 'shocks': shocks, 'bg': bg,
        'regional': reg.round(2).to_dict('records'), 'calib': calib, 'mags': [5, 10, 20, 30]}
s = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
t = open('tools/dashboard_template.html').read()
open(f'{SP}/night_dashboard.html', 'w').write(t.replace('__DATA__', s.replace('</', '<\\/')))
print(len(s), 'bytes;', len(shocks), 'shocks;', len(bg), 'bg points')
print(approaches)
