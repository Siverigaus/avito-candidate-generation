"""Полный прогон: обучение ранкера на отложенных поисках из train и предсказание для бенчмарка.
Запуск:  python src/run.py <папка_с_parquet> <выходной_answer.csv>"""
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__))
from common import make_folds
from prep import item_fields
from pipeline import ItemIndex, TrainStats, generate, label, FEATS

DATA = sys.argv[1] if len(sys.argv) > 1 else 'dataset'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'answer.csv'
GEN = dict(K=300, A=4, C=2, G=0.3, H=0.15)          # параметры этапа 1 (подобраны на валидации)
LGB = dict(objective='lambdarank', learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=60,
           seed=0, deterministic=True, force_col_wise=True, num_threads=2, verbose=-1)
N_ROUNDS = 300
t0 = time.time(); log = lambda *a: print(f'[{time.time() - t0:6.0f}s]', *a, flush=True)

items = pd.read_parquet(f'{DATA}/benchmark_items.parquet')
train = pd.read_parquet(f'{DATA}/train.parquet', columns=[
    'search_query', 'search_location_id', 'search_is_delivery_search', 'search_infm_params_text', 'search_category',
    'item_id', 'item_location_id', 'item_microcat_id', 'item_latitude', 'item_longitude'])
bench = pd.read_parquet(f'{DATA}/benchmark_queries.parquet')
log('loaded', train.shape, items.shape, bench.shape)

idx = ItemIndex(items, item_fields(items, cache='cache/items_tok.pkl')); log('item index')

# 1) обучающая выборка ранкера: отложенные поиски, признаки по статистикам без них
folds = make_folds(train, set(items.item_id), n_folds=3, per_fold=5000)
C = []
for k, (fit, q, rel) in enumerate(folds):
    st = TrainStats(fit, idx)
    d = generate(q, st, **GEN); d['y'] = label(d, rel, idx)
    d[FEATS] = d[FEATS].astype(np.float32); d['qi'] += k * 10 ** 6
    C.append((d, rel)); log(f'fold {k}: fit rows {len(fit)}, candidates {len(d)}')
del folds, st

def fit_ranker(d):
    d = d[d.groupby('qi').y.transform('max') > 0].sort_values(['qi', 'row'], kind='stable')
    return lgb.train(LGB, lgb.Dataset(d[FEATS], d.y, group=d.groupby('qi', sort=True).size().values), N_ROUNDS)

def recall50(d, rel, off=0):
    d = d.assign(qi=d['qi'] - off)
    d = d.sort_values(['qi', 'p', 'row'], ascending=[True, False, True], kind='stable').groupby('qi').head(50)
    hit = d.assign(h=label(d, rel, idx)).groupby('qi').h.sum().reindex(range(len(rel)), fill_value=0)
    return float(np.mean(hit.values / np.array([len(r) for r in rel])))

# 1a) честная оценка: ранкер на фолдах 1-2, метрика на фолде 0
m_eval = fit_ranker(pd.concat([C[1][0], C[2][0]], ignore_index=True))
dv, rv = C[0]; dv['p'] = m_eval.predict(dv[FEATS])
ceil = dv.groupby('qi').y.sum().reindex(range(len(rv)), fill_value=0).values / np.array([len(r) for r in rv])
log('VALIDATION recall@50 =', round(recall50(dv, rv), 4),
    '| stage-1 only =', round(recall50(dv.assign(p=dv['base']), rv), 4),
    '| candidate ceiling =', round(float(ceil.mean()), 4))
# 1b) финальный ранкер на всех трёх фолдах
model = fit_ranker(pd.concat([c[0] for c in C], ignore_index=True)); log('final ranker trained')
del C, dv, m_eval

# 2) бенчмарк: статистики по всему train
st = TrainStats(train, idx); log('stats on full train')
d = generate(bench, st, **GEN); log('bench candidates', len(d))
d['p'] = model.predict(d[FEATS])
d = d.sort_values(['qi', 'p', 'row'], ascending=[True, False, True], kind='stable').groupby('qi').head(50)
ans = d.groupby('qi')['row'].agg(lambda r: ' '.join(idx.ids[r.values]))
out = pd.DataFrame({'query_id': bench['query_id'].values,
                    'answer': [ans.get(i, '') for i in range(len(bench))]})
out.to_csv(OUT, index=False)
log('saved', OUT)
