"""Кандидатогенерация: каналы первого этапа + признаки для ранкера.

Этап 1 (дешёвый, на всём корпусе, плотные скоры по батчам запросов):
  * BM25 по полям объявления: заголовок / параметры / описание;
  * BM25 по «расширению документа»: тексты запросов из train, по которым выбирали
    это объявление (работает для объявлений, встречавшихся в train);
  * априорная вероятность локации объявления при локации поиска  P(item_loc | search_loc);
  * классификатор подкатегории по тексту запроса  P(microcat | query, фильтры).
  Кандидаты = top-K по комбинированному скору ∪ top-M по чистому тексту ∪ «память» train.
Этап 2: LightGBM переупорядочивает кандидатов, берём top-50.
"""
import numpy as np, pandas as pd, scipy.sparse as sp
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from common import tokens
from bm25 import BM25

EPS_LOC, EPS_MC = 1e-4, 1e-3


class ItemIndex:
    """Всё, что зависит только от корпуса объявлений."""
    def __init__(self, items, fields):
        self.items = items
        self.ids = items['item_id'].values
        self.id2row = {x: i for i, x in enumerate(self.ids)}
        self.bm = {k: BM25().fit(v) for k, v in fields.items()}
        # бинарные матрицы «объявление x токен» для признаков покрытия
        self.title_bin = self.bm['title'].W.T.tocsr(); self.title_bin.data[:] = 1
        self.params_bin = self.bm['params'].W.T.tocsr(); self.params_bin.data[:] = 1
        self.loc_codes, self.loc_idx = np.unique(items['item_location_id'].values, return_inverse=True)
        self.mc = items['item_microcat_id'].values
        self.rating = items['item_rating'].fillna(-1).values.astype(np.float32)
        self.reviews = items['item_rating_reviews_count'].fillna(0).values.astype(np.float32)
        self.has_price = items['item_price'].notna().values.astype(np.float32)
        self.desc_len = items['item_description_raw'].fillna('').str.len().values.astype(np.float32)
        # координаты (радианы) для расстояния до центра локации поиска
        self.lat = np.radians(items['item_latitude'].astype(float).fillna(0).values).astype(np.float32)
        self.lon = np.radians(items['item_longitude'].astype(float).fillna(0).values).astype(np.float32)
        # символьные n-граммы заголовка: ловят опечатки и склейки («бобкет» ~ «бобкат»)
        self.char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, sublinear_tf=True, dtype=np.float32)
        self.char_T = self.char_vec.fit_transform(_norm_titles(items['item_title_raw'])).T.tocsr()


def _norm_titles(xs):
    return [x.lower().replace('ё', 'е') if isinstance(x, str) else '' for x in xs]


def haversine_km(lat1, lon1, lat2, lon2):
    """lat1/lon1: (n,1), lat2/lon2: (m,) в радианах -> (n,m) км."""
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return (2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).astype(np.float32)


def _clf_text(q, p):
    return ' '.join(tokens(q)) + ' ' + ' '.join('p_' + t for t in tokens(p))


class TrainStats:
    """Статистики и модели, обученные на (части) train."""
    def __init__(self, fit, idx: ItemIndex, seed=0):
        self.idx = idx
        f = fit.assign(search_infm_params_text=fit['search_infm_params_text'].fillna(''))
        # --- локации: P(item_loc | search_loc)
        cnt = f.groupby(['search_location_id', 'item_location_id']).size()
        self.loc_prior = {}
        for sl, c in cnt.groupby(level=0):
            c = c.droplevel(0)
            pos = np.searchsorted(idx.loc_codes, c.index.values).clip(0, len(idx.loc_codes) - 1)
            ok = idx.loc_codes[pos] == c.index.values
            v = np.zeros(len(idx.loc_codes), np.float32)
            v[pos[ok]] = c.values[ok] / c.values.sum()
            self.loc_prior[sl] = v
        # --- «центр» локации поиска = медиана координат выбранных в ней объявлений
        cc = f.assign(la=f.item_latitude.astype(float), lo=f.item_longitude.astype(float)).groupby('search_location_id')[['la', 'lo']].median()
        self.sl_center = {k: (np.radians(a), np.radians(b)) for k, a, b in zip(cc.index, cc.la, cc.lo)}
        # --- классификатор подкатегории по (запрос + фильтры)
        g = f.groupby(['search_query', 'search_infm_params_text', 'item_microcat_id']).size().reset_index(name='n')
        X = [_clf_text(a, b) for a, b in zip(g.search_query, g.search_infm_params_text)]
        self.vec = TfidfVectorizer(token_pattern=r'\S+', ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        Xv = self.vec.fit_transform(X)
        self.clf = SGDClassifier(loss='log_loss', alpha=2e-6, max_iter=15, tol=None, random_state=seed)
        self.clf.fit(Xv, g.item_microcat_id.values, sample_weight=np.log1p(g.n.values))
        cls = self.clf.classes_
        self.mc_col = np.full(len(idx.mc), -1)
        m = {c: i for i, c in enumerate(cls)}
        self.mc_col = np.array([m.get(x, -1) for x in idx.mc])
        # --- популярность объявления и «память» запрос -> объявление
        inc = f[f.item_id.isin(idx.id2row)]
        pop = inc.item_id.map(idx.id2row).value_counts()
        self.pop = np.zeros(len(idx.ids), np.float32); self.pop[pop.index.values] = pop.values
        self.memo = defaultdict(Counter)
        for q, i in zip(inc.search_query, inc.item_id):
            self.memo[q][idx.id2row[i]] += 1
        self.seen_q = set(f.search_query)
        # --- расширение документа: запросы, по которым выбирали объявление
        exp = inc.groupby('item_id').search_query.agg(lambda s: ' '.join(s))
        docs = [[] for _ in idx.ids]
        for i, txt in exp.items():
            docs[idx.id2row[i]] = tokens(txt)
        self.bm_exp = BM25().fit(docs)

    def mc_proba(self, q, p):
        P = self.clf.predict_proba(self.vec.transform([_clf_text(a, b) for a, b in zip(q, p)])).astype(np.float32)
        P = np.hstack([P, np.zeros((len(P), 1), np.float32)])   # колонка -1 -> 0
        return P[:, self.mc_col]


FEATS = ['dist_km', 's_char', 's_title', 's_params', 's_desc', 's_exp', 'loc_p', 'mc_p', 'base', 'base_rank', 'text', 'text_rank',
         'pop', 'memo', 'same_loc', 'rating', 'reviews', 'has_price', 'desc_len', 'title_cov', 'fparam_cov',
         'q_len', 'q_seen', 'rating_filter_ok']
FEATS += [f + '_dmax' for f in ['base', 'text', 's_title', 's_desc', 's_char', 's_params', 'mc_p', 'loc_p', 'reviews', 'rating']]
FEATS += ['mc_p_rank', 'reviews_rank', 's_desc_rank', 'n_same_loc', 'mc_top']


def generate(queries, st: TrainStats, K=200, M=30, W=(1, 1, 1, 1), A=4.0, C=1.0, G=0.02, R=30.0, H=1.0, B=100):
    """Кандидаты + признаки. queries: DataFrame с колонками search_*.
    Возвращает DataFrame (qi, row, признаки...)."""
    idx = st.idx
    qs = queries['search_query'].values
    ps = queries['search_infm_params_text'].fillna('').values
    locs = queries['search_location_id'].values
    qt = [tokens(x) for x in qs]
    pt = [set(tokens(x)) - {'вид', 'услуг', 'тип', 'кто', 'оказыва'} for x in ps]
    out = []
    for s in range(0, len(qs), B):
        sl = slice(s, s + B)
        S = {k: m.score(m.query_matrix(qt[sl])) for k, m in idx.bm.items()}
        S['exp'] = st.bm_exp.score(st.bm_exp.query_matrix(qt[sl]))
        text = W[0] * S['title'] + W[1] * S['params'] + W[2] * S['desc'] + W[3] * S['exp']
        zero = np.zeros(len(idx.loc_codes), np.float32)
        LP = np.stack([st.loc_prior.get(l, zero) for l in locs[sl]])[:, idx.loc_idx]
        MC = st.mc_proba(qs[sl], ps[sl])
        cen = np.array([st.sl_center.get(l, (np.nan, np.nan)) for l in locs[sl]], np.float32)
        DK = haversine_km(cen[:, :1], cen[:, 1:], idx.lat, idx.lon)
        DK = np.nan_to_num(DK, nan=5000.0)
        # гео-сглаживание: объявление из соседнего города получает небольшую массу,
        # даже если такая пара локаций ни разу не встречалась в train
        LG = LP + G * np.exp(-DK / R)
        CH = (idx.char_vec.transform(_norm_titles(qs[sl])) @ idx.char_T).toarray()
        text = text + H * 10 * CH
        base = text + A * np.log(LG + EPS_LOC) + C * np.log(MC + EPS_MC)
        topb = np.argpartition(-base, K, axis=1)[:, :K]
        topt = np.argpartition(-text, M, axis=1)[:, :M]
        for j in range(len(text)):
            qi = s + j
            memo = st.memo.get(qs[qi], {})
            rows = np.unique(np.concatenate([topb[j], topt[j], np.fromiter(memo.keys(), int, len(memo))]))
            b = base[j]; t = text[j]
            qv = idx.bm['title'].query_matrix([qt[qi]]); n_q = max(1, len(set(qt[qi])))
            tcov = np.asarray(idx.title_bin[rows] @ qv.T.toarray()).ravel() / n_q
            if pt[qi]:
                pv = idx.bm['params'].query_matrix([list(pt[qi])])
                fcov = np.asarray(idx.params_bin[rows] @ pv.T.toarray()).ravel() / len(pt[qi])
            else:
                fcov = np.full(len(rows), -1, np.float32)
            d = {'qi': qi, 'dist_km': DK[j, rows], 's_char': CH[j, rows], 'title_cov': tcov, 'fparam_cov': fcov, 'row': rows, 's_title': S['title'][j, rows], 's_params': S['params'][j, rows],
                 's_desc': S['desc'][j, rows], 's_exp': S['exp'][j, rows], 'loc_p': LP[j, rows], 'mc_p': MC[j, rows],
                 'base': b[rows], 'text': t[rows]}
            out.append(pd.DataFrame(d))
    df = pd.concat(out, ignore_index=True)
    # ранги внутри запроса
    df['base_rank'] = df.groupby('qi')['base'].rank(ascending=False, method='first')
    df['text_rank'] = df.groupby('qi')['text'].rank(ascending=False, method='first')
    r = df['row'].values; qi = df['qi'].values
    df['pop'] = st.pop[r]
    df['memo'] = [st.memo.get(qs[a], {}).get(b, 0) for a, b in zip(qi, r)]
    df['same_loc'] = (idx.loc_codes[idx.loc_idx[r]] == locs[qi]).astype(np.float32)
    df['rating'] = idx.rating[r]; df['reviews'] = idx.reviews[r]
    df['has_price'] = idx.has_price[r]; df['desc_len'] = idx.desc_len[r]
    df['q_len'] = np.array([len(x) for x in qt])[qi]
    df['q_seen'] = np.array([x in st.seen_q for x in qs], np.float32)[qi]
    need4 = np.array(['4 звезды' in x for x in ps])[qi]
    df['rating_filter_ok'] = np.where(need4, (idx.rating[r] >= 4).astype(np.float32), -1)
    add_group_feats(df)
    return df


def add_group_feats(df):
    """Признаки относительно других кандидатов того же запроса: ранжированию важен
    не абсолютный скор, а насколько объявление лучше/хуже лидера."""
    g = df.groupby('qi')
    for f in ['base', 'text', 's_title', 's_desc', 's_char', 's_params', 'mc_p', 'loc_p', 'reviews', 'rating']:
        df[f + '_dmax'] = df[f] - g[f].transform('max')
    for f in ['mc_p', 'reviews', 's_desc']:
        df[f + '_rank'] = g[f].rank(ascending=False, method='first')
    df['n_same_loc'] = g['same_loc'].transform('sum')
    df['mc_top'] = g['mc_p'].transform('max')


def label(df, rel, idx):
    ids = idx.ids
    return np.array([ids[b] in rel[a] for a, b in zip(df['qi'].values, df['row'].values)], np.int8)


def top50(df, score_col, n_queries, idx):
    d = df[['qi', 'row', score_col]].sort_values(['qi', score_col], ascending=[True, False])
    d = d.groupby('qi').head(50)
    res = [[] for _ in range(n_queries)]
    for a, b in zip(d['qi'].values, d['row'].values):
        res[a].append(idx.ids[b])
    return res
