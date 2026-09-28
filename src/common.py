"""Общие утилиты: загрузка данных, токенизация, валидация, метрика."""
import re, numpy as np, pandas as pd
import Stemmer

_ru = Stemmer.Stemmer('russian'); _en = Stemmer.Stemmer('english')
_tok_re = re.compile(r'[a-zа-я0-9]+')
_cache = {}

def stem(w):
    s = _cache.get(w)
    if s is None:
        s = _en.stemWord(w) if w.isascii() else _ru.stemWord(w)
        _cache[w] = s
    return s

def tokens(text):
    """lower + ё->е + токены + стемминг Snowball (ru/en)."""
    if not isinstance(text, str) or not text:
        return []
    return [stem(w) for w in _tok_re.findall(text.lower().replace('ё', 'е'))]

GROUP = ['search_query', 'search_location_id', 'search_infm_params_text', 'search_category']

def recall_at_k(pred, rel):
    """pred: list[list[item_id]], rel: list[set[item_id]] -> mean recall."""
    return float(np.mean([len(set(p) & r) / len(r) for p, r in zip(pred, rel)]))

def make_validation(train, corpus_ids, n=3000, seed=42, p_unseen=0.63):
    """Отложенные «поиски» из train, чьи выбранные объявления есть в корпусе бенчмарка.
    В бенчмарке только ~37% текстов запросов встречаются в train, поэтому для доли
    p_unseen валидационных поисков из train_fit удаляются ВСЕ строки с тем же текстом
    запроса — иначе валидация была бы оптимистичнее теста.
    Возвращает (train_fit, val_queries, val_rel)."""
    t = train.copy()
    t['search_infm_params_text'] = t['search_infm_params_text'].fillna('')
    t['gid'] = t.groupby(GROUP, sort=False).ngroup()
    in_c = t['item_id'].isin(corpus_ids)
    cand = np.unique(t.loc[in_c, 'gid'])
    rng = np.random.RandomState(seed)
    val_g = set(rng.choice(cand, size=min(n, len(cand)), replace=False))
    vmask = t['gid'].isin(val_g)
    vtexts = t.loc[vmask, ['gid', 'search_query']].drop_duplicates('gid')
    unseen_txt = set(vtexts.loc[rng.rand(len(vtexts)) < p_unseen, 'search_query'])
    drop = vmask | t['search_query'].isin(unseen_txt)
    v = t[vmask & in_c]
    rel = v.groupby('gid')['item_id'].agg(set)
    vq = v.drop_duplicates('gid').set_index('gid').loc[rel.index, GROUP].reset_index()
    return t[~drop].drop(columns='gid'), vq, list(rel.values)


def make_splits(train, corpus_ids, n_rank=6000, n_val=3000, seed=42, p_unseen=0.45, p_cold=0.7):
    """Две непересекающиеся отложенные выборки поисков: rank (обучение ранкера) и val
    (честная оценка). Статистики/модели первого этапа строятся только по train_fit."""
    t = train.copy()
    t['search_infm_params_text'] = t['search_infm_params_text'].fillna('')
    t['gid'] = t.groupby(GROUP, sort=False).ngroup()
    in_c = t['item_id'].isin(corpus_ids)
    cand = np.unique(t.loc[in_c, 'gid'])
    rng = np.random.RandomState(seed)
    chosen = rng.choice(cand, size=n_rank + n_val, replace=False)
    parts = {'rank': set(chosen[:n_rank]), 'val': set(chosen[n_rank:])}
    hold = t['gid'].isin(parts['rank'] | parts['val'])
    vtexts = t.loc[hold, ['gid', 'search_query']].drop_duplicates('gid')
    unseen_txt = set(vtexts.loc[rng.rand(len(vtexts)) < p_unseen, 'search_query'])
    drop = hold | t['search_query'].isin(unseen_txt)
    # В корпусе бенчмарка только ~10% объявлений встречаются в train, а у отложенных
    # поисков выбранное объявление «знакомо» train в ~32% случаев. Чтобы признаки
    # истории объявления не переоценивались, для доли p_cold отложенных объявлений
    # удаляем из train_fit всю их историю.
    h_items = np.unique(t.loc[hold & in_c, 'item_id'])
    cold = set(h_items[rng.rand(len(h_items)) < p_cold])
    drop = drop | t['item_id'].isin(cold)
    out = {}
    for name, g in parts.items():
        v = t[t['gid'].isin(g) & in_c]
        rel = v.groupby('gid')['item_id'].agg(set)
        q = v.drop_duplicates('gid').set_index('gid').loc[rel.index, GROUP].reset_index(drop=True)
        out[name] = (q, list(rel.values))
    return t[~drop].drop(columns='gid'), out


def make_folds(train, corpus_ids, n_folds=3, per_fold=5000, seed=42, p_unseen=0.45, p_cold=0.7):
    """K непересекающихся фолдов отложенных поисков. Для каждого фолда свой train_fit
    (весь train минус поиски фолда, минус тексты «новых» запросов, минус история
    «холодных» объявлений) — так статистики почти не теряют данных, а ранкер
    получает n_folds * per_fold обучающих запросов.
    Возвращает список (train_fit_k, queries_k, rel_k)."""
    t = train.copy()
    t['search_infm_params_text'] = t['search_infm_params_text'].fillna('')
    t['gid'] = t.groupby(GROUP, sort=False).ngroup()
    in_c = t['item_id'].isin(corpus_ids)
    cand = np.unique(t.loc[in_c, 'gid'])
    rng = np.random.RandomState(seed)
    chosen = rng.choice(cand, size=n_folds * per_fold, replace=False)
    out = []
    for k in range(n_folds):
        g = set(chosen[k * per_fold:(k + 1) * per_fold])
        hold = t['gid'].isin(g)
        vtexts = t.loc[hold, ['gid', 'search_query']].drop_duplicates('gid')
        unseen_txt = set(vtexts.loc[rng.rand(len(vtexts)) < p_unseen, 'search_query'])
        h_items = np.unique(t.loc[hold & in_c, 'item_id'])
        cold = set(h_items[rng.rand(len(h_items)) < p_cold])
        drop = hold | t['search_query'].isin(unseen_txt) | t['item_id'].isin(cold)
        v = t[hold & in_c]
        rel = v.groupby('gid')['item_id'].agg(set)
        q = v.drop_duplicates('gid').set_index('gid').loc[rel.index, GROUP].reset_index(drop=True)
        out.append((t[~drop].drop(columns='gid'), q, list(rel.values)))
    return out
