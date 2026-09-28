"""Проверка answer.csv по требованиям платформы."""
import sys, re, pandas as pd
ans_path, data = sys.argv[1], sys.argv[2]
a = pd.read_csv(ans_path, dtype=str, keep_default_na=False)
q = pd.read_parquet(f'{data}/benchmark_queries.parquet', columns=['query_id'])
ids = set(pd.read_parquet(f'{data}/benchmark_items.parquet', columns=['item_id']).item_id)
assert list(a.columns) == ['query_id', 'answer'], a.columns
assert a.query_id.is_unique and set(a.query_id) == set(q.query_id) and len(a) == len(q)
assert a.query_id.str.len().eq(16).all()
lens = []
for s in a.answer:
    x = s.split(' ')
    assert len(x) <= 50 and len(x) == len(set(x)), 'dup or >50'
    assert all(re.fullmatch(r'[0-9a-f]{16}', i) and i in ids for i in x), 'bad id'
    lens.append(len(x))
print('OK', len(a), 'rows; ids per row min/mean', min(lens), sum(lens) / len(lens))
