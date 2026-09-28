"""Токенизация корпуса объявлений (кэшируется на диск)."""
import os, pickle, pandas as pd
from common import tokens

def item_fields(items, cache='cache/items_tok.pkl', desc_chars=3000):
    if os.path.exists(cache):
        return pickle.load(open(cache, 'rb'))
    f = {
        'title': [tokens(x) for x in items['item_title_raw']],
        'params': [tokens(x) for x in items['item_infm_params_text']],
        'desc': [tokens(x[:desc_chars] if isinstance(x, str) else '') for x in items['item_description_raw']],
    }
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    pickle.dump(f, open(cache, 'wb'))
    return f
