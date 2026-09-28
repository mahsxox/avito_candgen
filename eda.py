"""
Быстрый EDA — ответы на вопросы, от которых зависит дизайн решения.

  python eda.py --data data/
"""
import argparse

import numpy as np
import pandas as pd

from cg.data import build_groups, load_raw
from cg.pool import haversine_km, location_centroids
from cg.text import normalize

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
a = ap.parse_args()
train, bq, bi = load_raw(a.data)
groups = build_groups(train)

print("== размеры")
print(f"train pairs={len(train)}, query-groups={len(groups)}, bench queries={len(bq)}, bench items={len(bi)}")
print("релевантных на группу:\n", groups["relevant"].map(len).describe().round(2).to_string())

print("\n== пересечения (насколько полезна история train)")
tr_items, b_items = set(train["item_id"]), set(bi["item_id"])
print(f"доля объявлений бенчмарка, встречавшихся в train: {len(tr_items & b_items) / len(b_items):.3f}")
tq = set(train["search_query"].map(normalize))
print(f"доля запросов бенчмарка, чей текст точно встречался в train: {bq['search_query'].map(normalize).isin(tq).mean():.3f}")

print("\n== гео (насколько услуги локальны)")
same = (train["search_location_id"] == train["item_location_id"]).mean()
print(f"search_location_id == item_location_id: {same:.3f}")
cent = location_centroids(pd.concat([train, bi])[["item_location_id", "item_latitude", "item_longitude"]])
c = train["search_location_id"].map(cent)
ok = c.notna() & train["item_latitude"].notna()
cc = np.array(c[ok].tolist())
d = haversine_km(cc[:, 0], cc[:, 1], train.loc[ok, "item_latitude"].values, train.loc[ok, "item_longitude"].values)
print("расстояние от центра локации поиска до выбранного объявления, км (квантили):")
print(pd.Series(d).quantile([.5, .75, .9, .95, .99]).round(1).to_string())
print(f"доля поисков с доставкой: {train['search_is_delivery_search'].mean():.3f}")

print("\n== фильтры и рубрики")
print(train["search_infm_params_text"].value_counts().head(15).to_string())
print(f"уникальных microcat: {train['item_microcat_id'].nunique()}, category: {train['item_category_id'].nunique()}")
print("\nтоп запросов:\n", train["search_query"].value_counts().head(15).to_string())
