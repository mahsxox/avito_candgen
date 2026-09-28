"""
Проверка сдвига между валидацией и бенчмарком (~3-5 мин).

  python check_shift.py --data data

Зачем: v3 на валидации 0.937, на бенчмарке 0.861. Проверяем две гипотезы:
  1) запросы бенчмарка «длиннохвостее», чем случайные запросы train (там много «маникюр»);
  2) объявления из train отличимы от объявлений бенчмарка по своим признакам
     (в валидации v3 все искомые — из train, остальные кандидаты — из бенчмарка -> утечка).
"""
import argparse

import numpy as np
import pandas as pd

from cg.data import build_groups, build_items, load_raw
from cg.text import normalize

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
a = ap.parse_args()
train, bq, bi = load_raw(a.data)
groups = build_groups(train)
tn = groups["search_query"].map(normalize)
freq = tn.value_counts()


def describe(texts: pd.Series, self_count: int, name: str):
    """self_count=1 для запросов из train: не считаем сам запрос в его частоте."""
    f = texts.map(freq).fillna(0).values - self_count
    print(f"  {name:28s} unique_texts={texts.nunique() / len(texts):.2f} | "
          f"частота текста в train: median={np.median(f):.0f}, "
          f"=0: {np.mean(f == 0):.2f}, >=10: {np.mean(f >= 10):.2f}, >=100: {np.mean(f >= 100):.2f}")


print("== 1. Распределение запросов")
bn = bq["search_query"].map(normalize)
describe(bn, 0, "benchmark")
rng = np.random.default_rng(0)
describe(tn.iloc[rng.choice(len(tn), min(5000, len(tn)), replace=False)], 1, "val v3 (случайные группы)")
uniq = tn.drop_duplicates()
describe(uniq.iloc[rng.choice(len(uniq), min(5000, len(uniq)), replace=False)], 1, "val by_text (1 на текст)")
print("  топ запросов бенчмарка:", bn.value_counts().head(8).to_dict())

print("\n== 2. Отличимы ли объявления train от объявлений бенчмарка (AUC; 0.5 = неотличимы)")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

items = build_items(train, bi)
tr_ids = set(train["item_id"])
items = items[items["in_bench"] ^ items["item_id"].isin(tr_ids)]  # без пересечения
X = pd.DataFrame({
    "price": items["item_price"], "rating": items["item_rating"],
    "reviews": items["item_rating_reviews_count"],
    "has_geo": items["item_latitude"].notna().astype(int),
    "title_len": items["item_title_raw"].fillna("").str.len(),
    "desc_len": items["item_description_raw"].fillna("").str.len(),
    "params_len": items["item_infm_params_text"].fillna("").str.len(),
    "microcat": items["item_microcat_id"].fillna(-1),
    "category": items["item_category_id"].fillna(-1),
    "phone_hidden": items["item_is_phone_hidden"].astype(float),
    "msg_forbidden": items["item_is_message_forbidden"].astype(float),
})
y = items["in_bench"].astype(int).values
idx = rng.permutation(len(X))
cut = int(len(idx) * 0.7)
m = lgb.train(dict(objective="binary", verbose=-1, num_leaves=31, learning_rate=0.1),
              lgb.Dataset(X.iloc[idx[:cut]], y[idx[:cut]]), 200)
auc = roc_auc_score(y[idx[cut:]], m.predict(X.iloc[idx[cut:]]))
print(f"  AUC = {auc:.3f}")
imp = pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
print("  чем отличаются:", (imp / imp.sum()).round(2).head(5).to_dict())
print("  медианы train-only vs bench:")
print(X.groupby(y).median().T.rename(columns={0: "train", 1: "bench"}).round(1).to_string())
