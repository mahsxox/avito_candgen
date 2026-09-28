"""
Загрузка данных, «запросы-группы», валидационный сплит и единый корпус объявлений.

Ключевые решения:
1. В train нет query_id, есть только пары «запрос — выбранное объявление».
   Считаем одним запросом группу строк с одинаковыми признаками search_*.
   Все объявления группы = релевантные (как в бенчмарке: обычно 1–2, иногда больше).
2. Валидация: откладываем n_val случайных групп. Для них строим те же признаки,
   что и для бенчмарка, но всё «историческое» (какие объявления выбирали по каким
   запросам) считаем ТОЛЬКО по оставшимся группам — иначе будет утечка ответа.
3. Корпус = объединение объявлений бенчмарка и train (без дублей по item_id).
   Для валидации ищем по всему объединению (релевантные валидации — объявления
   из train), для сабмита маскируем всё, что не входит в benchmark_items.
"""
import os

import numpy as np
import pandas as pd

Q_KEYS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
ITEM_COLS = [
    "item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
    "item_category_id", "item_microcat_id", "item_price", "item_rating",
    "item_rating_reviews_count", "item_location_id", "item_latitude",
    "item_longitude", "item_is_phone_hidden", "item_is_message_forbidden",
]

NUM_COLS = [
    "search_location_id", "search_is_delivery_search", "item_price", "item_rating",
    "item_rating_reviews_count", "item_location_id", "item_latitude", "item_longitude",
    "item_category_id", "item_microcat_id",
]


def load_raw(data_dir: str):
    train = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
    bq = pd.read_parquet(os.path.join(data_dir, "benchmark_queries.parquet"))
    bi = pd.read_parquet(os.path.join(data_dir, "benchmark_items.parquet"))
    # Идентификаторы — строго строки (иначе можно потерять ведущие нули).
    train["item_id"] = train["item_id"].astype(str)
    bi["item_id"] = bi["item_id"].astype(str)
    bq["query_id"] = bq["query_id"].astype(str)
    for df in (train, bq):
        df["search_infm_params_text"] = df["search_infm_params_text"].fillna("")
        df["search_query"] = df["search_query"].fillna("")
    # В parquet координаты/цены лежат как decimal.Decimal (dtype=object) — numpy с ними
    # не работает (np.radians падает). Приводим все числовые признаки к float64.
    for df in (train, bq, bi):
        for c in NUM_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return train, bq, bi


def build_items(train: pd.DataFrame, bench_items: pd.DataFrame) -> pd.DataFrame:
    """Единый корпус. Версия объявления из бенчмарка приоритетнее версии из train."""
    tr_items = train[ITEM_COLS].drop_duplicates("item_id")
    items = pd.concat([bench_items[ITEM_COLS], tr_items], ignore_index=True)
    items = items.drop_duplicates("item_id", keep="first").reset_index(drop=True)
    items["in_bench"] = items["item_id"].isin(set(bench_items["item_id"]))
    return items


def build_groups(train: pd.DataFrame) -> pd.DataFrame:
    """Группируем пары в запросы: одна строка = один запрос со списком релевантных."""
    g = (
        train.groupby(Q_KEYS, sort=False, dropna=False)["item_id"]
        .agg(lambda s: list(dict.fromkeys(s)))
        .reset_index()
        .rename(columns={"item_id": "relevant"})
    )
    g["gid"] = np.arange(len(g))
    return g


def split_val(groups: pd.DataFrame, n_val: int, seed: int = 42, mode: str = "uniform"):
    """uniform — случайные группы (как трафик: много «маникюр»);
    by_text — не больше одной группы на текст запроса (длинный хвост, как, возможно, в бенчмарке)."""
    from .text import normalize

    rng = np.random.default_rng(seed)
    if mode == "by_text":
        perm = rng.permutation(len(groups))
        tn = groups["search_query"].map(normalize).values[perm]
        _, first = np.unique(tn, return_index=True)
        cand = perm[first]
        val_gids = rng.choice(cand, size=min(n_val, len(cand)), replace=False)
    else:
        val_gids = rng.choice(len(groups), size=min(n_val, len(groups)), replace=False)
    is_val = np.zeros(len(groups), bool)
    is_val[val_gids] = True
    return groups[~is_val].reset_index(drop=True), groups[is_val].reset_index(drop=True)


def history_pairs(groups: pd.DataFrame) -> pd.DataFrame:
    """Группы -> пары (запрос, item_id) для «исторических» признаков."""
    return groups[Q_KEYS + ["relevant"]].explode("relevant").rename(columns={"relevant": "item_id"})
