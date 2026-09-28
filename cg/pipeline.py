"""
Сборка всего пайплайна для режима 'val' (оценка на отложенных запросах train)
и 'submit' (ответ для benchmark_queries).
"""
import json
import os

import numpy as np
import pandas as pd

from .data import build_groups, build_items, history_pairs, load_raw, split_val
from .pool import PoolBuilder, add_derived
from .text import normalize
from .retrievers import BM25Retriever, CharRetriever, DenseRetriever, HistoryRetriever, ItemText, qhist_tokens

DEFAULT_CFG = {
    # Состав каналов и их K задаются в cg/pool.py: DEFAULT_CHANNELS.
    "local_radius_km": 30,    # что считаем «рядом» по расстоянию
    "aff_min": 0.01,          # ... или локация, куда >=1% выборов из этой локации поиска
    "mc_min": 0.05,           # порог prior подкатегории для канала mc
    "geo_scale_km": 30,       # масштаб затухания geo_prox = exp(-dist/scale)
    "n_val": 10000,
    "seed": 42,
    # --- валидация «как бенчмарк» (v3) ---
    # "bench": ищем среди объявлений бенчмарка + искомых объявлений валидации (~190k, как в
    #          бенчмарке). "all": по всем 516k (v2; пессимистично: корпус в 2.7 раза больше).
    "val_corpus": "train_sample",
    # В корпусе бенчмарка лишь ~10% объявлений встречались в train, а искомые объявления
    # валидации в v2 имели историю в 42% случаев -> признаки истории выглядели сильнее,
    # чем будут на бенчмарке. Поэтому у (1 - warm_rate) искомых объявлений валидации
    # стираем всю историю выборов («холодный старт»), как у большинства объявлений бенчмарка.
    "val_warm_rate": 0.10,
    # --- v4 ---
    # "train_sample": корпус = искомые + случайные объявления из train (всего ~ размер бенчмарка).
    #   Искомые и прочие кандидаты одного происхождения (в v3 искомые — из train, прочие —
    #   из бенчмарка: если их можно различить, модель учится на утечке). История выборов
    #   остаётся у val_warm_rate ВСЕХ объявлений корпуса — как у бенчмарка (~10%).
    # val_sampling: "uniform" — группы случайно; "by_text" — одна группа на текст запроса.
    "val_sampling": "by_text",
}


def make_split(train, bq, items, groups, mode, cfg):
    """Разбиение на историю и запросы + корпус поиска. Вынесено в функцию, чтобы
    make_ft_data.py (данные для дообучения e5) использовал РОВНО то же разбиение
    и не учил модель на искомых объявлениях валидации.
    Возвращает: queries, hp_full (все пары истории), hp (пары «тёплых» объявлений),
    allowed (маска корпуса поиска), cold (id объявлений без истории)."""
    N = len(items)
    cold = set()
    if mode == "val":
        hist_groups, queries = split_val(groups, cfg["n_val"], cfg["seed"], cfg["val_sampling"])
        val_items = sorted(set(i for r in queries["relevant"] for i in r))
        is_val = items["item_id"].isin(val_items).values
        rng = np.random.default_rng(cfg["seed"])
        if cfg["val_corpus"] == "train_sample":
            cand = np.flatnonzero(items["item_id"].isin(set(train["item_id"])).values & ~is_val)
            n_neg = max(int(items["in_bench"].sum()) - int(is_val.sum()), 0)
            allowed = is_val.copy()
            allowed[rng.choice(cand, size=min(n_neg, len(cand)), replace=False)] = True
            keep = rng.random(N) < cfg["val_warm_rate"]
            cold = set(items["item_id"].values[~keep])
        else:
            allowed = (items["in_bench"].values | is_val) if cfg["val_corpus"] == "bench" else np.ones(N, bool)
            cold = {i for i in val_items if rng.random() >= cfg["val_warm_rate"]}
    else:
        hist_groups, queries = groups, bq.copy()
        allowed = items["in_bench"].values
    queries = queries.reset_index(drop=True)
    queries["qid"] = np.arange(len(queries))
    hp_full = history_pairs(hist_groups)  # для статистик уровня запроса/локации
    hp = hp_full[~hp_full["item_id"].isin(cold)] if cold else hp_full  # уровень объявления
    return queries, hp_full, hp, allowed, cold


def prepare(data_dir, mode, cfg, art_dir="artifacts", use_dense=True):
    print("[1] loading data", flush=True)
    train, bq, bi = load_raw(data_dir)
    items = build_items(train, bi)
    items["mc_code"] = pd.factorize(items["item_microcat_id"].fillna(-1))[0]
    N = len(items)
    item_idx = dict(zip(items["item_id"].values, range(N)))
    groups = build_groups(train)
    print(f"    items in corpus: {N} (bench: {items['in_bench'].sum()}), train query-groups: {len(groups)}")

    queries, hp_full, hp, allowed, cold = make_split(train, bq, items, groups, mode, cfg)
    if mode == "val":
        seen = queries["search_query"].map(normalize).isin(set(hp_full["search_query"].map(normalize)))
        print(f"    val queries whose text is in history: {seen.mean():.2f} (benchmark: 0.37)")
    print(f"    corpus for search: {int(allowed.sum())} items; history pairs: {len(hp)}"
          + (f"; cold val items: {len(cold)}" if cold else ""))

    print("[2] building indexes", flush=True)
    it = ItemText(items, extra_query_texts=groups["search_query"].unique())
    retr = {
        "bm25": BM25Retriever(it, qhist_tokens(hp, item_idx, N)),
        "char": CharRetriever(items),
    }
    hist = HistoryRetriever(hp, item_idx, items["mc_code"].values, N, query_pairs=hp_full)
    emb_path = os.path.join(art_dir, "emb_items.npy")
    if use_dense and os.path.exists(emb_path):
        meta = json.load(open(os.path.join(art_dir, "emb_meta.json")))
        ids = np.load(os.path.join(art_dir, "emb_item_ids.npy"), allow_pickle=True)
        E = np.load(emb_path)
        pos = pd.Series(np.arange(len(ids)), index=ids).reindex(items["item_id"]).values
        assert not np.isnan(pos).any(), "эмбеддинги посчитаны не для всех объявлений — перезапустите embed.py"
        retr["dense"] = DenseRetriever(E[pos.astype(int)], meta["model"])
        print(f"    dense retriever: {meta['model']}")

    print("[3] building candidate pools", flush=True)
    rel_lists = None
    if mode == "val":
        rel_lists = [[item_idx[i] for i in r if i in item_idx] for r in queries["relevant"]]
    builder = PoolBuilder(items, it, retr, hist, allowed, cfg, hp, item_idx, hp_full)
    pool, diag = builder.build(queries, rel_lists)
    channels = [c[0] for c in builder.channels]
    print("    derived features ...", flush=True)
    add_derived(pool, channels)

    n_rel = None
    if mode == "val":
        rel = pd.DataFrame({"qid": np.repeat(queries["qid"].values, [len(r) for r in rel_lists]),
                            "ii": np.concatenate([np.asarray(r, np.int64) for r in rel_lists])})
        pool = pool.merge(rel.assign(label=1).astype({"ii": np.int32}), on=["qid", "ii"], how="left")
        pool["label"] = pool["label"].fillna(0).astype(np.int8)
        n_rel = rel.groupby("qid").size()
        diag.attrs["channels"] = channels
    else:
        pool["label"] = 0
    return items, queries, pool, n_rel, diag


def validate_answer(ans: pd.DataFrame, bq: pd.DataFrame, bench_ids: set):
    """Проверки формата из условия задачи — до отправки."""
    assert list(ans.columns) == ["query_id", "answer"], "колонки должны быть query_id,answer"
    assert ans["query_id"].is_unique and set(ans["query_id"]) == set(bq["query_id"]), "query_id не совпадают"
    for a in ans["answer"]:
        ids = a.split(" ") if a else []
        assert len(ids) <= 50 and len(set(ids)) == len(ids), "больше 50 или повторы"
        assert all(len(i) == 16 and i in bench_ids for i in ids), "item_id не из корпуса"
    print("    answer.csv format: OK")
