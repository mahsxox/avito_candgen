"""
Шаг B: как из пула выбрать 50. Три семейства:
  * одиночные методы: один ретривер (+ опционально гео) — бейзлайны для сравнения;
  * линейный гибрид: score = w · features, веса подбираются координатным спуском
    напрямую по Recall@50 (метрика недифференцируема, но спуск по сетке работает);
  * LightGBM LambdaRank по всем признакам (open-source, MIT).
"""
import numpy as np
import pandas as pd

from .pool import FEATURES

K = 50

# Методы для сравнения: (признаки, начальные веса). Для одиночных без гео веса не тюним.
METHODS = {
    "bm25_only":   ({"bm25": 1.0}, False),
    "char_only":   ({"char": 1.0}, False),
    "hist_only":   ({"hist": 1.0}, False),
    "dense_only":  ({"dense": 1.0}, False),
    "bm25_geo":    ({"bm25": 1.0, "same_loc": 0.5, "geo_prox": 0.5, "remote": 0.1}, True),
    "char_geo":    ({"char": 1.0, "same_loc": 0.5, "geo_prox": 0.5, "remote": 0.1}, True),
    "dense_geo":   ({"dense": 1.0, "same_loc": 0.5, "geo_prox": 0.5, "remote": 0.1}, True),
    "hybrid_linear": ({f: (1.0 if f in ("bm25", "char", "hist", "dense") else 0.2) for f in FEATURES}, True),
}


def recall_at_k(qid, score, label, n_rel: pd.Series, k=K):
    """Recall@k. n_rel — полное число релевантных у запроса (в т.ч. не попавших в пул)."""
    order = np.lexsort((-score, qid))
    q_sorted = qid[order]
    starts = np.r_[0, np.flatnonzero(np.diff(q_sorted)) + 1]
    rank = np.arange(len(q_sorted)) - np.repeat(starts, np.diff(np.r_[starts, len(q_sorted)]))
    hit = (label[order] > 0) & (rank < k)
    hits = pd.Series(hit.astype(np.float32)).groupby(q_sorted).sum()
    per_q = (hits.reindex(n_rel.index).fillna(0) / n_rel).values
    return float(np.mean(per_q))


def pool_recall(pool, n_rel):
    return float((pool.groupby("qid")["label"].sum().reindex(n_rel.index).fillna(0) / n_rel).mean())


def linear_score(pool, w: dict):
    s = np.zeros(len(pool), np.float32)
    for f, v in w.items():
        s += v * pool[f].values
    return s


def tune_linear(pool, n_rel, w0: dict,
                grid=(-0.5, -0.2, -0.1, -0.05, 0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3),
                passes=3):
    """Координатный спуск по Recall@50. Один вес (первый текстовый) фиксирован = масштаб."""
    w = dict(w0)
    qid, lab = pool["qid"].values, pool["label"].values
    best = recall_at_k(qid, linear_score(pool, w), lab, n_rel)
    anchor = next(iter(w))
    for p in range(passes):
        improved = False
        for f in w:
            if f == anchor:
                continue
            for v in grid:
                cand = dict(w, **{f: v})
                r = recall_at_k(qid, linear_score(pool, cand), lab, n_rel)
                if r > best + 1e-5:
                    best, w, improved = r, cand, True
        print(f"    pass {p + 1}: recall={best:.4f}")
        if not improved:
            break
    return w, best


# ----------------------------------------------------------------------------- LightGBM
LGB_PARAMS = dict(
    objective="lambdarank", metric="ndcg", eval_at=[K], learning_rate=0.05,
    num_leaves=63, min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.8,
    bagging_freq=1, lambdarank_truncation_level=K + 20, verbose=-1,
    seed=42, deterministic=True, force_row_wise=True,  # воспроизводимость между запусками
)


# v5: «большой» вариант — все негативы, меньше шаг, больше листьев.
LGB_BIG = dict(LGB_PARAMS, learning_rate=0.03, num_leaves=127, min_data_in_leaf=30, feature_fraction=0.8)

# Ранкеры для сравнения: имя -> (без признаков истории?, параметры, негативов на запрос или None=все)
RANKERS = {
    "lgbm": (False, LGB_PARAMS, 400),
    "lgbm_nohist": (True, LGB_PARAMS, 400),
    "lgbm_big": (True, LGB_BIG, None),
    # v7: ещё один член ансамбля — другой seed и подвыборка признаков (разнообразие ошибок)
    "lgbm_nohist_s2": (True, dict(LGB_PARAMS, seed=7, feature_fraction=0.7, num_leaves=95), 400),
}
# Ансамбли: среднее перцентилей мест внутри запроса (шкалы скоров у моделей разные,
# а места сравнимы). Разные модели ошибаются по-разному — среднее устойчивее.
ENSEMBLES = {
    "ens_nohist": ["lgbm_nohist", "lgbm_big"],
    "ens_all": ["lgbm", "lgbm_nohist", "lgbm_big"],
    "ens_all4": ["lgbm", "lgbm_nohist", "lgbm_big", "lgbm_nohist_s2"],
}


def rank_avg(qid, scores: list) -> np.ndarray:
    out = np.zeros(len(qid), np.float64)
    for sc in scores:
        out += pd.Series(sc).groupby(qid).rank(pct=True).values
    return out / len(scores)


def _lgb_data(pool, feats, neg_per_query=400, seed=0):
    import lightgbm as lgb

    # в обучении нужны только запросы, где в пуле есть хотя бы один положительный
    pos_q = pool.loc[pool["label"] > 0, "qid"].unique()
    p = pool[pool["qid"].isin(pos_q)]
    # Пул большой (тысячи кандидатов на запрос) — для обучения оставляем все позитивы
    # и случайные neg_per_query негативов на запрос. Экономит память и время в разы.
    rnd = np.random.default_rng(seed).random(len(p))
    rk = pd.Series(rnd).groupby(p["qid"].values).rank(method="first").values
    if neg_per_query is not None:
        p = p[(p["label"].values > 0) | (rk <= neg_per_query)]
    p = p.sort_values("qid", kind="stable")
    groups = p.groupby("qid", sort=True).size().values
    return lgb.Dataset(p[feats], label=p["label"].values, group=groups, free_raw_data=False)


def train_lgbm(pool_tr, feats, pool_va=None, rounds=2000, params=None, neg_per_query=400):
    """Список признаков сохраняется в модели (model.feature_name()) — сабмит берёт его оттуда."""
    import lightgbm as lgb

    params = params or LGB_PARAMS
    dtr = _lgb_data(pool_tr, feats, neg_per_query)
    valid = [_lgb_data(pool_va, feats, neg_per_query)] if pool_va is not None else []
    cb = [lgb.early_stopping(100, verbose=False)] if valid else []
    return lgb.train(params, dtr, num_boost_round=rounds, valid_sets=valid, callbacks=cb)


def top_k_per_query(pool, score, k=K):
    """Возвращает {qid: [индексы объявлений]} — top-k по score."""
    df = pd.DataFrame({"qid": pool["qid"].values, "ii": pool["ii"].values, "s": score})
    df = df.sort_values(["qid", "s"], ascending=[True, False])
    df = df.groupby("qid").head(k)
    return df.groupby("qid")["ii"].apply(list).to_dict()
