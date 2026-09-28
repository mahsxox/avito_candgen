"""
Точка входа.

  # 1) валидация: строит пулы для отложенных запросов train, сравнивает все методы,
  #    подбирает веса и обучает LightGBM, сохраняет всё в artifacts/
  python run.py val --data data/

  # 2) ответ для бенчмарка лучшим (или любым) методом
  python run.py submit --data data/ --method lgbm
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

from cg.models import (ENSEMBLES, METHODS, RANKERS, linear_score, pool_recall, rank_avg, recall_at_k,
                       top_k_per_query, train_lgbm, tune_linear)
from cg.pipeline import DEFAULT_CFG, prepare, validate_answer
from cg.pool import feature_cols


def _sub(pool, n_rel, n_q=2000, seed=1):
    """Подвыборка запросов для координатного спуска: пул большой, а спуск делает
    сотни пересчётов метрики. 2000 запросов хватает, чтобы веса были стабильны."""
    if len(n_rel) <= n_q:
        return pool, n_rel
    keep = np.random.default_rng(seed).choice(n_rel.index.values, n_q, replace=False)
    return pool[pool["qid"].isin(keep)], n_rel[n_rel.index.isin(keep)]


def run_val(args, cfg):
    os.makedirs(args.art, exist_ok=True)
    pool_path = os.path.join(args.art, "pool_val.parquet")
    if args.reuse_pool and os.path.exists(pool_path):
        pool = pd.read_parquet(pool_path)
        n_rel = pd.read_parquet(os.path.join(args.art, "n_rel_val.parquet"))["n"]
    else:
        _, _, pool, n_rel, diag = prepare(args.data, "val", cfg, args.art, use_dense=not args.no_dense)
        # строки in_pool=0 — релевантные, НЕ пойманные пулом; нужны только для диагностики
        diag.to_parquet(os.path.join(args.art, "diag_val.parquet"))
        json.dump(diag.attrs["channels"], open(os.path.join(args.art, "channels.json"), "w"))
        pool = pool[pool["in_pool"] == 1].reset_index(drop=True)
        pool.to_parquet(pool_path)
        n_rel.rename("n").to_frame().to_parquet(os.path.join(args.art, "n_rel_val.parquet"))
        from diagnose import report
        report(args.art)
    has_dense = pool["dense"].abs().sum() > 0

    # Честная оценка: подбор весов / обучение на половине A, замер на половине B.
    rng = np.random.default_rng(0)
    qids = n_rel.index.values
    qa = set(rng.choice(qids, size=len(qids) // 2, replace=False))
    in_a = pool["qid"].isin(qa)
    pa, pb = pool[in_a], pool[~in_a]
    na, nb = n_rel[n_rel.index.isin(qa)], n_rel[~n_rel.index.isin(qa)]

    print(f"\nPool recall (верхняя граница для любого отбора из пула): {pool_recall(pool, n_rel):.4f}"
          f"  | средний размер пула: {pool.groupby('qid').size().mean():.0f}")
    rows, final_w = [], {}
    for name, (w0, tunable) in METHODS.items():
        if not args.baselines:
            break  # линейные бейзлайны уже сравнены в v1–v4; включить: --baselines
        if "dense" in name and not has_dense:
            continue
        print(f"\n== {name}")
        w = dict(w0)
        if tunable:
            w, _ = tune_linear(*_sub(pa, na), w0)
        r = recall_at_k(pb["qid"].values, linear_score(pb, w), pb["label"].values, nb)
        rows.append({"method": name, "recall@50 (B)": round(r, 4), "weights": json.dumps(w)})
        print(f"   recall@50 on B = {r:.4f}   weights = {w}")
        # финальные веса — по всей валидации
        final_w[name] = tune_linear(*_sub(pool, n_rel), w)[0] if tunable else w

    qa_list = sorted(qa)
    qa1 = set(qa_list[: len(qa_list) * 4 // 5])  # внутри A: train / early-stopping
    final_models, preds_b = {}, {}
    for name, (excl, params, neg) in RANKERS.items():
        print(f"\n== {name} (LambdaRank)")
        feats = feature_cols(pool, exclude_hist=excl)
        model = train_lgbm(pa[pa["qid"].isin(qa1)], feats, pa[~pa["qid"].isin(qa1)], params=params, neg_per_query=neg)
        preds_b[name] = model.predict(pb[feats])
        r = recall_at_k(pb["qid"].values, preds_b[name], pb["label"].values, nb)
        rows.append({"method": name, "recall@50 (B)": round(r, 4), "weights": f"best_iter={model.best_iteration}"})
        print(f"   recall@50 on B = {r:.4f}  (best_iter={model.best_iteration}, features={len(feats)})")
        imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
        print("   top-15 features (gain):\n" + imp.head(15).round(0).to_string())
        # финальная модель — на всей валидации с найденным числом деревьев
        final_models[name] = train_lgbm(pool, feats, None, rounds=max(model.best_iteration, 50),
                                        params=params, neg_per_query=neg)
    for name, members in ENSEMBLES.items():
        sc = rank_avg(pb["qid"].values, [preds_b[m] for m in members])
        r = recall_at_k(pb["qid"].values, sc, pb["label"].values, nb)
        rows.append({"method": name, "recall@50 (B)": round(r, 4), "weights": "+".join(members)})
        print(f"\n== {name}: recall@50 on B = {r:.4f}")

    res = pd.DataFrame(rows).sort_values("recall@50 (B)", ascending=False)
    print("\n================ RESULTS ================\n" + res[["method", "recall@50 (B)"]].to_string(index=False))
    res.to_csv(os.path.join(args.art, "results.csv"), index=False)
    json.dump(final_w, open(os.path.join(args.art, "linear_weights.json"), "w"), ensure_ascii=False, indent=1)
    for name, m in final_models.items():
        pickle.dump(m, open(os.path.join(args.art, f"{name}.pkl"), "wb"))


def run_submit(args, cfg):
    items, queries, pool, _, _ = prepare(args.data, "submit", cfg, args.art, use_dense=not args.no_dense)
    def _predict(name):
        model = pickle.load(open(os.path.join(args.art, f"{name}.pkl"), "rb"))
        return model.predict(pool[model.feature_name()])

    if args.method in ENSEMBLES:
        score = rank_avg(pool["qid"].values, [_predict(m) for m in ENSEMBLES[args.method]])
    elif args.method.startswith("lgbm"):
        score = _predict(args.method)
    else:
        path = os.path.join(args.art, "linear_weights.json")
        weights = json.load(open(path)) if os.path.exists(path) else {}
        w = weights.get(args.method, METHODS[args.method][0])
        print(f"    weights: {w}")
        score = linear_score(pool, w)
    top = top_k_per_query(pool, score)

    # Добивка до 50: если кандидатов меньше, дополняем популярными объявлениями
    # той же локации (порядок не важен для метрики, а лишний шанс попасть — полезен).
    bench = items[items["in_bench"]]
    pop = bench.sort_values("item_rating_reviews_count", ascending=False)
    pop_by_loc = pop.groupby("item_location_id").head(50).groupby("item_location_id").apply(
        lambda d: d.index.tolist()).to_dict()
    ids = items["item_id"].values
    answers = []
    for _, q in queries.iterrows():
        cand = list(top.get(q["qid"], []))
        if len(cand) < 50:
            seen = set(cand)
            for ii in pop_by_loc.get(q["search_location_id"], []) + pop.index[:100].tolist():
                if ii not in seen:
                    cand.append(ii); seen.add(ii)
                if len(cand) == 50:
                    break
        answers.append(" ".join(ids[cand[:50]]))
    ans = pd.DataFrame({"query_id": queries["query_id"].astype(str).values, "answer": answers})
    validate_answer(ans, queries, set(bench["item_id"]))
    # lineterminator="\n": на Windows pandas пишет \r\n — «\r» может прилипнуть к последнему item_id
    ans.to_csv(args.out, index=False, lineterminator="\n")
    print(f"    saved {args.out}: {len(ans)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["val", "submit"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--art", default="artifacts")
    ap.add_argument("--method", default="lgbm", help="lgbm | lgbm_nohist | lgbm_big | ens_nohist | ens_all | hybrid_linear | bm25_geo | ... (см. cg/models.py)")
    ap.add_argument("--out", default="answer.csv")
    ap.add_argument("--n-val", type=int, default=DEFAULT_CFG["n_val"])
    ap.add_argument("--reuse-pool", action="store_true", help="не пересчитывать пул валидации")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--baselines", action="store_true", help="также сравнить линейные методы (медленно)")
    ap.add_argument("--val-corpus", default=DEFAULT_CFG["val_corpus"], choices=["bench", "all", "train_sample"])
    ap.add_argument("--val-sampling", default=DEFAULT_CFG["val_sampling"], choices=["uniform", "by_text"])
    ap.add_argument("--val-warm-rate", type=float, default=DEFAULT_CFG["val_warm_rate"])
    a = ap.parse_args()
    cfg = dict(DEFAULT_CFG, n_val=a.n_val, val_corpus=a.val_corpus, val_warm_rate=a.val_warm_rate,
               val_sampling=a.val_sampling)
    run_val(a, cfg) if a.mode == "val" else run_submit(a, cfg)
