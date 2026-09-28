"""
Диагностика пула кандидатов: почему релевантные объявления в него не попали.

  python diagnose.py --art artifacts        # после `python run.py val`

Все доли взвешены как в метрике: объявление запроса с n релевантными весит 1/n,
поэтому «доля» здесь = сколько Recall@50 можно отыграть, починив этот случай.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd



def report(art: str, n_examples: int = 25):
    diag = pd.read_parquet(os.path.join(art, "diag_val.parquet"))
    n_rel = pd.read_parquet(os.path.join(art, "n_rel_val.parquet"))["n"]
    channels = json.load(open(os.path.join(art, "channels.json")))
    nq = len(n_rel)
    diag["w"] = 1.0 / diag["qid"].map(n_rel).values / nq  # сумма w по всем = 1
    diag["w_total"] = 1.0
    hit, miss = diag[diag["in_pool"] == 1], diag[diag["in_pool"] == 0]
    W = lambda d: d["w"].sum()

    print("\n================ POOL DIAGNOSTICS ================")
    print(f"Pool recall: {W(hit):.4f}   (упущено: {W(miss):.4f})")
    pool_path = os.path.join(art, "pool_val.parquet")
    if os.path.exists(pool_path):
        sz = pd.read_parquet(pool_path, columns=["qid"]).groupby("qid").size()
        print(f"Размер пула: mean={sz.mean():.0f}, p50={sz.median():.0f}, p90={sz.quantile(.9):.0f}")

    print("\n-- Каналы: recall канала в одиночку / уникальный вклад (ловит только он)")
    src = diag["src"].values
    rows = []
    for b, name in enumerate(channels):
        m = (src >> b) & 1 == 1
        only = src == (1 << b)
        rows.append((name, diag.loc[m, "w"].sum(), diag.loc[only, "w"].sum()))
    print(pd.DataFrame(rows, columns=["channel", "recall_alone", "unique"]).round(4).to_string(index=False))

    print("\n-- Кто упущен (доли — от всей метрики)")
    no_text = (miss["rank_bm25_all"] < 0) & (miss["rank_char_all"] < 0)
    facts = {
        "объявление НЕ из корпуса бенчмарка (только train)": ~miss["in_bench"],
        "у локации поиска нет координат (гео не работает)": ~miss["geo_ok"],
        "та же location_id": miss["same_loc"] > 0,
        "в области near": miss["near"],
        "дальше 100 км": miss["dist_km"] > 100,
        "удалённая услуга": miss["remote"] > 0,
        "ни одного общего слова/н-граммы с запросом": no_text,
        "hist ничего не дал (похожих запросов не выбирали)": miss["rank_hist_all"] < 0,
        "prior подкатегории < 0.05": miss["mc_prior"] < 0.05,
        "объявление никогда не выбирали в истории": miss["item_pop"] == 0,
    }
    for k, m in facts.items():
        print(f"  {k:55s} {miss.loc[m, 'w'].sum():.4f}  ({m.mean():.0%} упущенных)")

    print("\n-- Кривые полноты: доля релевантных, попадающих в top-K ретривера (all / near)")
    Ks = [50, 100, 300, 1000, 3000, 10000]
    rows = []
    for c in [c for c in diag.columns if c.startswith("rank_")]:
        r = diag[c].values
        ok = (r >= 0) & (diag["near"].values if c.endswith("_near") else True)
        rows.append([c] + [diag.loc[ok & (r < K), "w"].sum() for K in Ks])
    print(pd.DataFrame(rows, columns=["retriever"] + [f"top{K}" for K in Ks]).round(3).to_string(index=False))

    print("\n-- Упущенные: квантили расстояния, км")
    print(miss["dist_km"].quantile([.25, .5, .75, .9]).round(0).to_string())

    print(f"\n-- Примеры упущенных ({n_examples}):")
    ex = miss.sample(min(n_examples, len(miss)), random_state=0)
    for r in ex.itertuples():
        print(f"  [{r.query}] {('{' + r.filters + '}') if r.filters else ''} -> «{str(r.title)[:60]}» "
              f"| dist={r.dist_km:.0f}км same={int(r.same_loc)} bm25_near#{r.rank_bm25_near} "
              f"hist_near#{r.rank_hist_near} mc={r.mc_prior:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default="artifacts")
    report(ap.parse_args().art)
