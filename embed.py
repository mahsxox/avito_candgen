"""
Эмбеддинги всех объявлений (train + бенчмарк) dense-моделью.

  python embed.py --data data --model models/e5-ft --art artifacts_v7

Результат (run.py подхватит его автоматически из той же папки --art):
  emb_items.npy     — нормированные эмбеддинги (float16, экономим место)
  emb_item_ids.npy  — их item_id
  emb_meta.json     — путь к модели: ею же кодируются запросы при поиске
Работает на CPU или GPU (если есть).
"""
import argparse
import json
import os

import numpy as np

from cg.data import build_items, load_raw
from cg.dense_text import passage

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--art", default="artifacts_v7")
    ap.add_argument("--model", default="models/e5-ft")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-seq", type=int, default=128, help="должно совпадать с train_dense.py")
    a = ap.parse_args()
    import torch
    from sentence_transformers import SentenceTransformer

    train, _, bi = load_raw(a.data)
    items = build_items(train, bi)
    texts = ["passage: " + passage(t, d) for t, d in zip(items["item_title_raw"], items["item_description_raw"])]
    model = SentenceTransformer(a.model, device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = a.max_seq
    E = model.encode(texts, batch_size=a.batch, normalize_embeddings=True, show_progress_bar=True,
                     convert_to_numpy=True)
    os.makedirs(a.art, exist_ok=True)
    np.save(os.path.join(a.art, "emb_items.npy"), E.astype(np.float16))
    np.save(os.path.join(a.art, "emb_item_ids.npy"), items["item_id"].values.astype(object))
    json.dump({"model": a.model}, open(os.path.join(a.art, "emb_meta.json"), "w"))
    print("saved", E.shape)
