"""
(Опционально) Dense-эмбеддинги объявлений локальной open-source моделью.

  python embed.py --data data/ --model intfloat/multilingual-e5-small
  python embed.py --data data/ --model models/e5-finetuned      # после train_dense.py

Модель скачивается один раз с HuggingFace и дальше работает локально (без внешних API).
На CPU e5-small по ~400k объявлениям — порядка часа, на GPU — несколько минут.
Результат: artifacts/emb_items.npy (+ ids, meta) — run.py подхватит его автоматически.
"""
import argparse
import json
import os

import numpy as np

from cg.data import build_items, load_raw


def item_passage(r, titles_only: bool) -> str:
    # e5 обучен с префиксами "query: " / "passage: "
    if titles_only:
        return f"passage: {r.item_title_raw}"
    return f"passage: {r.item_title_raw}. {str(r.item_description_raw or '')[:400]}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--art", default="artifacts")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--batch", type=int, default=256)
    # На CPU: --titles-only --max-seq 32 — в ~4-5 раз быстрее (заголовки короткие),
    # на полном корпусе ~20-40 минут вместо нескольких часов.
    ap.add_argument("--titles-only", action="store_true")
    ap.add_argument("--max-seq", type=int, default=128)
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    train, _, bi = load_raw(a.data)
    items = build_items(train, bi)
    texts = [item_passage(r, a.titles_only) for r in items.itertuples()]
    model = SentenceTransformer(a.model)
    model.max_seq_length = a.max_seq  # заголовок (+ начало описания); длиннее — дорого и мало пользы
    E = model.encode(texts, batch_size=a.batch, normalize_embeddings=True, show_progress_bar=True)
    os.makedirs(a.art, exist_ok=True)
    np.save(os.path.join(a.art, "emb_items.npy"), E.astype(np.float16))
    np.save(os.path.join(a.art, "emb_item_ids.npy"), items["item_id"].values.astype(object))
    json.dump({"model": a.model}, open(os.path.join(a.art, "emb_meta.json"), "w"))
    print("saved", E.shape)
