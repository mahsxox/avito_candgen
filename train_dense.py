"""
Дообучение dense-модели multilingual-e5-small на парах «запрос -> выбранное объявление».
Работает локально: на CPU (по умолчанию) или на GPU, если он есть в системе.

  python train_dense.py --data data --out models/e5-ft             # как в итоговом решении
  python train_dense.py --data data --out models/e5-ft --cpu-fast  # быстрее на CPU, чуть хуже

Какие пары берём. Ровно те, что run.py в режиме валидации считает «историей»
(функция make_split): без запросов валидации и только по «тёплым» объявлениям
(~10% — как доля объявлений бенчмарка, встречавшихся в train). Поэтому модель не видит
искомые объявления валидации: валидация остаётся честной. Одна и та же модель
используется и в валидации, и при построении ответа для бенчмарка.

Лосс: MultipleNegativesRankingLoss — для каждой пары остальные объявления батча служат
негативами; это прямо задача поиска (притянуть запрос к выбранному объявлению).
Базовая модель intfloat/multilingual-e5-small (MIT) скачивается один раз с HuggingFace,
дальше всё работает офлайн из папки --out.
"""
import argparse
import os
import random

import numpy as np
import pandas as pd

from cg.data import build_groups, build_items, load_raw
from cg.dense_text import passage
from cg.pipeline import DEFAULT_CFG, make_split


def build_pairs(data_dir: str) -> pd.DataFrame:
    train, bq, bi = load_raw(data_dir)
    items = build_items(train, bi)
    groups = build_groups(train)
    _, _, hp, _, _ = make_split(train, bq, items, groups, "val", DEFAULT_CFG)
    text = pd.Series([passage(t, d) for t, d in zip(items["item_title_raw"], items["item_description_raw"])],
                     index=items["item_id"].values)
    pairs = pd.DataFrame({"query": hp["search_query"].values, "passage": hp["item_id"].map(text).values})
    return pairs.dropna().drop_duplicates().sample(frac=1.0, random_state=0).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--out", default="models/e5-ft")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-seq", type=int, default=128)
    ap.add_argument("--cpu-fast", action="store_true",
                    help="1 эпоха и max-seq 64: примерно в 3-4 раза быстрее на CPU, качество немного ниже")
    a = ap.parse_args()
    if a.cpu_fast:
        a.epochs, a.max_seq = 1, 64

    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    # воспроизводимость: фиксируем все генераторы случайных чисел
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = build_pairs(a.data)
    print(f"pairs: {len(pairs)} (уникальных запросов {pairs['query'].nunique()}) | device: {device} | "
          f"epochs={a.epochs} batch={a.batch} max_seq={a.max_seq}", flush=True)

    model = SentenceTransformer(a.base, device=device)
    model.max_seq_length = a.max_seq
    examples = [InputExample(texts=["query: " + q, "passage: " + p]) for q, p in zip(pairs["query"], pairs["passage"])]
    loader = DataLoader(examples, shuffle=True, batch_size=a.batch)
    model.fit(train_objectives=[(loader, losses.MultipleNegativesRankingLoss(model))],
              epochs=a.epochs, warmup_steps=100, show_progress_bar=True, use_amp=(device == "cuda"))
    os.makedirs(a.out, exist_ok=True)
    model.save(a.out)
    print("saved", a.out)
