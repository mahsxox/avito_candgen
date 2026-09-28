"""
(Опционально, лучше на GPU) Дообучение bi-encoder'а на парах «запрос — выбранное объявление».

Лосс MultipleNegativesRankingLoss: для каждой пары остальные объявления в батче —
негативы. Это ровно задача кандидатогенерации: притянуть запрос к выбранному
объявлению и оттолкнуть от прочих. Валидационные группы из обучения исключаются
(тот же сплит, что в run.py), чтобы сравнение методов было честным.

  python train_dense.py --data data/ --out models/e5-finetuned
  python embed.py --data data/ --model models/e5-finetuned
  python run.py val --data data/
"""
import argparse

from cg.data import build_groups, history_pairs, load_raw, split_val
from cg.pipeline import DEFAULT_CFG

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--out", default="models/e5-ft")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-pairs", type=int, default=300_000)
    a = ap.parse_args()

    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    train, _, _ = load_raw(a.data)
    groups = build_groups(train)
    hist_groups, _ = split_val(groups, DEFAULT_CFG["n_val"], DEFAULT_CFG["seed"], DEFAULT_CFG["val_sampling"])
    hp = history_pairs(hist_groups).merge(
        train.drop_duplicates("item_id")[["item_id", "item_title_raw", "item_description_raw"]], on="item_id")
    hp = hp.sample(min(a.max_pairs, len(hp)), random_state=0)
    ex = [InputExample(texts=[f"query: {r.search_query}",
                              f"passage: {r.item_title_raw}. {str(r.item_description_raw or '')[:400]}"])
          for r in hp.itertuples()]
    model = SentenceTransformer(a.base)
    model.max_seq_length = 128
    dl = DataLoader(ex, shuffle=True, batch_size=a.batch)
    model.fit(train_objectives=[(dl, losses.MultipleNegativesRankingLoss(model))],
              epochs=a.epochs, warmup_steps=200, show_progress_bar=True)
    model.save(a.out)
    print("saved", a.out)
