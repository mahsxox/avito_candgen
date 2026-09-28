"""
Ретриверы — независимые способы оценить «насколько объявление подходит запросу».
Каждый возвращает для батча запросов матрицу скоров размера (batch x N_items).

1. BM25 (классический IR, собственная реализация на scipy.sparse):
   несколько полей объявления (заголовок, параметры, описание, «история запросов»),
   каждое взвешено по BM25 и сложено с весом поля (упрощённый BM25F).
2. Char n-gram TF-IDF по заголовку: устойчив к опечаткам, слитному/раздельному
   написанию и словоформам, которые стеммер не склеил («автоподбор» vs «авто подбор»).
3. History / Query-neighbours (коллаборативный сигнал из логов train):
   находим похожие запросы из train (по char n-gram), берём объявления, которые по ним
   выбирали, и рубрики (microcat) этих объявлений -> prior по подкатегории.
4. Dense (опционально): multilingual-e5 (sentence-transformers) — семантика,
   синонимы («обзвон по базе» ~ «холодные звонки»). Эмбеддинги считаются embed.py.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import normalize, tokenize


# ----------------------------------------------------------------------------- vocab
class Vocab:
    """Общий словарь основ для всех полей и запросов (нужен, чтобы матрицы складывались)."""

    def __init__(self, token_lists_iter):
        self.index = {}
        for toks in token_lists_iter:
            for t in toks:
                if t not in self.index:
                    self.index[t] = len(self.index)
        self.size = len(self.index)

    def to_csr(self, token_lists, binary=False) -> sp.csr_matrix:
        voc = self.index
        indptr = [0]
        indices = []
        for toks in token_lists:
            indices.extend(voc[t] for t in toks if t in voc)
            indptr.append(len(indices))
        m = sp.csr_matrix(
            (np.ones(len(indices), np.float32), np.asarray(indices, np.int32), np.asarray(indptr, np.int64)),
            shape=(len(token_lists), self.size),
        )
        m.sum_duplicates()
        if binary:
            m.data[:] = 1.0
        return m


def bm25_weight(tf: sp.csr_matrix, idf: np.ndarray, k1=1.2, b=0.75) -> sp.csr_matrix:
    """TF-матрицу (docs x terms) превращаем в матрицу BM25-весов термов."""
    tf = tf.tocsr().astype(np.float32, copy=True)
    dl = np.asarray(tf.sum(1)).ravel()
    avgdl = max(dl.mean(), 1e-6)
    rows = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
    denom = tf.data + k1 * (1 - b + b * dl[rows] / avgdl)
    tf.data = (idf[tf.indices] * tf.data * (k1 + 1) / denom).astype(np.float32)
    return tf


# ----------------------------------------------------------------------------- text fields
class ItemText:
    """Токенизированные поля объявлений — считаются один раз и переиспользуются."""

    def __init__(self, items, extra_query_texts):
        print("  tokenizing items ...", flush=True)
        self.title = [tokenize(t) for t in items["item_title_raw"].values]
        # В параметрах много чисел (цены, секунды графика работы) — это шум.
        self.params = [tokenize(t, drop_numbers=True) for t in items["item_infm_params_text"].values]
        # Длинные описания обрезаем: всё самое важное обычно в начале,
        # а SEO-хвосты «меня находят по запросам ...» тоже помещаются в 3000 символов.
        self.desc = [tokenize(t, max_chars=3000) for t in items["item_description_raw"].values]
        qtok = [tokenize(t) for t in extra_query_texts]
        self.vocab = Vocab(self.title + self.params + self.desc + qtok)
        self.T = self.vocab.to_csr(self.title)
        self.P = self.vocab.to_csr(self.params)
        self.D = self.vocab.to_csr(self.desc)
        # Бинарные матрицы для признаков «покрытия» (доля слов запроса в заголовке и т.п.)
        self.T_bin = self.T.copy(); self.T_bin.data[:] = 1
        self.P_bin = self.P.copy(); self.P_bin.data[:] = 1
        del self.title, self.params, self.desc  # списки токенов больше не нужны — экономим RAM
        print(f"  vocab size: {self.vocab.size}", flush=True)


# ----------------------------------------------------------------------------- BM25
class BM25Retriever:
    """
    Упрощённый BM25F. Поле `qhist` — «doc2query из логов»: к объявлению дописываем
    тексты запросов, по которым его выбирали в train. Это сильный сигнал: объявление
    «Скупка б/у техники» находится по «скупка телевизоров», хотя слова «телевизор» в нём нет.
    """

    def __init__(self, it: ItemText, qhist_tokens, field_w=None, k1=1.2, b=0.75):
        field_w = field_w or {"title": 3.0, "qhist": 2.0, "params": 1.0, "desc": 1.0}
        H = it.vocab.to_csr(qhist_tokens)
        n = it.T.shape[0]
        present = ((it.T + it.P + it.D + H) > 0).astype(np.float32)
        df = np.asarray(present.sum(0)).ravel()
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        Wf = {
            "title": bm25_weight(it.T, idf, k1, b),
            "params": bm25_weight(it.P, idf, k1, b),
            "desc": bm25_weight(it.D, idf, k1, b),
            "qhist": bm25_weight(H, idf, k1, b),
        }
        W = sum(field_w[k] * m for k, m in Wf.items())
        self.WT = W.T.tocsr()  # terms x docs: запрос @ WT = скоры
        # v5: скоры по полям отдельно — признаки для ранкера (совпадение в заголовке
        # и в описании значат разное, а сумма BM25F это смешивает)
        self.WT_fields = {k: m.T.tocsr() for k, m in Wf.items()}
        self.vocab = it.vocab

    def score(self, query_texts) -> np.ndarray:
        Q = self.vocab.to_csr([tokenize(q) for q in query_texts], binary=True)
        return (Q @ self.WT).toarray()

    def score_fields(self, query_texts) -> dict:
        Q = self.vocab.to_csr([tokenize(q) for q in query_texts], binary=True)
        return {k: (Q @ m).toarray() for k, m in self.WT_fields.items()}


# ----------------------------------------------------------------------------- char tf-idf
class CharRetriever:
    """Char n-gram (3–5, внутри слов) TF-IDF по заголовку + косинусная близость."""

    def __init__(self, items):
        self.vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=400_000,
            sublinear_tf=True, dtype=np.float32,
        )
        X = self.vec.fit_transform([normalize(t) for t in items["item_title_raw"].values])
        self.XT = X.T.tocsr()

    def score(self, query_texts) -> np.ndarray:
        Q = self.vec.transform([normalize(q) for q in query_texts])
        return (Q @ self.XT).toarray()


# ----------------------------------------------------------------------------- history
class HistoryRetriever:
    """
    Коллаборативный сигнал: «пользователи с похожим запросом выбирали вот это».
    q -> top-k похожих запросов из train (char TF-IDF) -> их выбранные объявления.
    Дополнительно отдаёт распределение по microcat (prior подкатегории для запроса).
    """

    def __init__(self, hist_pairs, item_idx: dict, item_mc: np.ndarray, n_items: int, topk=30, min_sim=0.35,
                 query_pairs=None):
        """hist_pairs  — пары для сигнала «какие объявления выбирали» (item-level);
        query_pairs — пары для сигнала «какие подкатегории выбирали» (query-level).
        В валидации «холодные» объявления убираются только из первого набора: рубрики
        по запросу на бенчмарке считаются по всему train, пусть и в валидации так будет."""
        query_pairs = hist_pairs if query_pairs is None else query_pairs

        def _prep(pairs):
            hp = pairs.assign(ii=pairs["item_id"].map(item_idx)).dropna(subset=["ii"])
            return hp["search_query"].map(normalize).values, hp["ii"].values.astype(np.int64)

        qn_q, ii_q = _prep(query_pairs)
        uq, qcode_q = np.unique(qn_q, return_inverse=True)
        self.uq = uq
        # item-level: какие объявления выбирали по запросу (строки нормируем: вклад
        # популярного запроса не должен забивать всё остальное)
        qn_i, ii_i = _prep(hist_pairs)
        qcode_i = np.searchsorted(uq, qn_i)
        ok = (qcode_i < len(uq)) & (uq[np.minimum(qcode_i, len(uq) - 1)] == qn_i)
        M = sp.csr_matrix((np.ones(ok.sum(), np.float32), (qcode_i[ok], ii_i[ok])), shape=(len(uq), n_items))
        M.sum_duplicates()
        rs = np.asarray(M.sum(1)).ravel(); rs[rs == 0] = 1
        self.M = sp.diags(1 / rs).dot(M).tocsr()
        # query-level: распределение по подкатегориям
        self.n_mc = int(item_mc.max()) + 1
        MC = sp.csr_matrix((np.ones(len(ii_q), np.float32), (qcode_q, item_mc[ii_q])), shape=(len(uq), self.n_mc))
        MC.sum_duplicates()
        rq = np.asarray(MC.sum(1)).ravel(); rq[rq == 0] = 1
        self.MC = sp.diags(1 / rq).dot(MC).tocsr()
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, dtype=np.float32)
        self.UQT = self.vec.fit_transform(uq).T.tocsr()
        self.topk, self.min_sim = topk, min_sim

    def _neighbours(self, query_texts) -> sp.csr_matrix:
        S = (self.vec.transform([normalize(q) for q in query_texts]) @ self.UQT).toarray()
        k = min(self.topk, S.shape[1])
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        rows = np.repeat(np.arange(S.shape[0]), k)
        vals = S[rows, top.ravel()]
        keep = vals >= self.min_sim
        # квадрат близости: точные/почти точные совпадения запроса весят заметно больше
        return sp.csr_matrix((vals[keep] ** 2, (rows[keep], top.ravel()[keep])), shape=(S.shape[0], S.shape[1]))

    def score(self, query_texts):
        K = self._neighbours(query_texts)
        items = (K @ self.M).toarray()
        mc = (K @ self.MC).toarray()
        mc /= np.maximum(mc.sum(1, keepdims=True), 1e-9)
        return items, mc


def qhist_tokens(hist_pairs, item_idx: dict, n_items: int, max_q_per_item=20):
    """Для каждого объявления — токены запросов, по которым его выбирали (поле qhist)."""
    out = [[] for _ in range(n_items)]
    hp = hist_pairs.assign(ii=hist_pairs["item_id"].map(item_idx)).dropna(subset=["ii"])
    hp["qn"] = hp["search_query"].map(normalize)
    # частота пары (объявление, запрос); оставляем max_q_per_item самых частых запросов
    cnt = hp.groupby(["ii", "qn"]).size().rename("c").reset_index()
    cnt = cnt.sort_values(["ii", "c"], ascending=[True, False]).groupby("ii").head(max_q_per_item)
    tok_cache = {q: tokenize(q) for q in cnt["qn"].unique()}
    for ii, q, c in zip(cnt["ii"].values.astype(np.int64), cnt["qn"].values, cnt["c"].values):
        # повторяем запрос до 3 раз по частоте — частые запросы важнее
        out[ii].extend(tok_cache[q] * int(min(c, 3)))
    return out


# ----------------------------------------------------------------------------- dense
class DenseRetriever:
    """Скалярное произведение нормированных эмбеддингов (считаются заранее embed.py)."""

    def __init__(self, emb_items: np.ndarray, model_name: str, query_prefix="query: "):
        from sentence_transformers import SentenceTransformer  # опциональная зависимость

        self.E = emb_items.astype(np.float32)
        self.model = SentenceTransformer(model_name)
        self.prefix = query_prefix

    def score(self, query_texts) -> np.ndarray:
        q = self.model.encode([self.prefix + t for t in query_texts], normalize_embeddings=True,
                              batch_size=128, show_progress_bar=False)
        return q.astype(np.float32) @ self.E.T
