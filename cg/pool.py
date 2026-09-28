"""
Пул кандидатов и признаки.

Схема (двухступенчатая, как в проде, только внутри кандидатогенерации):
  Шаг A. Набор КАНАЛОВ «ретривер × область». Каждый канал отдаёт свой top-K,
         объединение = пул. Полнота пула — потолок Recall@50 для любого отбора из него.
  Шаг B. Из пула выбираем 50 (линейная формула или LightGBM, см. models.py).

Области (почему их несколько — главный урок первого прогона):
  * all    — вся страна (редкие запросы, где исполнителей мало);
  * same   — объявления ровно той же location_id, что и поиск;
  * near   — «рядом»: та же локация, ИЛИ < R км, ИЛИ локация, куда исторически
             «ходят» пользователи этой локации поиска (loc affinity, см. ниже).
             Удалённые услуги сюда НЕ входят: в v1 они попадали в «local» и,
             т.к. «Удалённо» стоит у огромной доли объявлений, вытесняли местных;
  * remote — только удалённые/онлайн услуги, отдельный маленький канал.

Loc affinity. search_location_id бывает регионом, а объявления привязаны к городам/районам.
Тогда «та же локация» пуста, а центр локации по объявлениям не найти, и гео ломается.
Поэтому по истории train считаем P(локация объявления | локация поиска) и центр
локации поиска как медиану координат выбранных оттуда объявлений.

Каналы без текста (для запросов, где слова запроса и объявления не пересекаются,
например «укладка плитки» → «Ремонт квартир под ключ»):
  * mc     — объявления рядом из подкатегорий, которые выбирали по похожим запросам;
  * locpop — объявления, которые чаще всего выбирали из этой локации поиска
             в тех же подкатегориях.
"""
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .text import normalize, tokenize

EARTH_R = 6371.0
FAR_KM = 3000.0


def haversine_km(lat, lon, lat_arr, lon_arr):
    lat, lon, la, lo = (np.radians(np.asarray(x, dtype=np.float64)) for x in (lat, lon, lat_arr, lon_arr))
    a = np.sin((la - lat) / 2) ** 2 + np.cos(lat) * np.cos(la) * np.sin((lo - lon) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def location_centroids(items: pd.DataFrame) -> dict:
    """location_id -> (медианная широта, долгота) объявлений с этой локацией."""
    d = items.dropna(subset=["item_latitude", "item_longitude", "item_location_id"])
    med = d.groupby("item_location_id")[["item_latitude", "item_longitude"]].median()
    return {int(k): (float(r.item_latitude), float(r.item_longitude)) for k, r in med.iterrows()}


_REMOTE_RE = re.compile(r"удален|онлайн|по всей росси")
_RATING_RE = re.compile(r"рейтинг пользователя\s*(\d)")

FEATURES = [
    # текстовые ретриверы (нормированы в [0,1] внутри запроса)
    "bm25", "char", "hist", "dense",
    # гео
    "same_loc", "geo_prox", "log_dist", "aff", "remote", "is_delivery",
    # совпадение с фильтрами и рубрикой
    "title_cov", "filter_cov", "rating_ok", "mc_prior",
    # популярность из истории выборов (только по «исторической» части train)
    "item_pop", "pop_loc",
    # качество объявления
    "log_reviews", "rating",
    # в скольких каналах нашёлся кандидат
    "n_src",
    # v6: цена (для признаков «относительно соседей»)
    "log_price",
    # v5: BM25 по полям отдельно (нормированы на максимум общего BM25 запроса)
    "bm25f_title", "bm25f_params", "bm25f_desc", "bm25f_qhist",
]

# Признаки уровня запроса (одинаковы для всех кандидатов запроса; полезны деревьям:
# «насколько запрос специфичен» и «насколько плотный рынок в этой локации»).
Q_FEATURES = ["q_ntok", "q_bm25max", "q_n_same", "q_n_near", "q_pool_size"]

# Ранги внутри пула запроса (log1p). Ранг устойчивее нормированного скора:
# «лучший по BM25 из 800 кандидатов» значит одно и то же для любых запросов.
RANK_OF = [("bm25", False), ("char", False), ("hist", False), ("dense", False),
           ("log_dist", True), ("mc_prior", False), ("title_cov", False),
           # v6: «относительно соседей» — когда в пуле сотни похожих мастеров, выбирают
           # обычно того, у кого больше отзывов / выше рейтинг / цена ниже, чем у остальных
           ("log_reviews", False), ("rating", False), ("log_price", True), ("bm25f_title", False)]

# Признаки, построенные на истории выборов train. На бенчмарке у ~90% объявлений
# истории нет, поэтому держим вариант модели без них (lgbm_nohist) — страховка.
HIST_FEATURES = {"hist", "item_pop", "pop_loc", "r_hist", "n_src", "bm25f_qhist",
                 "ch_hist_all", "ch_hist_same", "ch_hist_near", "ch_locpop"}


def feature_cols(pool, exclude_hist=False):
    cols = [c for c in pool.columns
            if c in FEATURES or c in Q_FEATURES or c.startswith(("ch_", "r_", "d_"))]
    return [c for c in cols if not (exclude_hist and c in HIST_FEATURES)]


def add_derived(pool: pd.DataFrame, channels):
    """Признаки, которые считаются по всему пулу запроса (in-place)."""
    src = pool["src"].values
    for b, name in enumerate(channels):          # какой канал нашёл кандидата
        pool[f"ch_{name}"] = ((src >> b) & 1).astype(np.float32)
    g = pool.groupby("qid", sort=False)
    for c, asc in RANK_OF:
        pool[f"r_{c}"] = np.log1p(g[c].rank(method="min", ascending=asc).values - 1).astype(np.float32)
    pool["q_pool_size"] = np.log1p(g["ii"].transform("size").values).astype(np.float32)
    # отклонение от медианы пула: «дешевле/дороже, популярнее/менее популярен, чем типичный кандидат»
    for c in ("log_price", "log_reviews", "rating"):
        pool[f"d_{c}"] = (pool[c] - g[c].transform("median")).astype(np.float32)


# Каналы: (имя, источник скора, область, K). K — главные ручки полноты пула.
DEFAULT_CHANNELS = [
    # v6: bm25_all 100->300 (в v4 давал 1% уникального recall, кривая top100->top300 растёт),
    #     mc_near 200->300 (самый большой уникальный вклад), + mc_remote: треть промахов —
    #     удалённые услуги / дальше 100 км, а для них подкатегория работает и без слов запроса.
    ("bm25_all", "bm25", "all", 300),
    ("bm25_same", "bm25", "same", 250),
    ("bm25_near", "bm25", "near", 300),
    ("bm25_remote", "bm25", "remote", 50),
    ("char_all", "char", "all", 100),
    ("char_same", "char", "same", 200),
    ("char_near", "char", "near", 250),
    ("hist_all", "hist", "all", 100),
    ("hist_same", "hist", "same", 150),
    ("hist_near", "hist", "near", 250),
    ("mc_near", "mc", "near", 300),
    ("mc_remote", "mc", "remote", 100),
    ("locpop", "locpop", "near", 150),
    ("dense_all", "dense", "all", 100),
    ("dense_same", "dense", "same", 200),
    ("dense_near", "dense", "near", 250),
]


def _loc_int(x) -> int:
    return int(x) if pd.notna(x) else -1


class PoolBuilder:
    def __init__(self, items, it, retrievers: dict, hist, allowed_mask, cfg, hist_pairs, item_idx, loc_pairs=None):
        """hist_pairs — для признаков объявлений (популярность); loc_pairs — для статистик
        локаций (affinity, центры): они не зависят от того, «холодное» ли объявление."""
        self.items, self.it, self.retr, self.hist, self.cfg = items, it, retrievers, hist, cfg
        N = self.N = len(items)
        self.allowed = allowed_mask
        self.allowed_idx = None if allowed_mask.all() else np.flatnonzero(allowed_mask)
        self.channels = [c for c in cfg.get("channels", DEFAULT_CHANNELS)
                         if c[1] in ("mc", "locpop") or c[1] in retrievers or c[1] == "hist"]
        assert len(self.channels) <= 31

        # ---- статические признаки объявлений
        self.lat = items["item_latitude"].values.astype(np.float64)
        self.lon = items["item_longitude"].values.astype(np.float64)
        self.has_geo = ~np.isnan(self.lat) & ~np.isnan(self.lon)
        self.loc = items["item_location_id"].fillna(-1).values.astype(np.int64)
        self.remote = items["item_infm_params_text"].map(normalize).str.contains(_REMOTE_RE).values
        self.rating = items["item_rating"].fillna(0).values.astype(np.float32)
        self.log_rev = np.log1p(items["item_rating_reviews_count"].fillna(0).values).astype(np.float32)
        self.log_price = np.log1p(items["item_price"].fillna(0).clip(lower=0).values).astype(np.float32)
        self.mc_code = items["mc_code"].values
        self.in_bench = items["in_bench"].values

        # ---- индексы по локациям (только разрешённые объявления)
        self.loc_uni, self.loc_code = np.unique(self.loc, return_inverse=True)
        al = np.flatnonzero(allowed_mask)
        order = al[np.argsort(self.loc_code[al], kind="stable")]
        bounds = np.searchsorted(self.loc_code[order], np.arange(len(self.loc_uni) + 1))
        self.loc_items = {int(self.loc_uni[c]): order[bounds[c]:bounds[c + 1]] for c in range(len(self.loc_uni))}
        self.remote_idx = np.flatnonzero(self.remote & allowed_mask)
        self.centroids = location_centroids(items)

        # ---- статистики локаций из истории выборов
        loc_pairs = hist_pairs if loc_pairs is None else loc_pairs
        hp = loc_pairs.assign(ii=loc_pairs["item_id"].map(item_idx)).dropna(subset=["ii"])
        ii = hp["ii"].values.astype(np.int64)
        sloc = hp["search_location_id"].fillna(-1).values.astype(np.int64)
        # P(локация объявления | локация поиска)
        t = pd.DataFrame({"s": sloc, "l": self.loc_code[ii]}).value_counts().rename("c").reset_index()
        t["p"] = t["c"] / t.groupby("s")["c"].transform("sum")
        self.aff = {int(s): (g["l"].values, g["p"].values.astype(np.float32)) for s, g in t.groupby("s")}
        # запасной центр локации поиска: медиана координат выбранных оттуда объявлений
        ok = self.has_geo[ii]
        hc = pd.DataFrame({"s": sloc[ok], "la": self.lat[ii[ok]], "lo": self.lon[ii[ok]]}).groupby("s").median()
        self.hist_centroids = {int(s): (float(r.la), float(r.lo)) for s, r in hc.iterrows()}
        # ---- популярность объявлений (только «тёплые» объявления)
        hp = hist_pairs.assign(ii=hist_pairs["item_id"].map(item_idx)).dropna(subset=["ii"])
        ii = hp["ii"].values.astype(np.int64)
        sloc = hp["search_location_id"].fillna(-1).values.astype(np.int64)
        self.item_pop = np.log1p(np.bincount(ii, minlength=N)).astype(np.float32)
        # сколько раз объявление выбирали из данной локации поиска (строки = локации поиска)
        self.sloc_index = {int(s): i for i, s in enumerate(np.unique(sloc))}
        rows = np.array([self.sloc_index[int(s)] for s in sloc])
        LP = sp.csr_matrix((np.ones(len(ii), np.float32), (rows, ii)), shape=(len(self.sloc_index), N))
        LP.sum_duplicates()
        LP.data = np.log1p(LP.data)
        self.LP = LP

    # ------------------------------------------------------------------ scores for a batch
    def _batch(self, qb: pd.DataFrame):
        texts = qb["search_query"].tolist()
        S = {"bm25": self.retr["bm25"].score(texts), "char": self.retr["char"].score(texts)}
        S["hist"], mc_prior = self.hist.score(texts)
        S.update({f"bm25f_{k}": v for k, v in self.retr["bm25"].score_fields(texts).items()})
        if "dense" in self.retr:
            S["dense"] = self.retr["dense"].score(texts)
        qt = [tokenize(t) for t in texts]
        ft = [tokenize(t, drop_numbers=True) for t in qb["search_infm_params_text"]]
        V = self.it.vocab
        tcov = (V.to_csr(qt, binary=True) @ self.it.T_bin.T).toarray()
        tcov /= np.maximum([len(set(x)) for x in qt], 1)[:, None]
        fcov = (V.to_csr(ft, binary=True) @ self.it.P_bin.T).toarray()
        fcov /= np.maximum([len(set(x)) for x in ft], 1)[:, None]
        return S, mc_prior, tcov.astype(np.float32), fcov.astype(np.float32)

    def build(self, queries: pd.DataFrame, rel_lists=None, batch_size=48):
        """rel_lists (только валидация): индексы релевантных объявлений для каждого запроса —
        для них дополнительно считаются признаки и ранги, даже если они не попали в пул
        (строки с in_pool=0 идут только в диагностику, не в обучение и не в метрику)."""
        chunks, diags = [], []
        for s in range(0, len(queries), batch_size):
            qb = queries.iloc[s: s + batch_size]
            S, mc_prior, tcov, fcov = self._batch(qb)
            for j in range(len(qb)):
                q = qb.iloc[j]
                rel = None if rel_lists is None else np.asarray(rel_lists[s + j], np.int64)
                df, dg = self._one(q, {k: v[j] for k, v in S.items()}, mc_prior[j], tcov[j], fcov[j], rel)
                chunks.append(df)
                if dg is not None:
                    diags.append(dg)
            print(f"\r  pool: {min(s + batch_size, len(queries))}/{len(queries)}", end="", flush=True)
        print()
        pool = pd.concat(chunks, ignore_index=True)
        diag = pd.concat(diags, ignore_index=True) if diags else None
        return pool, diag

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _top(v, idx, k):
        """top-k положительных значений v; idx — какие позиции v соответствуют объявлениям."""
        if len(v) == 0 or k <= 0:
            return np.empty(0, np.int64)
        k = min(k, len(v))
        t = np.argpartition(-v, k - 1)[:k]
        t = t[v[t] > 0]
        return t if idx is None else idx[t]

    def _geo(self, loc_id):
        c = self.centroids.get(loc_id) or self.hist_centroids.get(loc_id)
        if c is None:
            return np.full(self.N, FAR_KM, np.float32), False
        d = haversine_km(c[0], c[1], np.nan_to_num(self.lat), np.nan_to_num(self.lon))
        d[~self.has_geo] = FAR_KM
        return d.astype(np.float32), True

    # ------------------------------------------------------------------ one query
    def _one(self, q, S, mc_prior, tcov, fcov, rel):
        cfg = self.cfg
        loc_id = _loc_int(q["search_location_id"])
        dist, geo_ok = self._geo(loc_id)
        aff = np.zeros(len(self.loc_uni), np.float32)
        if loc_id in self.aff:
            codes, p = self.aff[loc_id]
            aff[codes] = p
        aff = aff[self.loc_code]
        same = self.loc == loc_id
        near = (same | (aff >= cfg["aff_min"]) | (dist < cfg["local_radius_km"])) & self.allowed
        areas = {
            "all": self.allowed_idx,
            "same": self.loc_items.get(loc_id, np.empty(0, np.int64)),
            "near": np.flatnonzero(near),
            "remote": self.remote_idx,
        }
        # нормированные скоры (удобно и для признаков, и для смешанных каналов)
        Sn = {}
        for name, s in S.items():
            mx = 1.0 if name == "dense" else (S["bm25"].max() if name.startswith("bm25f_") else s.max())
            Sn[name] = (s / mx) if mx > 0 else np.zeros_like(s)
        mcs = mc_prior[self.mc_code].astype(np.float32)
        r = self.sloc_index.get(loc_id)
        pop_loc = self.LP[r].toarray().ravel() if r is not None else np.zeros(self.N, np.float32)

        # --- шаг A: каналы
        src = np.zeros(self.N, np.int32)
        for bit, (_, source, area, k) in enumerate(self.channels):
            idx = areas[area]
            if source == "mc":
                # подкатегория похожих запросов + немного текста + популярность
                sel = idx if idx is not None else np.arange(self.N)
                m = mcs[sel]
                v = np.where(m >= cfg["mc_min"], m + 0.3 * Sn["bm25"][sel] + 0.02 * self.item_pop[sel], 0)
                t = self._top(v, sel, k)
            elif source == "locpop":
                sel = pop_loc.nonzero()[0]
                sel = sel[self.allowed[sel]]
                t = self._top(pop_loc[sel] * (mcs[sel] + 0.01) + 0.1 * Sn["bm25"][sel], sel, k)
            else:
                s = Sn[source]
                t = self._top(s if idx is None else s[idx], idx, k)
            src[t] |= 1 << bit
        in_pool = src > 0
        idx = np.flatnonzero(in_pool)
        if rel is not None and len(rel):
            idx = np.union1d(idx, rel)
        if len(idx) == 0:
            return pd.DataFrame(columns=["qid", "ii", "src", "in_pool"] + FEATURES + Q_FEATURES[:-1]), None

        # --- признаки
        n = len(idx)
        f = {"qid": np.full(n, q["qid"], np.int32), "ii": idx.astype(np.int32),
             "src": src[idx], "in_pool": in_pool[idx].astype(np.int8)}
        for name in ("bm25", "char", "hist", "dense", "bm25f_title", "bm25f_params", "bm25f_desc", "bm25f_qhist"):
            f[name] = Sn[name][idx].astype(np.float32) if name in Sn else np.zeros(n, np.float32)
        f["same_loc"] = same[idx].astype(np.float32)
        f["geo_prox"] = np.exp(-dist[idx] / cfg["geo_scale_km"]).astype(np.float32)
        f["log_dist"] = np.log1p(dist[idx]).astype(np.float32)
        f["aff"] = aff[idx]
        f["remote"] = self.remote[idx].astype(np.float32)
        f["is_delivery"] = np.full(n, float(q["search_is_delivery_search"] or 0), np.float32)
        f["title_cov"] = tcov[idx]
        f["filter_cov"] = fcov[idx]
        m = _RATING_RE.search(normalize(q["search_infm_params_text"]))
        f["rating_ok"] = (self.rating[idx] >= int(m.group(1))).astype(np.float32) if m else np.ones(n, np.float32)
        f["mc_prior"] = mcs[idx]
        f["item_pop"] = self.item_pop[idx]
        f["pop_loc"] = pop_loc[idx].astype(np.float32)
        f["log_reviews"] = self.log_rev[idx]
        f["log_price"] = self.log_price[idx]
        f["rating"] = self.rating[idx]
        f["n_src"] = np.array([bin(x).count("1") for x in src[idx]], np.float32)
        f["q_ntok"] = np.full(n, len(tokenize(q["search_query"])), np.float32)
        f["q_bm25max"] = np.full(n, np.log1p(max(S["bm25"].max(), 0)), np.float32)
        f["q_n_same"] = np.full(n, np.log1p(len(areas["same"])), np.float32)
        f["q_n_near"] = np.full(n, np.log1p(len(areas["near"])), np.float32)
        df = pd.DataFrame(f)

        # --- диагностика: где в каждом ретривере находятся релевантные объявления
        diag = None
        if rel is not None and len(rel):
            d = df[df["ii"].isin(rel)].copy()
            d["in_bench"] = self.in_bench[d["ii"].values]
            d["geo_ok"] = geo_ok
            d["dist_km"] = dist[d["ii"].values]
            d["near"] = near[d["ii"].values]
            d["query"] = q["search_query"]
            d["filters"] = q["search_infm_params_text"]
            d["title"] = self.items["item_title_raw"].values[d["ii"].values]
            for name, s in S.items():
                if name.startswith("bm25f_"):
                    continue
                sa = s if self.allowed_idx is None else s[self.allowed_idx]
                sn = s[areas["near"]]
                sr = s[d["ii"].values]
                d[f"rank_{name}_all"] = [(sa > x).sum() if x > 0 else -1 for x in sr]
                d[f"rank_{name}_near"] = [(sn > x).sum() if x > 0 else -1 for x in sr]
            diag = d
        return df, diag
