"""
Нормализация и токенизация русского текста.

Идея: запросы короткие («автоподбор», «баня на дровах»), а объявления длинные.
Чтобы «бани»/«баня»/«баню» совпадали, приводим слова к основе стеммером Snowball
(open-source библиотека PyStemmer). Стемминг грубее лемматизации, но в ~50 раз
быстрее, а для кандидатогенерации (полнота важнее точности) этого достаточно.
"""
import re

import Stemmer  # PyStemmer, BSD license

_STEMMER = Stemmer.Stemmer("russian")
_TOKEN_RE = re.compile(r"[a-zа-я0-9]+")

# Короткий стоп-лист: предлоги/союзы, которые не несут смысла в запросах услуг.
STOPWORDS = {
    "и", "в", "во", "на", "с", "со", "по", "для", "от", "до", "из", "к", "ко", "за",
    "о", "об", "у", "не", "а", "но", "или", "же", "то", "это", "как", "что",
    "при", "под", "над", "без", "the", "a", "of",
}

# Кэш «слово -> основа»: уникальных слов на порядки меньше, чем словоупотреблений,
# поэтому стеммим каждое слово один раз.
_stem_cache: dict = {}


def normalize(s) -> str:
    if s is None or (isinstance(s, float) and s != s):  # None / NaN
        return ""
    return str(s).lower().replace("ё", "е")


def tokenize(s, drop_numbers: bool = False, max_chars: int | None = None) -> list:
    """Строка -> список основ слов (стоп-слова выброшены)."""
    s = normalize(s)
    if max_chars:
        s = s[:max_chars]
    out = []
    for w in _TOKEN_RE.findall(s):
        if w in STOPWORDS:
            continue
        if drop_numbers and w.isdigit():
            continue
        st = _stem_cache.get(w)
        if st is None:
            st = _STEMMER.stemWord(w) if not w.isdigit() else w
            _stem_cache[w] = st
        out.append(st)
    return out
