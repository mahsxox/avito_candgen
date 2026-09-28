"""
Тексты для dense-модели (e5). Одна функция и для дообучения (train_dense.py),
и для эмбеддингов корпуса (embed.py): текст объявления должен выглядеть одинаково
при обучении и при поиске, иначе дообучение теряет смысл.

e5 обучена с префиксами «query: » / «passage: » — их добавляем при кодировании.
"""
PASSAGE_DESC_CHARS = 300  # заголовок + начало описания: основной смысл объявления


def passage(title, desc) -> str:
    d = "" if desc is None or desc != desc else str(desc)[:PASSAGE_DESC_CHARS]
    return f"{title}. {d}".strip()
