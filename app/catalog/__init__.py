"""Каталог объявлений: чтение документа, индекс и фильтры.

Каталог приходит ТОЛЬКО через ``--catalog`` при каждом запуске: путь не
хардкодится, индекс строится от содержимого файла (см. ``app.catalog.index``).
"""

from __future__ import annotations

from pathlib import Path

from app.config import MAX_CATALOG_LISTINGS
from app.jsonio import JsonError, load_json_file
from app.model import CatalogError

__all__ = ["CatalogError", "read_catalog_document"]


def read_catalog_document(path: Path) -> list[dict[str, object]]:
    """Читает ``catalog.json`` и проверяет верхний уровень документа.

    Зачем: путь передаётся извне, файл может отсутствовать, быть не-массивом
    или содержать больше объявлений, чем разрешено. Ошибка каталога —
    технический сбой (ненулевой exit), а не ``no_match``.

    Вход:  path — путь к JSON-массиву объявлений.
    Выход: список словарей; при проблеме — :class:`CatalogError`.

    Алгоритм: строгий разбор JSON (без NaN и дублей ключей) + проверка контейнера.
    Сложность: время O(n) по числу объявлений (n <= 250), память O(n).
    Почему именно он: типы всех семи полей проверяет
    ``app.catalog.index.parse_listing`` — здесь достаточно быстро отсечь
    «файл не тот», не дублируя валидацию объявления.
    Крайние случаи: пустой массив допустим; пустой файл, объект вместо массива
    и более 250 записей вызывают ошибку каталога.
    Вызывается из: app/main.py::main.
    """
    try:
        document = load_json_file(path, source=f"каталог {path}")
    except JsonError as exc:
        raise CatalogError(str(exc)) from exc

    if not isinstance(document, list):
        raise CatalogError(f"каталог {path}: ожидался JSON-массив объявлений")
    if len(document) > MAX_CATALOG_LISTINGS:
        raise CatalogError(f"каталог {path}: {len(document)} объявлений при лимите {MAX_CATALOG_LISTINGS}")
    return document
