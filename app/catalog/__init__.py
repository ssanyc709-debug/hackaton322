"""Каталог объявлений: чтение документа, индекс и фильтры."""

from __future__ import annotations

from http.client import HTTPException
from pathlib import Path
from time import time_ns
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from app.config import MAX_CATALOG_LISTINGS
from app.jsonio import JsonError, load_json_file, loads_strict
from app.model import CatalogError

__all__ = ["CatalogError", "load_json_data", "parse_catalog_source", "read_catalog_document"]


def parse_catalog_source(value: str) -> Path | str:
    """Различает URL каталога и путь к локальному файлу."""
    if value.lower().startswith(("https://", "http://")):
        return value
    return Path(value).expanduser()


def load_json_data(source: Path | str) -> Any:
    """Читает текущий JSON с диска или по URL с уникальным параметром запроса."""
    if isinstance(source, Path):
        try:
            return load_json_file(source, source=f"каталог {source}")
        except JsonError as exc:
            raise CatalogError(str(exc)) from exc

    try:
        parts = urlsplit(source)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("ожидался URL HTTP(S) с именем сервера")
        refresh = f"t={time_ns()}"
        query = f"{parts.query}&{refresh}" if parts.query else refresh
        url = parts._replace(query=query, fragment="").geturl()
        request = Request(url, headers={"Accept": "application/json", "Cache-Control": "no-cache"})
        with urlopen(request, timeout=5) as response:
            payload = response.read()
        return loads_strict(payload.decode("utf-8-sig"), source="каталог по URL")
    except HTTPError as exc:
        exc.close()
        raise CatalogError(f"сервер каталога вернул HTTP {exc.code}") from exc
    except (OSError, HTTPException, ValueError) as exc:
        raise CatalogError(f"не удалось прочитать каталог по URL: {exc}") from exc


def read_catalog_document(source: Path | str) -> list[dict[str, object]]:
    """Читает ``catalog.json`` и проверяет верхний уровень документа.

    Зачем: путь передаётся извне, файл может отсутствовать, быть не-массивом
    или содержать больше объявлений, чем разрешено. Ошибка каталога —
    технический сбой (ненулевой exit), а не ``no_match``.

    Вход:  source — путь или URL JSON-массива объявлений.
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
    document = load_json_data(source)
    label = f"каталог {source}" if isinstance(source, Path) else "каталог по URL"

    if not isinstance(document, list):
        raise CatalogError(f"{label}: ожидался JSON-массив объявлений")
    if len(document) > MAX_CATALOG_LISTINGS:
        raise CatalogError(f"{label}: {len(document)} объявлений при лимите {MAX_CATALOG_LISTINGS}")
    return document
