"""Строгий разбор JSON: без NaN/Infinity и без повторяющихся ключей.

Контракт (CONTRACT.md §2.1) прямо запрещает NaN, Infinity и повторяющиеся ключи.
Стандартный ``json.loads`` молча берёт последнее значение повторяющегося ключа
и принимает ``NaN``/``Infinity``, поэтому вход, каталог и кэш разбираются только
через этот модуль.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonError(ValueError):
    """Текст не является строгим JSON по требованиям контракта."""


def loads_strict(text: str, *, source: str = "json") -> Any:
    """Разбирает JSON-текст, отклоняя NaN/Infinity и дубли ключей.

    Сложность: O(len(text)) — единственный проход парсера; память O(depth) на стек.
    """
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_no_duplicate_keys,
        )
    except JsonError:
        raise
    except json.JSONDecodeError as exc:
        raise JsonError(f"{source}: некорректный JSON (строка {exc.lineno}, колонка {exc.colno}): {exc.msg}") from exc


def load_json_file(path: Path, *, source: str | None = None) -> Any:
    """Читает UTF-8 файл и разбирает его как строгий JSON."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise JsonError(f"{source or path}: не удалось прочитать файл: {exc}") from exc
    return loads_strict(text, source=source or str(path))


def _reject_constant(name: str) -> Any:
    """Хук ``parse_constant``: NaN/Infinity/-Infinity в JSON недопустимы."""
    raise JsonError(f"JSON содержит запрещённое значение {name}")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Хук ``object_pairs_hook``: последний дубль ключа — уже неоднозначность."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise JsonError(f"JSON содержит повторяющийся ключ {key!r}")
        result[key] = value
    return result
