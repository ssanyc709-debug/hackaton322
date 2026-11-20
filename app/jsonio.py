"""Строгий разбор JSON: без NaN/Infinity и без повторяющихся ключей.

Контракт (CONTRACT.md §2.1) прямо запрещает NaN, Infinity и повторяющиеся ключи.
Стандартный ``json.loads`` молча берёт последнее значение повторяющегося ключа
и принимает ``NaN``/``Infinity``, поэтому вход, каталог и кэш разбираются только
через этот модуль.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


class JsonError(ValueError):
    """Текст не является строгим JSON по требованиям контракта."""


#: Сколько кандидатов-скобок просматривать, прежде чем сдаться: защита от
#: текста, целиком состоящего из «{».
MAX_CANDIDATES = 32
#: Сколько символов текста вообще просматривать в поисках объекта.
MAX_SCAN_CHARS = 64_000


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


#: Декодер для поиска объекта внутри текста: те же запреты, что у ``loads_strict``.
_DECODER = json.JSONDecoder(
    parse_constant=_reject_constant,
    object_pairs_hook=_no_duplicate_keys,
)


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


def json_from(
    text: str,
    *,
    source: str = "json",
    accept: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """CORE. Достаёт первый объект JSON из текста с шумом вокруг (обёртки модели).

    Зачем: даже при ``response_format: json_schema`` модель может добавить
    обрамление — ```json … ```, «Вот результат: {…}», «{…} надеюсь помог».
    Строгий разбор такого ответа падает, а токены уже потрачены: запрос
    теряется целиком (exit 5). Здесь мы снимаем обёртку локально, без сети,
    без токенов и без ослабления требований к самому JSON.

    Вход:  text   — содержимое ``choices[0].message.content`` (может быть с шумом);
           source — откуда текст, для сообщения об ошибке;
           accept — необязательный предикат «этот объект пригоден»: если он
                    задан, перебираются все найденные объекты, и возвращается
                    первый пригодный (а если пригодных нет — первый найденный).

    Выход: первый разобранный объект JSON (``dict``), прошедший те же запреты,
    что и :func:`loads_strict` (нет NaN/Infinity, нет повторяющихся ключей).

    Алгоритм: идём по тексту в поисках ``{`` или ``[`` → ``JSONDecoder.raw_decode``
    с начала найденной позиции (он сам определяет конец объекта, поэтому хвост
    после JSON не мешает) → успех и это ``dict`` → проверить ``accept`` → вернуть.
    Ошибка разбора — сдвиг на один символ и следующая позиция (скобка могла быть
    в прозе или в мусоре); нарушения строгости (NaN, дубли ключей) — сразу ошибка,
    без перебора: молча взять «похожий» объект было бы угадыванием.
    Сложность: O(L · попытки) в худшем случае, где L — длина просмотренного
    текста; на практике одна попытка, потому что ``raw_decode`` съедает весь
    объект. Память O(глубина JSON).
    Почему именно он: ``json.loads`` требует, чтобы весь текст был JSON;
    резать по первой ``{`` и последней ``}`` нельзя — скобки бывают в прозе и
    внутри строк. ``raw_decode`` — единственный способ узнать настоящую границу
    объекта, не написав свой парсер.
    Крайние случаи: массив вместо объекта → ошибка (это не ответ помощника);
    текст целиком из скобок → ограничение числом попыток; вложенные скобки
    в строке (``{"note": "см. {пример}"}``) разбираются корректно; пустой текст
    и текст без скобок → ошибка; BOM и markdown-заборы игнорируются как шум;
    два объекта (``{"error":"x"}{…}``) → берётся пригодный, а не первый.
    Вызывается из: app/llm/client.py::_extraction_content, тесты.
    """
    limit = min(len(text), MAX_SCAN_CHARS)
    position = 0
    attempts = 0
    saw_array = False
    first_object: dict[str, Any] | None = None
    while position < limit and attempts < MAX_CANDIDATES:
        brace = text.find("{", position, limit)
        bracket = text.find("[", position, limit)
        starts = [index for index in (brace, bracket) if index >= 0]
        if not starts:
            break
        start = min(starts)
        attempts += 1
        try:
            value, _ = _DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            position = start + 1
            continue
        except JsonError as exc:
            raise JsonError(f"{source}: {exc}") from exc
        if isinstance(value, dict):
            if accept is None or accept(value):
                return value
            if first_object is None:
                first_object = value
            position = start + 1
            continue
        saw_array = saw_array or isinstance(value, list)
        position = start + 1
    if first_object is not None:
        return first_object
    if saw_array:
        raise JsonError(f"{source}: ответ не является объектом JSON (найден массив)")
    raise JsonError(f"{source}: в тексте нет объекта JSON")


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
