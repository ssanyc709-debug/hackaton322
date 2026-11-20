"""Исходящий ответ: сборка, самопроверка и сериализация (CONTRACT.md §2.4).

Ответ формируется только через функции этого модуля, поэтому «лишние поля»
и неверные типы физически не могут попасть в stdout: перед печатью документ
проверяется структурно (stdlib) и, если установлен ``jsonschema``, ещё и по
``schemas/response.schema.json`` (Draft 2020-12).
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.config import MAX_OPTIONS, MAX_PRICE_RUB, MIN_PRICE_RUB, SCHEMA_DIR

STATUS_OK = "ok"
STATUS_CLARIFY = "clarify"
STATUS_NO_MATCH = "no_match"

QUESTION_BUDGET = "budget_rub"
QUESTION_CITY = "city_id"
QUESTION_CATEGORIES = "categories"
QUESTION_FIELDS: tuple[str, ...] = (QUESTION_BUDGET, QUESTION_CITY, QUESTION_CATEGORIES)
MAX_QUESTIONS = 3
MAX_LISTING_IDS_PER_OPTION = MAX_OPTIONS

QUESTION_TEXTS: dict[str, str] = {
    QUESTION_BUDGET: "Какой у вас бюджет на весь комплект?",
    QUESTION_CITY: "В каком городе нужен комплект: Таганрог, Ростов-на-Дону или Азов?",
    QUESTION_CATEGORIES: "Какие товары подобрать? Поддерживаются: стол, кресло, лампа.",
}

_RESPONSE_SCHEMA_PATH = SCHEMA_DIR / "response.schema.json"


class ResponseError(ValueError):
    """Собранный ответ не прошёл самопроверку (это баг, а не ошибка входа)."""


def clarify(fields: Iterable[str]) -> dict[str, Any]:
    """``clarify``: по одному вопросу на каждое недостающее основное условие.

    Порядок вопросов канонический (бюджет → город → категории), повторы полей
    отбрасываются: контракт требует спросить обо всех недостающих условиях ровно раз.
    """
    ordered = [field for field in QUESTION_FIELDS if field in set(fields)]
    if not ordered:
        raise ResponseError("clarify без вопросов невозможен")
    return {
        "status": STATUS_CLARIFY,
        "questions": [{"field": field, "text": QUESTION_TEXTS[field]} for field in ordered],
    }


def option(listing_ids: Sequence[str], total_rub: int, explanation: str) -> dict[str, Any]:
    """Один вариант комплекта: ровно по одному объявлению каждой категории."""
    return {
        "listing_ids": list(listing_ids),
        "total_rub": total_rub,
        "explanation": explanation,
    }


def ok(options: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """``ok``: 1..3 варианта комплектов."""
    return {"status": STATUS_OK, "options": list(options)}


def no_match(message: str) -> dict[str, Any]:
    """``no_match``: объяснение препятствия и что можно изменить (без ID)."""
    return {"status": STATUS_NO_MATCH, "message": message}


def validate(doc: Any) -> None:
    """CORE. Структурная самопроверка ответа без внешних зависимостей.

    Зачем: stdout — единственный канал контракта. Ответ с лишним полем,
    пустым ``explanation``, дублем ID внутри варианта или ``total_rub = 0``
    автопроверка отклонит целиком, поэтому такой ответ нельзя даже печатать.

    Вход:  doc — словарь ответа, собранный функциями этого модуля.
    Выход: None, если документ валиден; иначе ResponseError с указанием места.

    Алгоритм: диспетчеризация по ``status`` и проверка точных наборов ключей
    на каждом уровне вложенности (три независимые ветки).
    Сложность: время O(p + s), где p — число вариантов (<= 3), s — число ID
    в одном варианте (<= 3), то есть O(1) по факту; память O(1) на проверку.
    Почему именно он: ``isinstance(True, int)`` в Python истинно, а JSON-схема
    различает integer и boolean; здесь ``bool`` отсекается явно, чтобы ответ
    не проходил проверку из-за типа. Плюс проверка работает без установленного
    ``jsonschema`` — контракт не должен ломаться на чужой машине.
    Крайние случаи: ``True`` вместо цены, пустая строка в ``listing_ids``,
    одинаковые варианты, два одинаковых поля в ``questions``, дубли ID.
    Вызывается из: app/contract/response.py::finalize, тесты.
    """
    if not isinstance(doc, dict):
        raise ResponseError(f"ответ должен быть объектом, получено {type(doc).__name__}")

    status = doc.get("status")
    if status == STATUS_OK:
        _validate_ok(doc)
    elif status == STATUS_CLARIFY:
        _validate_clarify(doc)
    elif status == STATUS_NO_MATCH:
        _validate_no_match(doc)
    else:
        raise ResponseError(f"status должен быть одним из ok/clarify/no_match, получено {status!r}")


def validate_with_schema(doc: Any) -> None:
    """Дополнительная проверка по ``schemas/response.schema.json``.

    Если ``jsonschema`` не установлен, проверка пропускается: структурная
    :func:`validate` уже отсекает нарушения. Отсутствие пакета не является
    техническим сбоем (README ставит его заранее).
    """
    validator = _schema_validator()
    if validator is None:
        return
    errors = sorted(validator.iter_errors(doc), key=lambda error: list(error.absolute_path))
    if errors:
        details = "; ".join(
            f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in errors[:3]
        )
        raise ResponseError(f"ответ не соответствует схеме: {details}")


def dumps(doc: dict[str, Any]) -> str:
    """Каноническая однострочная сериализация: UTF-8, без NaN, без пробелов.

    ``allow_nan=False`` — требование контракта: NaN/Infinity в JSON недопустимы.
    ``ensure_ascii=False`` — кириллица остаётся читаемой и не раздувает байты.
    """
    try:
        return json.dumps(doc, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except ValueError as exc:  # NaN/Infinity или неподдерживаемый тип
        raise ResponseError(f"ответ не сериализуется в корректный JSON: {exc}") from exc


def finalize(doc: dict[str, Any]) -> str:
    """Самопроверка (структура + схема) и сериализация — единая точка выхода."""
    validate(doc)
    validate_with_schema(doc)
    return dumps(doc)


def _validate_ok(doc: dict[str, Any]) -> None:
    _require_keys(doc, {"status", "options"}, "ответ ok")
    options = doc["options"]
    if not isinstance(options, list):
        raise ResponseError("options должно быть массивом")
    if not 1 <= len(options) <= MAX_OPTIONS:
        raise ResponseError(f"options: ожидалось 1..{MAX_OPTIONS} вариантов, получено {len(options)}")

    seen_compositions: set[frozenset[str]] = set()
    for index, item in enumerate(options, start=1):
        if not isinstance(item, dict):
            raise ResponseError(f"вариант #{index}: ожидался объект")
        _require_keys(item, {"listing_ids", "total_rub", "explanation"}, f"вариант #{index}")

        listing_ids = item["listing_ids"]
        if not isinstance(listing_ids, list) or not listing_ids:
            raise ResponseError(f"вариант #{index}: listing_ids должен быть непустым массивом")
        if len(listing_ids) > MAX_LISTING_IDS_PER_OPTION:
            raise ResponseError(f"вариант #{index}: не более {MAX_LISTING_IDS_PER_OPTION} ID в комплекте")
        for listing_id in listing_ids:
            if not isinstance(listing_id, str) or not listing_id:
                raise ResponseError(f"вариант #{index}: ID объявления должен быть непустой строкой")
        if len(set(listing_ids)) != len(listing_ids):
            raise ResponseError(f"вариант #{index}: ID внутри комплекта должны быть уникальны")

        composition = frozenset(listing_ids)
        if composition in seen_compositions:
            raise ResponseError(f"вариант #{index}: дублирует состав ID другого варианта")
        seen_compositions.add(composition)

        total = item["total_rub"]
        if isinstance(total, bool) or not isinstance(total, int):
            raise ResponseError(f"вариант #{index}: total_rub должен быть целым числом")
        if not MIN_PRICE_RUB <= total <= MAX_PRICE_RUB:
            raise ResponseError(f"вариант #{index}: total_rub вне допустимого диапазона: {total}")

        explanation = item["explanation"]
        if not isinstance(explanation, str) or not explanation.strip():
            raise ResponseError(f"вариант #{index}: explanation должен быть непустой строкой")


def _validate_clarify(doc: dict[str, Any]) -> None:
    _require_keys(doc, {"status", "questions"}, "ответ clarify")
    questions = doc["questions"]
    if not isinstance(questions, list):
        raise ResponseError("questions должно быть массивом")
    if not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ResponseError(f"questions: ожидалось 1..{MAX_QUESTIONS} вопросов, получено {len(questions)}")

    seen_fields: set[str] = set()
    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ResponseError(f"вопрос #{index}: ожидался объект")
        _require_keys(item, {"field", "text"}, f"вопрос #{index}")
        field = item["field"]
        if field not in QUESTION_FIELDS:
            raise ResponseError(f"вопрос #{index}: недопустимое поле {field!r}")
        if field in seen_fields:
            raise ResponseError(f"вопрос #{index}: поле {field!r} спрошено повторно")
        seen_fields.add(field)
        text = item["text"]
        if not isinstance(text, str) or not text.strip():
            raise ResponseError(f"вопрос #{index}: text должен быть непустой строкой")


def _validate_no_match(doc: dict[str, Any]) -> None:
    _require_keys(doc, {"status", "message"}, "ответ no_match")
    message = doc["message"]
    if not isinstance(message, str) or not message.strip():
        raise ResponseError("message должен быть непустой строкой")
    if "listing_ids" in doc:
        raise ResponseError("no_match не должен содержать ID объявлений")


def _require_keys(container: dict[str, Any], expected: set[str], where: str) -> None:
    actual = set(container)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise ResponseError(f"{where}: неверный набор полей (нет {missing}, лишние {extra})")


@lru_cache(maxsize=1)
def _schema_validator() -> Any:
    """Ленивая загрузка валидатора Draft 2020-12; None, если пакета нет."""
    try:
        import jsonschema
    except ImportError:
        print(
            "[buyer-assistant] jsonschema не установлен: используется встроенная структурная проверка",
            file=sys.stderr,
            flush=True,
        )
        return None
    try:
        schema = json.loads(Path(_RESPONSE_SCHEMA_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResponseError(f"не удалось прочитать схему ответа {_RESPONSE_SCHEMA_PATH}: {exc}") from exc
    return jsonschema.Draft202012Validator(schema)
