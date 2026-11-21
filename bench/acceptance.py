"""Репетиция автопроверки жюри по CONTRACT.md §1–§4 (независимый стенд).

Зачем этот файл
---------------
Жюри проверяет не «наши тесты», а соответствие ответа каталогу и условиям
запроса. Поэтому стенд устроен как независимый проверяльщик, а не как ещё один
набор утверждений о нашем же коде:

1. **Генератор каталога** строит валидный каталог по ``schemas/catalog.schema.json``:
   те же семь полей, те же типы, ``attributes`` с граничными значениями,
   пустыми объектами и «лишними» ключами (строка/число/boolean).
2. **Генератор запроса** строит реплику по §2.2/§3 и ЗНАЕТ свой замысел: город,
   бюджет, категории и ограничения. Формулировки — только те, что закреплены
   в §2.2: «до», «не более», «не шире», «уже», «шире», «от … до …».
3. **Oracle** — своя реализация §4, не связанная с ``app/catalog``: по каталогу
   и замыслу решает, должен ли быть ``ok`` или ``no_match``, и полным перебором
   доказывает существование/отсутствие комплекта.
4. **Запуск как у жюри**: отдельный процесс ``python -m app.main --catalog …``,
   stdin → stdout. Проверяется код 0, ровно один JSON в stdout и все инварианты §4.
5. **Диалоги** (§3): сначала реплика с недостающим основным условием, затем —
   история ``user → assistant (текст вопросов) → user (ответ покупателя)``.
   Уточнённое условие обязано «доехать» до подбора без потерь.

Модель не вызывается: ``ASSISTANT_OFFLINE=1``. Стоимость прогона — 0 токенов,
сеть не нужна.

Запуск::

    python bench/acceptance.py                  # боевой каталог + 2 синтетических
    python bench/acceptance.py --cases 40       # больше запросов на каталог
    python bench/acceptance.py --seed 7         # другой набор каталогов и запросов
    python bench/acceptance.py --dialogs 6      # больше диалогов «уточнение → ответ»
    python bench/acceptance.py --json           # отчёт в JSON
    python bench/acceptance.py --no-inputs      # без проверок §1 (битый вход)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CATALOG_PATH = REPO_ROOT / "data" / "catalog.json"
CATALOG_SCHEMA_PATH = REPO_ROOT / "schemas" / "catalog.schema.json"

#: Категории и города — ровно как в §2 и data/cities.json.
CATEGORIES: tuple[str, ...] = ("table", "chair", "lamp")
CITIES: tuple[str, ...] = ("taganrog", "rostov-na-donu", "azov")

CATEGORY_WORDS = {"table": "стол", "chair": "кресло", "lamp": "лампа"}
CITY_PHRASES = {
    "taganrog": "в Таганроге",
    "rostov-na-donu": "в Ростове-на-Дону",
    "azov": "в Азове",
}
ATTRIBUTE_OF = {
    "table": "width_cm",
    "chair": "adjustable_height",
    "lamp": "color_temperature_k",
}
QUESTION_FIELDS = ("budget_rub", "city_id", "categories")

#: Правдоподобные значения, если в каталоге у категории нет ни одного значения.
WIDTH_POOL = (1, 60, 70, 80, 90, 95, 100, 110, 120, 140, 160, 200)
TEMPERATURE_POOL = (1, 2700, 3000, 3500, 4000, 5000, 6500, 9000)
#: Граница неоднозначности «N к»: до 200 — это тысячи рублей («15 к» = 15 000 ₽),
#: выше — кельвины. Поэтому в ЗАПРОСАХ температура берётся реалистичной
#: (лампы ниже 2000 K не бывает), а в каталоге граничное значение 1 K остаётся:
#: это проверяет, что значение характеристики не путается с деньгами.
KELVIN_QUERY_MIN = 200
TEMPERATURE_QUERY_POOL = (2000, 2700, 3000, 3500, 4000, 5000, 6500, 9000)

VALID_REQUEST = json.dumps(
    {"messages": [{"role": "user", "text": "Стол в Таганроге до 15 000 ₽"}]},
    ensure_ascii=False,
).encode("utf-8")


# --- Модель условий и замысла ----------------------------------------------


@dataclass(frozen=True, slots=True)
class Bound:
    """Граница условия: значение и признак включительности (§2.2)."""

    value: int
    inclusive: bool = True


@dataclass(frozen=True, slots=True)
class NumericCondition:
    """Ограничение на числовую характеристику (ширина стола, температура лампы)."""

    lower: Bound | None = None
    upper: Bound | None = None


@dataclass(frozen=True, slots=True)
class FlagCondition:
    """Ограничение на boolean-характеристику (регулировка высоты кресла)."""

    value: bool


Condition = NumericCondition | FlagCondition


@dataclass(frozen=True, slots=True)
class Intent:
    """Что покупатель попросил — в структурном виде, независимо от текста.

    ``city_id=None`` означает «город неизвестен» (не назван или назван вне
    справочника). ``categories=()`` — «категории неизвестны».
    """

    text: str
    city_id: str | None
    budget_rub: int | None
    categories: tuple[str, ...]
    conditions: Mapping[str, Condition] = field(default_factory=dict)

    def missing_fields(self) -> tuple[str, ...]:
        """Недостающие основные условия ровно в терминах §4."""
        fields: list[str] = []
        if self.budget_rub is None:
            fields.append("budget_rub")
        if self.city_id is None:
            fields.append("city_id")
        if not self.categories:
            fields.append("categories")
        return tuple(fields)


# --- Oracle: независимая проверка §4 ---------------------------------------


def matches(listing: Mapping[str, Any], condition: Condition | None) -> bool:
    """Удовлетворяет ли объявление условию (``None`` — условия нет).

    Отсутствующий ключ — «неизвестное значение»: при заданном ограничении такое
    объявление не подходит (§2.2). ``False`` и ``0`` — валидные значения,
    поэтому ``bool`` не считается числом.
    """
    if condition is None:
        return True
    key = ATTRIBUTE_OF[str(listing["category"])]
    attributes = listing["attributes"]
    if key not in attributes:
        return False
    value = attributes[key]
    if isinstance(condition, FlagCondition):
        return isinstance(value, bool) and value == condition.value
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    lower, upper = condition.lower, condition.upper
    if lower is not None:
        if lower.inclusive and value < lower.value:
            return False
        if not lower.inclusive and value <= lower.value:
            return False
    if upper is not None:
        if upper.inclusive and value > upper.value:
            return False
        if not upper.inclusive and value >= upper.value:
            return False
    return True


def candidates_for(
    intent: Intent, catalog: Sequence[Mapping[str, Any]]
) -> dict[str, list[Mapping[str, Any]]]:
    """Подходящие объявления по каждой запрошенной категории (§2, §3)."""
    per_category: dict[str, list[Mapping[str, Any]]] = {}
    for category in intent.categories:
        condition = intent.conditions.get(category)
        per_category[category] = [
            listing
            for listing in catalog
            if listing["city_id"] == intent.city_id
            and listing["category"] == category
            and matches(listing, condition)
        ]
    return per_category


def complete_bundles(
    intent: Intent, catalog: Sequence[Mapping[str, Any]]
) -> list[tuple[Mapping[str, Any], ...]]:
    """Все полные комплекты в бюджете — грубым перебором, без нашего подбора."""
    if intent.budget_rub is None or not intent.categories:
        return []
    per_category = candidates_for(intent, catalog)
    if any(not items for items in per_category.values()):
        return []
    budget = int(intent.budget_rub)
    bundles: list[tuple[Mapping[str, Any], ...]] = []
    for bundle in itertools.product(*(per_category[category] for category in intent.categories)):
        if sum(int(listing["price_rub"]) for listing in bundle) <= budget:
            bundles.append(bundle)
    return bundles


def cheapest_bundle(intent: Intent, catalog: Sequence[Mapping[str, Any]]) -> int | None:
    """Минимально возможная сумма комплекта (без перебора) или ``None``."""
    if not intent.categories:
        return None
    per_category = candidates_for(intent, catalog)
    if any(not items for items in per_category.values()):
        return None
    return sum(
        min(int(listing["price_rub"]) for listing in per_category[category])
        for category in intent.categories
    )


def expected_status(intent: Intent, catalog: Sequence[Mapping[str, Any]]) -> str:
    """Ожидаемый статус по §4: сначала ``clarify``, потом ``ok`` / ``no_match``."""
    if intent.missing_fields():
        return "clarify"
    return "ok" if complete_bundles(intent, catalog) else "no_match"


# --- Проверка документа ответа ---------------------------------------------


def check_document(
    document: Any,
    intent: Intent,
    catalog: Sequence[Mapping[str, Any]],
    expected: str,
) -> list[str]:
    """Все инварианты §4 для одного ответа. Возвращает список нарушений."""
    problems: list[str] = []
    if not isinstance(document, dict):
        return [f"ответ не объект JSON: {type(document).__name__}"]

    status = document.get("status")
    if status not in ("ok", "clarify", "no_match"):
        return [f"недопустимый status: {status!r}"]

    expected_keys = {
        "ok": {"status", "options"},
        "clarify": {"status", "questions"},
        "no_match": {"status", "message"},
    }[status]
    if set(document) != expected_keys:
        problems.append(
            f"поля ответа {sorted(document)} вместо {sorted(expected_keys)}"
        )

    if status != expected:
        detail = ""
        if status == "clarify":
            detail = " (вопросы: " + ", ".join(
                str(question.get("field")) for question in document.get("questions", [])
            ) + ")"
        problems.append(f"ожидался status={expected!r}, получен {status!r}{detail}")

    if status == "clarify":
        problems.extend(_check_clarify(document, intent))
    elif status == "ok":
        problems.extend(_check_ok(document, intent, catalog))
    else:
        problems.extend(_check_no_match(document, intent, catalog))
    return problems


def _check_clarify(document: Mapping[str, Any], intent: Intent) -> list[str]:
    problems: list[str] = []
    questions = document.get("questions")
    if not isinstance(questions, list) or not questions:
        return ["questions должен быть непустым массивом"]
    if len(questions) > 3:
        problems.append(f"вопросов больше трёх: {len(questions)}")

    fields: list[str] = []
    for question in questions:
        if not isinstance(question, dict) or set(question) != {"field", "text"}:
            problems.append(f"вопрос неверной формы: {question!r}")
            continue
        name = question["field"]
        if name not in QUESTION_FIELDS:
            problems.append(f"недопустимое поле вопроса: {name!r}")
        if not isinstance(question["text"], str) or not question["text"].strip():
            problems.append("пустой текст вопроса")
        fields.append(str(name))

    if len(fields) != len(set(fields)):
        problems.append(f"поле спрошено повторно: {fields}")
    if set(fields) != set(intent.missing_fields()):
        problems.append(
            f"спрошены поля {sorted(set(fields))} вместо {sorted(intent.missing_fields())}"
        )
    return problems


def _check_ok(
    document: Mapping[str, Any],
    intent: Intent,
    catalog: Sequence[Mapping[str, Any]],
) -> list[str]:
    problems: list[str] = []
    options = document.get("options")
    if not isinstance(options, list) or not 1 <= len(options) <= 3:
        return [f"options: ожидалось 1..3 варианта, получено {options!r}"]

    by_id = {str(listing["id"]): listing for listing in catalog}
    seen: set[frozenset[str]] = set()

    for number, option in enumerate(options, start=1):
        if not isinstance(option, dict) or set(option) != {"listing_ids", "total_rub", "explanation"}:
            problems.append(f"вариант #{number}: неверная форма {option!r}")
            continue
        ids = option["listing_ids"]
        if not isinstance(ids, list) or not ids:
            problems.append(f"вариант #{number}: listing_ids пуст или не массив")
            continue
        if len(ids) > 3:
            problems.append(f"вариант #{number}: больше трёх ID в комплекте")
        if len(set(ids)) != len(ids):
            problems.append(f"вариант #{number}: повторяющиеся ID {ids}")

        composition = frozenset(str(item) for item in ids)
        if composition in seen:
            problems.append(f"вариант #{number}: состав повторяет другой вариант")
        seen.add(composition)

        missing = [str(item) for item in ids if str(item) not in by_id]
        if missing:
            problems.append(f"вариант #{number}: ID нет в каталоге {missing}")
            continue

        chosen = [by_id[str(item)] for item in ids]
        got_categories = sorted(str(listing["category"]) for listing in chosen)
        if got_categories != sorted(intent.categories):
            problems.append(
                f"вариант #{number}: категории {got_categories} вместо {sorted(intent.categories)}"
            )

        for listing in chosen:
            if listing["city_id"] != intent.city_id:
                problems.append(
                    f"вариант #{number}: {listing['id']} в городе {listing['city_id']!r}, "
                    f"а не {intent.city_id!r}"
                )
            condition = intent.conditions.get(str(listing["category"]))
            if not matches(listing, condition):
                problems.append(
                    f"вариант #{number}: {listing['id']} не удовлетворяет условию {condition!r}"
                )

        total = option["total_rub"]
        if isinstance(total, bool) or not isinstance(total, int):
            problems.append(f"вариант #{number}: total_rub не целое: {total!r}")
        else:
            actual = sum(int(listing["price_rub"]) for listing in chosen)
            if total != actual:
                problems.append(f"вариант #{number}: total_rub={total}, а сумма по каталогу {actual}")
            if intent.budget_rub is not None and total > int(intent.budget_rub):
                problems.append(
                    f"вариант #{number}: total_rub={total} больше бюджета {intent.budget_rub}"
                )

        explanation = option["explanation"]
        if not isinstance(explanation, str) or not explanation.strip():
            problems.append(f"вариант #{number}: пустое explanation")
    return problems


def _check_no_match(
    document: Mapping[str, Any],
    intent: Intent,
    catalog: Sequence[Mapping[str, Any]],
) -> list[str]:
    problems: list[str] = []
    message = document.get("message")
    if not isinstance(message, str) or not message.strip():
        problems.append("no_match: пустое message")
    if "listing_ids" in document:
        problems.append("no_match: содержит listing_ids")
    if complete_bundles(intent, catalog):
        problems.append("no_match, хотя допустимый комплект в каталоге существует")
    return problems


# --- Генератор каталога -----------------------------------------------------


def generate_catalog(
    rng: random.Random,
    size: int,
    cities: Sequence[str] = CITIES,
    categories: Sequence[str] = CATEGORIES,
) -> list[dict[str, Any]]:
    """Валидный каталог по §2: семь полей, типы, граничные характеристики."""
    listings: list[dict[str, Any]] = []
    for index in range(size):
        city_id = rng.choice(list(cities))
        category = rng.choice(list(categories))
        attributes: dict[str, Any] = {}

        if category == "table":
            if rng.random() < 0.85:
                attributes["width_cm"] = rng.choice(WIDTH_POOL)
        elif category == "chair":
            if rng.random() < 0.85:
                attributes["adjustable_height"] = rng.random() < 0.5
        else:
            if rng.random() < 0.85:
                attributes["color_temperature_k"] = rng.choice(TEMPERATURE_POOL)

        # §2: дополнительные ключи разрешены, значения — строка/число/boolean,
        # вложенных объектов, массивов и null нет.
        if rng.random() < 0.25:
            attributes["material"] = rng.choice(("дуб", "сосна", "металл", ""))
        if rng.random() < 0.20:
            attributes["warranty_months"] = rng.choice((0, 12, 36))
        if rng.random() < 0.20:
            attributes["folding"] = rng.random() < 0.5

        listings.append(
            {
                "id": f"gen-{index:04d}",
                "category": category,
                "title": f"{CATEGORY_WORDS[category].capitalize()} {index}",
                "description": rng.choice(("", f"Описание {index}")),
                "price_rub": rng.choice((rng.randint(1, 3_000), rng.randint(1, 60_000))),
                "city_id": city_id,
                "attributes": attributes,
            }
        )
    return listings


def catalog_schema_errors(catalog: Any) -> list[str]:
    """Проверка сгенерированного каталога по схеме (§2). Без jsonschema — пропуск."""
    try:
        import jsonschema
    except ImportError:
        return []
    schema = json.loads(CATALOG_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in validator.iter_errors(catalog)
    ]


# --- Генератор запросов -----------------------------------------------------


def _numeric_values(
    catalog: Sequence[Mapping[str, Any]], city_id: str | None, category: str
) -> list[int]:
    """Значения характеристики, реально присутствующие в каталоге для города."""
    key = ATTRIBUTE_OF[category]
    found: set[int] = set()
    for listing in catalog:
        if listing["city_id"] != city_id or listing["category"] != category:
            continue
        attributes = listing["attributes"]
        if key not in attributes:
            continue
        value = attributes[key]
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        found.add(int(value))
    return sorted(found)


def _generate_condition(
    rng: random.Random,
    catalog: Sequence[Mapping[str, Any]],
    city_id: str | None,
    category: str,
) -> Condition | None:
    """Случайное ограничение по §2.2 (иногда — его отсутствие)."""
    if rng.random() < 0.45:
        return None
    if category == "chair":
        return FlagCondition(rng.random() < 0.5)

    world = _numeric_values(catalog, city_id, category)
    if category == "lamp":
        world = [value for value in world if value > KELVIN_QUERY_MIN]
    pool = WIDTH_POOL if category == "table" else TEMPERATURE_QUERY_POOL
    values = world or list(pool)

    kinds = ["upper_inc", "lower_inc", "range", "exact"]
    if category == "table":
        kinds += ["upper_strict", "lower_strict"]
    else:
        kinds += ["lower_strict"]  # «выше N K» закреплено тестами, «ниже» — нет
    kind = rng.choice(kinds)

    if kind == "upper_inc":
        return NumericCondition(upper=Bound(rng.choice(values), True))
    if kind == "upper_strict":
        return NumericCondition(upper=Bound(rng.choice(values), False))
    if kind == "lower_inc":
        return NumericCondition(lower=Bound(rng.choice(values), True))
    if kind == "lower_strict":
        return NumericCondition(lower=Bound(rng.choice(values), False))
    if kind == "exact":
        value = rng.choice(values)
        return NumericCondition(lower=Bound(value, True), upper=Bound(value, True))

    if len(values) < 2:
        value = values[0]
        return NumericCondition(lower=Bound(value, True), upper=Bound(value, True))
    low, high = sorted(rng.sample(values, 2))
    return NumericCondition(lower=Bound(low, True), upper=Bound(high, True))


def render_condition(category: str, condition: Condition) -> str:
    """Формулировка ограничения обычным языком (§2.2)."""
    if isinstance(condition, FlagCondition):
        return "с регулировкой высоты" if condition.value else "без регулировки высоты"

    lower, upper = condition.lower, condition.upper
    unit = "см" if category == "table" else "K"
    noun = "шириной" if category == "table" else "с температурой"

    if lower is not None and upper is not None:
        if lower.value == upper.value and lower.inclusive and upper.inclusive:
            if category == "lamp":
                return f"{noun} {lower.value} {unit}"
            return f"{noun} ровно {lower.value} {unit}"
        return f"{noun} от {lower.value} до {upper.value} {unit}"

    if upper is not None:
        if category == "table":
            return f"не шире {upper.value} {unit}" if upper.inclusive else f"уже {upper.value} {unit}"
        return f"не выше {upper.value} {unit}"

    if lower is not None:
        if lower.inclusive:
            return f"от {lower.value} {unit}"
        return f"выше {lower.value} {unit}" if category == "lamp" else f"шире {lower.value} {unit}"

    return ""


def format_rub(value: int) -> str:
    """Сумма с разрядами: 15 000 ₽."""
    return f"{int(value):,}".replace(",", " ") + " ₽"


def join_clauses(clauses: Sequence[str]) -> str:
    """Перечисление: «a», «a и b», «a, b и c»."""
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]
    return ", ".join(clauses[:-1]) + " и " + clauses[-1]


def render_intent(
    city_id: str | None,
    budget_rub: int | None,
    categories: Sequence[str],
    conditions: Mapping[str, Condition],
    *,
    unknown_city: bool = False,
    unknown_category: bool = False,
) -> str:
    """Текст запроса из структурного замысла."""
    clauses: list[str] = []
    for category in categories:
        condition = conditions.get(category)
        word = CATEGORY_WORDS[category]
        clauses.append(word if condition is None else f"{word} {render_condition(category, condition)}")
    if unknown_category:
        clauses.append("диван")

    parts: list[str] = []
    body = join_clauses(clauses)
    if body:
        parts.append(body)
    if unknown_city:
        parts.append("в Москве")
    elif city_id is not None:
        parts.append(CITY_PHRASES[city_id])
    if budget_rub is not None:
        parts.append("до " + format_rub(budget_rub))

    text = " ".join(parts).strip()
    return text[:1].upper() + text[1:] if text else "Помогите"


def generate_intent(
    rng: random.Random,
    catalog: Sequence[Mapping[str, Any]],
    *,
    allow_incomplete: bool = True,
) -> Intent:
    """Случайный запрос по §3 вместе с его структурным замыслом."""
    category_count = rng.choices((1, 2, 3), weights=(3, 4, 5))[0]
    categories = tuple(rng.sample(list(CATEGORIES), category_count))
    city_id = rng.choice(list(CITIES))

    conditions: dict[str, Condition] = {}
    for category in categories:
        condition = _generate_condition(rng, catalog, city_id, category)
        if condition is not None:
            conditions[category] = condition

    probe = Intent(
        text="",
        city_id=city_id,
        budget_rub=None,
        categories=categories,
        conditions=conditions,
    )
    budget_rub = _generate_budget(rng, probe, catalog)

    if allow_incomplete:
        roll = rng.random()
        if roll < 0.06:
            city_id = None  # город не назван
        elif roll < 0.12:
            budget_rub = None  # бюджет не назван
        elif roll < 0.18:
            categories = ()  # категории не названы
            conditions = {}
        elif roll < 0.23:
            # Город вне справочника (§4: уточнить city_id).
            return Intent(
                text=render_intent(None, budget_rub, categories, conditions, unknown_city=True),
                city_id=None,
                budget_rub=budget_rub,
                categories=categories,
                conditions=conditions,
            )
        elif roll < 0.28:
            # Неподдерживаемая категория (§4: уточнить categories).
            return Intent(
                text=render_intent(city_id, budget_rub, categories, conditions, unknown_category=True),
                city_id=city_id,
                budget_rub=budget_rub,
                categories=(),
                conditions={},
            )

    return Intent(
        text=render_intent(city_id, budget_rub, categories, conditions),
        city_id=city_id,
        budget_rub=budget_rub,
        categories=categories,
        conditions=conditions,
    )


def _generate_budget(
    rng: random.Random,
    intent: Intent,
    catalog: Sequence[Mapping[str, Any]],
) -> int:
    """Бюджет: часто — точная граница минимального комплекта (§3: равенство допустимо)."""
    cheapest = cheapest_bundle(intent, catalog)
    if cheapest is None or rng.random() < 0.2:
        return rng.choice((rng.randint(1, 2_000), rng.randint(1, 120_000), 1_000))

    return rng.choice(
        (
            cheapest,                          # ровно хватает
            max(1, cheapest - 1),              # на рубль меньше → no_match
            cheapest + rng.randint(0, 3_000),  # запас
            rng.randint(1, max(2, cheapest + 5_000)),
        )
    )


# --- Диалог: clarify → ответ покупателя → комплект (§3) --------------------


@dataclass(frozen=True, slots=True)
class Dialogue:
    """Сценарий уточнения: что покупатель сказал сначала, что ответил, что вышло.

    ``first`` — замысел первой реплики (с недостающим основным условием),
    ``answer`` — текст ответа покупателя, ``final`` — замысел после ответа.
    """

    name: str
    first: Intent
    answer: str
    final: Intent


def make_dialogue(rng: random.Random, catalog: Sequence[Mapping[str, Any]], missing: str) -> Dialogue:
    """Собирает сценарий «не хватает одного основного условия» (§4, §3)."""
    category_count = rng.choices((1, 2, 3), weights=(3, 4, 5))[0]
    categories = tuple(rng.sample(list(CATEGORIES), category_count))
    city_id = rng.choice(list(CITIES))

    conditions: dict[str, Condition] = {}
    for category in categories:
        condition = _generate_condition(rng, catalog, city_id, category)
        if condition is not None:
            conditions[category] = condition

    probe = Intent(
        text="",
        city_id=city_id,
        budget_rub=None,
        categories=categories,
        conditions=conditions,
    )
    budget_rub = _generate_budget(rng, probe, catalog)

    if missing == "budget_rub":
        first = Intent(
            text=render_intent(city_id, None, categories, conditions),
            city_id=city_id,
            budget_rub=None,
            categories=categories,
            conditions=conditions,
        )
        answer = "До " + format_rub(budget_rub)
    elif missing == "city_id":
        first = Intent(
            text=render_intent(None, budget_rub, categories, conditions),
            city_id=None,
            budget_rub=budget_rub,
            categories=categories,
            conditions=conditions,
        )
        answer = CITY_PHRASES[city_id]
    else:
        # Категории не названы: первая реплика — «подберите что-нибудь» с городом
        # и бюджетом. Бюджет и город обязаны пережить уточнение (§3), а условия
        # на характеристики в ответе покупателя не звучали — их быть не должно.
        conditions = {}
        first = Intent(
            text=f"Помогите подобрать {CITY_PHRASES[city_id]} до {format_rub(budget_rub)}",
            city_id=city_id,
            budget_rub=budget_rub,
            categories=(),
            conditions={},
        )
        answer = join_clauses([CATEGORY_WORDS[category] for category in categories])

    final = Intent(
        text=first.text + " " + answer,
        city_id=city_id,
        budget_rub=budget_rub,
        categories=categories,
        conditions=conditions,
    )
    return Dialogue(name=f"уточнение {missing}", first=first, answer=answer, final=final)


def _questions_text(document: Any) -> str:
    """Текст вопросов помощника — так проверяющий добавляет реплику assistant."""
    if not isinstance(document, dict):
        return ""
    questions = document.get("questions")
    if not isinstance(questions, list):
        return ""
    return " ".join(
        str(question.get("text", "")) for question in questions if isinstance(question, dict)
    ).strip()


# --- Запуск решения как у жюри ---------------------------------------------


def cli_env(cache_dir: Path) -> dict[str, str]:
    """Окружение прогона: офлайн (0 токенов), UTF-8, отдельный кэш."""
    env = dict(os.environ)
    env.update(
        {
            "ASSISTANT_OFFLINE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": str(REPO_ROOT),
            "ASSISTANT_CACHE_DIR": str(cache_dir),
        }
    )
    return env


@dataclass
class CaseOutcome:
    """Результат одного запроса: что ожидали, что получили, чем не сошлось."""

    label: str
    text: str
    expected: str
    actual: str | None
    code: int
    problems: list[str]
    elapsed_ms: float


def ask(
    catalog_path: Path,
    messages: Sequence[Mapping[str, str]],
    env: Mapping[str, str],
    timeout: float,
) -> tuple[int | None, Any, list[str], float]:
    """Общается с решением как проверяющий: ``messages`` → код, документ, нарушения.

    Проверяет только процессный контракт §1 (код 0, ровно один JSON в stdout);
    смысл ответа проверяет вызывающий код по §4.
    """
    payload = json.dumps({"messages": list(messages)}, ensure_ascii=False).encode("utf-8")
    problems: list[str] = []

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "app.main", "--catalog", str(catalog_path)],
            input=payload,
            cwd=str(REPO_ROOT),
            capture_output=True,
            timeout=timeout,
            env=dict(env),
        )
    except subprocess.TimeoutExpired:
        return None, None, [f"таймаут {timeout:.0f} с"], 0.0
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if completed.returncode != 0:
        tail = completed.stderr.decode("utf-8", "replace").strip().splitlines()[-1:]
        problems.append(f"код выхода {completed.returncode}, ожидался 0; stderr: {tail}")

    try:
        stdout = completed.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        stdout = ""
        problems.append(f"stdout не UTF-8: {exc}")

    document: Any = None
    if not stdout.strip():
        problems.append("stdout пуст")
    else:
        try:
            document = json.loads(stdout)
        except json.JSONDecodeError as exc:
            problems.append(f"stdout не один JSON-объект: {exc}")

    if document is not None and not isinstance(document, dict):
        problems.append(f"ответ не объект JSON: {type(document).__name__}")

    return completed.returncode, document, problems, elapsed_ms


def run_case(
    label: str,
    catalog_path: Path,
    catalog: Sequence[Mapping[str, Any]],
    intent: Intent,
    env: Mapping[str, str],
    timeout: float,
) -> CaseOutcome:
    """Запускает один запрос отдельным процессом и проверяет ответ по §4."""
    expected = expected_status(intent, catalog)
    code, document, problems, elapsed_ms = ask(
        catalog_path, [{"role": "user", "text": intent.text}], env, timeout
    )
    actual: str | None = None
    if isinstance(document, dict):
        actual = document.get("status")
        problems.extend(check_document(document, intent, catalog, expected))
    return CaseOutcome(
        label,
        intent.text,
        expected,
        actual,
        -1 if code is None else code,
        problems,
        elapsed_ms,
    )


def check_dialogue(
    catalog_path: Path,
    catalog: Sequence[Mapping[str, Any]],
    dialogue: Dialogue,
    env: Mapping[str, str],
    timeout: float,
) -> dict[str, Any]:
    """§3: уточнение → ответ покупателя в истории → комплект или ``no_match``.

    Проверяющий после ``clarify`` добавляет текст вопросов как реплику
    ``assistant`` и ответ покупателя как ``user``. Стенд повторяет этот сценарий
    буквально и требует, чтобы уточнённое условие ничего не потеряло.
    """
    problems: list[str] = []

    code, first, issues, _ = ask(
        catalog_path, [{"role": "user", "text": dialogue.first.text}], env, timeout
    )
    problems.extend(issues)
    if code is not None and code != 0:
        problems.append(f"первая реплика: код {code}")
    if not isinstance(first, dict):
        problems.extend(["первая реплика: нет JSON-ответа"])
        return {"name": dialogue.name, "text": dialogue.first.text, "ok": False, "detail": "; ".join(problems)}

    missing = list(dialogue.first.missing_fields())
    fields = [
        str(question.get("field"))
        for question in first.get("questions", [])
        if isinstance(question, dict)
    ]
    if first.get("status") != "clarify":
        problems.append(f"первая реплика: status={first.get('status')!r} вместо 'clarify'")
    elif sorted(fields) != sorted(missing):
        problems.append(f"первая реплика: спрошены {sorted(fields)} вместо {sorted(missing)}")

    assistant_text = _questions_text(first) or "Уточните, пожалуйста."
    history = [
        {"role": "user", "text": dialogue.first.text},
        {"role": "assistant", "text": assistant_text},
        {"role": "user", "text": dialogue.answer},
    ]
    code, second, issues, _ = ask(catalog_path, history, env, timeout)
    problems.extend(issues)
    if code is not None and code != 0:
        problems.append(f"после ответа: код {code}")
    if isinstance(second, dict):
        problems.extend(
            check_document(second, dialogue.final, catalog, expected_status(dialogue.final, catalog))
        )

    return {
        "name": dialogue.name,
        "text": f"{dialogue.first.text} → «{dialogue.answer}»",
        "ok": not problems,
        "detail": "; ".join(problems),
    }


def check_technical_inputs(
    folder: Path, env: Mapping[str, str], timeout: float
) -> list[dict[str, Any]]:
    """§1 и §2: технический сбой — ненулевой код, пустой stdout, без трассировки."""

    def write_catalog(name: str, **patch: Any) -> Path:
        """Каталог из одного объявления с подменённым полем — для битых случаев."""
        listing: dict[str, Any] = {
            "id": "ok-1",
            "category": "table",
            "title": "Стол",
            "description": "",
            "price_rub": 1_000,
            "city_id": "taganrog",
            "attributes": {},
        }
        listing.update(patch)
        path = folder / f"{name}.json"
        path.write_text(json.dumps([listing], ensure_ascii=False), encoding="utf-8")
        return path

    valid_catalog = write_catalog("valid-catalog")

    utf16_catalog = folder / "utf16-catalog.json"
    utf16_catalog.write_bytes(
        json.dumps(
            [
                {
                    "id": "u-1",
                    "category": "table",
                    "title": "Стол",
                    "description": "",
                    "price_rub": 1_000,
                    "city_id": "taganrog",
                    "attributes": {},
                }
            ],
            ensure_ascii=False,
        ).encode("utf-16")
    )

    not_array_catalog = folder / "not-array.json"
    not_array_catalog.write_text('{"listings": []}', encoding="utf-8")

    empty_file = folder / "empty-file.json"
    empty_file.write_bytes(b"")

    cases: list[tuple[str, bytes, Path]] = [
        # §1: сбой на входе.
        ("пустой stdin", b"", valid_catalog),
        ("битый JSON на входе", b'{"messages": [}', valid_catalog),
        ("на входе массив", b"[]", valid_catalog),
        ("NaN на входе", b'{"messages": [{"role": "user", "text": "x", "n": NaN}]}', valid_catalog),
        # §1/§2: сбой каталога.
        ("каталог не найден", VALID_REQUEST, folder / "missing-catalog.json"),
        ("каталог не массив", VALID_REQUEST, not_array_catalog),
        ("пустой файл каталога", VALID_REQUEST, empty_file),
        ("каталог не в UTF-8", VALID_REQUEST, utf16_catalog),
        ("глубокая вложенность", b"[" * 50_000, valid_catalog),
        # §2: типы характеристик — тоже часть контракта каталога.
        ("ширина строкой", VALID_REQUEST, write_catalog("width-str", attributes={"width_cm": "90"})),
        ("ширина 0", VALID_REQUEST, write_catalog("width-zero", attributes={"width_cm": 0})),
        ("ширина boolean", VALID_REQUEST, write_catalog("width-bool", attributes={"width_cm": True})),
        (
            "температура строкой",
            VALID_REQUEST,
            write_catalog("temp-str", category="lamp", attributes={"color_temperature_k": "4000"}),
        ),
        (
            "регулировка числом",
            VALID_REQUEST,
            write_catalog("flag-int", category="chair", attributes={"adjustable_height": 1}),
        ),
        (
            "характеристика объектом",
            VALID_REQUEST,
            write_catalog("attr-object", attributes={"extra": {"cm": 90}}),
        ),
        ("характеристика null", VALID_REQUEST, write_catalog("attr-null", attributes={"material": None})),
    ]

    results: list[dict[str, Any]] = []
    for name, payload, catalog_path in cases:
        problems: list[str] = []
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "app.main", "--catalog", str(catalog_path)],
                input=payload,
                cwd=str(REPO_ROOT),
                capture_output=True,
                timeout=timeout,
                env=dict(env),
            )
        except subprocess.TimeoutExpired:
            results.append({"name": name, "ok": False, "detail": "таймаут"})
            continue

        stderr = completed.stderr.decode("utf-8", "replace")
        if completed.returncode == 0:
            problems.append("код выхода 0, ожидался ненулевой")
        if completed.stdout.strip():
            problems.append(f"stdout не пуст: {completed.stdout[:80]!r}")
        if "Traceback" in stderr:
            problems.append("в stderr трассировка исключения вместо сообщения об ошибке")

        results.append(
            {
                "name": name,
                "ok": not problems,
                "detail": "; ".join(problems) if problems else f"код {completed.returncode}",
            }
        )
    return results


def check_determinism(
    label: str,
    catalog_path: Path,
    intent: Intent,
    env: Mapping[str, str],
    timeout: float,
) -> dict[str, Any]:
    """§1: тот же вход обязан давать побайтово тот же stdout."""
    payload = json.dumps(
        {"messages": [{"role": "user", "text": intent.text}]}, ensure_ascii=False
    ).encode("utf-8")
    command = [sys.executable, "-m", "app.main", "--catalog", str(catalog_path)]
    first = subprocess.run(
        command, input=payload, cwd=str(REPO_ROOT), capture_output=True, timeout=timeout, env=dict(env)
    )
    second = subprocess.run(
        command, input=payload, cwd=str(REPO_ROOT), capture_output=True, timeout=timeout, env=dict(env)
    )
    same = first.returncode == second.returncode and first.stdout == second.stdout
    return {
        "label": label,
        "text": intent.text,
        "ok": same,
        "detail": "" if same else f"код {first.returncode}/{second.returncode}, stdout отличается",
    }


# --- Полный прогон ----------------------------------------------------------


#: Поля, которых может не хватать: порядок задаёт сценарии диалогового прогона.
DIALOG_FIELDS: tuple[str, ...] = ("budget_rub", "city_id", "categories")


def run_acceptance(
    case_count: int = 20,
    seed: int = 2026_11_21,
    catalog_path: Path = CATALOG_PATH,
    *,
    include_inputs: bool = True,
    synthetic_catalogs: int = 2,
    dialog_count: int = 3,
    quiet: bool = False,
    timeout: float = 150.0,
) -> dict[str, Any]:
    """Полный прогон стенда. Возвращает отчёт; ``report["ok"]`` — итог."""
    started = time.perf_counter()
    rng = random.Random(seed)
    real_catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))

    outcomes: list[CaseOutcome] = []
    technical: list[dict[str, Any]] = []
    determinism: list[dict[str, Any]] = []
    dialogs: list[dict[str, Any]] = []
    schema_problems: list[str] = []

    with tempfile.TemporaryDirectory(prefix="accept-") as folder:
        root = Path(folder)
        env = cli_env(root / "cache")

        catalogs: list[tuple[str, Path, Sequence[Mapping[str, Any]]]] = [
            ("боевой каталог", Path(catalog_path), real_catalog)
        ]
        for number in range(max(0, synthetic_catalogs)):
            synthetic = generate_catalog(rng, rng.randint(18, 40))
            problems = catalog_schema_errors(synthetic)
            if problems:
                schema_problems.extend(f"каталог #{number + 1}: {problem}" for problem in problems)
            path = root / f"catalog-{number + 1}.json"
            path.write_text(json.dumps(synthetic, ensure_ascii=False), encoding="utf-8")
            catalogs.append((f"синтетический #{number + 1}", path, synthetic))

        cases: list[tuple[str, Path, Sequence[Mapping[str, Any]], Intent]] = []
        for label, path, catalog in catalogs:
            for _ in range(case_count):
                cases.append((label, path, catalog, generate_intent(rng, catalog)))

        for label, path, catalog, intent in cases:
            outcomes.append(run_case(label, path, catalog, intent, env, timeout))

        for number in range(max(0, dialog_count)):
            missing = DIALOG_FIELDS[number % len(DIALOG_FIELDS)]
            label, path, catalog = catalogs[number % len(catalogs)]
            dialogue = make_dialogue(rng, catalog, missing)
            item = check_dialogue(path, catalog, dialogue, env, timeout)
            item["label"] = label
            dialogs.append(item)

        if include_inputs:
            technical = check_technical_inputs(root, env, timeout)

        if cases:
            determinism.append(
                check_determinism(cases[0][0], cases[0][1], cases[0][3], env, timeout)
            )
            determinism.append(
                check_determinism(cases[-1][0], cases[-1][1], cases[-1][3], env, timeout)
            )

    failures = [outcome for outcome in outcomes if outcome.problems]
    by_status: dict[str, int] = {}
    for outcome in outcomes:
        by_status[outcome.expected] = by_status.get(outcome.expected, 0) + 1

    technical_failures = [item for item in technical if not item["ok"]]
    determinism_failures = [item for item in determinism if not item["ok"]]
    dialog_failures = [item for item in dialogs if not item["ok"]]
    elapsed = time.perf_counter() - started

    report: dict[str, Any] = {
        "seed": seed,
        "cases": len(outcomes),
        "elapsed_seconds": elapsed,
        "by_status": by_status,
        "schema_problems": schema_problems,
        "failures": [
            {
                "label": outcome.label,
                "text": outcome.text,
                "expected": outcome.expected,
                "actual": outcome.actual,
                "code": outcome.code,
                "problems": outcome.problems,
            }
            for outcome in failures
        ],
        "technical": technical,
        "determinism": determinism,
        "dialogs": dialogs,
        "ok": (
            not failures
            and not technical_failures
            and not determinism_failures
            and not dialog_failures
            and not schema_problems
        ),
    }

    if not quiet:
        print_report(report)
    return report


def print_report(report: Mapping[str, Any]) -> None:
    """Человекочитаемый отчёт: что проверено, что не сошлось."""
    print("=" * 78)
    print("Приёмка по CONTRACT §1–§4: независимая репетиция автопроверки")
    print("=" * 78)
    print(
        f"запросов: {report['cases']}, "
        f"по ожидаемому статусу: {report['by_status']}, "
        f"время: {report['elapsed_seconds']:.1f} с"
    )
    if report["schema_problems"]:
        print("каталоги не соответствуют схеме §2:")
        for problem in report["schema_problems"]:
            print(f"  ! {problem}")

    if report["failures"]:
        print(f"нарушения §4: {len(report['failures'])}")
        for failure in report["failures"]:
            print(f"  [{failure['label']}] {failure['text']!r}")
            print(
                f"    ожидалось {failure['expected']!r}, получено {failure['actual']!r}, "
                f"код {failure['code']}"
            )
            for problem in failure["problems"]:
                print(f"    - {problem}")
    else:
        print("нарушений §4 нет: все ответы соответствуют каталогу и запросу")

    if report["technical"]:
        failed = [item for item in report["technical"] if not item["ok"]]
        print(f"технические входы §1: {len(report['technical'])}, нарушений: {len(failed)}")
        for item in report["technical"]:
            mark = "ok" if item["ok"] else "! "
            print(f"  [{mark}] {item['name']}: {item['detail']}")

    if report["determinism"]:
        print("детерминизм §1:")
        for item in report["determinism"]:
            mark = "ok" if item["ok"] else "! "
            print(f"  [{mark}] {item['label']}: {item['text']!r} {item['detail']}")

    if report["dialogs"]:
        failed = [item for item in report["dialogs"] if not item["ok"]]
        print(f"диалоги §3 (clarify → ответ покупателя): {len(report['dialogs'])}, нарушений: {len(failed)}")
        for item in report["dialogs"]:
            mark = "ok" if item["ok"] else "! "
            print(f"  [{mark}] [{item['label']}] {item['name']}: {item['text']}")
            if item["detail"]:
                print(f"      {item['detail']}")

    print("-" * 78)
    print("ИТОГ:", "всё сходится" if report["ok"] else "есть расхождения")


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа стенда: печатает отчёт и возвращает 0/1."""
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--cases", type=int, default=20, help="запросов на каждый каталог")
    parser.add_argument("--seed", type=int, default=2026_11_21, help="зерно генератора")
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH, help="боевой каталог §2")
    parser.add_argument("--synthetic", type=int, default=2, help="сколько синтетических каталогов")
    parser.add_argument("--dialogs", type=int, default=3, help="сколько диалогов «уточнение → ответ»")
    parser.add_argument("--no-inputs", action="store_true", help="без проверок технического входа §1")
    parser.add_argument("--json", action="store_true", help="отчёт в JSON")
    args = parser.parse_args(argv)

    if args.cases < 1:
        parser.error("--cases должен быть >= 1")

    report = run_acceptance(
        case_count=args.cases,
        seed=args.seed,
        catalog_path=args.catalog,
        include_inputs=not args.no_inputs,
        synthetic_catalogs=args.synthetic,
        dialog_count=args.dialogs,
        quiet=args.json,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
