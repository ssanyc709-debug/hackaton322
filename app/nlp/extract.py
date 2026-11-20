"""Извлечение условий из реплики: lexicon-first, модель — только fallback.

Порядок (§5):

1. кэш разборов — попадание даёт 0 токенов;
2. локальный разбор: лексикон + числа + границы + флаги (0 токенов);
3. если обязательные условия всё ещё не закрыты И передан resolver (клиент модели),
   делается ОДИН компактный вызов модели; локальные результаты им не перетираются,
   кроме полей, которые локально не определены;
4. удачный разбор пишется в кэш.

Модель не нужна и не вызывается, если ``city_id``, ``budget_rub`` и ``categories``
определены однозначно — это и есть целевые ≥ 60 % запросов без модели.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable

from app.model import (
    FIELD_ADJUSTABLE,
    FIELD_BUDGET,
    FIELD_CATEGORIES,
    FIELD_CITY,
    FIELD_TEMPERATURE,
    FIELD_WIDTH,
    NumericRange,
    Slots,
)
from app.nlp.cache import LexiconCache
from app.nlp.lexicon import get_lexicon, order_categories
from app.nlp.numbers import (
    collect_constraints,
    find_amounts,
    normalize_text,
    to_bounds,
    tokenize,
)

#: Верхняя граница «мелких» чисел: меньше этого числа не считаются бюджетом.
MIN_BUDGET_GUESS = 1000
#: Нижняя/верхняя границы правдоподобной цветовой температуры, K.
MIN_KELVIN = 1500
MAX_KELVIN = 9000
#: Верхняя граница правдоподобной ширины стола, см (защита от «100 метров»).
MAX_WIDTH_CM = 400
#: Максимальная длина реплики, которую имеет смысл отдавать в модель.
MAX_MODEL_TEXT_BYTES = 4_000

#: Регулярные выражения для булевой характеристики кресла.
ADJUSTABLE_TRUE_RE = re.compile(
    r"(с|есть|нужн\w*|хочу)\s+(регулировк|подъем|подъём)|регулируем\w*|"
    r"регулировк\w*\s+высот|изменяем\w*\s+высот|с\s+подъемн"
)
ADJUSTABLE_FALSE_RE = re.compile(
    r"без\s+регулировк|не\s+регулир\w*|фиксированн\w*\s+высот|"
    r"без\s+подъем|жестк\w*\s+высот"
)

#: Основы названий категорий для поиска отказа от товара.
CATEGORY_STEM_TO_ID: dict[str, str] = {"стол": "table", "кресл": "chair", "ламп": "lamp"}
_STEMS = "|".join(CATEGORY_STEM_TO_ID)

#: Отказ от категории: «без лампы», «стол не нужен», «убери кресло», «кроме стола».
REMOVAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"(?:без|убери|уберите|исключи|исключите|кроме)\s+(?:мне\s+)?({_STEMS})"),
    re.compile(rf"(?:не\s+надо|не\s+нужн\w*|не\s+хочу|не\s+интересн\w*)\s+(?:мне\s+)?({_STEMS})"),
    re.compile(rf"({_STEMS})\w*\s+(?:не\s+(?:надо|нужн\w*|хочу|интересн\w*)|убери|уберите)"),
)

#: «Добавь ещё», а не «замени всё»: при таких словах категории объединяются с прежними.
ADDITIVE_PATTERN = re.compile(r"\b(?:еще|также|плюс|вдобавок|добавь|добавьте|дополнительно)\b")

#: Тип resolver-а: реплика + уже известные слоты → частичный результат модели.
Resolver = Callable[[str, Slots], "dict[str, object] | None"]


def extract_message(
    text: str,
    *,
    cache: LexiconCache | None = None,
    resolver: Resolver | None = None,
    allow_model: bool = True,
) -> Slots:
    """CORE. Извлекает условия из одной реплики, минимизируя вызовы модели.

    Зачем: это точка, где решается бюджет токенов. Всё, что можно понять
    локально (город, категории, числа, границы, флаг регулировки), разбирается
    здесь бесплатно; модель подключается только для того, что лексикон не покрыл.

    Вход:  text        — реплика покупателя (сырая);
           cache       — кэш разборов (повтор = 0 токенов);
           resolver    — функция вызова модели (появляется в блоке клиента);
           allow_model — разрешены ли исходящие вызовы (офлайн-режим, тесты).

    Выход: :class:`Slots` с заполненными условиями и множеством ``mentioned``
    (какие поля были названы именно в этой реплике).

    Алгоритм: нормализация → токенизация → лексикон (город, категории,
    неподдерживаемые слова) → числа и границы → классификация «нейтральных»
    чисел по составу категорий → флаг регулировки регулярным выражением →
    (опционально) один вызов модели для незакрытых полей → запись в кэш.
    Сложность: время O(len(text)) на нормализацию и проход по токенам;
    память O(числа токенов). Модель — единственная сетевая операция.
    Почему именно он: lexicon-first даёт 0 токенов на большинстве реплик
    (цель ≥ 60 %), а fallback изолирован в одном месте и не влияет на ядро.
    Крайние случаи: пустая реплика (все поля пустые); город вне справочника →
    ``city_unknown``; предмет вне таксономии → ``categories_unknown``;
    «2 кресла» → число не становится бюджетом; ответ «15 тысяч» на уточнение
    без города и категорий даёт только бюджет (остальное подтянет история).
    Вызывается из: app/nlp/merge.py, app/pipeline.py, тесты, bench/load.py.
    """
    normalized = normalize_text(text)
    if cache is not None:
        cached = cache.get(text)
        if cached is not None:
            return cached

    slots = _lexicon_slots(normalized)
    if resolver is not None and allow_model and not is_complete(slots):
        slots = _apply_resolver(slots, text, resolver)

    if cache is not None:
        cache.put(text, slots)
    return slots


def _lexicon_slots(normalized: str) -> Slots:
    """Локальный разбор нормализованной реплики (без сети и без модели)."""
    lexicon = get_lexicon()
    tokens = tokenize(normalized)

    city_hits = lexicon.scan_cities(tokens)
    foreign_city_hits = lexicon.scan_unsupported_cities(tokens)
    category_hits = lexicon.scan_categories(tokens)
    unsupported_hits = lexicon.scan_unsupported_categories(tokens)

    # Явное изменение города внутри реплики: последнее упоминание побеждает.
    city_id = city_hits[-1].value if city_hits else None
    city_unknown = None
    if city_id is None and foreign_city_hits:
        city_unknown = foreign_city_hits[-1].value

    categories = order_categories(hit.value for hit in category_hits)
    categories_unknown = tuple(dict.fromkeys(hit.value for hit in unsupported_hits))
    categories_removed = _detect_removals(normalized)
    categories_additive = bool(ADDITIVE_PATTERN.search(normalized))

    constraints = collect_constraints(find_amounts(normalized))
    width = constraints.width
    temperature = constraints.temperature
    budget = constraints.budget_rub

    if width is not None and not _plausible_width(width):
        width = None  # явно не ширина стола (например, «100 метров»)

    # Числа без единицы: контекст категорий решает, деньги это или кельвины.
    # Приоритет у денег: бюджет — обязательное условие, а температура без «К»
    # почти всегда означает пару «от…до» вокруг лампы.
    for group in constraints.neutral_groups:
        values = [amount.value for amount in group]
        if (
            len(group) >= 2
            and "lamp" in categories
            and temperature is None
            and all(MIN_KELVIN <= value <= MAX_KELVIN for value in values)
        ):
            temperature = to_bounds(group)
        elif max(values) >= MIN_BUDGET_GUESS and budget is None:
            budget = max(values)

    adjustable = _detect_adjustable(normalized)

    mentioned: set[str] = set()
    if city_id is not None or city_unknown is not None:
        mentioned.add(FIELD_CITY)
    if budget is not None:
        mentioned.add(FIELD_BUDGET)
    if categories:
        mentioned.add(FIELD_CATEGORIES)
    if width is not None:
        mentioned.add(FIELD_WIDTH)
    if temperature is not None:
        mentioned.add(FIELD_TEMPERATURE)
    if adjustable is not None:
        mentioned.add(FIELD_ADJUSTABLE)

    return Slots(
        city_id=city_id,
        budget_rub=budget,
        categories=categories,
        width=width,
        color_temperature=temperature,
        adjustable_height=adjustable,
        city_unknown=city_unknown,
        categories_unknown=categories_unknown,
        categories_removed=categories_removed,
        categories_explicit=bool(category_hits) or bool(categories_removed) or bool(categories_unknown),
        categories_additive=categories_additive,
        mentioned=frozenset(mentioned),
        source="lexicon",
    )


def _detect_removals(normalized: str) -> tuple[str, ...]:
    """Находит категории, от которых покупатель отказался в этой реплике.

    «Без лампы», «стол не нужен», «убери кресло», «кроме стола» — это явное
    изменение условия: при слиянии истории такая категория удаляется из набора.
    Отрицание рядом с числовым ограничением («стол не шире метра») сюда не
    попадает: шаблоны требуют связки «отказ + название категории».
    """
    removed: list[str] = []
    for pattern in REMOVAL_PATTERNS:
        for match in pattern.finditer(normalized):
            category = CATEGORY_STEM_TO_ID.get(match.group(1))
            if category is not None and category not in removed:
                removed.append(category)
    return tuple(removed)


def _plausible_width(width: NumericRange) -> bool:
    """Правдоподобна ли ширина стола: 1..400 см."""
    for bound in (width.lower, width.upper):
        if bound is None:
            continue
        if bound.value <= 0 or bound.value > MAX_WIDTH_CM:
            return False
    return True


def _detect_adjustable(normalized: str) -> bool | None:
    """Определяет требование к регулировке высоты кресла.

    ``False`` и ``True`` — равноправные значения: «без регулировки» даёт ``False``,
    и это не должно теряться при проверке условий (никакой truthiness).
    """
    if ADJUSTABLE_FALSE_RE.search(normalized):
        return False
    if ADJUSTABLE_TRUE_RE.search(normalized):
        return True
    return None


def _apply_resolver(slots: Slots, text: str, resolver: Resolver) -> Slots:
    """Дополняет слоты результатом одного вызова модели, не перетирая локальное.

    Исключения клиента модели (:class:`app.llm.client.LlmError`) намеренно
    НЕ глушатся: сбой внешнего сервиса — это техническая ошибка запуска
    (stderr + ненулевой код, CONTRACT.md §1), а не повод выдать ``no_match``
    или «уточните заново». Пустой ответ модели (``None``/``{}``) — не ошибка:
    она просто ничего не нашла, и условия остаются пустыми.
    """
    if len(text.encode("utf-8")) > MAX_MODEL_TEXT_BYTES:
        return slots
    raw = resolver(text, slots)
    if not raw:
        return slots
    return merge_model_slots(slots, raw)


def merge_model_slots(slots: Slots, raw: dict[str, object]) -> Slots:
    """Дополняет слоты полями модели; локально найденные значения не меняются.

    Модель — вспомогательный источник: если она «увидела» город, которого нет
    в справочнике, значение всё равно проверяется по ``CITY_WORDS``. Это часть
    защиты от манипуляций: текст объявления и ответ модели не могут переписать
    правила, только дополнить недостающие условия.
    """
    city_id = slots.city_id
    mentioned = set(slots.mentioned)
    model_city = raw.get("city_id")
    if city_id is None and isinstance(model_city, str):
        from app.nlp.lexicon import CITY_WORDS

        if model_city in CITY_WORDS:
            city_id = model_city
            mentioned.add(FIELD_CITY)

    budget = slots.budget_rub
    model_budget = raw.get("budget_rub")
    if budget is None and isinstance(model_budget, int) and not isinstance(model_budget, bool) and model_budget > 0:
        budget = model_budget
        mentioned.add(FIELD_BUDGET)

    categories = slots.categories
    categories_explicit = slots.categories_explicit
    model_categories = raw.get("categories")
    if not categories and isinstance(model_categories, list):
        valid = [item for item in model_categories if isinstance(item, str) and item in ("table", "chair", "lamp")]
        if valid:
            categories = order_categories(valid)
            categories_explicit = True
            mentioned.add(FIELD_CATEGORIES)

    return replace(
        slots,
        city_id=city_id,
        budget_rub=budget,
        categories=categories,
        categories_explicit=categories_explicit,
        mentioned=frozenset(mentioned),
        source="model",
    )


def is_complete(slots: Slots) -> bool:
    """Все обязательные условия известны: город, бюджет, категории.

    Используется для решения «звать ли модель»: если True — вызова нет.
    """
    return bool(slots.city_id and slots.budget_rub and slots.categories)


def missing_fields(slots: Slots) -> tuple[str, ...]:
    """Недостающие основные условия в каноническом порядке (для ``clarify``).

    Если назван город вне справочника — ``city_id`` тоже считается недостающим;
    если названы неподдерживаемые предметы — ``categories``. Уже известное
    не спрашивается повторно: поле попадает сюда только при пустом значении.
    """
    missing: list[str] = []
    if slots.city_id is None:
        missing.append(FIELD_CITY)
    if slots.budget_rub is None:
        missing.append(FIELD_BUDGET)
    if not slots.categories or slots.categories_unknown:
        missing.append(FIELD_CATEGORIES)
    return tuple(missing)
