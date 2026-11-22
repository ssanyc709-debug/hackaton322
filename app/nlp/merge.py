"""Слияние слотов по истории диалога (CONTRACT.md §2.3).

Правило контракта: учитывать ВСЮ историю; явное изменение условия ЗАМЕНЯЕТ
прежнее значение этого условия, остальные сохраняются; не спрашивать повторно
то, что уже известно. Именно поэтому слияние — отдельный слой с одной чистой
функцией и набором явных правил, а не «склеивание строк».
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from app.model import (
    FIELD_ADJUSTABLE,
    FIELD_BUDGET,
    FIELD_CATEGORIES,
    FIELD_CITY,
    FIELD_TEMPERATURE,
    FIELD_WIDTH,
    Slots,
)
from app.nlp.cache import LexiconCache
from app.nlp.extract import (
    MAX_MODEL_TEXT_BYTES,
    Resolver,
    extract_message,
    has_product_context,
    model_can_help,
    resolve_categories,
)
from app.nlp.lexicon import order_categories
from app.nlp.product_intent import describes_product, requests_product

#: Разделитель реплик в одном тексте для модели: модель видит диалог одним куском.
MODEL_TEXT_SEPARATOR = " / "
_SENTENCE_BREAK = re.compile(r"(?<=[.!?;])\s+|[\r\n]+")
_CLAUSE_BOUNDARY = re.compile(r"[.;!?]\s*|,\s*|\s+(?:и|а|но)\s+", re.I)
_EXPLICIT_REPLACEMENT = re.compile(r"\b(?:только|вместо|теперь|передумал(?:а)?)\b", re.I)


@dataclass(frozen=True)
class PendingProduct:
    position: int
    text: str
    excluded: tuple[str, ...]


def merge_history(
    user_texts: Sequence[str],
    *,
    cache: LexiconCache | None = None,
    resolver: Resolver | None = None,
    allow_model: bool = True,
) -> Slots:
    """CORE. Собирает итоговые условия из всех реплик покупателя.

    Зачем: помощник ведёт диалог, а не отвечает на одну строку. «Таганрог» может
    прийти в первом сообщении, бюджет — во втором, а «давай в Азове» — в третьем.
    Без явного правила слияния легко получить «город из первой реплики навсегда»
    или потерю ограничения, о котором покупатель не повторял.

    Вход:  user_texts — тексты реплик покупателя в хронологическом порядке;
           cache/resolver/allow_model — параметры разбора (см. ``extract_message``).
    Выход: :class:`Slots` — накопленные условия; ``mentioned`` — объединение
    полей, названных в любой реплике.

    Алгоритм: левая свёртка (fold) по репликам. Для каждой реплики извлекаются
    слоты, затем :func:`merge_step` применяет только ЯВНО названные поля:
    город заменяется, бюджет заменяется, категории заменяются либо объединяются
    при слове «ещё/также/добавь», отказанные категории удаляются, ограничения
    по характеристикам заменяются или снимаются явной командой. Неназванные
    поля сохраняются.
    Модель подключается один раз на запрос, если товар надо выяснить по
    репликам покупателя. Новая просьба об описанном товаре обновляет эту
    часть диалога. Отдельные уточнения сохраняют уже разобранные условия.
    Сложность: время O(сумма длин реплик) + O(1) на слияние шага; память O(1).
    Почему именно он: свёртка даёт ровно семантику «последнее явное упоминание
    побеждает, остальное живёт», её легко протестировать по одному правилу на тест
    и невозможно «потерять» условие при длинном диалоге (до 20 реплик).
    Крайние случаи: пустая история → пустые слоты; реплика без условий («спасибо»)
    ничего не меняет; отказ от единственной категории → категории пусты → clarify;
    повторное называние того же города идемпотентно; город вне справочника
    заменяет прежний корректный город (и требует уточнения, а не подстановки);
    история длиннее предела модели в байтах → в модель уходит последняя реплика.
    Вызывается из: app/pipeline.py::answer, тесты, bench/load.py.
    """
    state = Slots()
    product_start = 0
    active_resolver = resolver if allow_model else None
    pending: PendingProduct | None = None
    for position, text in enumerate(user_texts):
        # Локальный разбор каждой реплики: модель здесь не нужна, решение о вызове
        # принимается ниже — по СОСТОЯНИЮ ПОСЛЕ всей истории, а не по одной реплике.
        step = extract_message(text, cache=cache, resolver=None, allow_model=False)
        description = _trailing_product_description(text, bool(step.categories_removed)) if step.categories_explicit else None
        if (pending is not None and description is None and step.categories
                and not step.categories_removed and not step.categories_unknown
                and not _EXPLICIT_REPLACEMENT.search(text) and not requests_product(text)):
            step = replace(step, categories_additive=True)
        state = merge_step(state, step)
        if step.categories_explicit:
            product_start = position + 1
            pending = PendingProduct(position, description, step.categories_removed) if description else None
        elif active_resolver is not None:
            replacement = _EXPLICIT_REPLACEMENT.search(text) and not step.categories_additive
            new_request = describes_product(text) or (
                step.mentioned <= {FIELD_CITY, FIELD_BUDGET} and requests_product(text)
            )
            if not new_request or not has_product_context(text, step):
                continue
            if step.categories_additive and state.categories and pending is None:
                pending = PendingProduct(position, text, ())
            elif replacement or (pending is None and not model_can_help(state)):
                product_start = position
                pending = None
                state = replace(state, categories=(), categories_unknown=(), categories_explicit=False)
    if pending is not None:
        if active_resolver is None:
            return replace(state, categories_uncertain=True)
        descriptions = (pending.text, *user_texts[pending.position + 1:])
        resolved = resolve_missing(replace(state, categories=(), categories_unknown=()), descriptions, active_resolver)
        extra = set(resolved.categories) - set(pending.excluded) - set(state.categories)
        return replace(state, categories=order_categories((*state.categories, *extra)),
                       categories_uncertain=not extra, source="model" if extra else state.source)
    if active_resolver is not None:
        state = resolve_missing(state, user_texts[product_start:], active_resolver)
    return state


def _trailing_product_description(text: str, after_removal: bool) -> str | None:
    """Находит просьбу о ещё одном товаре после явного названия или отказа."""
    for boundary in reversed(list(_CLAUSE_BOUNDARY.finditer(text))):
        description = text[boundary.end():].strip()
        independent = after_removal or boundary.group().strip().casefold() == "и"
        if not (describes_product(description) or (independent and requests_product(description))):
            continue
        parsed = extract_message(description)
        if not parsed.categories_explicit and has_product_context(description, parsed):
            return description
    return None


def dialogue_text(user_texts: Sequence[str]) -> str:
    """Текст диалога одним куском для единственного вызова модели.

    Склеивает реплики покупателя разделителем ``" / "``, чтобы модель видела
    диалог целиком («Стол и кресло в Таганроге» + «12 тысяч» = полный набор).
    Если суммарный текст не влезает в предел входа модели, берётся последняя
    реплика, которая влезает: лучше один осмысленный вызов, чем обрезанный
    посередине фрагмент. Возвращает пустую строку, если реплик нет.
    """
    joined = MODEL_TEXT_SEPARATOR.join(text.strip() for text in user_texts if text.strip())
    if len(joined.encode("utf-8")) <= MAX_MODEL_TEXT_BYTES:
        return joined
    for text in reversed(user_texts):
        candidate = text.strip()
        if candidate and len(candidate.encode("utf-8")) <= MAX_MODEL_TEXT_BYTES:
            return candidate
    return ""


def resolve_missing(state: Slots, user_texts: Sequence[str], resolver: Resolver) -> Slots:
    """CORE. Извлекает недостающие категории по содержательным фрагментам истории.

    Фрагменты с описанием товара передаются в хронологическом порядке
    одним запросом. Город и бюджет можно уточнить после разбора описания.
    """
    if not model_can_help(state):
        return state
    product_texts = [
        part.strip()
        for turn in user_texts
        for part in _SENTENCE_BREAK.split(turn)
        if has_product_context(part)
    ]
    return resolve_categories(state, dialogue_text(product_texts), resolver)


def merge_step(state: Slots, step: Slots) -> Slots:
    """Применяет слоты одной реплики к накопленному состоянию.

    Заменяются только поля из ``step.mentioned``; всё остальное остаётся прежним.
    Отдельно обрабатываются категории: они заменяются целиком (явное «нужен стол»
    после «стол, кресло и лампа» означает новый набор), объединяются при
    «ещё/также/добавь» и удаляются при «без лампы»/«стол не нужен».
    Ограничения по характеристикам заменяются новым значением или снимаются
    при явной команде («расширим ширину», «убери регулировку»).
    """
    mentioned = set(state.mentioned) | set(step.mentioned)
    city_id = state.city_id
    city_unknown = state.city_unknown
    budget = state.budget_rub
    categories = state.categories
    categories_unknown = state.categories_unknown
    width = state.width
    temperature = state.color_temperature
    adjustable = state.adjustable_height

    if FIELD_CITY in step.mentioned:
        city_id = step.city_id
        # Явно названный город вне справочника отменяет прежний корректный город.
        city_unknown = step.city_unknown if step.city_id is None else None

    if FIELD_BUDGET in step.mentioned and step.budget_rub is not None:
        budget = step.budget_rub

    if step.categories_explicit:
        removed = set(step.categories_removed)
        positive = [category for category in step.categories if category not in removed]
        if positive and not step.categories_additive:
            # Явный новый набор: «нужен только стол» заменяет прежние категории.
            categories = order_categories(positive)
        else:
            # Только отказ («без лампы») или добавление («ещё лампа»): меняем прежний набор.
            base = set(state.categories) | set(positive)
            categories = order_categories(base)
        categories = tuple(category for category in categories if category not in removed)
        if step.categories_additive:
            categories_unknown = tuple(dict.fromkeys(state.categories_unknown + step.categories_unknown))
        else:
            categories_unknown = step.categories_unknown
        mentioned.add(FIELD_CATEGORIES)

    if FIELD_WIDTH in step.mentioned:
        width = step.width
    if FIELD_TEMPERATURE in step.mentioned:
        temperature = step.color_temperature
    if FIELD_ADJUSTABLE in step.mentioned:
        adjustable = step.adjustable_height

    # Явное снятие ограничения («расширим ширину», «убери регулировку»):
    # обнуляем поле, даже если в этой реплике ему не дали нового значения.
    for field in step.fields_removed:
        if field == FIELD_WIDTH:
            width = None
        elif field == FIELD_TEMPERATURE:
            temperature = None
        elif field == FIELD_ADJUSTABLE:
            adjustable = None
        mentioned.add(field)

    return replace(
        state,
        city_id=city_id,
        city_unknown=city_unknown,
        budget_rub=budget,
        categories=categories,
        categories_unknown=categories_unknown,
        width=width,
        color_temperature=temperature,
        adjustable_height=adjustable,
        categories_removed=step.categories_removed,
        categories_explicit=state.categories_explicit or step.categories_explicit,
        categories_additive=step.categories_additive,
        mentioned=frozenset(mentioned),
        source=step.source if step.mentioned else state.source,
    )


def history_fields(texts: Iterable[str]) -> frozenset[str]:
    """Какие поля упоминались в истории (диагностика без побочных эффектов).

    Используется только в логах и тестах, поэтому не кэшируется.
    """
    fields: set[str] = set()
    for text in texts:
        fields |= extract_message(text).mentioned
    return frozenset(fields)
