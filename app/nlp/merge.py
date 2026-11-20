"""Слияние слотов по истории диалога (CONTRACT.md §2.3).

Правило контракта: учитывать ВСЮ историю; явное изменение условия ЗАМЕНЯЕТ
прежнее значение этого условия, остальные сохраняются; не спрашивать повторно
то, что уже известно. Именно поэтому слияние — отдельный слой с одной чистой
функцией и набором явных правил, а не «склеивание строк».
"""

from __future__ import annotations

from dataclasses import replace
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
    merge_model_slots,
    model_can_help,
)
from app.nlp.lexicon import order_categories

#: Разделитель реплик в одном тексте для модели: модель видит диалог одним куском.
MODEL_TEXT_SEPARATOR = " / "


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
    по характеристикам заменяются. Неназванные поля сохраняются.
    Модель подключается ОДИН раз на запрос и только если после слияния всей
    истории обязательные условия всё ещё не закрыты (§5) И модель физически
    способна их закрыть (:func:`model_can_help`): реплика «12 тысяч» сама по
    себе не содержит города, но вместе с предыдущей даёт полный набор,
    поэтому вызов модели здесь был бы лишним расходом токенов.
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
    for text in user_texts:
        # Локальный разбор каждой реплики: модель здесь не нужна, решение о вызове
        # принимается ниже — по СОСТОЯНИЮ ПОСЛЕ всей истории, а не по одной реплике.
        step = extract_message(text, cache=cache, resolver=None, allow_model=False)
        state = merge_step(state, step)
    if resolver is not None and allow_model:
        state = resolve_missing(state, user_texts, resolver)
    return state


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
    """CORE. Дополняет условия ОДНИМ вызовом модели по всему диалогу (§5).

    Зачем: fallback нужен только когда лексикон не закрыл обязательные условия
    даже с учётом истории. Один вызов на запрос вместо вызова на каждую реплику
    уменьшает расход токенов и число обращений к шлюзу (лимит 20 rpm), а модель
    при этом видит диалог целиком, а не обрывок.

    Вход:  state      — накопленные условия после слияния истории;
           user_texts — реплики покупателя;
           resolver   — клиент модели (``app.llm.client.make_resolver``).
    Выход: слоты, дополненные тем, что нашла модель; при пустом ответе — те же.

    Алгоритм: гейт :func:`model_can_help` (можно ли вообще закрыть недостающее) →
    склейка диалога (:func:`dialogue_text`) → один вызов resolver →
    :func:`merge_model_slots` (локальные значения не перетираются, город
    проверяется по справочнику). Ошибка клиента не глушится: это техническая
    ошибка запуска (CONTRACT.md §1), а не повод выдать ``no_match``.
    Сложность: O(длина диалога) + один сетевой обмен; память O(слоты).
    Почему именно он: решение «звать ли модель» принимается там, где видна вся
    история, — в слиянии; поэтому диалог «Стол в Таганроге» + «12 тысяч» стоит
    0 токенов, хотя ни одна его реплика по отдельности условий не закрывает.
    Гейт делает следующий шаг: если бюджет не назван вовсе (чисел в тексте нет)
    или город не к чему привязать, вызов не состоится — модель не может
    придумать условие, которого покупатель не называл.
    Крайние случаи: пустой текст → вызова нет; все условия известны → False
    из гейта; непустой ответ модели без нужных полей → условия остаются как
    были (дальше ``clarify``).
    Вызывается из: app/nlp/merge.py::merge_history, тесты, bench/load.py.
    """
    if not model_can_help(state, user_texts):
        return state
    text = dialogue_text(user_texts)
    if not text:
        return state
    raw = resolver(text, state)
    if not raw:
        return state
    return merge_model_slots(state, raw)


def merge_step(state: Slots, step: Slots) -> Slots:
    """Применяет слоты одной реплики к накопленному состоянию.

    Заменяются только поля из ``step.mentioned``; всё остальное остаётся прежним.
    Отдельно обрабатываются категории: они заменяются целиком (явное «нужен стол»
    после «стол, кресло и лампа» означает новый набор), объединяются при
    «ещё/также/добавь» и удаляются при «без лампы»/«стол не нужен».
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
