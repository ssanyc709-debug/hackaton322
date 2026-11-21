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
from typing import Callable, Sequence

from app.model import (
    CATEGORIES,
    FIELD_ADJUSTABLE,
    FIELD_BUDGET,
    FIELD_CATEGORIES,
    FIELD_CITY,
    FIELD_TEMPERATURE,
    FIELD_WIDTH,
    Slots,
)
from app.nlp.cache import LexiconCache
from app.nlp.lexicon import (
    NOISE_WORDS,
    STOPWORD_GUARDS,
    get_lexicon,
    order_categories,
)
from app.nlp.numbers import (
    LOWER_INC,
    LOWER_STRICT,
    MONEY_HINTS,
    UNIT_RUB,
    UNIT_WORDS,
    FILLER_WORDS,
    collect_constraints,
    find_amounts,
    normalize_text,
    tokenize,
)

#: Верхняя граница «мелких» чисел: меньше этого числа не считаются бюджетом.
MIN_BUDGET_GUESS = 1000
#: Виды границ, которые ограничивают значение снизу и потолком бюджета быть не могут.
NON_CAPPING_KINDS = frozenset({LOWER_INC, LOWER_STRICT})
#: Нижняя/верхняя границы правдоподобной цветовой температуры, K.
MIN_KELVIN = 1500
MAX_KELVIN = 9000

#: Явные счётные и размерные обозначения, которые не могут обозначать рубли.
NON_MONEY_UNITS = frozenset({"шт", "штука", "штуки", "штук", "мм", "mm"})

#: Максимальная длина реплики, которую имеет смысл отдавать в модель.
MAX_MODEL_TEXT_BYTES = 4_000

#: Регулярные выражения для булевой характеристики кресла.
ADJUSTABLE_TRUE_RE = re.compile(
    r"(с|есть|нужн\w*|хочу)\s+(регулировк|подъем|подъём)|регулируем\w*|"
    r"регулировк\w*\s+высот|изменяем\w*\s+высот|с\s+подъемн|"
    # глагольные формы: «чтобы регулировалось по высоте», «умеет регулироваться»
    r"регулировал\w*|регулирующ\w*"
)
ADJUSTABLE_FALSE_RE = re.compile(
    r"без\s+регулировк|нерегулир\w*|не\s+регулир\w*|фиксированн\w*\s+высот|"
    r"без\s+подъем|жестк\w*\s+высот|"
    r"регулировк\w*\s+(?:мне\s+)?не\s+(?:нужн|требу\w*|интересн)\w*|"
    r"подъем\w*\s+(?:мне\s+)?не\s+(?:нужн|требу\w*|интересн)\w*"
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

#: Фразы-описания без названия категории: «на чём мне сидеть» — кресло,
#: «что светится» — лампа. Существительное не названо, поэтому поиск по
#: отдельным словам их не видит. Такая фраза ДОБАВЛЯЕТ категорию, а не заменяет
#: набор: «нужен стол и на чём сидеть» — это стол и кресло.
PHRASE_CATEGORIES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"на\s+ч[её]м\s+(?:мне\s+)?сидеть|где\s+(?:мне\s+)?сидеть|"
            r"посид\w*|посадочн\w*\s+мест\w*|мест\w*\s+для\s+сидения"
        ),
        "chair",
    ),
    (
        re.compile(r"что(?:\s*-?\s*то)?\s+светит\w*|что-то\s+светящ\w*|источник\w*\s+света"),
        "lamp",
    ),
)

# Название отдельного предмета в короткой реплике. Узкие границы исключают,
# например, «чехол на сиденье со спинкой» и «не нужно сиденье со спинкой».
_SEAT_WITH_BACK = re.compile(
    r"(?:^|[.!?]\s*)(?:нужн[оа]|хочу|ищу)\s+сиден(?:ье|ие)\s+со\s+спинкой\b"
)

#: Основы команд «снять ограничение»: «расширим ширину», «убери регулировку».
REMOVAL_CUE_STEMS: tuple[str, ...] = (
    "расшир", "убер", "убир", "убра", "сним", "снят", "откаж", "неважн",
)

#: Основы названий характеристик → поле условия, которое снимается.
REMOVAL_FIELD_STEMS: tuple[tuple[str, str], ...] = (
    ("ширин", FIELD_WIDTH),
    ("температур", FIELD_TEMPERATURE),
    ("регулиров", FIELD_ADJUSTABLE),
    ("подъем", FIELD_ADJUSTABLE),
    ("подъём", FIELD_ADJUSTABLE),
)

#: Окно поиска названия характеристики вокруг команды снятия (в токенах).
REMOVAL_WINDOW_BEFORE = 5
REMOVAL_WINDOW_AFTER = 7

#: Предлоги, после которых слово категории означает МЕСТО, а не товар:
#: «посветить над столом», «поставить под лампу», «лампа на стол».
PLACE_PREPOSITIONS: frozenset[str] = frozenset({"над", "под", "на"})

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
    снятие ограничений («расширим ширину», «убери регулировку» → ``fields_removed``)
    → (опционально) один вызов модели для незакрытых полей, но только если
    :func:`model_can_help` подтвердил, что она может их закрыть → запись в кэш.
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
    city_positions = {hit.position for hit in (*city_hits, *foreign_city_hits)}
    category_hits = tuple(
        hit
        for hit in lexicon.scan_categories(tokens)
        if hit.position not in city_positions and not _is_place_use(tokens, hit.position)
    )
    unsupported_hits = lexicon.scan_unsupported_categories(tokens)

    # Берём последнее упоминание города, включая явный переход к городу вне справочника.
    city_id = city_hits[-1].value if city_hits else None
    city_unknown = None
    if foreign_city_hits and (not city_hits or foreign_city_hits[-1].position >= city_hits[-1].position):
        city_id = None
        city_unknown = foreign_city_hits[-1].value

    categories = order_categories(hit.value for hit in category_hits)
    seat = _SEAT_WITH_BACK.search(normalized)
    if seat:
        tail = normalized[seat.end():].lstrip()
        if not tail or tail[0] in ",.!?" or (
            tail.startswith(("в ", "во ")) and len(tokenize(normalized[:seat.end()])) + 1 in city_positions
        ):
            categories = order_categories((*categories, "chair"))
    phrase_categories = tuple(
        category for pattern, category in PHRASE_CATEGORIES if pattern.search(normalized)
    )
    if phrase_categories:
        categories = order_categories((*categories, *phrase_categories))
    categories_unknown = tuple(dict.fromkeys(hit.value for hit in unsupported_hits))
    categories_removed = _detect_removals(normalized)
    # Фраза-описание добавляет категорию к уже названным, а не заменяет набор.
    categories_additive = bool(ADDITIVE_PATTERN.search(normalized)) or bool(phrase_categories)

    noun_positions = {
        hit.position
        for hit in (*category_hits, *unsupported_hits)
        if hit.how != "fuzzy" and hit.word not in lexicon.category_adjectives
    }
    amounts = (
        amount
        for amount in find_amounts(normalized)
        if amount.unit is not None
        or amount.following_position is None
        or (
            amount.following_position not in noun_positions
            and tokens[amount.following_position] not in NON_MONEY_UNITS
        )
    )
    constraints = collect_constraints(amounts)
    width = constraints.width
    temperature = constraints.temperature
    budget = constraints.budget_rub

    if budget is None:
        for group in constraints.neutral_groups:
            if "lamp" in categories and temperature is None and len(group) > 1 and all(
                MIN_KELVIN <= amount.value <= MAX_KELVIN for amount in group
            ):
                continue
            caps = [amount.value for amount in group if amount.kind not in NON_CAPPING_KINDS]
            if caps and max(caps) >= MIN_BUDGET_GUESS:
                budget = max(caps)
                break

    adjustable = _detect_adjustable(normalized)

    # Снятие ограничения: «расширим ширину», «убери регулировку». Если в этой же
    # реплике полю дали значение, это замена, а не снятие.
    fields_removed = _detect_field_removals(normalized)
    if width is not None:
        fields_removed = tuple(field for field in fields_removed if field != FIELD_WIDTH)
    if temperature is not None:
        fields_removed = tuple(field for field in fields_removed if field != FIELD_TEMPERATURE)
    if adjustable is not None:
        fields_removed = tuple(field for field in fields_removed if field != FIELD_ADJUSTABLE)

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
    mentioned.update(fields_removed)

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
        fields_removed=fields_removed,
        categories_explicit=bool(categories)
        or bool(categories_removed)
        or bool(categories_unknown),
        categories_additive=categories_additive,
        mentioned=frozenset(mentioned),
        source="lexicon",
    )


def _is_place_use(tokens: Sequence[str], position: int) -> bool:
    """Слово категории стоит после предлога места («над столом», «на стол»).

    Зачем: «посветить над столом» — запрос лампы, а не стола; «лампа на стол» —
    лампа. Предлог даёт роль слова бесплатно, без модели и без разбора смысла.
    Ложное срабатывание безопасно: категория просто не добавляется, и работает
    обычный путь — уточнение или (если есть бюджет и город) вызов модели.

    Исключение — деньги: «бюджет 20 000 рублей **на стол** и кресло» — это набор
    из стола и кресла, а не место. Если перед предлогом стоит сумма или денежное
    слово, слово категории означает товар. Без этой ветки покупатель просил стол,
    а получал только кресла: категория пропадала молча.

    Сложность: O(1). Крайние случаи: слово в начале фразы (position=0) → False;
    «стол под ноутбук» → False (предлог стоит ПОСЛЕ названия товара).
    """
    if position == 0 or tokens[position - 1] not in PLACE_PREPOSITIONS:
        return False
    return not _money_context_before(tokens, position - 1)


def _money_context_before(tokens: Sequence[str], preposition: int) -> bool:
    """Перед предлогом стоит сумма или денежное слово («20 000 рублей на стол»)."""
    if preposition == 0:
        return False
    previous = tokens[preposition - 1]
    return (
        previous.isdigit()
        or previous in MONEY_HINTS
        or UNIT_WORDS.get(previous) == UNIT_RUB
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


def _removal_cue(tokens: Sequence[str], index: int) -> str | None:
    """Команда «снять ограничение» в позиции ``index`` (или ``None``).

    «Без» — команда только для ширины и температуры: «без регулировки высоты»
    означает ``adjustable_height = False`` (значение), а не снятие условия.
    """
    token = tokens[index]
    if token.startswith(REMOVAL_CUE_STEMS):
        return token
    if token == "без":
        return "без"
    if token == "не" and index + 1 < len(tokens) and tokens[index + 1].startswith("важн"):
        return "неважно"
    return None


def _removal_field(token: str) -> str | None:
    """Поле условия по названию характеристики («ширину» → ``width_cm``)."""
    for stem, field in REMOVAL_FIELD_STEMS:
        if token.startswith(stem):
            return field
    return None


def _detect_field_removals(normalized: str) -> tuple[str, ...]:
    """Находит ограничения, которые покупатель снял в этой реплике.

    «Давай расширим допустимую ширину», «убери требование к регулировке»,
    «без ограничений по температуре» — это явное изменение условия, а не новое
    значение: при слиянии истории такое поле обнуляется (по образцу
    :data:`REMOVAL_PATTERNS` для категорий). Ограничение считается снятым,
    только если в этой же реплике ему не дали значение — иначе
    «расширим ширину до 250 см» потеряло бы новую ширину.
    """
    tokens = tokenize(normalized)
    removed: list[str] = []
    for index in range(len(tokens)):
        cue = _removal_cue(tokens, index)
        if cue is None:
            continue
        start = max(0, index - REMOVAL_WINDOW_BEFORE)
        end = min(len(tokens), index + REMOVAL_WINDOW_AFTER)
        for other in tokens[start:end]:
            field = _removal_field(other)
            if field is None:
                continue
            if cue == "без" and field == FIELD_ADJUSTABLE:
                continue
            if field not in removed:
                removed.append(field)
    return tuple(removed)


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
    """Дополняет категории результатом одного вызова модели.

    Перед вызовом проверяется :func:`model_can_help`: если модель всё равно не
    может закрыть недостающую категорию (например, бюджет ещё неизвестен),
    вызова нет. Исключения клиента модели (:class:`app.llm.client.LlmError`)
    намеренно НЕ глушатся: сбой внешнего сервиса — это техническая ошибка запуска
    (stderr + ненулевой код, CONTRACT.md §1), а не повод выдать ``no_match``
    или «уточните заново». Пустой ответ модели (``None``/``{}``) — не ошибка:
    она просто ничего не нашла, и условия остаются пустыми.
    """
    if len(text.encode("utf-8")) > MAX_MODEL_TEXT_BYTES:
        return slots
    if not model_can_help(slots):
        return slots
    raw = resolver(text, slots)
    if not raw:
        return slots
    return merge_model_slots(slots, raw)


def merge_model_slots(slots: Slots, raw: dict[str, object]) -> Slots:
    """За O(числа вариантов модели) дополняет пропущенную категорию из таксономии."""
    model_categories = raw.get("categories")
    if model_can_help(slots) and isinstance(model_categories, list):
        valid = [item for item in model_categories if isinstance(item, str) and item in CATEGORIES]
        if valid:
            return replace(
                slots,
                categories=order_categories(valid),
                categories_explicit=True,
                mentioned=slots.mentioned | {FIELD_CATEGORIES},
                source="model",
            )
    return slots


def model_can_help(slots: Slots) -> bool:
    """CORE. Разрешает вызов модели для описательного запроса о товаре.

    Вход: локально разобранные условия одной реплики или всего диалога.
    Выход: True, когда город и бюджет известны, категория отсутствует и
    покупатель не назвал неподдерживаемый предмет. Проверка занимает O(1).
    Так «деревяшка с четырьмя ножками» с указанными городом и бюджетом
    попадает в модель, а «диван», «в Азовском» и «до 900» ведут к уточнению.
    Явно заданный нулевой бюджет считается известным.
    """
    return (
        slots.city_id is not None
        and slots.budget_rub is not None
        and not slots.categories
        and not slots.categories_unknown
    )


#: Короткие осмысленные ответы покупателя: подтверждение, благодарность,
#: приветствие. Такая реплика НЕ является мусором, даже если в ней нет ни одного
#: условия и ни одного числа: «спасибо» — это ответ, а не непонятный текст.
CONFIRMATION_WORDS: frozenset[str] = frozenset(
    {
        "спасибо", "благодарю", "пожалуйста", "да", "нет", "ок", "окей", "хорошо",
        "ладно", "понятно", "ясно", "ага", "угу", "привет", "здравствуйте",
        "беру", "подходит",
        # «лады» — живое согласие, все буквы которого случайно лежат в одном
        # ряду клавиатуры («фывапролджэ»): без белого списка признак ряда
        # (:func:`_single_keyboard_row`) объявил бы его мусором. «норм» — просто
        # живое согласие, добавлено для полноты словаря подтверждений
        # (правило клавиатуры его не задевает: буквы из двух рядов).
        "лады", "норм",
    }
)

#: Минимальная длина «содержательного» слова для детектора непонятной реплики:
#: короткие слова («ок», «ну») слишком шумные и мусором не считаются.
MIN_UNINTELLIGIBLE_TOKEN = 4

#: Гласные (русские и латинские) и предел цепочки согласных.
#: Слово, в котором 4+ согласных подряд, — это набор букв, а не слово:
#: «ыаллфдылааввы» (ллфд), «asdfgh» (dfgh). Обычная русская речь такого не даёт.
_VOWELS: frozenset[str] = frozenset("аеиоуыэюяaeiouy")
MAX_CONSONANT_RUN = 4

#: Признак «набор букв», а не слова: слишком мало разных букв («фффффф»).
MIN_DISTINCT_LETTERS = 3

#: Ряды клавиатуры: русская раскладка ЙЦУКЕН и латинская QWERTY.
#: Мусор часто получается прогоном пальцами по одному ряду, поэтому слово,
#: целиком лежащее в одном ряду, — это набор, а не слово языка.
_KEYBOARD_ROWS: tuple[frozenset[str], ...] = (
    frozenset("йцукенгшщзхъ"),
    frozenset("фывапролджэ"),
    frozenset("ячсмитьбю"),
    frozenset("qwertyuiop"),
    frozenset("asdfghjkl"),
    frozenset("zxcvbnm"),
)


def _single_keyboard_row(token: str) -> bool:
    """Все буквы слова лежат в одном ряду клавиатуры (слово из 4+ букв).

    Зачем: «аваыла», «фывафыва», «фывапролд» состоят из гласных и согласных в
    правильном чередовании, поэтому проверки «мало разных букв» и «4+ согласных
    подряд» их пропускают. Общий признак у них другой — все буквы с одного ряда
    клавиатуры: это след прогона пальцами, а не слово. Правило описывает
    генератор мусора (ряд клавиш), поэтому не требует словаря всей лексики.

    Сложность: O(len(token)). Крайние случаи: «отлично» (2 ряда) → False,
    «дорого» (2 ряда) → False; слова короче 4 букв не проверяются — среди них
    слишком много законных («ок», «да»).
    """
    if len(token) < 4:
        return False
    rows = 0
    for row in _KEYBOARD_ROWS:
        if token[0] in row:
            rows += 1
            if any(letter not in row for letter in token):
                return False
    return rows == 1


def _looks_like_gibberish(token: str) -> bool:
    """Похоже ли слово на случайный набор букв, а не на слово языка.

    Зачем: без этого детектор считал мусором любое незнакомое слово, и законная
    реплика «Покажи варианты» (условий не меняет, но осмысленна) обнуляла
    категории. Проверка идёт по форме слова, а не по словарю: словаря всей
    русской лексики у нас нет, а «ссылку на словарь» заменить нечем.

    Сложность: O(len(token)). Крайние случаи: «фффффф» → True (мало разных букв);
    «ыаллфдылааввы» → True (цепочка «ллфд»); «аваыла» и «фывафыва» → True (все
    буквы с одного ряда клавиатуры); «Покажи» → False; слово без гласных
    («взгляд» → «взгл») тоже считается набором букв — как самостоятельный ответ
    на уточнение оно бессмысленно и требует переспроса.
    """
    if len(set(token)) < MIN_DISTINCT_LETTERS:
        return True
    if _single_keyboard_row(token):
        return True
    run = 0
    for letter in token:
        if letter in _VOWELS:
            run = 0
            continue
        run += 1
        if run >= MAX_CONSONANT_RUN:
            return True
    return False


def is_unintelligible(text: str) -> bool:
    """CORE. Является ли реплика непонятной (не подтверждает прежние условия).

    Зачем: реплика, в которой не распознано НИ ОДНОГО условия и нет ни одного
    числа, не должна молча подтверждать прежний подбор — иначе «ыаллфдылааввы»
    читается как «продолжай, я согласен», и покупатель получает старый комплект
    вместо уточнения (CONTRACT.md §3: явное изменение заменяет, остальное живёт).

    Вход:  text — сырая реплика покупателя.
    Выход: True, если реплика непонятна; False для осмысленных ответов
    («спасибо», «ок», «15 тысяч», «нужен Азов», «добавь лампу»).

    Алгоритм (0 токенов модели): нормализация → если есть числа, это ответ
    (False) → локальный разбор лексиконом: любое распознанное условие (город,
    категория, бюджет, ширина, температура, регулировка, снятие ограничения)
    означает понятную реплику → иначе проход по словам: слово «понятно», если
    это город/категория (включая неподдерживаемые), шум, служебное слово или
    подтверждение; слово «мусор», если оно похоже на набор букв
    (:func:`_looks_like_gibberish`). Реплика непонятна, когда есть мусор, нет
    ни одного понятного слова и ВСЕ остальные значимые слова — тоже мусор:
    «ыаллфдылааввы», «фффффф» и «аваыла» — мусор, а «Покажи варианты»,
    «Спасибо!» и «Правда, дорого» — осмысленные ответы без новых условий.
    Сложность: O(len(text)) на нормализацию и проход по токенам.
    Почему именно он: сначала бесплатный разбор, который уже есть в конвейере,
    и только потом проверка «остались ли необъяснённые слова» — так детектор
    не расходится с извлечением условий и не требует второй модели правил.
    Крайние случаи: пустая реплика → False (нечего подтверждать); «ок» и другие
    короткие ответы → False; «Хочу мебель» без прочих условий → False (есть
    служебное слово «хочу», реплика осмысленная, но условий не даёт); опечатка
    категории («стлл») → False, её распознаёт нечёткий поиск лексикона; живые
    реакции («отлично», «дорого», «лады») → False, их защищает форма слова или
    белый список подтверждений, а не словарь всей лексики; «правда» и «дрова»
    по отдельности похожи на набор букв (весь ряд клавиатуры), поэтому «Правда,
    дорого» защищено вторым словом обычной формы, а не словарём.
    Вызывается из: app/pipeline.py::answer, тесты.
    """
    normalized = normalize_text(text)
    if not normalized:
        return False
    if find_amounts(normalized):
        # Любое число — это ответ покупателя (бюджет, ширина, количество),
        # а не мусор: «15 тысяч» обязано сохранять прежний подбор.
        return False

    slots = _lexicon_slots(normalized)
    if slots.mentioned:
        return False

    tokens = tokenize(normalized)
    lexicon = get_lexicon()
    known_positions: set[int] = set()
    for hits in (
        lexicon.scan_cities(tokens),
        lexicon.scan_categories(tokens),
        lexicon.scan_unsupported_cities(tokens),
        lexicon.scan_unsupported_categories(tokens),
    ):
        known_positions.update(hit.position for hit in hits)

    garbage = 0  # значимых слов, похожих на набор букв
    unknown = 0  # значимых слов обычной формы, но не из словарей проекта
    understood = False
    for position, token in enumerate(tokens):
        if not token.isalpha():
            continue
        if (
            position in known_positions
            or token in CONFIRMATION_WORDS
            or token in NOISE_WORDS
            or token in STOPWORD_GUARDS
            or token in FILLER_WORDS
        ):
            understood = True
            continue
        if len(token) < MIN_UNINTELLIGIBLE_TOKEN:
            # Короткое слово («ну», «э») не мусор, но и не защита: шума в нём
            # слишком много, чтобы делать по нему вывод о смысле реплики.
            continue
        if _looks_like_gibberish(token):
            garbage += 1
        else:
            unknown += 1
    # Непонятна только реплика, в которой ВСЕ значимые слова похожи на набор
    # букв. Если рядом есть слово обычной формы («Правда, дорого»), реплика
    # осмысленна, даже когда этого слова нет в наших словарях: «правда» —
    # живая реакция, просто все её буквы случайно лежат в одном ряду клавиатуры.
    return garbage > 0 and unknown == 0 and not understood


def is_complete(slots: Slots) -> bool:
    """Все обязательные условия известны: город, бюджет, категории.

    Используется для решения «звать ли модель»: если True — вызова нет.
    """
    return not missing_fields(slots)


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
