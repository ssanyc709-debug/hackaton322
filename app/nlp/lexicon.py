"""Лексикон: города, категории, сленг — распознавание слов без модели (0 токенов).

Порядок распознавания (по возрастанию цены):

1. хеш-словарь точных форм — O(1);
2. Snowball-стемминг русских окончаний и повторный хеш-поиск — O(len слова);
3. триграммный инвертированный индекс + расстояние Левенштейна — только по
   узкому списку кандидатов из пересечения триграмм, а не по всему словарю.

Слова вне закрытой таксономии не превращаются в товар: они помечаются как
«неподдерживаемая категория/город», и помощник уточняет соответствующее поле.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from app.model import CATEGORIES

#: Города из data/cities.json: формы, которые встречаются в живой речи.
CITY_WORDS: dict[str, tuple[str, ...]] = {
    "taganrog": (
        "таганрог", "таганроге", "таганрога", "таганрогу", "таганрогом",
        "таган", "таганрожцы", "тгн", "taganrog",
    ),
    "rostov-na-donu": (
        "ростов", "ростове", "ростова", "ростову", "ростовом",
        "ростов-на-дону", "ростов на дону", "рнд", "rostov", "rostov-na-donu",
    ),
    "azov": ("азов", "азове", "азова", "азову", "азовом", "azov"),
}

#: Категории закрытой таксономии: синонимы, сленг и просторечия.
CATEGORY_WORDS: dict[str, tuple[str, ...]] = {
    "table": (
        "стол", "стола", "столы", "столов", "столе", "столик", "столика", "столиков",
        "столешница", "парта", "парты", "бюро", "секретер", "консоль", "верстак",
        "компьютерный", "письменный", "обеденный", "рабочий",
        "table", "desk",
    ),
    "chair": (
        "кресло", "кресла", "кресло-качалка", "креслице", "стул", "стула", "стулья",
        "стульев", "стульчик", "стульчика", "табурет", "табуретка", "табурета",
        "пуф", "пуфик", "банкетка", "кресло-мешок", "седло",
        "chair", "armchair",
    ),
    "lamp": (
        "лампа", "лампы", "лампу", "лампой", "лампочка", "лампочки", "светильник",
        "светильника", "торшер", "торшера", "ночник", "бра", "люстра", "люстры",
        "прожектор", "подсветка", "lamp", "light",
    ),
}

#: Конкретные предметы вне таксономии: покупатель явно просит не то, что умеем —
#: это ``clarify`` по полю ``categories``, а не «нет совпадений».
UNSUPPORTED_CATEGORY_WORDS: dict[str, tuple[str, ...]] = {
    "диван": ("диван", "диваны", "дивана", "софа", "кушетка"),
    "шкаф": ("шкаф", "шкафы", "шкафа", "гардероб", "шифоньер"),
    "полка": ("полка", "полки", "полок", "стеллаж", "стеллажа", "этажерка"),
    "кровать": ("кровать", "кровати", "кроватка", "койка"),
    "тумба": ("тумба", "тумбы", "тумбочка", "комод", "комода"),
    "вешалка": ("вешалка", "вешалки", "вешалок"),
    "зеркало": ("зеркало", "зеркала", "зеркал"),
    "ковер": ("ковер", "ковра", "ковры", "палас"),
    "матрас": ("матрас", "матраса", "матрасы"),
}

#: Города, которых нет в справочнике: упоминание города вне справочника —
#: это ``clarify`` по ``city_id``, а не молчаливая подстановка Таганрога.
UNSUPPORTED_CITY_WORDS: dict[str, tuple[str, ...]] = {
    "москва": ("москва", "москве", "москвы", "москву", "moscow"),
    "санкт-петербург": ("питер", "петербург", "петербурге", "спб", "санкт-петербург", "spb"),
    "краснодар": ("краснодар", "краснодаре", "краснодара"),
    "сочи": ("сочи",),
    "воронеж": ("воронеж", "воронеже"),
    "екатеринбург": ("екатеринбург", "екатеринбурге"),
    "новосибирск": ("новосибирск", "новосибирске"),
    "казань": ("казань", "казани"),
    "самара": ("самара", "самаре"),
    "уфа": ("уфа", "уфе"),
    "пермь": ("пермь", "перми"),
    "волгоград": ("волгоград", "волгограде"),
    "красноярск": ("красноярск", "красноярске"),
    "омск": ("омск", "омске"),
    "тюмень": ("тюмень", "тюмени"),
    "саратов": ("саратов", "саратове"),
    "иркутск": ("иркутск", "иркутске"),
    "хабаровск": ("хабаровск", "хабаровске"),
    "владивосток": ("владивосток", "владивостоке"),
    "ярославль": ("ярославль", "ярославле"),
    "тольятти": ("тольятти",),
    "махачкала": ("махачкала", "махачкале"),
    "батайск": ("батайск", "батайске"),
    "новочеркасск": ("новочеркасск", "новочеркасске"),
    "шахты": ("шахты", "шахтах"),
}

#: Мат и шум: не должны ни ломать разбор, ни становиться товаром.
NOISE_WORDS: frozenset[str] = frozenset(
    {
        "бля", "блять", "блядь", "блин", "нахуй", "нахуя", "хуй", "хуя", "хуйня",
        "пиздец", "пизда", "пиздато", "ебать", "ебаный", "ебанный", "хер", "херня",
        "сука", "сучка", "черт", "чорт", "дерьмо", "говно", "жопа", "уебок", "мразь",
        "дурак", "идиот", "тупой", "срочно", "быстро", "срочняк", "плиз", "пожалуйста",
    }
)

#: Слова, которые не должны считаться категорией, даже если похожи на неё.
STOPWORD_GUARDS: frozenset[str] = frozenset({"свет", "света", "светло", "столовая"})

#: Минимальная длина слова для нечёткого поиска: короткие слова слишком шумные.
MIN_FUZZY_LENGTH = 4

#: Триграммы ищут только слова не короче этого значения.
MIN_TRIGRAM_LENGTH = 3

_VOWELS = frozenset("аеиоуыэюя")

# --- Окончания русского языка (Snowball, шаг 1) ----------------------------

_PERFECTIVE_GERUND = ("вшись", "вши", "в")
_REFLEXIVE = ("ся", "сь")
_ADJECTIVE = (
    "ими", "ыми", "его", "ого", "ему", "ому", "ее", "ие", "ые", "ое", "ей", "ий",
    "ый", "ой", "ем", "им", "ым", "ом", "их", "ых", "ую", "юю", "ая", "яя", "ою",
    "ею",
)
_PARTICIPLE_1 = ("ющ", "ем", "нн", "вш", "щ")
_PARTICIPLE_2 = ("ивш", "ывш", "ующ")
_VERB_1 = (
    "ила", "ыла", "ена", "ейте", "уйте", "ите", "или", "ыли", "ей", "уй", "ил",
    "ыл", "им", "ым", "ен", "ило", "ыло", "ено", "ят", "ует", "уют", "ит", "ыт",
    "ены", "ить", "ыть", "ишь", "ую", "ю",
)
_VERB_2 = (
    "ла", "на", "ете", "йте", "ли", "й", "л", "ем", "н", "ло", "но", "ет", "ют",
    "ны", "ть", "ешь", "нно",
)
_NOUN = (
    "иями", "ями", "ами", "иях", "иям", "ием", "ев", "ов", "ие", "ье", "еи", "ии",
    "ией", "ей", "ой", "ий", "й", "ям", "ем", "ам", "ом", "ах", "ях", "ию", "ью",
    "ия", "ья", "а", "е", "и", "о", "у", "ы", "ь", "ю", "я",
)


@dataclass(frozen=True, slots=True)
class Hit:
    """Найденное в тексте слово: что оно значит и как распознано."""

    value: str
    position: int
    word: str
    how: str  # exact | stem | fuzzy


def stem(word: str) -> str:
    """CORE. Стеммер русского языка (Snowball/Porter) — сведение словоформ.

    Зачем: покупатели пишут «кресла», «лампу», «в Таганроге», «столик». Хеш-словарь
    работает только с точными строками, поэтому нужна нормализация словоформ
    до основы: «кресла» → «кресл», «в Таганроге» → «таганрог».

    Вход:  word — одно слово в нижнем регистре («таганроге», «кресла», «стол»).
    Выход: основа слова без словоизменительных окончаний.

    Алгоритм: точная реализация шагов русского Snowball — вычисление областей
    RV/R1/R2, удаление возвратных/прилагательных/причастных/глагольных/существительных
    окончаний, шаг «и», деривационное «ость», затем «ейш», «нн» → «н» и мягкий знак.
    Сложность: время O(len(word)) на слово (окончания — короткие строки,
    сравнение за константу), память O(len(word)).
    Почему именно он: Snowball — стандарт для русского и даёт совместимость
    с классическими стеммерами; регулярная обрезка вручную («убрать 2 буквы»)
    склеивает разные слова и не покрывает возвратные формы.
    Крайние случаи: односложные слова; слова без гласных; уже нормальная форма;
    «стол» → «сто» (особенность Porter — поэтому в словаре регистрируются и
    точные формы, см. :meth:`WordIndex.add`), «ростове» → «ростов» → «рост».
    Вызывается из: app/nlp/lexicon.py::WordIndex.lookup, _candidate_forms.
    """
    if len(word) <= 2:
        return word
    rv, r1, r2 = _regions(word)

    # Шаг 1: перфектный деепричастный оборот, иначе возвратное + (прил.|прич.|глаг.|сущ.)
    result = _strip_ending(word, _PERFECTIVE_GERUND, rv, require_preceding=("а", "я"))
    if result == word:
        reflexive = _strip_ending(word, _REFLEXIVE, rv)
        current = reflexive
        for group in (_ADJECTIVE, _PARTICIPLE_1, _PARTICIPLE_2, _VERB_1, _VERB_2, _NOUN):
            preceding = ("а", "я") if group in (_PARTICIPLE_1, _VERB_1) else None
            candidate = _strip_ending(current, group, rv, require_preceding=preceding)
            if candidate != current:
                current = candidate
                break
        result = current

    # Шаг 2: конечная «и».
    if result.endswith("и") and len(result) - 1 >= rv:
        result = result[:-1]

    # Шаг 3: деривационное «ость»/«ост» только в области R2.
    for suffix in ("ость", "ост"):
        if result.endswith(suffix) and len(result) - len(suffix) >= r2:
            result = result[:-len(suffix)]
            break

    # Шаг 4: превосходная степень, «нн» → «н», мягкий знак.
    for suffix in ("ейше", "ейш"):
        if result.endswith(suffix) and len(result) - len(suffix) >= rv:
            result = result[:-len(suffix)]
            break
    if result.endswith("нн"):
        result = result[:-1]
    if result.endswith("ь"):
        result = result[:-1]
    return result or word


def _regions(word: str) -> tuple[int, int, int]:
    """Вычисляет начала областей RV, R1 и R2 (как в спецификации Snowball).

    RV — после первой гласной; R1 — после первой пары «гласная + согласная»;
    R2 — то же правило, начиная с R1. Если пара не найдена, область равна длине слова.
    """
    length = len(word)
    rv = length
    for index, char in enumerate(word):
        if char in _VOWELS:
            rv = index + 1
            break
    r1 = length
    for index in range(1, length):
        if word[index - 1] in _VOWELS and word[index] not in _VOWELS:
            r1 = index + 1
            break
    r2 = length
    for index in range(r1 + 1, length):
        if word[index - 1] in _VOWELS and word[index] not in _VOWELS:
            r2 = index + 1
            break
    return rv, r1, r2


def _strip_ending(
    word: str,
    endings: Sequence[str],
    region_start: int,
    require_preceding: Sequence[str] | None = None,
) -> str:
    """Удаляет самое длинное окончание из списка, если оно целиком лежит в области.

    ``require_preceding`` реализует условие Snowball «among (а, я)»: перед
    окончанием должна стоять одна из указанных букв.
    """
    best: str | None = None
    for ending in endings:
        if not word.endswith(ending):
            continue
        start = len(word) - len(ending)
        if start < region_start:
            continue
        if require_preceding is not None:
            if start == 0 or word[start - 1] not in require_preceding:
                continue
        if best is None or len(ending) > len(best):
            best = ending
    return word[: len(word) - len(best)] if best else word


def candidate_forms(word: str) -> tuple[str, ...]:
    """Формы слова для поиска в словаре: точная, стем, стем от стема.

    Двойной стем нужен из-за непостоянства Porter: «ростов» → «рост», а
    «ростове» → «ростов» → «рост» уже на втором шаге. Порядок — от дешёвого к дорогому.
    """
    forms = [word]
    once = stem(word)
    if once != word:
        forms.append(once)
    twice = stem(once)
    if twice not in forms:
        forms.append(twice)
    return tuple(forms)


def trigrams(word: str) -> frozenset[str]:
    """Триграммы слова с граничными маркерами (для инвертированного индекса).

    Сложность: O(len(word)); память O(len(word)).
    """
    padded = f"  {word} "
    return frozenset(padded[index:index + 3] for index in range(len(padded) - 2))


def levenshtein(left: str, right: str, limit: int) -> int:
    """Расстояние Левенштейна с ранним выходом по ``limit``.

    Зачем: опечатки («таганрг», «креслр», «лампаа») не ловятся ни словарём,
    ни стеммером. Расстояние редактирования — стандартная метрика для этого.

    Алгоритм: динамическое программирование по двум строкам с двумя рядами
    вместо матрицы (память O(min(m, n))); если минимум в ряду превысил ``limit``,
    возвращается ``limit + 1`` (дальше считать незачем).
    Сложность: время O(m·n), память O(min(m, n)).
    Почему именно он: Левенштейн учитывает вставки/удаления/замены и на словах
    длиной <= 12 считается за микросекунды; но он применяется ТОЛЬКО к 3–5
    кандидатам из триграммного индекса, поэтому O(m·n) не превращается в O(N·m·n).
    Крайние случаи: пустые строки; одинаковые строки → 0; превышение лимита →
    ``limit + 1`` без полного расчёта.
    Вызывается из: app/nlp/lexicon.py::WordIndex.lookup.
    """
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


class WordIndex:
    """CORE. Хеш-словарь форм + триграммный индекс с нечётким поиском.

    Зачем: распознавание города/категории/сленга обязано работать без модели и
    стоить 0 токенов. Хеш даёт O(1) на точную форму и на стем, а нечёткий поиск
    нужен только для опечаток — и тогда он идёт по узкому списку кандидатов.

    Регистрация (:meth:`add`) кладёт в словарь и точную форму, и стем, и стем от
    стема: это снимает непостоянство Porter («ростов» → «рост») и заодно
    поддерживает формы, которые стеммер портит («стол» → «сто»).

    Вход/выход ``lookup``: слово → ``(значение, способ)``; ``("", "")``, если
    ничего не найдено.

    Алгоритм поиска: точный хеш → формы из :func:`candidate_forms` (O(1) каждая) →
    триграммы запроса → пересечение с инвертированным индексом → Левенштейн по
    кандидатам (3–5 штук) → порог по длине слова.
    Сложность: попадание O(len(word)); промах O(len + k·m·n), где k — число
    кандидатов по триграммам (мало), m·n — длины слов.
    Почему именно он: НЕ бинарный поиск — ключи словаря не отсортированы и не
    упорядочены лексикографически по смыслу; сравнение строк идёт по хешу, а
    нечёткое сравнение — по триграммам. Сортированный массив тут не помог бы:
    «кресл» и «кресло» соседствуют только при точном префиксном поиске, а опечатка
    в середине слова ломает и его.
    Крайние случаи: короткие слова (< 4) не ищутся нечётко; два равных кандидата
    по расстоянию → отказ (неоднозначность важнее догадки); слово из одних
    согласных; пустой индекс.
    Вызывается из: app/nlp/extract.py::extract_message.
    """

    def __init__(self) -> None:
        self._exact: dict[str, str] = {}
        self._trigram_index: dict[str, set[str]] = {}
        self._keys: set[str] = set()

    def add(self, word: str, value: str) -> None:
        """Регистрирует слово и его формы; конфликт значений не перетирается.

        Если одна форма уже привязана к другому значению (например «свет» может
        означать и лампу, и «свет в комнате»), форма помечается как неоднозначная
        и удаляется из точного словаря — лучше не распознать, чем угадать неверно.
        """
        for form in candidate_forms(word):
            if not form:
                continue
            existing = self._exact.get(form)
            if existing is None:
                self._exact[form] = value
            elif existing != value:
                self._exact.pop(form, None)
            for gram in trigrams(form):
                self._trigram_index.setdefault(gram, set()).add(form)
            self._keys.add(form)

    def lookup(self, word: str) -> tuple[str, str]:
        """Ищет значение слова: точная форма → стем → нечёткий поиск."""
        for form in candidate_forms(word):
            value = self._exact.get(form)
            if value is not None:
                return value, ("exact" if form == word else "stem")
        if len(word) < MIN_FUZZY_LENGTH:
            return "", ""
        return self._fuzzy(word)

    def _fuzzy(self, word: str) -> tuple[str, str]:
        """Нечёткий поиск по триграммному индексу с ограничением кандидатов."""
        gram_counts: dict[str, int] = {}
        for gram in trigrams(word):
            for candidate_key in self._trigram_index.get(gram, ()):
                gram_counts[candidate_key] = gram_counts.get(candidate_key, 0) + 1
        if not gram_counts:
            return "", ""
        # Сначала самые похожие по триграммам; берём не больше пяти кандидатов,
        # чтобы Левенштейн не считался по всему словарю.
        candidates = sorted(gram_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        limit = 1 if len(word) <= 5 else 2
        best_value = ""
        best_distance = limit + 1
        ambiguous = False
        for candidate_key, _ in candidates:
            distance = levenshtein(word, candidate_key, limit)
            if distance > limit:
                continue
            value = self._exact.get(candidate_key, "")
            if not value:
                continue
            if distance < best_distance:
                best_distance = distance
                best_value = value
                ambiguous = False
            elif distance == best_distance and value != best_value:
                ambiguous = True
        return ("", "") if ambiguous else (best_value, "fuzzy")


class Lexicon:
    """Готовый лексикон: индексы городов, категорий и неподдерживаемых предметов."""

    def __init__(self) -> None:
        self.cities = WordIndex()
        for city_id, words in CITY_WORDS.items():
            for word in words:
                self.cities.add(word, city_id)

        self.categories = WordIndex()
        for category, words in CATEGORY_WORDS.items():
            for word in words:
                self.categories.add(word, category)

        self.unsupported_categories = WordIndex()
        for name, words in UNSUPPORTED_CATEGORY_WORDS.items():
            for word in words:
                self.unsupported_categories.add(word, name)

        self.unsupported_cities = WordIndex()
        for canonical, words in UNSUPPORTED_CITY_WORDS.items():
            for word in words:
                self.unsupported_cities.add(word, canonical)

        self.noise = NOISE_WORDS
        self.guards = STOPWORD_GUARDS

    def scan_cities(self, tokens: Sequence[str]) -> tuple[Hit, ...]:
        """Все упоминания города в порядке текста."""
        return self._scan(tokens, self.cities)

    def scan_categories(self, tokens: Sequence[str]) -> tuple[Hit, ...]:
        """Все упоминания поддерживаемых категорий в порядке текста."""
        return self._scan(tokens, self.categories)

    def scan_unsupported_categories(self, tokens: Sequence[str]) -> tuple[Hit, ...]:
        """Упоминания предметов вне таксономии (диван, шкаф, кровать…)."""
        return self._scan(tokens, self.unsupported_categories)

    def scan_unsupported_cities(self, tokens: Sequence[str]) -> tuple[Hit, ...]:
        """Упоминания городов вне справочника (Москва, Питер…)."""
        return self._scan(tokens, self.unsupported_cities)

    def _scan(self, tokens: Sequence[str], index: WordIndex) -> tuple[Hit, ...]:
        """Линейный проход по токенам; шум и служебные слова пропускаются."""
        hits: list[Hit] = []
        for position, token in enumerate(tokens):
            if token in self.noise or token in self.guards or len(token) < 3:
                continue
            if not token.isalpha():
                continue
            value, how = index.lookup(token)
            if value:
                hits.append(Hit(value=value, position=position, word=token, how=how))
        return tuple(hits)


_LEXICON: Lexicon | None = None


def get_lexicon() -> Lexicon:
    """Единственный экземпляр лексикона на процесс (словари строятся один раз)."""
    global _LEXICON
    if _LEXICON is None:
        _LEXICON = Lexicon()
    return _LEXICON


def supported_categories() -> tuple[str, ...]:
    """Категории закрытой таксономии (для сообщений покупателю)."""
    return CATEGORIES


def order_categories(values: Iterable[str]) -> tuple[str, ...]:
    """Упорядочивает категории канонически: стол → кресло → лампа."""
    unique = set(values)
    return tuple(category for category in CATEGORIES if category in unique)


def describe_words(mapping: Mapping[str, Iterable[str]], key: str) -> str:
    """Человеческое описание синонимов категории (для ``explanation`` и логов)."""
    return ", ".join(mapping.get(key, ()))
