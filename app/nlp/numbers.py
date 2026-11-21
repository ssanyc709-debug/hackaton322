"""Числа, единицы измерения и сравнительные конструкции (0 токенов модели).

Модуль отвечает на вопрос «какое условие сформулировал покупатель и с какой
границей»: «не шире метра» → ``width_cm <= 100``, «уже 100 см» → ``width < 100``,
«от 3000 до 5000» → ``3000 <= K <= 5000``, «15 тысяч» → ``15000``.

Правила контракта §2.2: «до», «не более», «не шире», «от», «не менее» включают
границу; «меньше», «больше», «уже», «шире» — строгая граница; «от … до …»
включает оба конца. Всё воспроизводится здесь, без обращения к модели.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

from app.model import Bound, NumericRange

#: Виды сравнительных конструкций (по включительности границы).
LOWER_INC = "lower_inc"
LOWER_STRICT = "lower_strict"
UPPER_INC = "upper_inc"
UPPER_STRICT = "upper_strict"
EXACT = "exact"
NO_TRIGGER = "none"

#: Единицы: см и метры — ширина стола, кельвины — цветовая температура, ₽ — деньги.
UNIT_CM = "cm"
UNIT_M = "m"
UNIT_K = "k"
UNIT_RUB = "rub"

#: Неразрывные и узкие пробелы, которые приходят из веба и Excel.
SPACE_CHARS = "\u00a0\u2007\u2009\u200a\u202f\u2060\u3000"

#: Символы тире, которые надо свести к «-».
DASH_CHARS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"

#: Глаголы и служебные слова, которые не мешают, но и не значат ничего.
FILLER_WORDS = frozenset(
    {"пожалуйста", "плиз", "надо", "нужно", "нужен", "нужна", "нужны", "хочу", "ищу",
     "подбери", "подберите", "найди", "найдите", "дай", "дайте", "можно", "бы", "мне",
     "нам", "я", "мы", "есть", "что", "то", "чтобы", "было", "был", "была", "если",
     "также", "ещё", "еще", "и", "а", "или", "но", "в", "во", "на", "за", "по", "с",
     "со", "для", "из", "к", "ко", "у", "о", "об", "при", "же", "ли", "не"}
)

WORD_NUMBERS: dict[str, float] = {
    "ноль": 0, "нуль": 0,
    "один": 1, "одна": 1, "одно": 1, "одну": 1, "одного": 1,
    "два": 2, "две": 2, "двух": 2, "двум": 2,
    "три": 3, "трех": 3, "трёх": 3, "трем": 3, "трём": 3,
    "четыре": 4, "четырех": 4, "четырёх": 4,
    "пять": 5, "пяти": 5,
    "шесть": 6, "шести": 6,
    "семь": 7, "семи": 7,
    "восемь": 8, "восьми": 8,
    "девять": 9, "девяти": 9,
    "десять": 10, "десяти": 10,
    "одиннадцать": 11, "одиннадцати": 11,
    "двенадцать": 12, "двенадцати": 12, "тринадцать": 13, "тринадцати": 13,
    "четырнадцать": 14, "четырнадцати": 14,
    "пятнадцать": 15, "пятнадцати": 15, "шестнадцать": 16, "шестнадцати": 16,
    "семнадцать": 17, "семнадцати": 17, "восемнадцать": 18, "восемнадцати": 18,
    "девятнадцать": 19, "девятнадцати": 19,
    "двадцать": 20, "двадцати": 20, "тридцать": 30, "тридцати": 30,
    "сорок": 40, "сорока": 40, "пятьдесят": 50, "пятидесяти": 50,
    "шестьдесят": 60, "шестидесяти": 60, "семьдесят": 70, "семидесяти": 70,
    "восемьдесят": 80, "восьмидесяти": 80, "девяносто": 90, "девяноста": 90,
    "сто": 100, "ста": 100, "двести": 200, "двухсот": 200,
    "триста": 300, "трёхсот": 300, "трехсот": 300, "четыреста": 400, "четырёхсот": 400,
    "пятьсот": 500, "пятисот": 500, "шестьсот": 600, "шестисот": 600,
    "семьсот": 700, "семисот": 700, "восемьсот": 800, "восьмисот": 800,
    "девятьсот": 900, "девятисот": 900,
    "полтора": 1.5, "полторы": 1.5, "полтораста": 150,
}

#: «С половиной» после числа: «двенадцать с половиной тысяч» → 12 500.
HALF_WORDS = frozenset({"половиной", "половина", "половину"})

#: Множители, которые умножают накопленное число. Падежные формы важны:
#: «располагаю 15 тысячами» без творительного падежа давало бы бюджет 15 ₽.
MULTIPLIER_WORDS: dict[str, int] = {
    "тысяча": 1000, "тысячи": 1000, "тысяч": 1000, "тысячу": 1000, "тысяче": 1000,
    "тысячам": 1000, "тысячами": 1000,
    "тыща": 1000, "тыщи": 1000, "тыщ": 1000, "тыщами": 1000, "тысячек": 1000, "тыс": 1000,
    "миллион": 1_000_000, "миллиона": 1_000_000, "миллионов": 1_000_000,
    "миллионе": 1_000_000, "миллионам": 1_000_000, "миллионами": 1_000_000,
    "млн": 1_000_000,
}

UNIT_WORDS: dict[str, str] = {
    # сантиметры
    "см": UNIT_CM, "сантиметр": UNIT_CM, "сантиметра": UNIT_CM, "сантиметров": UNIT_CM,
    "сантиметрах": UNIT_CM, "сантиметрами": UNIT_CM, "cm": UNIT_CM,
    # метры
    "м": UNIT_M, "метр": UNIT_M, "метра": UNIT_M, "метров": UNIT_M, "метре": UNIT_M,
    "метрах": UNIT_M, "метрами": UNIT_M, "м.": UNIT_M, "метрa": UNIT_M,
    # кельвины
    "к": UNIT_K, "кельвин": UNIT_K, "кельвина": UNIT_K, "кельвинов": UNIT_K,
    "кельвинах": UNIT_K, "k": UNIT_K, "kelvin": UNIT_K, "kelvins": UNIT_K,
    # деньги
    "₽": UNIT_RUB, "р": UNIT_RUB, "руб": UNIT_RUB, "руб.": UNIT_RUB, "рубль": UNIT_RUB,
    "рубля": UNIT_RUB, "рублей": UNIT_RUB, "рублях": UNIT_RUB, "рублями": UNIT_RUB,
    "рублям": UNIT_RUB, "rub": UNIT_RUB, "rur": UNIT_RUB, "₽.": UNIT_RUB,
}

#: Фразы-подсказки, что речь о деньгах, даже если единица не написана.
MONEY_HINTS = frozenset(
    {"бюджет", "бюджетом", "бюджете", "потратить", "уложиться", "уложимся", "укладываемся",
     "лимит", "лимитом", "всего", "итого", "максимум", "потрачу", "располагаю", "есть"}
)

#: Виды сравнения, из которых может получиться бюджет-потолок (§3: бюджет — это
#: верхний предел суммы). Нижние границы сюда не входят: «от 5000 ₽» покупатель
#: сказал, чтобы ему не предлагали дешевле, а не чтобы ограничить сверху.
CAPPING_KINDS = (UPPER_INC, UPPER_STRICT, EXACT, NO_TRIGGER)

#: Сравнительные конструкции. Ключ — фраза (слова через пробел), значение — вид.
#: Порядок в словаре не важен: при разборе фразы проверяются от длинных к коротким,
#: поэтому «не уже» всегда побеждает «уже», а «не более» — «более».
TRIGGER_PHRASES: dict[str, str] = {
    # включительная нижняя граница
    "не менее": LOWER_INC, "не меньше": LOWER_INC, "не уже": LOWER_INC,
    "не уже чем": LOWER_INC, "от": LOWER_INC, "минимум": LOWER_INC,
    "начиная с": LOWER_INC, "от и": LOWER_INC, "минимум": LOWER_INC,
    # строгая нижняя граница
    "больше": LOWER_STRICT, "более": LOWER_STRICT, "шире": LOWER_STRICT,
    "выше": LOWER_STRICT, "дороже": LOWER_STRICT, "свыше": LOWER_STRICT,
    "побольше": LOWER_STRICT,
    # включительная верхняя граница
    "не более": UPPER_INC, "не больше": UPPER_INC, "не шире": UPPER_INC,
    "не выше": UPPER_INC, "не дороже": UPPER_INC, "не превышает": UPPER_INC,
    "в пределах": UPPER_INC, "уложиться в": UPPER_INC, "уложимся в": UPPER_INC,
    "до": UPPER_INC, "максимум": UPPER_INC, "около": UPPER_INC, "примерно": UPPER_INC,
    "где-то": UPPER_INC, "пределах": UPPER_INC,
    # строгая верхняя граница
    "меньше": UPPER_STRICT, "менее": UPPER_STRICT, "уже": UPPER_STRICT,
    "ниже": UPPER_STRICT, "дешевле": UPPER_STRICT, "поменьше": UPPER_STRICT,
    # точное значение
    "ровно": EXACT, "точно": EXACT, "именно": EXACT, "как раз": EXACT,
}

#: Триггеры, которые сами по себе означают «речь о ширине стола».
WIDTH_HINTS = frozenset({"не шире", "шире", "уже", "не уже"})

#: Слова «ширина» рядом с единицей без числа: «стол шириной метр» = ровно 100 см.
#: Без них единица без числа разбиралась только при сравнительной конструкции
#: («до метра», «не шире метра»), а контракт §2.2 разрешает выражать единицы
#: обычным языком.
WIDTH_WORDS = frozenset(
    {"ширина", "ширины", "ширине", "ширину", "шириной", "ширин"}
)

#: Максимальное число токенов между «от» и «до» внутри одного диапазона.
MAX_RANGE_GAP_TOKENS = 4

#: Верхняя граница «маленького» числа: до этого «к» считается множителем (15к),
#: выше — единицей кельвинов (4000 к), иначе терялся бы смысл лампы.
KELVIN_THRESHOLD = 200

_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?|[a-zа-яё]+|₽", re.IGNORECASE)
_DIGIT_GROUP_RE = re.compile(r"(?<!\d)(\d{1,3})(?:[ ](\d{3}))+(?!\d)")


@dataclass(frozen=True, slots=True)
class Amount:
    """Найденное в тексте число вместе с единицей и сравнительной конструкцией."""

    value: int
    unit: str | None
    kind: str
    trigger: str
    position: int  # индекс токена, с которого началось число
    raw: str


@dataclass(frozen=True, slots=True)
class Constraints:
    """Ограничения, извлечённые из ОДНОГО сообщения.

    ``neutral`` — числа без единицы, которые нельзя классифицировать без знания
    категорий («до 5000» — это деньги или кельвины?). Их решает
    ``app/nlp/extract.py``, где уже известно, что просил покупатель.
    """

    width: NumericRange | None = None
    temperature: NumericRange | None = None
    budget_rub: int | None = None
    neutral_groups: tuple[tuple[Amount, ...], ...] = ()
    amounts: tuple[Amount, ...] = ()

    @property
    def neutral(self) -> tuple[Amount, ...]:
        """Все нейтральные числа подряд (для логов и тестов)."""
        return tuple(amount for group in self.neutral_groups for amount in group)


def normalize_text(text: str) -> str:
    """CORE. Приводит реплику к каноническому виду для разбора без модели.

    Зачем: покупатели пишут «1 500 ₽», «15\u00a0000», «1,5 метра», «ДО 15000₽»,
    «Таганрог\u2014город». Без нормализации ломается каждый второй шаг (числа, единицы,
    поиск города), поэтому нормализация — первый шаг конвейера и стоит 0 токенов.

    Вход:  text — исходная строка реплики (UTF-8, любой регистр).
    Выход: строка в нижнем регистре, с «е» вместо «ё», обычными пробелами вместо
    неразрывных, «-» вместо тире, склеенными разрядами чисел («15 000» → «15000»).

    Алгоритм: NFKC → замена ё/пробелов/тире → нижний регистр → склейка групп
    разрядов регулярным выражением (повтор до стабилизации) → сжатие пробелов.
    Сложность: время O(len(text)) (склейка — не более длины строки итераций,
    фактически 1–2 прохода), память O(len(text)).
    Почему именно он: нормализация строки в один поток дешевле, чем учить каждый
    последующий шаг терпеть варианты написания; регулярное выражение для групп
    разрядов решает «15 000» без ручного разбора.
    Крайние случаи: пустая строка; «ё» в любом регистре; несколько неразрывных
    пробелов подряд; «1 500 000» (склеивается целиком); «руб.» (точка сохраняется,
    потому что не является разделителем разрядов).
    Вызывается из: app/nlp/extract.py::extract_message, app/nlp/lexicon.py.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("ё", "е").replace("Ё", "Е")
    for space in SPACE_CHARS:
        normalized = normalized.replace(space, " ")
    for dash in DASH_CHARS:
        normalized = normalized.replace(dash, "-")
    normalized = normalized.lower()
    normalized = _merge_digit_groups(normalized)
    return re.sub(r"[ ]+", " ", normalized).strip()


def _merge_digit_groups(text: str) -> str:
    """Склеивает «15 000» и «1 500 000» в «15000» / «1500000».

    Повторяет замену до стабилизации: за один проход «1 500 000» дал бы «1500 000».
    """
    previous = None
    current = text
    while previous != current:
        previous = current
        current = _DIGIT_GROUP_RE.sub(lambda match: match.group(0).replace(" ", ""), current)
    return current


def tokenize(text: str) -> tuple[str, ...]:
    """Разбивает нормализованный текст на токены: числа, слова, символ «₽».

    Сложность: O(len(text)); память O(числа токенов).
    """
    return tuple(_TOKEN_RE.findall(text))


def parse_digits(token: str) -> float | None:
    """Разбирает числовой токен: «15000», «1,5», «1.5», «15,000».

    Запятая считается десятичным разделителем только при 1–2 цифрах после неё
    (иначе это разделитель разрядов). Возвращает ``None``, если токен не число.
    """
    if not token or not token[0].isdigit():
        return None
    if "," in token or "." in token:
        whole, _, fraction = token.replace(",", ".").rpartition(".")
        if len(fraction) in (1, 2):
            try:
                return float(f"{whole}.{fraction}")
            except ValueError:
                return None
        try:
            return float(token.replace(",", "").replace(".", ""))
        except ValueError:
            return None
    try:
        return float(token)
    except ValueError:
        return None


def _consume_number(tokens: Sequence[str], start: int) -> tuple[float, int, bool]:
    """Читает число, начиная с токена ``start`` (цифры или числительные словами).

    Возвращает ``(значение, следующий индекс, найдено ли число)``.
    Поддерживает «15», «15 тысяч», «двадцать пять тысяч пятьсот», «полторы тысячи».
    """
    first = tokens[start] if start < len(tokens) else ""
    digits = parse_digits(first)
    if digits is not None:
        return digits, start + 1, True

    if first not in WORD_NUMBERS and first not in MULTIPLIER_WORDS:
        return 0.0, start, False

    total = 0.0
    current = 0.0
    index = start
    consumed = False
    while index < len(tokens):
        token = tokens[index]
        if token in WORD_NUMBERS:
            current += WORD_NUMBERS[token]
            consumed = True
        elif token in MULTIPLIER_WORDS:
            multiplier = MULTIPLIER_WORDS[token]
            total += (current if current else 1.0) * multiplier
            current = 0.0
            consumed = True
        else:
            break
        index += 1
    if not consumed:
        return 0.0, start, False
    # «двенадцать с половиной» = 12.5: половина стоит после числа, но до единицы.
    if index + 1 < len(tokens) and tokens[index] == "с" and tokens[index + 1] in HALF_WORDS:
        current += 0.5
        index += 2
    return total + current, index, True


#: Служебные слова между множителем и числом в обратном порядке («тысяч НА двенадцать»).
INVERTED_FILLERS = frozenset({"на", "с", "около", "эдак", "примерно", "этак"})


def _consume_inverted_multiplier(tokens: Sequence[str], start: int) -> tuple[float, int] | None:
    """Читает множитель ПЕРЕД числом: «тысяч на двенадцать» → 12 000.

    Обычный порядок («двенадцать тысяч») обрабатывает :func:`_consume_number`,
    но в разговорной речи множитель часто стоит первым. Без этой ветки «тысяч»
    превращалось в самостоятельное число 1000, и бюджет занижался в двенадцать
    раз — покупатель получал ``no_match`` при достаточном бюджете.

    Возвращает ``(значение, следующий индекс)`` или ``None``, если после
    множителя нет правдоподобного числа (1..999) — тогда работает обычный разбор.
    """
    if start >= len(tokens) or tokens[start] not in MULTIPLIER_WORDS:
        return None
    cursor = start + 1
    if cursor < len(tokens) and tokens[cursor] in INVERTED_FILLERS:
        cursor += 1
    value, next_index, found = _consume_number(tokens, cursor)
    if not found or not 1 <= value <= 999:
        return None
    return value * MULTIPLIER_WORDS[tokens[start]], next_index


def _append_amount(
    amounts: list[Amount],
    tokens: Sequence[str],
    start: int,
    value: float,
    next_index: int,
) -> int:
    """Досчитывает единицу и сравнительную конструкцию, пишет :class:`Amount`.

    Возвращает индекс, с которого продолжается проход. Метры приводятся к
    сантиметрам здесь же, чтобы все условия ширины жили в одной единице.
    """
    unit, next_index, value = _unit_after(tokens, next_index, value)
    if unit == UNIT_M:
        value *= 100
        unit = UNIT_CM
    kind, trigger = _lookback_trigger(tokens, start)
    amounts.append(
        Amount(
            value=int(round(value)),
            unit=unit,
            kind=kind,
            trigger=trigger,
            position=start,
            raw=" ".join(tokens[start:next_index]),
        )
    )
    return max(next_index, start + 1)


def _lookback_trigger(tokens: Sequence[str], number_index: int) -> tuple[str, str]:
    """Ищет сравнительную конструкцию перед числом (до трёх слов).

    Возвращает ``(вид, исходная фраза)``; ``(NO_TRIGGER, "")``, если конструкции нет.
    Проверка идёт от длинных фраз к коротким, поэтому «не уже» не распадается
    на «не» и «уже», а «не более» — на «не» и «более».

    Если сравнительной конструкции нет, но рядом стоит денежное слово («бюджет»,
    «за всё», «лимит»), оно возвращается как ``trigger``: это подсказка, что число
    относится к деньгам, а не к характеристике.
    """
    window = [token for token in tokens[max(0, number_index - 3):number_index] if token.isalpha()]
    for size in (3, 2, 1):
        if len(window) < size:
            continue
        phrase = " ".join(window[-size:])
        kind = TRIGGER_PHRASES.get(phrase)
        if kind is not None:
            return kind, phrase
    for token in reversed(window):
        if token in MONEY_HINTS:
            return NO_TRIGGER, token
    return NO_TRIGGER, ""


def _unit_after(tokens: Sequence[str], index: int, value: float) -> tuple[str | None, int, float]:
    """Определяет единицу измерения сразу после числа и возможные множители.

    Возвращает ``(единица, следующий индекс, значение)``. Отдельно решается
    неоднозначность «к»: «15 к» — это тысячи рублей, «4000 к» — кельвины.
    Множители («тысяч», «тыс», «млн») применяются до единицы, поэтому
    «15 тысяч рублей» даёт 15000 ₽, а не 15 без единицы.
    """
    while index < len(tokens) and tokens[index] in MULTIPLIER_WORDS:
        value *= MULTIPLIER_WORDS[tokens[index]]
        index += 1
    if index >= len(tokens):
        return None, index, value
    token = tokens[index]
    if token in ("к", "k") and value > KELVIN_THRESHOLD:
        return UNIT_K, index + 1, value
    if token in ("к", "k") and value <= KELVIN_THRESHOLD:
        return None, index + 1, value * 1000
    unit = UNIT_WORDS.get(token)
    if unit is not None:
        return unit, index + 1, value
    return None, index, value


def find_amounts(text: str) -> tuple[Amount, ...]:
    """CORE. Находит числа с единицами и сравнительными конструкциями.

    Зачем: это единственное место, где текст превращается в числа. Здесь
    реализованы правила §2.2 про включительность границ и словарь единиц
    («метр» → 100 см, «15 тысяч» → 15000, «₽»/«рублей» — деньги). Модель не нужна.

    Вход:  text — нормализованная или сырая реплика (нормализуется внутри).
    Выход: кортеж :class:`Amount` в порядке появления; пустой, если чисел нет.

    Алгоритм: токенизация → линейный проход; на каждом числовом токене читается
    число (цифрами или словами), затем множитель и единица справа, затем
    сравнительная конструкция слева (до 3 слов). Отдельная ветка — единица
    без числа («не шире метра» = 1 метр).
    Сложность: время O(tokens · 3), то есть O(len(text)); память O(числа чисел).
    Почему именно он: один проход с локальным окном ±3 токена дешевле, чем
    полноценный синтаксический разбор, и покрывает все формулировки из контракта.
    Крайние случаи: «1,5 метра» → 150 см; «15к» → 15000; «4000 к» → кельвины;
    «не шире метра» (без цифры) → 100 см; «стол шириной метр» → ровно 100 см;
    «от 3000 до 5000» — два отдельных Amount, их объединяет
    :func:`collect_constraints`; «15 000 ₽ за всё» → деньги.
    Вызывается из: app/nlp/extract.py::extract_message.
    """
    normalized = normalize_text(text)
    tokens = tokenize(normalized)
    amounts: list[Amount] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]

        # Единица без числа: «не шире метра», «до метра», «от метра»,
        # «стол шириной метр» (слово «ширина» рядом делает значение точным).
        if token in UNIT_WORDS and token not in ("к", "k"):
            kind, trigger = _lookback_trigger(tokens, index)
            if kind == NO_TRIGGER:
                window = [t for t in tokens[max(0, index - 3):index] if t.isalpha()]
                if any(t in WIDTH_WORDS for t in window):
                    kind, trigger = EXACT, "ширина"
            if kind != NO_TRIGGER:
                unit = UNIT_WORDS[token]
                value = 100 if unit == UNIT_M else 1
                if unit == UNIT_M:
                    unit = UNIT_CM  # ширина всегда в сантиметрах
                amounts.append(Amount(value=value, unit=unit, kind=kind, trigger=trigger,
                                      position=index, raw=token))
                index += 1
                continue

        # Обратный порядок множителя: «тысяч на двенадцать» = 12 000.
        inverted = _consume_inverted_multiplier(tokens, index)
        if inverted is not None:
            value, next_index = inverted
            index = _append_amount(amounts, tokens, index, value, next_index)
            continue

        value, next_index, found = _consume_number(tokens, index)
        if not found:
            index += 1
            continue

        index = _append_amount(amounts, tokens, index, value, next_index)
    return tuple(amounts)


def to_bounds(pair: Sequence[Amount]) -> NumericRange:
    """Строит :class:`NumericRange` из одной пары «нижняя/верхняя» (или одного числа).

    Включительность берётся из вида конструкции: ``lower_inc``/``upper_inc`` —
    граница входит, ``lower_strict``/``upper_strict`` — нет, ``exact`` — обе
    границы равны и включительны. Одиночное число без конструкции трактуется как
    точное значение («стол 95 см»), потому что иначе условие не имело бы смысла.
    """
    lower: Bound | None = None
    upper: Bound | None = None
    single = len(pair) == 1
    for amount in pair:
        if amount.kind == EXACT or (single and amount.kind == NO_TRIGGER):
            lower = Bound(amount.value, True)
            upper = Bound(amount.value, True)
        elif amount.kind in (LOWER_INC, LOWER_STRICT):
            lower = Bound(amount.value, amount.kind == LOWER_INC)
        elif amount.kind in (UPPER_INC, UPPER_STRICT):
            upper = Bound(amount.value, amount.kind == UPPER_INC)
    return NumericRange(lower=lower, upper=upper)


def _merge_pairs(items: Sequence[Amount]) -> list[tuple[Amount, ...]]:
    """Объединяет «от X … до Y» в пары; одиночные числа остаются сами по себе.

    Пара образуется, только если нижняя граница идёт раньше верхней, между ними
    не больше :data:`MAX_RANGE_GAP_TOKENS` токенов и у них нет противоречащих
    единиц измерения. Так «от 80 до 100 см» становится одним диапазоном, а
    «до 100 ₽, от 3000 до 5000 кельвинов» — тремя отдельными условиями.

    Одиночная нижняя граница — самостоятельное условие (§2.2), даже если верхняя
    граница далеко: в «Стол от 100 см в Таганроге до 15 000 ₽» слово «до»
    относится к бюджету, а не к ширине. Поэтому «пара не найдена» проверяется
    явным флагом ``paired``: ветка ``else`` у ``for`` здесь не годится, потому что
    при выходе по ``break`` она не выполняется и условие покупателя терялось молча.
    """
    result: list[tuple[Amount, ...]] = []
    used: set[int] = set()
    for position, amount in enumerate(items):
        if position in used:
            continue
        if amount.kind in (LOWER_INC, LOWER_STRICT):
            paired = False
            for other_position in range(position + 1, len(items)):
                if other_position in used:
                    continue
                other = items[other_position]
                if other.kind not in (UPPER_INC, UPPER_STRICT):
                    continue
                if other.position - amount.position > MAX_RANGE_GAP_TOKENS:
                    break
                if amount.unit is not None and other.unit is not None and amount.unit != other.unit:
                    continue
                result.append((amount, other))
                used.update({position, other_position})
                paired = True
                break
            if not paired:
                result.append((amount,))
            continue
        result.append((amount,))
    return result


def _pair_unit(pair: Sequence[Amount]) -> str | None:
    """Единица измерения пары: метры уже приведены к сантиметрам при разборе."""
    units = {amount.unit for amount in pair if amount.unit is not None}
    if not units:
        return None
    if len(units) == 1:
        return next(iter(units))
    return None  # противоречивые единицы в одной паре — условие не собираем


def _cap_value(pair: Sequence[Amount]) -> int | None:
    """Бюджет-потолок пары или ``None``, если верхней границы в ней нет.

    Бюджет в контракте — верхний предел суммы, поэтому потолком может стать
    только «до 5000 ₽», «не дороже 5000», «ровно 5000» или голое число
    («бюджет 15000», «располагаю 15 тысячами»). Одиночная нижняя граница
    («от 5000 ₽», «дороже 5000») ограничивает сумму снизу: превращать её
    в потолок нельзя, иначе покупателю предложат ровно то, что он просил
    не предлагать. В паре «от 3000 до 5000 ₽» потолок по-прежнему 5000.
    """
    caps = [amount.value for amount in pair if amount.kind in CAPPING_KINDS]
    if not caps:
        return None
    return max(caps)


def collect_constraints(amounts: Iterable[Amount]) -> Constraints:
    """CORE. Раскладывает найденные числа по условиям: ширина, кельвины, бюджет.

    Зачем: покупатель пишет всё в одном предложении («стол не шире метра, кресло
    и лампа. До 15 000 ₽ за всё»). Нужно разделить числа по смыслу, иначе бюджет
    станет шириной, а температура — ценой.

    Вход:  amounts — результат :func:`find_amounts` для одного сообщения.
    Выход: :class:`Constraints` — диапазон ширины в см, диапазон температуры в K,
    целый бюджет и список ``neutral`` чисел без единицы (их классифицирует
    ``app.nlp.extract`` по составу категорий).

    Алгоритм: объединение пар «от…до» → классификация по единице измерения
    (см → ширина, K → температура, ₽ → бюджет) → для чисел без единицы:
    подсказка «шире/уже» → ширина, денежное слово → бюджет, иначе в ``neutral``.
    Бюджетом становится только **потолок** пары (см. :func:`_cap_value`): бюджет
    по §3 — верхний предел суммы, поэтому «от 5000 ₽» его не задаёт.
    Сложность: время O(a²) на a чисел в реплике (a <= ~6, то есть константа).
    Почему именно он: реплика короткая, квадратичный перебор пар дешевле
    сложного парсера диапазонов, а нейтральные числа не угадываются здесь —
    их решение требует контекста категорий, который есть только выше по стеку.
    Крайние случаи: «до 15 000 ₽» → бюджет; «лампа 4000 к» → температура;
    «стол до 100 см» → ширина; «1 метр» → 100 см; «2 кресла» → количество,
    а не бюджет (попадает в ``neutral`` и отбрасывается как слишком маленькое);
    «от 5000 ₽» и «дороже 5000» → потолка нет, бюджет остаётся неизвестным.
    Вызывается из: app/nlp/extract.py::extract_message.
    """
    items = list(amounts)
    width: NumericRange | None = None
    temperature: NumericRange | None = None
    budget: int | None = None
    neutral_groups: list[tuple[Amount, ...]] = []

    for pair in _merge_pairs(items):
        unit = _pair_unit(pair)
        if unit == UNIT_RUB:
            cap = _cap_value(pair)
            if cap is not None:
                budget = max(budget or 0, cap)
            continue
        if unit == UNIT_CM:
            width = to_bounds(pair)
            continue
        if unit == UNIT_K:
            temperature = to_bounds(pair)
            continue
        triggers = " ".join(amount.trigger for amount in pair)
        if any(hint in triggers for hint in WIDTH_HINTS):
            width = to_bounds(pair)
            continue
        if any(amount.trigger in MONEY_HINTS for amount in pair):
            cap = _cap_value(pair)
            if cap is not None:
                budget = max(budget or 0, cap)
                continue
        neutral_groups.append(pair)

    return Constraints(
        width=width,
        temperature=temperature,
        budget_rub=budget,
        neutral_groups=tuple(neutral_groups),
        amounts=tuple(items),
    )


def parse_amount(text: str) -> int | None:
    """Удобная обёртка: первое денежное или «просто числовое» значение из строки.

    Используется для коротких ответов на уточнение («15 тысяч», «15к», «15000 ₽»):
    если единица не указана, а число большое, оно трактуется как бюджет.
    Сложность: O(len(text)).
    """
    amounts = find_amounts(text)
    if not amounts:
        return None
    for amount in amounts:
        if amount.unit == UNIT_RUB:
            return amount.value
    for amount in amounts:
        if amount.unit is None:
            return amount.value
    return amounts[0].value
