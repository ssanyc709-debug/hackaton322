"""Стресс-проверка детектора непонятной реплики: шире автопроверки.

Основной сценарий отчёта — «мусорная реплика молча подтверждает подбор» — закрыт
в ``tests/test_dialogue.py``. Здесь проверяется то, что на презентации спросят
«а если?»:

* клавиатурная геометрия: правило ряда обязано ловить ЛЮБОЙ набор букв одного
  ряда (все раскладки, все ряды, все 4-буквенные сочетания) и не трогать слова
  из двух и более рядов;
* живой корпус: 130+ реплик покупателя и 200+ слов товарной лексики не должны
  считаться мусором ни в детекторе, ни в конвейере;
* композиция правила: мусор переспрашивает категории только тогда, когда мусором
  является ВСЯ реплика, а не одно слово рядом с осмысленным;
* сквозной CLI: то же поведение в отдельном процессе, как в автопроверке;
* устойчивость: регистр, пунктуация, эмодзи, длина 100 000 символов, детерминизм;
* служебное слово не может стать товаром или городом через нечёткий поиск.
"""

from __future__ import annotations

import itertools
import json
import time
import unittest

from app.catalog.index import build_index
from app.config import Settings
from app.contract.request import parse_request
from app.nlp.extract import (
    CONFIRMATION_WORDS,
    MIN_UNINTELLIGIBLE_TOKEN,
    _looks_like_gibberish,
    _single_keyboard_row,
    is_unintelligible,
)
from app.nlp.lexicon import NOISE_WORDS, get_lexicon
from app.nlp.numbers import FILLER_WORDS
from app.pipeline import answer
from tests.support import CATALOG_PATH, run_cli

DOCUMENT = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
INDEX = build_index(DOCUMENT)
SETTINGS = Settings.load(environ={"ASSISTANT_OFFLINE": "1"})
BASE = "Стол до 5000. Таганрог."

#: Клавиатура как она есть: ряды ЙЦУКЕН и QWERTY. Наборы заданы независимо от
#: ``extract._KEYBOARD_ROWS``, чтобы тест проверял правило, а не повторял константу.
ROW_TOP = "йцукенгшщзхъ"
ROW_HOME = "фывапролджэ"
ROW_BOTTOM = "ячсмитьбю"
ROW_LATIN_TOP = "qwertyuiop"
ROW_LATIN_HOME = "asdfghjkl"
ROW_LATIN_BOTTOM = "zxcvbnm"
KEYBOARD_ROWS = (
    ROW_TOP, ROW_HOME, ROW_BOTTOM, ROW_LATIN_TOP, ROW_LATIN_HOME, ROW_LATIN_BOTTOM,
)

#: Мусор из отчёта и его родня: прогон пальцами по одному ряду.
ROW_MASH = (
    "аваыла", "фывафыва", "дваа", "фывапролд", "фывапролджэ", "цукен",
    "йцукенг", "йцук", "кгшщз", "ячсмит", "митьбю", "пролдж", "олджа", "фывап",
    "апролд", "ролджэ", "выаыва", "ававаы", "фыфыфы", "вваваы", "длолд", "жээээ",
    "asdfgh", "qwerty", "zxcvbn", "asdf", "qwer", "hjkl", "vbnm", "uiop",
)

#: Мусор другой формы: мало разных букв и длинные цепочки согласных.
OTHER_MASH = (
    "ыаллфдылааввы", "фффффф", "фффф", "ааааа", "ыыыыы", "яяяя", "жжжж",
    "рвсв", "крслр", "взглд", "пстх", "сдфг", "фгхдж",
)

#: Живые реплики покупателя: подтверждения, оценки, вопросы, опечатки.
#: Ни одна не должна считаться мусором — иначе детектор съест осмысленный ответ.
LIVE_REPLIES = (
    # короткие реакции
    "да", "нет", "ок", "окей", "хорошо", "отлично", "супер", "класс", "круто",
    "нормально", "норм", "лады", "ладно", "годится", "подходит", "беру", "берём",
    "спасибо", "благодарю", "понятно", "ясно", "ага", "угу", "привет",
    "здравствуйте", "добрый день", "покажи", "показывай", "покажите", "варианты",
    "ещё", "дальше", "вперёд", "продолжим", "погоди", "стоп", "назад", "отмена",
    "сброс", "покажи варианты", "давай дальше", "можно ещё", "ещё варианты",
    "а дальше", "что дальше", "повтори", "сначала",
    # оценка вариантов
    "дорого", "дешевле", "подешевле", "дороже", "не подходит", "не то", "первый",
    "второй", "третий", "крайний", "левый", "правый", "средний", "этот",
    "вот этот", "нравится", "не нравится", "устраивает", "не устраивает",
    "ужас", "кошмар", "идеально", "то что нужно", "вроде норм", "ну такое",
    "более менее", "пойдёт", "сойдёт", "неплохо", "хороший вариант", "взял бы",
    "это дорого", "это дешево", "мне нравится", "не очень", "так себе",
    "нормальный вариант", "вполне подходит", "беру этот", "давай первый",
    "посмотрим второй", "а третий?", "что лучше",
    # живые слова, случайно похожие на набор букв или на товар
    "Правда, дорого", "правда недешево", "дрова не нужны", "вода есть",
    "мне нужен повар", "пожар не нужен", "пока хватит", "просто покажи",
    "привет, покажи варианты", "стоп, не надо", "сама решу", "старая цена",
    # решения и отговорки
    "я подумаю", "надо подумать", "посоветуюсь", "позже", "потом", "не сейчас",
    "пока хватит", "хватит", "всё", "всё понятно", "всё ясно", "спасибо большое",
    # вопросы по товару и доставке
    "сколько стоит", "сколько стоит доставка", "есть ли скидки",
    "можно дешевле", "а что ещё", "а есть другие", "покажи ещё",
    "есть что-то другое", "чем отличаются", "почему так дорого",
    "какой размер", "сколько весит", "где находится", "как купить",
    "доставка есть", "а самовывоз", "есть гарантия", "можно в рассрочку",
    "а можно фото", "покажи фотографии", "какие цвета есть", "из чего сделан",
    "какой материал", "кто производитель", "а дешевле есть",
    # опечатки и живые оговорки
    "ничего не понял", "не понял", "покажы", "что-то не то", "ну и ну",
)

#: Реплики, которые МЕНЯЮТ условия: детектор обязан пропустить их к разбору.
CONDITION_REPLIES = (
    "в Таганроге", "в Азове", "до 15 тысяч", "бюджет 20000", "не шире метра",
    "нужен стол", "добавь кресло", "убери лампу", "без регулировки",
    "с регулировкой", "серый", "белый", "дубовый", "деревянный", "мягкое",
    "письменный", "обеденный", "журнальный", "компьютерный", "офисное",
    "кухонный", "нужна кровать", "нужен шкаф", "хочу диван", "давай стол",
    "только стол", "стол и кресло", "тысяч десять", "двенадцать тысяч",
    "до трёх тысяч", "не больше пяти", "где-то пятнадцать",
)

#: Товарная лексика проекта: слова, которые покупатель пишет о мебели.
#: Детектор обязан не считать их мусором — это живая речь, а не набор букв.
DOMAIN_WORDS = (
    "стол", "стул", "стулья", "кресло", "кресла", "диван", "шкаф", "полка",
    "кровать", "лампа", "светильник", "торшер", "ночник", "люстра", "комод",
    "тумба", "табурет", "пуф", "стеллаж", "гардероб", "матрас", "подушка",
    "одеяло", "простыня", "скатерть", "ковер", "коврик", "зеркало", "вешалка",
    "этажерка", "парта", "мойка", "плита", "холодильник", "техника", "мебель",
    "интерьер", "комната", "кухня", "спальня", "гостиная", "прихожая", "кабинет",
    "офис", "дача", "квартира", "гарнитур", "комплект", "набор", "вариант",
    "модель", "размер", "ширина", "высота", "глубина", "длина", "вес", "цвет",
    "материал", "ткань", "обивка", "каркас", "ножка", "спинка", "сиденье",
    "подлокотник", "ящик", "дверца", "фасад", "столешница", "колесико",
    "крепление", "сборка", "доставка", "гарантия", "скидка", "цена",
    "стоимость", "рубль", "тысяча", "бюджет", "рассрочка", "оплата",
    "самовывоз", "продавец", "производитель", "магазин", "склад", "адрес",
    "город", "улица",
    "деревянный", "дубовый", "сосновый", "ореховый", "ясеневый",
    "металлический", "стеклянный", "пластиковый", "кожаный", "тканевый",
    "велюровый", "микрофибра", "экокожа", "белый", "черный", "серый", "бежевый",
    "коричневый", "светлый", "темный", "синий", "зеленый", "красный",
    "оранжевый", "желтый", "розовый", "фиолетовый", "мятный", "пудровый",
    "графит", "венге", "орех", "дуб", "ясень", "сосна", "береза", "мдф", "шпон",
    "лак", "краска", "эмаль",
    "письменный", "обеденный", "журнальный", "компьютерный", "кухонный",
    "офисный", "мягкий", "жесткий", "удобный", "неудобный", "компактный",
    "широкий", "узкий", "высокий", "низкий", "регулируемый", "складной",
    "раскладной", "угловой", "прямой", "круглый", "овальный", "квадратный",
    "прямоугольный", "современный", "классический", "скандинавский",
    "практичный", "прочный", "надежный", "качественный", "дешевый", "дорогой",
    "недорогой", "бюджетный", "премиальный",
    "покажи", "покажите", "подбери", "подберите", "найди", "найдите", "добавь",
    "добавьте", "убери", "уберите", "замени", "измени", "оставь", "выбери",
    "предложи", "сравни", "посчитай", "посмотри", "расскажи", "объясни", "помоги",
    "помогите", "подскажи", "подскажите", "хочу", "нужен", "нужна", "нужно",
    "ищу", "ищем", "выбираю", "куплю", "закажу", "оформлю", "оплачу",
    "сегодня", "завтра", "сейчас", "потом", "позже", "сразу", "конечно",
    "возможно", "наверное", "обязательно", "желательно", "примерно", "около",
    "немного", "совсем", "очень", "слишком", "довольно", "достаточно",
    "хватит", "давай", "давайте", "можно", "нельзя", "спасибо", "благодарю",
    "пожалуйста", "здравствуйте", "привет", "добрый", "вечер", "утро", "отлично",
    "хорошо", "нормально", "плохо", "ужасно", "интересно", "понятно", "согласен",
    "ладно", "годится", "подходит", "устраивает", "нравится", "дешевле",
    "дороже", "дорого", "первый", "второй", "третий", "последний", "другой",
)

#: Мусор для сквозных проверок конвейера: каждое значение в отдельности
#: обязано переспросить категории, сохранив город и бюджет.
GARBAGE_REPLIES = (
    "аваыла", "фывафыва", "дваа", "фывапролд", "ыаллфдылааввы", "фффффф",
    "asdfgh", "цукен", "АВАЫЛА", "АВАЫЛА!!!", "аваыла 🛋️", "«аваыла»",
    "аваыла фывафыва", "фыва-фыва", "  аваыла  ", "йцукенг", "рвсв", "крслр",
    "апролд", "олджа", "жээээ", "длолд", "ячсмит", "митьбю",
)

#: Живые слова, которые правило ряда ловит по форме. Это известный долг
#: эвристики, а не список исключений: он измерен по товарному тексту проекта.
KNOWN_CATALOG_GIBBERISH = frozenset(
    {
        # 4 согласных подряд в живом слове — долг правила согласных
        "встрече", "встроенное", "инструкции", "инструкция", "надстройки",
        "надстройкой",
        # весь ряд клавиатуры — долг правила ряда
        "опора", "опоры", "пара", "провод", "проводов",
    }
)

#: Живые слова с законной русской цепочкой из четырёх согласных («вств»).
#: Правило согласных считает их набором букв; на продукт это не влияет, потому
#: что они защищены белым списком подтверждений или окружением в реплике.
LEGAL_CONSONANT_CLUSTERS = frozenset({"здравствуйте"})


def run(
    texts: list[str],
    cache: object | None = None,
    settings: Settings | None = None,
    resolver: object | None = None,
) -> dict:
    """Запрос контракта из реплик покупателя (между ними — ответ помощника)."""
    messages: list[dict[str, str]] = []
    for index, text in enumerate(texts):
        if index:
            messages.append({"role": "assistant", "text": "Уточняю."})
        messages.append({"role": "user", "text": text})
    request = parse_request({"messages": messages})
    return answer(
        request=request,
        index=INDEX,
        settings=settings if settings is not None else SETTINGS,
        deadline=1e18,
        cache=cache,
        resolver=resolver,
    )


def fields(document: dict) -> list[str]:
    """Поля уточнений в ответе."""
    return [question["field"] for question in document.get("questions", [])]


def catalog_vocabulary() -> set[str]:
    """Слова товарного текста проекта длиной от 4 букв (реальная лексика домена)."""
    from app.nlp.numbers import normalize_text, tokenize

    def strings(node: object, acc: list[str]) -> list[str]:
        if isinstance(node, str):
            acc.append(node)
        elif isinstance(node, dict):
            for value in node.values():
                strings(value, acc)
        elif isinstance(node, list):
            for value in node:
                strings(value, acc)
        return acc

    words: set[str] = set()
    for text in strings(DOCUMENT, []):
        for token in tokenize(normalize_text(text)):
            if token.isalpha() and len(token) >= MIN_UNINTELLIGIBLE_TOKEN:
                words.add(token)
    return words


class KeyboardGeometryTest(unittest.TestCase):
    """Правило «слово целиком с одного ряда» на всей клавиатуре."""

    def test_rows_cover_both_layouts(self) -> None:
        """Ряды правила совпадают с физическими: две раскладки, по три ряда."""
        for key in ("й", "ф", "я", "q", "a", "z"):
            with self.subTest(key=key):
                rows = [row for row in KEYBOARD_ROWS if key in row]
                self.assertEqual(len(rows), 1, "клавиша должна быть ровно в одном ряду")

    def test_every_four_letter_sequence_of_every_row_is_gibberish(self) -> None:
        """Все 4-буквенные сочетания любого ряда — мусор (около 70 000 строк)."""
        checked = 0
        for row in KEYBOARD_ROWS:
            for letters in itertools.product(row, repeat=4):
                checked += 1
                token = "".join(letters)
                if not _looks_like_gibberish(token):
                    self.fail(f"пропущен набор букв одного ряда: {token!r} (ряд {row!r})")
        self.assertGreater(checked, 60_000, "проверено слишком мало сочетаний")

    def test_three_letter_row_sequence_is_not_single_row(self) -> None:
        """Короче четырёх букв правило ряда не срабатывает: слишком много живых слов."""
        for row in KEYBOARD_ROWS:
            for letters in itertools.product(row, repeat=3):
                token = "".join(letters)
                self.assertFalse(_single_keyboard_row(token), token)

    def test_words_from_two_rows_are_not_single_row(self) -> None:
        for word in ("отлично", "дорого", "покажи", "добавь", "стол", "кресло",
                     "лампа", "спасибо", "дешевле", "Таганрог"):
            with self.subTest(word=word):
                self.assertFalse(_single_keyboard_row(word.lower()), word)

    def test_yo_is_not_a_row_key(self) -> None:
        """«ё» вне рядов правила: иначе пострадают живые «орёл», «придёшь»."""
        for word in ("орёл", "придёшь", "ещё бы"):
            with self.subTest(word=word):
                self.assertFalse(_single_keyboard_row(word), word)

    def test_latin_row_mash_is_single_row(self) -> None:
        for token in ("asdf", "qwer", "zxcv", "QWER", "AsDf"):
            with self.subTest(token=token):
                self.assertTrue(_single_keyboard_row(token.lower()), token)

    def test_mixed_layout_word_is_not_single_row(self) -> None:
        for token in ("фываf", "asdfы", "qwertyй"):
            with self.subTest(token=token):
                self.assertFalse(_single_keyboard_row(token), token)


class GibberishCorpusTest(unittest.TestCase):
    """Мусор ловится, живая лексика — нет."""

    def test_bug_report_replies_are_gibberish(self) -> None:
        for text in ("аваыла", "фывафыва", "дваа", "фывапролд"):
            with self.subTest(text=text):
                self.assertTrue(is_unintelligible(text), text)

    def test_row_rolling_mash_is_gibberish(self) -> None:
        for text in ROW_MASH:
            with self.subTest(text=text):
                self.assertTrue(_looks_like_gibberish(text.lower()), text)
                self.assertTrue(is_unintelligible(text), text)

    def test_other_mash_shapes_are_gibberish(self) -> None:
        for text in OTHER_MASH:
            with self.subTest(text=text):
                self.assertTrue(is_unintelligible(text), text)

    def test_domain_vocabulary_is_not_gibberish(self) -> None:
        """200+ слов товарной лексики: ни одно не похоже на набор букв.

        Исключение одно и оно измерено: «здравствуйте» содержит законную
        русскую цепочку «вств», а правило согласных ловит 4 согласных подряд.
        На продукт это не влияет — слово в белом списке подтверждений, и
        :meth:`test_domain_vocabulary_is_understood` это проверяет.
        """
        for word in DOMAIN_WORDS:
            with self.subTest(word=word):
                if word in LEGAL_CONSONANT_CLUSTERS:
                    continue
                self.assertFalse(_looks_like_gibberish(word), word)

    def test_domain_vocabulary_is_understood(self) -> None:
        """Продуктовое утверждение: ни одно живое слово не считается мусором."""
        for word in DOMAIN_WORDS:
            with self.subTest(word=word):
                self.assertFalse(is_unintelligible(word), word)

    def test_live_replies_are_not_gibberish(self) -> None:
        for text in LIVE_REPLIES:
            with self.subTest(text=text):
                self.assertFalse(is_unintelligible(text), text)

    def test_conditions_are_never_mistaken_for_garbage(self) -> None:
        for text in CONDITION_REPLIES:
            with self.subTest(text=text):
                self.assertFalse(is_unintelligible(text), text)

    def test_catalog_text_has_only_known_gibberish(self) -> None:
        """Снимок долга: новых «мусорных» слов в товарном тексте быть не должно."""
        vocabulary = catalog_vocabulary()
        self.assertGreater(len(vocabulary), 500, "каталог разобрался подозрительно мало")
        flagged = frozenset(word for word in vocabulary if _looks_like_gibberish(word))
        self.assertEqual(
            flagged - KNOWN_CATALOG_GIBBERISH,
            frozenset(),
            "появились новые «мусорные» слова из каталога — правило стало шире",
        )

    def test_two_row_rolling_mash_is_out_of_scope(self) -> None:
        """Осознанная граница: прогон по двум рядам («фывайцу») не ловится.

        Признак «четыре соседние клавиши подряд» поймал бы его, но на словаре
        русского языка даёт 3321 ложное срабатывание (0,22 %): «пролёт»,
        «апробация», «альтиметр». Один ряд как признак стоит 1341 (0,09 %) и при
        этом ловит все формы из отчёта, поэтому правило остаётся на одном ряду.
        """
        for token in ("фывайцу", "авайцу", "йфяцыч"):
            with self.subTest(token=token):
                self.assertFalse(_looks_like_gibberish(token), token)


class CompositionTest(unittest.TestCase):
    """Переспрашиваем только тогда, когда мусором является вся реплика."""

    def test_garbage_alone_is_unintelligible(self) -> None:
        for text in ("аваыла", "фывафыва", "ыаллфдылааввы", "фффффф"):
            with self.subTest(text=text):
                self.assertTrue(is_unintelligible(text), text)

    def test_two_garbage_tokens_are_still_unintelligible(self) -> None:
        for text in ("аваыла фывафыва", "фывапролд цукен", "аваыла, фывафыва"):
            with self.subTest(text=text):
                self.assertTrue(is_unintelligible(text), text)

    def test_garbage_next_to_word_of_normal_shape_is_not_unintelligible(self) -> None:
        """Рядом есть слово обычной формы — реплика осмысленна, сброса нет.

        «правда», «дрова», «вода» по форме похожи на набор букв (весь ряд
        клавиатуры), но в живой реплике стоят рядом с обычными словами.
        """
        for text in ("Правда, дорого", "дрова не нужны", "вода есть"):
            with self.subTest(text=text):
                self.assertFalse(is_unintelligible(text), text)

    def test_garbage_with_number_is_an_answer(self) -> None:
        for text in ("аваыла 5000", "5000 аваыла", "фывапролд 15 тысяч"):
            with self.subTest(text=text):
                self.assertFalse(is_unintelligible(text), text)

    def test_garbage_with_condition_is_not_unintelligible(self) -> None:
        for text in ("фыва в Азове", "аваыла не шире метра", "цукен до 10000"):
            with self.subTest(text=text):
                self.assertFalse(is_unintelligible(text), text)

    def test_garbage_with_confirmation_is_not_unintelligible(self) -> None:
        for text in ("аваыла лады", "фывафыва спасибо"):
            with self.subTest(text=text):
                self.assertFalse(is_unintelligible(text), text)

    def test_short_mash_is_below_threshold(self) -> None:
        """3 буквы — не мусор: «ок», «ну», «э» слишком шумные."""
        for text in ("фыв", "ава", "ячс"):
            with self.subTest(text=text):
                self.assertFalse(is_unintelligible(text), text)


class LiveRepliesKeepTheSelectionTest(unittest.TestCase):
    """Живая реплика после полного подбора не должна обнулять категории."""

    def test_reactions_keep_the_selection(self) -> None:
        broken: list[tuple[str, str, list[str]]] = []
        for text in LIVE_REPLIES:
            document = run([BASE, text])
            if document.get("status") != "ok":
                broken.append((text, str(document.get("status")), fields(document)))
        self.assertEqual(broken, [], "живые реплики потеряли подбор")

    def test_reactions_keep_options_non_empty(self) -> None:
        for text in ("лады", "отлично", "Правда, дорого", "пока хватит", "спасибо"):
            with self.subTest(text=text):
                document = run([BASE, text])
                self.assertEqual(document.get("status"), "ok", document)
                self.assertTrue(document["options"], "ответ ok без вариантов")

    def test_service_word_cannot_become_a_product_or_city(self) -> None:
        """Служебное слово не может стать товаром через нечёткий поиск.

        «пока» ↔ «полка», «стоп» ↔ «стол», «привет» ↔ «свет» (лампа),
        «просто» ↔ «Ростов-на-Дону» — расстояние 1–2, и без защиты реплика
        «просто покажи» молча меняла город на Ростов.
        """
        lexicon = get_lexicon()
        service_words = FILLER_WORDS | CONFIRMATION_WORDS | NOISE_WORDS
        for word in sorted(service_words):
            for name, scan in (
                ("city", lexicon.scan_cities),
                ("category", lexicon.scan_categories),
                ("unsupported_city", lexicon.scan_unsupported_cities),
                ("unsupported_category", lexicon.scan_unsupported_categories),
            ):
                with self.subTest(word=word, scan=name):
                    self.assertEqual(scan([word]), (), f"«{word}» стало {name}")

    def test_service_word_does_not_change_the_selection(self) -> None:
        for text in ("просто покажи", "пока хватит", "привет", "стоп"):
            with self.subTest(text=text):
                document = run([BASE, text])
                self.assertEqual(document.get("status"), "ok", document)


class GarbageDialogueTest(unittest.TestCase):
    """Сквозная реплика покупателя: мусор переспрашивает только категории."""

    def assert_asks_categories(self, document: dict) -> None:
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertNotIn("options", document, "в clarify комплектов быть не может")
        self.assertEqual(fields(document), ["categories"], document)
        for question in document["questions"]:
            self.assertTrue(question["text"].strip(), "вопрос без текста")

    def test_every_garbage_reply_asks_categories(self) -> None:
        for text in GARBAGE_REPLIES:
            with self.subTest(text=text):
                self.assert_asks_categories(run([BASE, text]))

    def test_city_and_budget_are_not_asked_again(self) -> None:
        document = run([BASE, "аваыла"])
        asked = fields(document)
        self.assertNotIn("city_id", asked, "город уже назван")
        self.assertNotIn("budget_rub", asked, "бюджет уже назван")

    def test_garbage_without_conditions_asks_everything(self) -> None:
        self.assertEqual(
            fields(run(["Хочу мебель", "аваыла"])),
            ["budget_rub", "city_id", "categories"],
        )

    def test_garbage_keeps_missing_city(self) -> None:
        self.assertEqual(fields(run(["Стол до 5000.", "аваыла"])), ["city_id", "categories"])

    def test_garbage_in_the_middle_does_not_reset_the_last_answer(self) -> None:
        document = run([BASE, "аваыла", "стол"])
        self.assertEqual(document.get("status"), "ok", document)
        self.assertTrue(document["options"], "ответ ok без вариантов")

    def test_garbage_recovers_on_the_next_answer(self) -> None:
        first = run([BASE, "аваыла"])
        self.assert_asks_categories(first)
        document = run([BASE, "аваыла", "стол"])
        self.assertEqual(document.get("status"), "ok", document)

    def test_garbage_with_number_keeps_the_selection(self) -> None:
        document = run([BASE, "аваыла 5000"])
        self.assertEqual(document.get("status"), "ok", document)

    def test_garbage_does_not_create_a_match_from_nothing(self) -> None:
        """Мусор не должен молча превращаться в подбор по старым условиям."""
        document = run(["Стол до 900. Таганрог.", "аваыла"])
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertIn("categories", fields(document))

    def test_garbage_resets_categories_and_nothing_else(self) -> None:
        """Сброс точечный: ширина, температура и регулировка выживают.

        Сброс «всего состояния» был бы проще, но он наказывает покупателя за
        одну непонятную реплику и заставляет заново называть три условия.
        Проверяем не только список вопросов, но и сам подбор после ответа.
        """
        base = (
            "Стол не шире 80 см, лампа 4000 K, кресло с регулировкой. "
            "Таганрог. До 20000."
        )
        document = run([base, "аваыла"])
        self.assert_asks_categories(document)

        recovered = run([base, "аваыла", "стол и кресло"])
        self.assertEqual(recovered.get("status"), "ok", recovered)
        by_id = {listing["id"]: listing for listing in DOCUMENT}
        for option in recovered["options"]:
            for listing_id in option["listing_ids"]:
                listing = by_id[listing_id]
                attributes = listing["attributes"]
                if listing["category"] == "table":
                    self.assertLessEqual(attributes["width_cm"], 80, "ширина потерялась")
                if listing["category"] == "chair":
                    self.assertTrue(
                        attributes["adjustable_height"], "регулировка потерялась"
                    )

    def test_garbage_inside_one_message_with_a_condition_is_not_garbage(self) -> None:
        """Реплика «аваыла, нужен стол в Таганроге до 5000» — это ответ, не мусор."""
        document = run(["аваыла, нужен стол в Таганроге до 5000"])
        self.assertEqual(document.get("status"), "ok", document)
        self.assertTrue(document["options"], "ответ ok без вариантов")

    def test_garbage_before_conditions_does_not_leak_into_them(self) -> None:
        document = run(["аваыла", BASE])
        self.assertEqual(document.get("status"), "ok", document)

    def test_assistant_reply_between_messages_changes_nothing(self) -> None:
        request = parse_request(
            {
                "messages": [
                    {"role": "user", "text": BASE},
                    {"role": "assistant", "text": "Вот варианты."},
                    {"role": "user", "text": "аваыла"},
                ]
            }
        )
        document = answer(request=request, index=INDEX, settings=SETTINGS, deadline=1e18)
        self.assert_asks_categories(document)


class CliEndToEndTest(unittest.TestCase):
    """То же поведение в отдельном процессе — как в автопроверке."""

    @staticmethod
    def payload(reply: str) -> dict:
        return {
            "messages": [
                {"role": "user", "text": BASE},
                {"role": "assistant", "text": "Готово."},
                {"role": "user", "text": reply},
            ]
        }

    def test_cli_asks_categories_for_garbage(self) -> None:
        result = run_cli(self.payload("аваыла"))
        self.assertEqual(result.code, 0, result.stderr)
        document = result.json()
        self.assertIsInstance(document, dict)
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertEqual(fields(document), ["categories"], document)
        self.assertNotIn("Traceback", result.stderr, "CLI не должен падать")

    def test_cli_keeps_the_selection_for_live_reaction(self) -> None:
        result = run_cli(self.payload("Правда, дорого"))
        self.assertEqual(result.code, 0, result.stderr)
        document = result.json()
        self.assertEqual(document.get("status"), "ok", document)
        self.assertTrue(document["options"], "ответ ok без вариантов")

    def test_cli_is_deterministic(self) -> None:
        first = run_cli(self.payload("аваыла"))
        second = run_cli(self.payload("аваыла"))
        self.assertEqual(first.code, 0)
        self.assertEqual(first.stdout, second.stdout, "один запрос — два разных ответа")


class RobustnessTest(unittest.TestCase):
    """Мусор нестандартной формы: регистр, пунктуация, длина, детерминизм."""

    def test_case_and_punctuation_do_not_change_the_verdict(self) -> None:
        variants = {
            "аваыла": ("АВАЫЛА", "Аваыла", "аваыла!", "«аваыла»", "  аваыла  ",
                        "аваыла...", "(аваыла)"),
            "фывапролд": ("ФЫВАПРОЛД", "фывапролд?", "фывапролд!"),
        }
        for base, texts in variants.items():
            for text in (base,) + texts:
                with self.subTest(text=text):
                    self.assertTrue(is_unintelligible(text), text)

    def test_emoji_and_symbols_do_not_break_the_verdict(self) -> None:
        for text in ("аваыла 🛋️", "🛋️ аваыла", "аваыла ###", "аваыла ..."):
            with self.subTest(text=text):
                self.assertTrue(is_unintelligible(text), text)

    def test_symbols_only_is_not_garbage(self) -> None:
        """Реплика без букв не подтверждает условий — но и мусором не является."""
        for text in ("🛋️", "!!", "???", "—", "\u200b\u200b", "  "):
            with self.subTest(text=text):
                self.assertFalse(is_unintelligible(text), text)

    def test_long_mash_is_linear_and_detected(self) -> None:
        """100 000 символов: детектор обязан остаться линейным и не упасть."""
        text = "фы" * 50_000
        started = time.perf_counter()
        self.assertTrue(is_unintelligible(text))
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 5.0, f"слишком долго: {elapsed:.2f} с")

    def test_long_live_text_is_not_garbage(self) -> None:
        text = " ".join(["нужен стол и кресло"] * 500)
        self.assertFalse(is_unintelligible(text))

    def test_verdict_is_deterministic(self) -> None:
        for text in ("аваыла", "Правда, дорого", "аваыла 5000"):
            verdicts = {is_unintelligible(text) for _ in range(50)}
            with self.subTest(text=text):
                self.assertEqual(len(verdicts), 1, verdicts)

    def test_detector_does_not_mutate_the_request(self) -> None:
        """Повторный запрос с тем же текстом даёт тот же ответ (нет скрытого состояния)."""
        for text in ("аваыла", "фывапролд"):
            with self.subTest(text=text):
                first = run([BASE, text])
                second = run([BASE, text])
                self.assertEqual(first, second)


class MashMutationTest(unittest.TestCase):
    """Мусор остаётся мусором после косметических изменений реплики."""

    def test_mash_survives_case_reverse_and_repetition(self) -> None:
        for token in ROW_MASH:
            mutations = {
                "верхний регистр": token.upper(),
                "обратный порядок": token[::-1],
                "удвоение букв": "".join(letter * 2 for letter in token),
                "повтор до длины 40": (token * 4)[:40],
            }
            for name, text in mutations.items():
                with self.subTest(token=token, mutation=name):
                    self.assertTrue(is_unintelligible(text), text)

    def test_mash_survives_truncation_to_four_letters(self) -> None:
        """Четыре буквы — нижняя граница детектора, и она тоже работает."""
        for token in ROW_MASH:
            text = token[:4]
            with self.subTest(token=token, text=text):
                self.assertGreaterEqual(len(text), 4)
                self.assertTrue(is_unintelligible(text), text)

    def test_mash_inside_quotes_or_parentheses_is_still_garbage(self) -> None:
        for token in ("аваыла", "фывапролд", "ыаллфдылааввы"):
            for wrapper in ("«{}»", "\"{}\"", "({})", "[{}]", "— {} —"):
                text = wrapper.format(token)
                with self.subTest(text=text):
                    self.assertTrue(is_unintelligible(text), text)


class ReferenceRuleTest(unittest.TestCase):
    """Правило ряда сверяется с независимой реализацией по тем же рядам."""

    @staticmethod
    def reference(token: str) -> bool:
        rows = [row for row in KEYBOARD_ROWS if all(letter in row for letter in token)]
        return len(token) >= 4 and len(rows) == 1

    def test_random_tokens_match_the_reference(self) -> None:
        import random

        rng = random.Random(20261121)
        alphabet = "абвгдежзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyzё"
        samples: list[str] = []
        for _ in range(1500):
            length = rng.randint(1, 12)
            samples.append("".join(rng.choice(alphabet) for _ in range(length)))
        for _ in range(1500):
            row = rng.choice(KEYBOARD_ROWS)
            length = rng.randint(1, 12)
            samples.append("".join(rng.choice(row) for _ in range(length)))
        mismatched = [
            token for token in samples if _single_keyboard_row(token) != self.reference(token)
        ]
        self.assertEqual(mismatched[:10], [], f"расхождений: {len(mismatched)}")

    def test_random_row_tokens_of_four_letters_are_always_garbage(self) -> None:
        import random

        rng = random.Random(42)
        for _ in range(2000):
            row = rng.choice(KEYBOARD_ROWS)
            token = "".join(rng.choice(row) for _ in range(rng.randint(4, 10)))
            with self.subTest(token=token):
                self.assertTrue(_looks_like_gibberish(token), token)


class LongDialogueTest(unittest.TestCase):
    """Мусор в длинной истории не копит ошибку и не ломает восстановление."""

    def test_alternating_garbage_and_answers(self) -> None:
        """Четыре пары «мусор → ответ»: история целиком проходит контракт."""
        texts = [BASE]
        for index in range(4):
            texts.append(GARBAGE_REPLIES[index % len(GARBAGE_REPLIES)])
            texts.append("стол" if index % 2 else "добавь кресло")
        self.assertLessEqual(len(texts) * 2 - 1, 20, "контракт разрешает не больше 20 сообщений")
        document = run(texts)
        self.assertEqual(document.get("status"), "ok", document)
        self.assertTrue(document["options"], "ответ ok без вариантов")

    def test_garbage_after_unknown_city_asks_city_and_categories(self) -> None:
        document = run(["Стол до 5000, Москва", "аваыла"])
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertEqual(sorted(fields(document)), ["categories", "city_id"])

    def test_garbage_when_budget_is_below_minimum(self) -> None:
        document = run(["Стол, кресло и лампа. Таганрог. До 900", "аваыла"])
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertEqual(sorted(fields(document)), ["budget_rub", "categories"])

    def test_cache_does_not_change_the_verdict(self) -> None:
        import tempfile
        from pathlib import Path

        from app.nlp.cache import LexiconCache

        with tempfile.TemporaryDirectory(prefix="stress-cache-") as folder:
            cache = LexiconCache(Path(folder), enabled=True)
            cold = run([BASE, "аваыла"], cache=cache)
            warm = run([BASE, "аваыла"], cache=cache)
        self.assertEqual(cold, warm, "кэш разборов изменил ответ")
        self.assertEqual(cold.get("status"), "clarify", cold)
        self.assertGreater(cache.hits, 0, "кэш не использовался — тест проверяет не то")

    def test_repeated_garbage_keeps_asking_categories(self) -> None:
        document = run([BASE] + ["аваыла"] * 5)
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertEqual(fields(document), ["categories"])


class ZeroTokensTest(unittest.TestCase):
    """Мусор не стоит ни одного токена модели — даже когда модель доступна."""

    @staticmethod
    def online_settings() -> Settings:
        return Settings.load(
            environ={
                "LLM_BASE_URL": "https://stress.invalid/v1",
                "LLM_MODEL": "google/gemma-4-31B-it",
                "LLM_API_KEY": "stress-no-network",
                "ASSISTANT_OFFLINE": "0",
            }
        )

    def test_garbage_does_not_add_model_calls(self) -> None:
        """Детектор и сброс категорий локальные: вызовов модели не прибавляется."""
        settings = self.online_settings()
        self.assertTrue(settings.llm_available, "нужен режим с доступной моделью")

        def count(texts: list[str]) -> list[str]:
            calls: list[str] = []

            def resolver(text: str, slots: object) -> dict:
                calls.append(text)
                return {}

            run(texts, settings=settings, resolver=resolver)
            return calls

        first = ["Нужна деревяшка. Таганрог. До 15000."]
        without = count(first)
        with_garbage = count(first + ["аваыла"])
        self.assertGreater(len(without), 0, "гейт не позвал модель — проверять нечего")
        self.assertEqual(with_garbage, without, "мусор изменил текст для модели")

    def test_garbage_never_reaches_the_model(self) -> None:
        """Если модель увидит мусор — тест падает, а не «ну почти»."""
        settings = self.online_settings()
        garbage = set(GARBAGE_REPLIES)
        seen: list[str] = []

        def resolver(text: str, slots: object) -> dict:
            seen.append(text)
            return {}

        for text in GARBAGE_REPLIES:
            with self.subTest(text=text):
                run([BASE, text], settings=settings, resolver=resolver)
                run([text], settings=settings, resolver=resolver)
        for prompt in seen:
            for text in garbage:
                self.assertNotIn(text, prompt, f"мусор в промпте модели: {text!r}")


class FuzzSafetyTest(unittest.TestCase):
    """Случайный юникодный мусор: без исключений, ответ детерминирован."""

    JUNK_POOL = (
        "абвгдежзийклмнопрстуфхцчшщъыьэюяёabcxyzABCXYZ0123456789"
        " !?.,;:-—()[]{}«»\"'/\\|@#$%^&*+=~`"
        "🛋️🪑💡📦✅❌\u200b\u200c\u200d\u202e\u0301\u00a0\t\n"
    )

    def test_random_junk_never_raises(self) -> None:
        import random

        rng = random.Random(777)
        for _ in range(500):
            text = "".join(rng.choice(self.JUNK_POOL) for _ in range(rng.randint(0, 40)))
            with self.subTest(text=text[:20]):
                verdict = is_unintelligible(text)
                self.assertIsInstance(verdict, bool)
                self.assertEqual(verdict, is_unintelligible(text), "ответ не детерминирован")

    def test_random_cyrillic_words_do_not_crash_the_pipeline(self) -> None:
        import random

        rng = random.Random(778)
        alphabet = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
        texts = [
            "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
            for _ in range(30)
        ]
        for text in texts:
            with self.subTest(text=text):
                document = run([BASE, text])
                status = document.get("status")
                self.assertIn(status, {"ok", "clarify", "no_match"}, document)
                if status == "ok":
                    self.assertTrue(document.get("options"), "ответ ok без вариантов")
                if status == "clarify":
                    self.assertTrue(document.get("questions"), "clarify без вопросов")
                    self.assertNotIn("options", document, "в clarify комплектов быть не может")


if __name__ == "__main__":
    unittest.main()
