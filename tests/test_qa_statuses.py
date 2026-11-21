"""Блок 6 — статусы ``clarify`` и ``no_match`` (CONTRACT §4).

Проверяется смысл, а не формулировки: текст вопросов и объяснений свободный
(§5), поэтому сравниваются поля, состав и числа из каталога.

Правила §4, зафиксированные здесь:

``clarify``
    * спрашиваются **все** недостающие основные условия (бюджет, город,
      категории) и каждое — ровно один раз;
    * комплектов в этом статусе нет;
    * город и бюджет не подставляются по умолчанию;
    * неподдерживаемая категория или город вне справочника — повод уточнить
      соответствующее поле;
    * при нескольких неизвестных условиях сначала возвращается ``clarify``,
      даже если по известным условиям комплекта заведомо нет.

``no_match``
    * есть непустое сообщение, ID объявлений отсутствуют;
    * названо препятствие (нехватка бюджета, жёсткое ограничение, пустая
      категория) и предложено изменить условие;
    * суммы в сообщении — реальные значения из каталога;
    * ``no_match`` не используется вместо ``clarify`` и не подменяет технический
      сбой (для сбоя — ненулевой код и пустой stdout, см. блок 1).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.test_qa_support import (
    EXIT_OK,
    ask,
    assert_clarify_covers,
    index_by_id,
    load_catalog,
    load_cities,
    make_temp_dir,
    request,
    run_shared,
    user,
    assistant,
    write_catalog,
)

#: Запросы, у которых известны не все основные условия: любой из них обязан
#: получить ``clarify``, а не ``ok`` и не ``no_match``.
INCOMPLETE_CONDITIONS: tuple[str, ...] = (
    "Стол",
    "Стол в Таганроге",
    "Стол в Азове",
    "Кресло",
    "Таганрог до 15000",
    "до 15000 ₽",
    "Стол до 15000 ₽",
    "Хочу мебель",
    "стол кресло лампа",
    "ааа",
    "Купи",
    "?",
)

_RESPONSES: dict[tuple[str, str], dict[str, Any]] = {}
_WORKDIR: tempfile.TemporaryDirectory[str] | None = None


def setUpModule() -> None:
    global _WORKDIR
    _WORKDIR = tempfile.TemporaryDirectory(prefix="qa-statuses-")


def tearDownModule() -> None:
    if _WORKDIR is not None:
        _WORKDIR.cleanup()
    _RESPONSES.clear()


def document_for(query: str, *, catalog: Path | str | None = None) -> dict[str, Any]:
    """Документ ответа на запрос; одинаковый запрос запускается ровно один раз."""
    key = (str(catalog or ""), query)
    if key not in _RESPONSES:
        result = run_shared(ask(query), catalog=catalog)
        if result.code != EXIT_OK:
            raise AssertionError(f"запрос {query!r}: код {result.code}, stderr: {result.stderr[-400:]}")
        _RESPONSES[key] = result.one_json_object()
    return _RESPONSES[key]


def dialogue(*messages: Mapping[str, str]) -> dict[str, Any]:
    """Ответ на диалог (для проверки «уточнили — ответили»)."""
    key = ("dialogue", "|".join(f"{message['role']}:{message['text']}" for message in messages))
    if key not in _RESPONSES:
        result = run_shared(request(*messages))
        if result.code != EXIT_OK:
            raise AssertionError(f"диалог {key[1]!r}: код {result.code}, stderr: {result.stderr[-400:]}")
        _RESPONSES[key] = result.one_json_object()
    return _RESPONSES[key]


def asked_fields(document: Mapping[str, Any]) -> set[str]:
    """Поля вопросов ``clarify`` как множество."""
    return {str(question["field"]) for question in document.get("questions", [])}


class AmbiguousQueryTest(unittest.TestCase):
    """Вопросы боевого пути соответствуют явно названным основным условиям."""

    def test_only_missing_primary_fields_are_asked(self) -> None:
        for text, fields in (
            ("В Азовском, стол до 5000", {"city_id"}),
            ("Стол в Таганроге, но лучше в Туле, до 5000", {"city_id"}),
            ("Лампа от 3000 до 5000, Таганрог", {"budget_rub"}),
            ("Стол 142 в Азове", {"budget_rub"}),
            ("До 1500 кресел в Азове", {"budget_rub"}),
            ("Стол и белая полоса сверху", {"city_id", "budget_rub"}),
        ):
            with self.subTest(text=text):
                document = document_for(text)
                self.assertEqual(document["status"], "clarify")
                self.assertEqual(asked_fields(document), fields)
                self.assertEqual(len(document["questions"]), len(fields))


def plain(text: str) -> str:
    """Текст без неразрывных пробелов: суммы в сообщениях пишутся через ``\\xa0``."""
    return text.replace("\u00a0", " ")


def formatted_price(value: int) -> str:
    """Сумма в том виде, в каком её печатает решение: 2 500 ₽ (разряды через NBSP)."""
    return f"{value:,}".replace(",", "\u00a0")


class CatalogFacts(unittest.TestCase):
    """Общие данные: каталог, справочник городов, минимумы по парам."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()
        cls.by_id = index_by_id(cls.catalog)
        cls.cities = load_cities()

    def minimum(self, city_id: str, category: str) -> int:
        """Самая низкая цена категории в городе по каталогу."""
        prices = [
            int(item["price_rub"])
            for item in self.catalog
            if item["city_id"] == city_id and item["category"] == category
        ]
        self.assertTrue(prices, f"в каталоге нет пары {city_id}/{category}")
        return min(prices)

    def minimum_bundle(self, city_id: str, categories: Sequence[str]) -> int:
        """Сумма минимумов: дешевле этого комплект стоить не может."""
        return sum(self.minimum(city_id, category) for category in categories)

    def catalog_without(self, city_id: str, category: str) -> Path:
        """Копия каталога без одной пары city/category (во временном каталоге)."""
        workdir = make_temp_dir(self)
        return write_catalog(
            workdir,
            [
                item
                for item in self.catalog
                if not (item["city_id"] == city_id and item["category"] == category)
            ],
        )


class ClarifyCompletenessTest(CatalogFacts):
    """§4: спрашиваются все недостающие основные условия, каждое один раз."""

    def test_only_budget_is_asked_when_city_and_categories_are_known(self) -> None:
        assert_clarify_covers(self, document_for("Стол в Таганроге"), ["budget_rub"])

    def test_only_city_is_asked_when_budget_and_categories_are_known(self) -> None:
        assert_clarify_covers(self, document_for("Стол до 15000 ₽"), ["city_id"])

    def test_only_categories_are_asked_when_budget_and_city_are_known(self) -> None:
        assert_clarify_covers(self, document_for("Таганрог до 15000"), ["categories"])

    def test_two_missing_conditions_are_asked_together(self) -> None:
        assert_clarify_covers(self, document_for("Стол"), ["budget_rub", "city_id"])
        assert_clarify_covers(self, document_for("до 15000 ₽"), ["city_id", "categories"])

    def test_all_three_conditions_are_asked_at_once(self) -> None:
        """Три вопроса сразу: помощник не заставляет угадывать по одному кругу."""
        assert_clarify_covers(
            self,
            document_for("Хочу мебель"),
            ["budget_rub", "city_id", "categories"],
        )

    def test_meaningless_text_asks_all_three(self) -> None:
        assert_clarify_covers(
            self,
            document_for("ааа"),
            ["budget_rub", "city_id", "categories"],
        )

    def test_clarify_contains_no_bundles(self) -> None:
        document = document_for("Стол")
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertNotIn("options", document, "в clarify не может быть вариантов")
        self.assertNotIn("message", document, "у clarify нет поля message")

    def test_each_question_has_a_field_and_a_nonempty_text(self) -> None:
        document = document_for("Хочу мебель")
        for question in document["questions"]:
            with self.subTest(field=question.get("field")):
                self.assertEqual(set(question), {"field", "text"}, "у вопроса ровно два поля")
                self.assertIsInstance(question["text"], str)
                self.assertTrue(question["text"].strip(), "текст вопроса пуст")


class ClarifyNoDefaultsTest(CatalogFacts):
    """§4: «Не подставляйте город или бюджет по умолчанию»."""

    def test_unknown_city_is_not_replaced_by_a_default(self) -> None:
        document = document_for("Стол до 15 000 ₽")
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertEqual(asked_fields(document), {"city_id"})

    def test_unknown_budget_is_not_replaced_by_a_default(self) -> None:
        document = document_for("Стол в Азове")
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertEqual(asked_fields(document), {"budget_rub"})

    def test_unknown_categories_are_not_replaced_by_a_default(self) -> None:
        document = document_for("Таганрог до 15000")
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertEqual(asked_fields(document), {"categories"})

    def test_budget_question_mentions_the_budget(self) -> None:
        document = document_for("Стол в Таганроге")
        question = next(item for item in document["questions"] if item["field"] == "budget_rub")
        self.assertIn("бюджет", question["text"].lower(), "вопрос о бюджете должен быть о бюджете")

    def test_city_question_offers_the_directory_cities(self) -> None:
        """Смысл вопроса — выбор города, значит в тексте должны быть города справочника."""
        document = document_for("Стол до 15000 ₽")
        question = next(item for item in document["questions"] if item["field"] == "city_id")
        mentioned = [name for name in self.cities.values() if name in question["text"]]
        self.assertTrue(mentioned, f"в вопросе о городе нет названий городов: {question['text']!r}")

    def test_categories_question_names_supported_categories(self) -> None:
        document = document_for("Таганрог до 15000")
        question = next(item for item in document["questions"] if item["field"] == "categories")
        text = question["text"].lower()
        self.assertTrue(
            any(word in text for word in ("стол", "кресл", "ламп")),
            f"в вопросе о категориях нет поддерживаемых товаров: {question['text']!r}",
        )

    def test_known_conditions_are_not_asked(self) -> None:
        """Названные условия не переспрашиваются (§3 + §4)."""
        document = document_for("Таганрог до 15000")
        fields = asked_fields(document)
        self.assertNotIn("city_id", fields, "город назван — спрашивать нельзя")
        self.assertNotIn("budget_rub", fields, "бюджет назван — спрашивать нельзя")


class ClarifyUnknownValuesTest(CatalogFacts):
    """§4: неподдерживаемая категория и город вне справочника — уточнение поля."""

    def test_city_outside_directory_is_asked_as_city(self) -> None:
        document = document_for("Стол в Москве до 15 000 ₽")
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertEqual(asked_fields(document), {"city_id"})

    def test_unsupported_category_is_asked_as_categories(self) -> None:
        document = document_for("Диван в Таганроге до 15 000 ₽")
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertEqual(asked_fields(document), {"categories"})

    def test_supported_and_unsupported_categories_together_ask_categories(self) -> None:
        """«Стол и диван» — состав неоднозначен, подбирать «стол вместо дивана» нельзя."""
        for query in (
            "Стол и диван в Таганроге до 15 000 ₽",
            "Стол, диван и лампа в Таганроге до 15 000 ₽",
            "Стол и шкаф в Таганроге до 15 000 ₽",
        ):
            with self.subTest(query=query):
                document = document_for(query)
                self.assertEqual(document.get("status"), "clarify", document)
                self.assertEqual(asked_fields(document), {"categories"})

    def test_unknown_city_and_unknown_category_ask_all_missing_fields(self) -> None:
        document = document_for("Диван в Москве")
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertEqual(asked_fields(document), {"budget_rub", "city_id", "categories"})

    def test_unknown_city_with_known_budget_asks_only_city(self) -> None:
        document = document_for("Стол в Москве до 100 ₽")
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertEqual(
            asked_fields(document),
            {"city_id"},
            "бюджет известен, а невыполнимость по бюджету проверяется только после уточнения города",
        )


class ClarifyPriorityTest(CatalogFacts):
    """§4: «При нескольких неизвестных основных условиях сначала возвращается clarify»."""

    def test_unknown_budget_wins_over_an_impossible_constraint(self) -> None:
        """Стола шириной ровно 90 см в каталоге нет, но бюджет не назван — спрашиваем бюджет."""
        document = document_for("Стол шириной ровно 90 см в Таганроге")
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertEqual(asked_fields(document), {"budget_rub"})

    def test_two_unknown_conditions_win_over_an_impossible_constraint(self) -> None:
        document = document_for("Стол шириной ровно 90 см")
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertEqual(asked_fields(document), {"budget_rub", "city_id"})

    def test_unknown_budget_wins_over_an_empty_category(self) -> None:
        """В Азове нет ламп вообще, но бюджет не назван — сначала уточнение."""
        catalog = self.catalog_without("azov", "lamp")
        document = document_for("Лампа в Азове", catalog=catalog)
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertEqual(asked_fields(document), {"budget_rub"})

    def test_every_incomplete_query_is_clarify_never_no_match(self) -> None:
        for query in INCOMPLETE_CONDITIONS:
            with self.subTest(query=query):
                document = document_for(query)
                self.assertEqual(
                    document.get("status"),
                    "clarify",
                    f"{query!r}: при неизвестных условиях ожидался clarify, получено {document.get('status')}",
                )


class NoMatchReasonTest(CatalogFacts):
    """§4: сообщение ``no_match`` называет препятствие и предлагает изменение."""

    def test_budget_shortage_names_the_amount_and_suggests_a_higher_budget(self) -> None:
        document = document_for("Стол, кресло и лампа в Азове до 1500 ₽")
        self.assertEqual(document.get("status"), "no_match", document)
        message = plain(document["message"])
        expected = self.minimum_bundle("azov", ("table", "chair", "lamp"))
        self.assertIn(
            formatted_price(expected).replace("\u00a0", " "),
            message,
            f"в сообщении нет минимальной суммы {expected} ₽ из каталога: {message!r}",
        )
        self.assertIn("бюджет", message.lower(), "не названо препятствие (бюджет)")

    def test_message_lists_the_minimum_of_every_requested_category(self) -> None:
        document = document_for("Стол, кресло и лампа в Азове до 1500 ₽")
        message = plain(document["message"])
        for category in ("table", "chair", "lamp"):
            with self.subTest(category=category):
                price = self.minimum("azov", category)
                self.assertIn(
                    formatted_price(price).replace("\u00a0", " "),
                    message,
                    f"нет минимальной цены категории {category} ({price} ₽): {message!r}",
                )

    def test_hard_width_constraint_names_the_nearest_value(self) -> None:
        document = document_for("Стол шириной ровно 90 см в Таганроге до 15 000 ₽")
        self.assertEqual(document.get("status"), "no_match", document)
        message = document["message"]
        self.assertIn("ширин", message.lower(), "не названо препятствие (ширина)")
        widths = sorted(
            int(item["attributes"]["width_cm"])
            for item in self.catalog
            if item["city_id"] == "taganrog" and item["category"] == "table" and "width_cm" in item["attributes"]
        )
        nearest = min(widths, key=lambda value: (abs(value - 90), value))
        self.assertIn(str(nearest), message, f"не подсказано ближайшее значение {nearest} см: {message!r}")

    def test_hard_temperature_constraint_names_the_nearest_value(self) -> None:
        document = document_for("Лампа 1000 K в Таганроге до 15 000 ₽")
        self.assertEqual(document.get("status"), "no_match", document)
        message = document["message"]
        self.assertIn("температур", message.lower(), "не названо препятствие (температура)")
        temperatures = sorted(
            int(item["attributes"]["color_temperature_k"])
            for item in self.catalog
            if item["city_id"] == "taganrog"
            and item["category"] == "lamp"
            and "color_temperature_k" in item["attributes"]
        )
        nearest = min(temperatures, key=lambda value: (abs(value - 1000), value))
        self.assertIn(str(nearest), message, f"не подсказано ближайшее значение {nearest} K: {message!r}")

    def test_closed_category_names_the_category_and_offers_alternatives(self) -> None:
        catalog = self.catalog_without("azov", "lamp")
        document = document_for("Лампа в Азове до 15 000 ₽", catalog=catalog)
        self.assertEqual(document.get("status"), "no_match", document)
        message = document["message"].lower()
        self.assertIn("ламп", message, "не названа категория, которой нет в городе")
        self.assertTrue(
            "город" in message or "категор" in message,
            f"не предложено изменить условие: {document['message']!r}",
        )

    def test_message_contains_no_listing_ids(self) -> None:
        for query in (
            "Стол в Таганроге до 100 ₽",
            "Стол, кресло и лампа в Азове до 1500 ₽",
            "Стол шириной ровно 90 см в Таганроге до 15 000 ₽",
            "Лампа 1000 K в Таганроге до 15 000 ₽",
        ):
            with self.subTest(query=query):
                document = document_for(query)
                message = document["message"]
                leaked = [listing_id for listing_id in self.by_id if listing_id in message]
                self.assertEqual(leaked, [], f"ID объявлений в no_match: {leaked}")

    def test_no_match_has_no_bundles_or_questions(self) -> None:
        document = document_for("Стол в Таганроге до 100 ₽")
        self.assertEqual(document.get("status"), "no_match", document)
        self.assertNotIn("options", document)
        self.assertNotIn("questions", document)

    def test_zero_budget_is_no_match_not_clarify(self) -> None:
        """Бюджет 0 назван явно (§3: неотрицательное целое), значит он известен."""
        document = document_for("Стол в Таганроге до 0 ₽")
        self.assertEqual(document.get("status"), "no_match", document)

    def test_no_match_explains_a_flag_constraint_too(self) -> None:
        """Ограничение по регулировке высоты тоже может отсечь всё."""
        document = document_for("Кресло с регулировкой высоты в Азове до 500 ₽")
        self.assertEqual(document.get("status"), "no_match", document)
        message = document["message"].lower()
        self.assertTrue(
            "бюджет" in message or "регулировк" in message,
            f"препятствие не названо: {document['message']!r}",
        )


class StatusFlowTest(CatalogFacts):
    """§4: после ответа на уточнение — комплект или no_match; условия не теряются."""

    def test_answer_after_clarify_gives_a_bundle(self) -> None:
        document = dialogue(
            user("Стол в Таганроге"),
            assistant("Какой у вас бюджет на весь комплект?"),
            user("до 5000"),
        )
        self.assertEqual(document.get("status"), "ok", document)

    def test_partial_answer_keeps_asking_only_the_rest(self) -> None:
        first = dialogue(user("до 5000 ₽"))
        self.assertEqual(first.get("status"), "clarify", first)
        self.assertEqual(asked_fields(first), {"city_id", "categories"})

        second = dialogue(
            user("до 5000 ₽"),
            assistant("В каком городе нужен комплект: Таганрог, Ростов-на-Дону или Азов?"),
            user("Азов"),
        )
        self.assertEqual(second.get("status"), "clarify", second)
        self.assertEqual(
            asked_fields(second),
            {"categories"},
            "после ответа про город бюджет спрашивать повторно нельзя",
        )

    def test_known_conditions_survive_a_no_match(self) -> None:
        """После no_match по бюджету поднятие бюджета даёт комплект в том же городе."""
        first = dialogue(user("Стол в Азове до 100 ₽"))
        self.assertEqual(first.get("status"), "no_match", first)

        second = dialogue(
            user("Стол в Азове до 100 ₽"),
            assistant(first["message"]),
            user("Хорошо, до 3000"),
        )
        self.assertEqual(second.get("status"), "ok", second)
        for option in second["options"]:
            for listing_id in option["listing_ids"]:
                self.assertEqual(self.by_id[listing_id]["city_id"], "azov", "город потерян после no_match")

    def test_unrelated_reply_after_clarify_does_not_invent_conditions(self) -> None:
        """Ответ не по теме не превращается в угаданный город или бюджет."""
        document = dialogue(
            user("Стол"),
            assistant("Какой у вас бюджет на весь комплект?"),
            user("не знаю"),
        )
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertNotIn("options", document)


if __name__ == "__main__":
    unittest.main()
