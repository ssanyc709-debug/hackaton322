"""Блок 2+3 — структура ответа (§4, схема) и смысл статуса ``ok``.

Проверяется две независимые вещи:

1. **Структура.** Каждый ответ — строгий JSON, проходящий
   ``_upstream/schemas/response.schema.json`` (Draft 2020-12): точные наборы
   ключей на всех уровнях, 1–3 варианта, 1–3 уникальных ID, ``total_rub`` —
   целое число, а не ``true``/строка.
2. **Смысл ``ok``.** Ни один ID не берётся на веру: все они ищутся в текущем
   каталоге, состав сверяется с запрошенными категориями, город — с городом
   покупателя, ``total_rub`` — с суммой ``price_rub`` по каталогу, бюджет — с
   ограничением покупателя. Проверяется КАЖДЫЙ вариант, а не только первый:
   контракт требует, чтобы допустимым был любой возвращённый комплект.

Чего здесь намеренно НЕТ: сравнения с эталонными ID из ``examples/01-selection``
(контракт §4 разрешает любой подходящий комплект), сравнения порядка вариантов
и текста ``explanation`` побайтово — формулировки свободные.
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tests.test_qa_support import (
    EXIT_OK,
    UPSTREAM_EXAMPLES,
    QaResult,
    ask,
    assert_documents_differ_by_composition,
    assert_option_matches_catalog,
    catalog_schema,
    index_by_id,
    load_catalog,
    option_ids,
    response_schema,
    run_shared,
    schema_errors,
)

#: Один запрос корпуса вместе с тем, что из него обязано следовать по контракту §3.
@dataclass(frozen=True)
class Bundle:
    """Запрос и его условия: категории, город, общий бюджет в рублях."""

    query: str
    categories: tuple[str, ...]
    city_id: str
    budget_rub: int


#: Корпус: покрыты все три города, все категории по одной, пары и полный комплект.
CASES: tuple[Bundle, ...] = (
    Bundle("Стол, кресло и лампа. Таганрог. До 15 000 ₽.", ("table", "chair", "lamp"), "taganrog", 15000),
    Bundle("Стол в Таганроге до 5000 ₽", ("table",), "taganrog", 5000),
    Bundle("Нужен стол не шире 100 см в Азове, бюджет 8000", ("table",), "azov", 8000),
    Bundle("Кресло с регулировкой высоты в Ростове-на-Дону до 5000", ("chair",), "rostov-na-donu", 5000),
    Bundle("Лампа в Таганроге до 2000 ₽", ("lamp",), "taganrog", 2000),
    Bundle("Стол и кресло в Таганроге до 10 000 ₽", ("table", "chair"), "taganrog", 10000),
    Bundle("Стол в Таганроге до 15 000 ₽ и лампа", ("table", "lamp"), "taganrog", 15000),
    Bundle("Кресло и лампа в Азове до 3000 ₽", ("chair", "lamp"), "azov", 3000),
    Bundle("Стол, кресло и лампа. Ростов-на-Дону. До 30 000 ₽.", ("table", "chair", "lamp"), "rostov-na-donu", 30000),
    Bundle("Кресло в Азове до 2000 ₽", ("chair",), "azov", 2000),
    Bundle("Стол, кресло и лампа в Азове до 20 000 ₽", ("table", "chair", "lamp"), "azov", 20000),
    Bundle("Нужна лампа в Ростове-на-Дону до 1000 ₽", ("lamp",), "rostov-na-donu", 1000),
    Bundle("Стол в Таганроге до 7000 ₽", ("table",), "taganrog", 7000),
)

#: Бюджет, точно равный самому дешёвому комплекту Таганрога (2500 + 1700 + 500).
#: Контракт §3: равенство бюджета допустимо, поэтому ответ обязан быть ``ok``.
EQUAL_BUDGET = Bundle("Стол, кресло и лампа. Таганрог. До 4700 ₽.", ("table", "chair", "lamp"), "taganrog", 4700)

#: Запросы, ответ на которые всегда обязан быть ``clarify``/``no_match``.
NON_OK_QUERIES: tuple[str, ...] = (
    "Стол в Таганроге",  # нет бюджета
    "Стол, кресло и лампа. До 15 000 ₽.",  # нет города
    "Стол, кресло и лампа. Таганрог. До 100 ₽.",  # бюджет заведомо мал
)

_RESPONSES: dict[str, QaResult] = {}
_WORKDIRS: list[tempfile.TemporaryDirectory[str]] = []


def setUpModule() -> None:
    """Один временный каталог кэша на модуль: процессы не зависят от ``data/cache``."""
    _WORKDIRS.append(tempfile.TemporaryDirectory(prefix="qa-response-"))


def tearDownModule() -> None:
    for holder in _WORKDIRS:
        holder.cleanup()
    _WORKDIRS.clear()
    _RESPONSES.clear()


def response_for(query: str) -> dict[str, Any]:
    """Ответ на запрос; каждый уникальный запрос запускается ровно один раз."""
    if query not in _RESPONSES:
        _RESPONSES[query] = run_shared(ask(query))
    result = _RESPONSES[query]
    if result.code != EXIT_OK:
        raise AssertionError(f"запрос {query!r}: код {result.code}, stderr: {result.stderr[-400:]}")
    return result.one_json_object()


def normalize_spaces(text: str) -> str:
    """Приводит неразрывные пробелы к обычным: числа в объяснениях форматируются ими."""
    return text.replace("\u00a0", " ").replace("\u202f", " ")


def money(price: int) -> str:
    """Цена в том же виде, в каком она попадает в объяснение: «2 500»."""
    return f"{price:,}".replace(",", " ")


def mentions_price(text: str, price: int) -> bool:
    """Есть ли в тексте именно это число (а не его часть внутри другого числа)."""
    pattern = rf"(?<!\d){re.escape(money(price))}(?!\d)"
    return re.search(pattern, normalize_spaces(text)) is not None


class ResponseSchemaTest(unittest.TestCase):
    """Структура ответа: схема ``_upstream`` и точные наборы ключей."""

    def test_every_ok_response_matches_upstream_schema(self) -> None:
        for bundle in CASES + (EQUAL_BUDGET,):
            with self.subTest(query=bundle.query):
                document = response_for(bundle.query)
                self.assertEqual(document.get("status"), "ok", document)
                self.assertEqual(schema_errors(document), [], "ответ не соответствует response.schema.json")

    def test_clarify_and_no_match_match_upstream_schema(self) -> None:
        for query in NON_OK_QUERIES:
            with self.subTest(query=query):
                document = response_for(query)
                self.assertIn(document.get("status"), {"clarify", "no_match"}, document)
                self.assertEqual(schema_errors(document), [], "ответ не соответствует response.schema.json")

    def test_upstream_example_response_is_accepted_by_the_schema(self) -> None:
        """Схема-источник истины должна принимать собственный пример кейса."""
        example = json.loads((UPSTREAM_EXAMPLES / "01-selection" / "response.json").read_text(encoding="utf-8"))
        self.assertEqual(schema_errors(example), [], "example response не проходит собственную схему")

    def test_validator_rejects_broken_documents(self) -> None:
        """Проверка самой проверки: битые ответы обязаны отсекаться, а не проходить молча."""
        broken = (
            {"status": "ok", "options": []},
            {"status": "ok", "options": [{"listing_ids": [], "total_rub": 1, "explanation": "x"}]},
            {"status": "ok", "options": [{"listing_ids": ["a", "a"], "total_rub": 1, "explanation": "x"}]},
            {"status": "ok", "options": [{"listing_ids": ["a"], "total_rub": 0, "explanation": "x"}]},
            {"status": "ok", "options": [{"listing_ids": ["a"], "total_rub": True, "explanation": "x"}]},
            {"status": "ok", "options": [{"listing_ids": ["a"], "total_rub": 1, "explanation": ""}]},
            {"status": "ok", "options": [{"listing_ids": ["a"], "total_rub": 1, "explanation": "x", "extra": 1}]},
            {"status": "ok", "options": [{"listing_ids": ["a"], "total_rub": 1}]},
            {"status": "ok"},
            {"status": "clarify", "questions": []},
            {"status": "clarify", "questions": [{"field": "city", "text": "?"}]},
            {"status": "no_match", "message": ""},
            {"status": "no_match", "message": "нет", "options": []},
            {"status": "ok", "options": [{"listing_ids": ["a"], "total_rub": 1, "explanation": "x"}], "message": "m"},
            {"status": "unknown"},
            [],
            "ok",
            None,
        )
        for document in broken:
            with self.subTest(document=document):
                self.assertTrue(schema_errors(document), f"битый документ прошёл проверку: {document!r}")

    def test_ok_has_exactly_the_documented_keys(self) -> None:
        for bundle in CASES:
            with self.subTest(query=bundle.query):
                document = response_for(bundle.query)
                self.assertEqual(set(document), {"status", "options"})
                self.assertIsInstance(document["options"], list)
                for option in document["options"]:
                    self.assertEqual(set(option), {"listing_ids", "total_rub", "explanation"})
                    self.assertIsInstance(option["listing_ids"], list)
                    self.assertIsInstance(option["explanation"], str)

    def test_total_rub_is_integer_not_boolean(self) -> None:
        """JSON различает ``1`` и ``true``; в Python ``isinstance(True, int)`` истинно."""
        for bundle in CASES:
            with self.subTest(query=bundle.query):
                for option in response_for(bundle.query)["options"]:
                    self.assertIs(type(option["total_rub"]), int)
                    self.assertGreaterEqual(option["total_rub"], 1)

    def test_clarify_has_questions_and_no_bundles(self) -> None:
        for query in NON_OK_QUERIES:
            with self.subTest(query=query):
                document = response_for(query)
                if document.get("status") != "clarify":
                    continue
                self.assertEqual(set(document), {"status", "questions"})
                self.assertNotIn("options", document)
                for question in document["questions"]:
                    self.assertEqual(set(question), {"field", "text"})
                    self.assertIn(question["field"], {"budget_rub", "city_id", "categories"})
                    self.assertTrue(question["text"].strip())

    def test_no_match_has_message_and_no_ids(self) -> None:
        query = "Стол, кресло и лампа. Таганрог. До 100 ₽."
        document = response_for(query)
        self.assertEqual(document.get("status"), "no_match", document)
        self.assertEqual(set(document), {"status", "message"})
        self.assertTrue(document["message"].strip(), "message не может быть пустым")
        self.assertNotIn("options", document)

    def test_catalog_of_this_repository_matches_catalog_schema(self) -> None:
        """Каталог, по которому считаются ожидания, сам обязан быть валидным."""
        self.assertEqual(schema_errors(load_catalog(), catalog_schema()), [])


class OkSemanticsTest(unittest.TestCase):
    """Смысл ``ok``: каждый вариант пересчитывается по каталогу, а не принимается на веру."""

    def test_every_option_is_a_valid_bundle(self) -> None:
        by_id = index_by_id(load_catalog())
        for bundle in CASES + (EQUAL_BUDGET,):
            with self.subTest(query=bundle.query):
                document = response_for(bundle.query)
                self.assertEqual(document.get("status"), "ok", document)
                self.assertTrue(1 <= len(document["options"]) <= 3, "вариантов должно быть 1..3")
                for index, option in enumerate(document["options"]):
                    with self.subTest(query=bundle.query, option=index):
                        assert_option_matches_catalog(
                            self,
                            option,
                            by_id=by_id,
                            categories=bundle.categories,
                            city_id=bundle.city_id,
                            budget_rub=bundle.budget_rub,
                        )

    def test_single_category_bundle_contains_exactly_one_item(self) -> None:
        """Один товар — тоже комплект (контракт §3), но ровно один."""
        for bundle in CASES:
            if len(bundle.categories) != 1:
                continue
            with self.subTest(query=bundle.query):
                for option in response_for(bundle.query)["options"]:
                    self.assertEqual(len(option_ids(option)), 1, "вариант из одной категории должен содержать один ID")

    def test_options_differ_by_composition(self) -> None:
        for bundle in CASES:
            with self.subTest(query=bundle.query):
                assert_documents_differ_by_composition(self, response_for(bundle.query)["options"])

    def test_no_option_lists_the_same_id_twice(self) -> None:
        for bundle in CASES:
            with self.subTest(query=bundle.query):
                for option in response_for(bundle.query)["options"]:
                    ids = option_ids(option)
                    self.assertEqual(len(ids), len(set(ids)), f"повтор ID внутри варианта: {ids}")

    def test_budget_equality_is_allowed(self) -> None:
        """Бюджет — верхний предел, равенство допустимо (контракт §3)."""
        document = response_for(EQUAL_BUDGET.query)
        self.assertEqual(document.get("status"), "ok", "комплект ровно за бюджет обязан подойти")
        totals = [option["total_rub"] for option in document["options"]]
        self.assertIn(EQUAL_BUDGET.budget_rub, totals, f"ни один вариант не использует бюджет целиком: {totals}")

    def test_explanation_mentions_prices_from_the_catalog(self) -> None:
        """``explanation`` — объяснение «с опорой на данные каталога», а не заглушка."""
        by_id = index_by_id(load_catalog())
        queries = (
            "Стол, кресло и лампа. Таганрог. До 15 000 ₽.",
            "Кресло и лампа в Азове до 3000 ₽",
            "Нужна лампа в Ростове-на-Дону до 1000 ₽",
        )
        for query in queries:
            with self.subTest(query=query):
                for option in response_for(query)["options"]:
                    explanation = option["explanation"]
                    self.assertGreaterEqual(len(explanation.strip()), 10, "объяснение слишком короткое")
                    for listing_id in option_ids(option):
                        price = int(by_id[listing_id]["price_rub"])
                        self.assertTrue(
                            mentions_price(explanation, price),
                            f"в объяснении нет цены {money(price)} из каталога: {explanation!r}",
                        )
                    self.assertTrue(
                        mentions_price(explanation, int(option["total_rub"])),
                        f"в объяснении нет итоговой суммы: {explanation!r}",
                    )

    def test_explanation_is_distinct_per_option(self) -> None:
        """Разные комплекты не могут объясняться одинаково: иначе объяснение бесполезно."""
        document = response_for("Стол, кресло и лампа. Таганрог. До 15 000 ₽.")
        explanations = [option["explanation"] for option in document["options"]]
        self.assertEqual(len(explanations), len(set(explanations)), "у разных вариантов одинаковое объяснение")

    def test_all_returned_ids_belong_to_the_given_catalog(self) -> None:
        """ID берутся из каталога, переданного аргументом, а не из памяти процесса."""
        by_id = index_by_id(load_catalog())
        for bundle in CASES:
            with self.subTest(query=bundle.query):
                for option in response_for(bundle.query)["options"]:
                    for listing_id in option_ids(option):
                        self.assertIn(listing_id, by_id, f"ID {listing_id} отсутствует в data/catalog.json")

    def test_city_of_every_item_equals_customer_city(self) -> None:
        by_id: Mapping[str, Mapping[str, Any]] = index_by_id(load_catalog())
        for bundle in CASES:
            with self.subTest(query=bundle.query, city=bundle.city_id):
                for option in response_for(bundle.query)["options"]:
                    for listing_id in option_ids(option):
                        self.assertEqual(by_id[listing_id]["city_id"], bundle.city_id)


if __name__ == "__main__":
    unittest.main()
