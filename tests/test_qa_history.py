"""Блок 5 — история диалога (CONTRACT §3).

Правила §3, которые здесь проверяются:

* «Учитывайте всю историю»: условия берутся из всех реплик покупателя;
* «Явное изменение условия в новой реплике заменяет предыдущее значение этого
  условия» — причём заменяет, а не сужает: «не шире 120» после «не шире 80»
  отменяет прежние 80, а не пересекается с ними;
* «Остальные условия сохраняются»: смена бюджета не сбрасывает город и состав;
* «Не запрашивайте повторно сведения, которые уже присутствуют в истории»;
* «Покупатель может ответить на уточнение короткой фразой, например „15 тысяч“»;
* «Прошлые ID объявлений не заменяют повторный подбор по текущему каталогу».

Каждый ответ проверяется по каталогу (город, состав, сумма, бюджет), поэтому
тест не зависит от того, какой именно комплект выберет решение.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tests.test_qa_support import (
    EXIT_OK,
    ask,
    assert_option_matches_catalog,
    index_by_id,
    load_catalog,
    option_ids,
    request,
    run_shared,
    user,
    assistant,
)

_RESPONSES: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
_WORKDIR: tempfile.TemporaryDirectory[str] | None = None


def setUpModule() -> None:
    global _WORKDIR
    _WORKDIR = tempfile.TemporaryDirectory(prefix="qa-history-")


def tearDownModule() -> None:
    if _WORKDIR is not None:
        _WORKDIR.cleanup()
    _RESPONSES.clear()


def dialogue(*messages: Mapping[str, str]) -> dict[str, Any]:
    """Ответ на диалог; одинаковый диалог запускается ровно один раз."""
    key = tuple((str(message["role"]), str(message["text"])) for message in messages)
    if key not in _RESPONSES:
        result = run_shared(request(*messages))
        if result.code != EXIT_OK:
            raise AssertionError(f"диалог {key!r}: код {result.code}, stderr: {result.stderr[-400:]}")
        _RESPONSES[key] = result.one_json_object()
    return _RESPONSES[key]


class HistoryCase(unittest.TestCase):
    """Общая часть: каталог и проверка варианта по нему."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.by_id = index_by_id(load_catalog())

    def options(self, query: str) -> list[dict[str, Any]]:
        """Варианты ответа ``ok`` (статус проверяется отдельно)."""
        document = dialogue(user(query))
        self.assertEqual(document.get("status"), "ok", document)
        return document["options"]

    def assert_bundle(
        self,
        document: Mapping[str, Any],
        *,
        categories: Sequence[str],
        city_id: str,
        budget_rub: int,
    ) -> None:
        """Все варианты ответа допустимы по каталогу и по условиям."""
        self.assertEqual(document.get("status"), "ok", document)
        self.assertTrue(1 <= len(document["options"]) <= 3)
        for index, option in enumerate(document["options"]):
            with self.subTest(option=index):
                assert_option_matches_catalog(
                    self,
                    option,
                    by_id=self.by_id,
                    categories=categories,
                    city_id=city_id,
                    budget_rub=budget_rub,
                )

    def categories_in(self, document: Mapping[str, Any]) -> set[str]:
        """Какие категории встречаются в ответе."""
        return {
            str(self.by_id[listing_id]["category"])
            for option in document["options"]
            for listing_id in option_ids(option)
        }

    def cities_in(self, document: Mapping[str, Any]) -> set[str]:
        return {
            str(self.by_id[listing_id]["city_id"])
            for option in document["options"]
            for listing_id in option_ids(option)
        }

    def totals(self, document: Mapping[str, Any]) -> list[int]:
        return [int(option["total_rub"]) for option in document["options"]]

    def widths_in(self, document: Mapping[str, Any]) -> set[int]:
        """Ширины столов в ответе (только известные значения)."""
        return {
            int(self.by_id[listing_id]["attributes"]["width_cm"])
            for option in document["options"]
            for listing_id in option_ids(option)
            if "width_cm" in self.by_id[listing_id]["attributes"]
        }


class ShortClarifyAnswerTest(HistoryCase):
    """§3: на уточнение можно ответить короткой фразой."""

    def test_short_budget_answer_completes_the_selection(self) -> None:
        document = dialogue(
            user("Нужен стол, кресло и лампа в Таганроге"),
            assistant("Какой у вас бюджет на весь комплект?"),
            user("15 тысяч"),
        )
        self.assert_bundle(
            document,
            categories=("table", "chair", "lamp"),
            city_id="taganrog",
            budget_rub=15000,
        )

    def test_short_city_answer_completes_the_selection(self) -> None:
        document = dialogue(
            user("Стол до 15 000 ₽"),
            assistant("В каком городе нужен комплект: Таганрог, Ростов-на-Дону или Азов?"),
            user("Азов"),
        )
        self.assert_bundle(document, categories=("table",), city_id="azov", budget_rub=15000)

    def test_short_category_answer_completes_the_selection(self) -> None:
        document = dialogue(
            user("В Таганроге до 15 000 ₽"),
            assistant("Какие товары подобрать? Поддерживаются: стол, кресло, лампа."),
            user("Стол и кресло"),
        )
        self.assert_bundle(document, categories=("table", "chair"), city_id="taganrog", budget_rub=15000)

    def test_known_condition_is_not_asked_again(self) -> None:
        """§3: не запрашивать повторно то, что уже есть в истории."""
        document = dialogue(
            user("Нужен стол в Таганроге"),
            assistant("Какой у вас бюджет на весь комплект?"),
            user("до 5000"),
        )
        self.assert_bundle(document, categories=("table",), city_id="taganrog", budget_rub=5000)

    def test_three_rounds_of_clarification(self) -> None:
        document = dialogue(
            user("Хочу мебель"),
            assistant("Какие товары подобрать?"),
            user("Стол"),
            assistant("В каком городе?"),
            user("Азов"),
            assistant("Какой бюджет?"),
            user("5 тысяч"),
        )
        self.assert_bundle(document, categories=("table",), city_id="azov", budget_rub=5000)


class ReplacementTest(HistoryCase):
    """§3: новое значение условия заменяет прежнее, остальные условия живут."""

    def test_new_budget_replaces_the_previous_one(self) -> None:
        document = dialogue(
            user("Стол, кресло и лампа. Таганрог. До 20 000 ₽."),
            assistant("Вот три варианта комплекта."),
            user("Давай до 5000"),
        )
        self.assert_bundle(
            document,
            categories=("table", "chair", "lamp"),
            city_id="taganrog",
            budget_rub=5000,
        )

    def test_budget_change_keeps_city_and_categories(self) -> None:
        """Смена бюджета не должна сбрасывать город и состав комплекта."""
        document = dialogue(
            user("Стол, кресло и лампа. Таганрог. До 20 000 ₽."),
            assistant("Вот три варианта комплекта."),
            user("Давай до 5000"),
        )
        self.assertEqual(self.categories_in(document), {"table", "chair", "lamp"})
        self.assertEqual(self.cities_in(document), {"taganrog"})

    def test_new_city_replaces_the_previous_one(self) -> None:
        document = dialogue(
            user("Стол в Таганроге до 15 000 ₽"),
            assistant("Вот варианты."),
            user("Нет, нужен в Азове"),
        )
        self.assert_bundle(document, categories=("table",), city_id="azov", budget_rub=15000)

    def test_last_city_mention_wins(self) -> None:
        document = dialogue(
            user("Стол в Азове до 10 000 ₽"),
            assistant("Вот варианты."),
            user("Нет, Таганрог"),
            assistant("Вот варианты."),
            user("Передумал, Азов"),
        )
        self.assert_bundle(document, categories=("table",), city_id="azov", budget_rub=10000)

    def test_new_width_replaces_instead_of_narrowing(self) -> None:
        """«Не шире 120» после «не шире 80» — замена, а не пересечение (80∩120 = 80)."""
        document = dialogue(
            user("Стол не шире 80 см в Таганроге до 15 000 ₽"),
            assistant("Вот варианты."),
            user("Ладно, не шире 120 см"),
        )
        self.assert_bundle(document, categories=("table",), city_id="taganrog", budget_rub=15000)
        widths = self.widths_in(document)
        self.assertTrue(widths, "в ответе нет стола с известной шириной")
        self.assertTrue(
            any(width > 80 for width in widths),
            f"условие «не шире 80» не отменено: все ширины {sorted(widths)} укладываются в старую границу",
        )

    def test_tightening_the_width_still_works(self) -> None:
        document = dialogue(
            user("Стол не шире 120 см в Таганроге до 15 000 ₽"),
            assistant("Вот варианты."),
            user("Нужен не шире 80"),
        )
        widths = self.widths_in(document)
        self.assertTrue(widths, "в ответе нет стола с известной шириной")
        self.assertTrue(max(widths) <= 80, f"новая граница 80 см не соблюдена: ширины {sorted(widths)}")


class CategoryHistoryTest(HistoryCase):
    """Категории: замена, исключение и добавление — три разных поведения."""

    def test_without_lamp_removes_the_category(self) -> None:
        document = dialogue(
            user("Стол, кресло и лампа в Таганроге до 15 000 ₽"),
            assistant("Вот варианты комплекта."),
            user("Без лампы"),
        )
        self.assert_bundle(document, categories=("table", "chair"), city_id="taganrog", budget_rub=15000)
        self.assertNotIn("lamp", self.categories_in(document))

    def test_not_needed_removes_the_category(self) -> None:
        document = dialogue(
            user("Стол, кресло и лампа в Таганроге до 15 000 ₽"),
            assistant("Вот варианты комплекта."),
            user("Кресло не нужно"),
        )
        self.assert_bundle(document, categories=("table", "lamp"), city_id="taganrog", budget_rub=15000)

    def test_additive_word_keeps_previous_categories(self) -> None:
        """«Ещё и лампа» добавляет категорию к прежнему составу, а не заменяет его."""
        document = dialogue(
            user("Стол в Таганроге до 15 000 ₽"),
            assistant("Вот варианты."),
            user("Ещё и лампа"),
        )
        self.assert_bundle(document, categories=("table", "lamp"), city_id="taganrog", budget_rub=15000)

    def test_new_statement_without_additive_word_replaces_categories(self) -> None:
        """«Нужна лампа» без слова «ещё» — явная замена состава (принцип §3)."""
        document = dialogue(
            user("Стол и кресло в Таганроге до 15 000 ₽"),
            assistant("Вот варианты."),
            user("Нужна лампа"),
        )
        self.assertEqual(self.categories_in(document), {"lamp"}, "состав должен быть заменён на одну лампу")

    def test_additive_word_does_not_duplicate_a_category(self) -> None:
        """«Добавь ещё стол», когда стол уже есть, не может дать два стола."""
        document = dialogue(
            user("Стол в Таганроге до 15 000 ₽"),
            assistant("Вот варианты."),
            user("Добавь ещё стол"),
        )
        self.assert_bundle(document, categories=("table",), city_id="taganrog", budget_rub=15000)
        for option in document["options"]:
            self.assertEqual(len(option_ids(option)), 1, "в варианте оказалось больше одного товара")


class MultipleCitiesTest(HistoryCase):
    """Несколько городов в запросе: подбор идёт в каждом, комплект — в одном.

    Контракт §2 требует, чтобы комплект собирался для самовывоза в ОДНОМ городе,
    и это правило соблюдается: внутри варианта города не смешиваются. Но §3
    говорит про «один город», поэтому «в Таганроге или в Азове» сейчас означает
    «взять последний упомянутый» — покупатель видит варианты только в Азове.
    Тесты ниже описывают ожидаемое поведение и помечены ``skip``:
    ``docs/QA-ERRORS.md``, БАГ №3.
    """

    def cities_per_option(self, document: Mapping[str, Any]) -> list[set[str]]:
        return [
            {str(self.by_id[listing_id]["city_id"]) for listing_id in option_ids(option)}
            for option in document["options"]
        ]

    def test_each_option_is_within_one_city(self) -> None:
        document = dialogue(user("Стол, кресло и лампа в Таганроге или в Азове до 12 000 ₽"))
        self.assertEqual(document.get("status"), "ok", document)
        for index, cities in enumerate(self.cities_per_option(document)):
            with self.subTest(option=index):
                self.assertEqual(len(cities), 1, f"вариант смешал города: {sorted(cities)}")

    @unittest.skip("БАГ №3 (docs/QA-ERRORS.md): из двух городов используется только последний")
    def test_both_cities_are_presented(self) -> None:
        document = dialogue(user("Стол в Таганроге или в Азове до 5000 ₽"))
        self.assertEqual(document.get("status"), "ok", document)
        presented = set().union(*self.cities_per_option(document))
        self.assertEqual(presented, {"taganrog", "azov"}, "покупатель ждал варианты в обоих городах")
        for index, (option, cities) in enumerate(zip(document["options"], self.cities_per_option(document))):
            with self.subTest(option=index):
                self.assertEqual(len(cities), 1, f"вариант смешал города: {sorted(cities)}")
                assert_option_matches_catalog(
                    self,
                    option,
                    by_id=self.by_id,
                    categories=("table",),
                    city_id=next(iter(cities)),
                    budget_rub=5000,
                )

    @unittest.skip("БАГ №3 (docs/QA-ERRORS.md): из трёх городов используется только последний")
    def test_three_cities_are_all_presented(self) -> None:
        document = dialogue(user("Стол в Таганроге, Азове или Ростове-на-Дону до 3000 ₽"))
        self.assertEqual(document.get("status"), "ok", document)
        presented = set().union(*self.cities_per_option(document))
        self.assertEqual(presented, {"taganrog", "azov", "rostov-na-donu"})

    @unittest.skip("БАГ №3 (docs/QA-ERRORS.md): no_match упоминает только последний город")
    def test_no_match_when_neither_city_fits(self) -> None:
        document = dialogue(user("Стол в Таганроге или в Азове до 100 ₽"))
        self.assertEqual(document.get("status"), "no_match", document)
        self.assertNotIn("options", document)
        message = document["message"]
        self.assertIn("Таганрог", message, "в объяснении нет первого города")
        self.assertIn("Азов", message, "в объяснении нет второго города")

    @unittest.skip("БАГ №3 (docs/QA-ERRORS.md): «или в Азове» заменяет город вместо добавления")
    def test_or_in_a_follow_up_adds_the_city(self) -> None:
        """«Или в Азове» — альтернатива, прежний город сохраняется."""
        document = dialogue(
            user("Стол в Таганроге до 5000 ₽"),
            assistant("Вот варианты."),
            user("Или в Азове"),
        )
        presented = set().union(*self.cities_per_option(document))
        self.assertEqual(presented, {"taganrog", "azov"})

    def test_plain_new_city_still_replaces(self) -> None:
        """«Нет, в Азове» — явное изменение условия: остаётся только Азов."""
        document = dialogue(
            user("Стол в Таганроге до 5000 ₽"),
            assistant("Вот варианты."),
            user("Нет, в Азове"),
        )
        self.assertEqual(set().union(*self.cities_per_option(document)), {"azov"})

    def test_negated_city_is_not_a_variant(self) -> None:
        """«Не в Азове, а в Таганроге» — Азов отвергнут, а не предложен."""
        document = dialogue(user("Не в Азове, а в Таганроге, стол до 5000 ₽"))
        self.assertEqual(set().union(*self.cities_per_option(document)), {"taganrog"})

    def test_single_city_is_unaffected(self) -> None:
        """Обычный запрос с одним городом не должен начать искать в других."""
        document = dialogue(user("Стол в Таганроге до 5000 ₽"))
        self.assertEqual(set().union(*self.cities_per_option(document)), {"taganrog"})


class UnknownValueTest(HistoryCase):
    """§4: город вне справочника и неподдерживаемая категория — повод уточнить."""

    def test_city_outside_directory_asks_city(self) -> None:
        document = dialogue(user("Стол в Москве до 15 000 ₽"))
        self.assertEqual(document.get("status"), "clarify", document)
        fields = {question["field"] for question in document["questions"]}
        self.assertIn("city_id", fields, "город вне справочника — уточняем город")
        self.assertNotIn("options", document)

    def test_unsupported_category_asks_categories(self) -> None:
        document = dialogue(user("Нужен диван в Таганроге до 15 000 ₽"))
        self.assertEqual(document.get("status"), "clarify", document)
        fields = {question["field"] for question in document["questions"]}
        self.assertIn("categories", fields, "неподдерживаемая категория — уточняем состав")

    def test_answer_after_unknown_city_produces_a_bundle(self) -> None:
        document = dialogue(
            user("Стол в Москве до 15 000 ₽"),
            assistant("В каком городе нужен комплект: Таганрог, Ростов-на-Дону или Азов?"),
            user("В Таганроге"),
        )
        self.assert_bundle(document, categories=("table",), city_id="taganrog", budget_rub=15000)

    def test_answer_after_unsupported_category_produces_a_bundle(self) -> None:
        document = dialogue(
            user("Диван в Таганроге до 15 000 ₽"),
            assistant("Какие товары подобрать? Поддерживаются: стол, кресло, лампа."),
            user("Стол"),
        )
        self.assert_bundle(document, categories=("table",), city_id="taganrog", budget_rub=15000)


class PreviousIdsTest(HistoryCase):
    """§3: прошлые ID в истории не заменяют повторный подбор."""

    RESENT_ANSWER = (
        '{"status":"ok","options":[{"listing_ids":["demo-1-1-07"],'
        '"total_rub":7000,"explanation":"самый дорогой стол города"}]}'
    )

    def test_resent_previous_ids_do_not_override_the_new_query(self) -> None:
        document = dialogue(
            user("Стол в Таганроге до 15 000 ₽"),
            assistant(self.RESENT_ANSWER),
            user("А теперь до 3000 ₽"),
        )
        self.assert_bundle(document, categories=("table",), city_id="taganrog", budget_rub=3000)
        returned = {listing_id for option in document["options"] for listing_id in option_ids(option)}
        self.assertNotIn(
            "demo-1-1-07",
            returned,
            "ID из прошлого ответа попал в новый подбор, хотя не проходит по бюджету (7 000 ₽ > 3 000 ₽)",
        )

    def test_resent_ids_do_not_appear_when_the_query_changes_city(self) -> None:
        document = dialogue(
            user("Стол в Таганроге до 15 000 ₽"),
            assistant(self.RESENT_ANSWER),
            user("Нужен в Азове"),
        )
        self.assert_bundle(document, categories=("table",), city_id="azov", budget_rub=15000)

    def test_selection_uses_the_current_catalog_only(self) -> None:
        """Все ID берутся из каталога — даже если история упоминает несуществующие."""
        document = dialogue(
            user("Стол в Таганроге до 15 000 ₽"),
            assistant('{"status":"ok","options":[{"listing_ids":["ghost-1","ghost-2"],'
            '"total_rub":1,"explanation":"нет такого"}]}'),
            user("Покажи варианты"),
        )
        self.assert_bundle(document, categories=("table",), city_id="taganrog", budget_rub=15000)
        returned = {listing_id for option in document["options"] for listing_id in option_ids(option)}
        self.assertTrue(returned <= set(self.by_id), f"в ответе есть ID вне каталога: {returned - set(self.by_id)}")


class StatusAfterHistoryTest(HistoryCase):
    """Смена условий меняет статус предсказуемо."""

    def test_budget_reduction_below_the_cheapest_bundle_gives_no_match(self) -> None:
        document = dialogue(
            user("Стол, кресло и лампа. Таганрог. До 15 000 ₽."),
            assistant("Вот три варианта комплекта."),
            user("Стоп, только 1000 ₽"),
        )
        self.assertEqual(document.get("status"), "no_match", document)
        self.assertNotIn("options", document)

    def test_long_history_still_produces_an_answer(self) -> None:
        """19 сообщений (предел §3 при чередовании ролей) — история не должна ломать подбор."""
        messages: list[dict[str, str]] = []
        for index in range(18):
            if index % 2 == 0:
                messages.append(user(f"Реплика {index}"))
            else:
                messages.append(assistant(f"Ответ {index}"))
        messages.append(user("Стол, кресло и лампа в Таганроге. До 15 000 ₽."))
        self.assertEqual(len(messages), 19)
        document = dialogue(*messages)
        self.assert_bundle(
            document,
            categories=("table", "chair", "lamp"),
            city_id="taganrog",
            budget_rub=15000,
        )


if __name__ == "__main__":
    unittest.main()
