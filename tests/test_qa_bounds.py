"""Блок 4 — границы, характеристики и числа (CONTRACT §2.2, §2.3).

Контракт задаёт четыре правила, и каждое здесь проверяется на данных каталога,
а не на коде:

* «до», «не более», «не шире», «от», «не менее» — граница **входит**;
* «меньше», «больше», «уже», «шире» — граница **строгая**;
* «от … до …» включает **оба** конца;
* ограничение действует только на объявления с известным значением: если ключа
  в ``attributes`` нет, объявление под ограничение **не подходит**;
* для ``false`` и ``0`` нельзя проверять наличие через truthiness.

Все ожидания считаются по ``data/catalog.json``: тест не знает, какой именно
комплект выберет решение, он проверяет, что **каждый** возвращённый товар
удовлетворяет условию, и что граничное значение достижимо.

Известные дефекты (БАГ №1 и БАГ №2 в ``docs/QA-ERRORS.md``) в этом блоке
описываются тестами с ``skip``: код продукта по решению владельца не правится.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from tests.test_qa_support import (
    EXIT_OK,
    ask,
    index_by_id,
    load_catalog,
    option_ids,
    run_shared,
)

#: Характеристика, которой ограничена категория (CONTRACT §2, таблица).
ATTRIBUTE_OF = {
    "table": "width_cm",
    "chair": "adjustable_height",
    "lamp": "color_temperature_k",
}

_RESPONSES: dict[str, dict[str, Any]] = {}
_WORKDIR: tempfile.TemporaryDirectory[str] | None = None


def setUpModule() -> None:
    global _WORKDIR
    _WORKDIR = tempfile.TemporaryDirectory(prefix="qa-bounds-")


def tearDownModule() -> None:
    if _WORKDIR is not None:
        _WORKDIR.cleanup()
    _RESPONSES.clear()


def document_for(query: str) -> dict[str, Any]:
    """Ответ на запрос; каждый уникальный запрос запускается ровно один раз."""
    if query not in _RESPONSES:
        result = run_shared(ask(query))
        if result.code != EXIT_OK:
            raise AssertionError(f"запрос {query!r}: код {result.code}, stderr: {result.stderr[-400:]}")
        _RESPONSES[query] = result.one_json_object()
    return _RESPONSES[query]


def selected_ids(query: str) -> list[str]:
    """Все ID из всех вариантов ответа ``ok``."""
    document = document_for(query)
    if document.get("status") != "ok":
        raise AssertionError(f"запрос {query!r}: ожидался ok, получено {document!r}")
    return [listing_id for option in document["options"] for listing_id in option_ids(option)]


def selected_composition(query: str) -> list[frozenset[str]]:
    """Составы вариантов — сравнение «одинаковые ли подборы» без учёта порядка."""
    document = document_for(query)
    if document.get("status") != "ok":
        raise AssertionError(f"запрос {query!r}: ожидался ok, получено {document!r}")
    return [frozenset(option_ids(option)) for option in document["options"]]


class AttributeTests(unittest.TestCase):
    """Общая часть: доступ к характеристике товара по каталогу."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.by_id = index_by_id(load_catalog())

    def attribute(self, listing_id: str, key: str) -> Any:
        """Значение характеристики или ``None``, если ключа нет (значение неизвестно)."""
        return self.by_id[listing_id]["attributes"].get(key)

    def assert_every_item(self, query: str, key: str, check: Callable[[Any], bool], message: str) -> None:
        """Каждый товар во всех вариантах удовлетворяет проверке (и характеристика известна)."""
        ids = selected_ids(query)
        self.assertTrue(ids, f"{query}: ответ ok без товаров")
        for listing_id in ids:
            with self.subTest(query=query, listing_id=listing_id):
                value = self.attribute(listing_id, key)
                self.assertIsNotNone(
                    value,
                    f"{query}: товар {listing_id} без характеристики {key} не подтверждает соответствие",
                )
                self.assertTrue(check(value), f"{query}: {listing_id} имеет {key}={value!r}, {message}")

    def values(self, query: str, key: str) -> list[Any]:
        return [self.attribute(listing_id, key) for listing_id in selected_ids(query)]


class WidthBoundaryTest(AttributeTests):
    """Ширина стола: включительные и строгие границы, интервал, точное значение."""

    def test_upper_inclusive_boundary_is_allowed(self) -> None:
        """«Не шире 100 см»: 100 см входит, 110/120 не входят, неизвестная ширина не подходит."""
        query = "Стол не шире 100 см в Таганроге до 15 000 ₽"
        self.assert_every_item(query, "width_cm", lambda value: value <= 100, "шире 100 см")
        self.assertIn(100, self.values(query, "width_cm"), "граница 100 см должна быть достижима: она входит")

    def test_upper_inclusive_phrasings_are_equivalent(self) -> None:
        """§2.2: «до», «не более», «не шире», «максимум» задают одну и ту же границу."""
        forms = (
            "Стол не шире 100 см в Таганроге до 15 000 ₽",
            "Стол не более 100 см в Таганроге до 15 000 ₽",
            "Стол шириной до 100 см в Таганроге до 15 000 ₽",
            "Стол максимум 100 см в Таганроге до 15 000 ₽",
        )
        reference = selected_composition(forms[0])
        for form in forms[1:]:
            with self.subTest(query=form):
                self.assertEqual(selected_composition(form), reference, "формулировки обязаны давать один подбор")

    def test_upper_strict_boundary_is_excluded(self) -> None:
        """«Уже 100 см»: 100 см не входит (строгая граница)."""
        query = "Стол уже 100 см в Таганроге до 15 000 ₽"
        self.assert_every_item(query, "width_cm", lambda value: value < 100, "не уже 100 см")
        self.assertNotIn(100, self.values(query, "width_cm"), "строгая граница: ровно 100 см не подходит")

    def test_lower_strict_boundary_is_excluded(self) -> None:
        """«Шире 100 см»: 100 см не входит, подходят только 110 и 120."""
        query = "Стол шире 100 см в Таганроге, бюджет 15 000 ₽"
        self.assert_every_item(query, "width_cm", lambda value: value > 100, "не шире 100 см")
        self.assertNotIn(100, self.values(query, "width_cm"), "строгая граница: ровно 100 см не подходит")

    def test_interval_includes_both_ends(self) -> None:
        """«От … до …» включает оба конца (§2.2)."""
        upper_query = "Стол шириной от 105 до 110 см в Таганроге до 15 000 ₽"
        self.assert_every_item(upper_query, "width_cm", lambda value: 105 <= value <= 110, "вне 105..110")
        self.assertIn(110, self.values(upper_query, "width_cm"), "верхний конец интервала обязан входить")

        lower_query = "Стол шириной от 95 до 96 см в Таганроге до 15 000 ₽"
        self.assert_every_item(lower_query, "width_cm", lambda value: 95 <= value <= 96, "вне 95..96")
        self.assertIn(95, self.values(lower_query, "width_cm"), "нижний конец интервала обязан входить")

    def test_interval_without_items_returns_no_match(self) -> None:
        """Интервал, в который не попадает ни один стол, — это ``no_match``, а не подмена."""
        document = document_for("Стол шириной от 105 до 109 см в Таганроге до 15 000 ₽")
        self.assertEqual(document.get("status"), "no_match", document)

    def test_exact_value_with_and_without_trigger(self) -> None:
        """«Ровно 95 см» и «шириной 95 см» — одно и то же условие."""
        for query in (
            "Стол шириной ровно 95 см в Таганроге до 15 000 ₽",
            "Стол шириной 95 см в Таганроге до 15 000 ₽",
        ):
            with self.subTest(query=query):
                self.assert_every_item(query, "width_cm", lambda value: value == 95, "не ровно 95 см")

    def test_unknown_width_never_satisfies_a_constraint(self) -> None:
        """§2.3: отсутствующий ключ — неизвестное значение, оно не подтверждает соответствие."""
        forbidden = "demo-1-1-11"  # 2 500 ₽, attributes = {} — самый дешёвый стол Таганрога
        self.assertIsNone(self.attribute(forbidden, "width_cm"), "у выбранного объявления должна быть пустая характеристика")
        for query in (
            "Стол не шире 100 см в Таганроге до 15 000 ₽",
            "Стол шириной ровно 95 см в Таганроге до 15 000 ₽",
            "Стол шириной от 60 до 120 см в Таганроге до 15 000 ₽",
        ):
            with self.subTest(query=query):
                self.assertNotIn(
                    forbidden,
                    selected_ids(query),
                    "стол без указания ширины попал в подбор при ограничении по ширине",
                )

    def test_meters_and_centimeters_are_the_same_constraint(self) -> None:
        """«Не шире метра» = «не шире 100 см», «1,5 метра» = «150 см»."""
        pairs = (
            ("Стол не шире 1 метра в Таганроге до 15 000 ₽", "Стол не шире 100 см в Таганроге до 15 000 ₽"),
            ("Стол не шире 1,5 метра в Таганроге до 15 000 ₽", "Стол не шире 150 см в Таганроге до 15 000 ₽"),
        )
        for meters, centimeters in pairs:
            with self.subTest(meters=meters):
                self.assertEqual(selected_composition(meters), selected_composition(centimeters))


class ChairAttributeTest(AttributeTests):
    """Кресло: ``adjustable_height`` — boolean, ``false`` не равно «нет значения»."""

    def test_true_means_adjustable(self) -> None:
        query = "Кресло с регулировкой высоты в Таганроге, бюджет 15 000 ₽"
        self.assert_every_item(query, "adjustable_height", lambda value: value is True, "регулировки нет")

    def test_false_is_respected_and_not_treated_as_missing(self) -> None:
        """§2.3: ``false`` нельзя проверять через truthiness — иначе «без регулировки» опустеет."""
        query = "Кресло без регулировки высоты в Таганроге, бюджет 15 000 ₽"
        self.assert_every_item(query, "adjustable_height", lambda value: value is False, "регулировка есть")
        self.assertGreaterEqual(len(selected_ids(query)), 1, "кресла без регулировки в каталоге есть")

    def test_unknown_is_different_from_false(self) -> None:
        """Кресло без ключа не подходит ни под «с регулировкой», ни под «без регулировки»."""
        unknown = [
            listing_id
            for listing_id, raw in self.by_id.items()
            if raw["category"] == "chair" and "adjustable_height" not in raw["attributes"]
        ]
        self.assertTrue(unknown, "в каталоге должны быть кресла без характеристики")
        for query in (
            "Кресло с регулировкой высоты в Таганроге, бюджет 15 000 ₽",
            "Кресло без регулировки высоты в Таганроге, бюджет 15 000 ₽",
        ):
            with self.subTest(query=query):
                for listing_id in unknown:
                    self.assertNotIn(listing_id, selected_ids(query), f"кресло {listing_id} без характеристики попало в подбор")


class LampAttributeTest(AttributeTests):
    """Лампа: ``color_temperature_k`` — целое число кельвинов."""

    def test_upper_inclusive_boundary_is_allowed(self) -> None:
        query = "Лампа не выше 4000 K в Таганроге, бюджет 15 000 ₽"
        self.assert_every_item(query, "color_temperature_k", lambda value: value <= 4000, "выше 4000 K")
        self.assertIn(4000, self.values(query, "color_temperature_k"), "граница 4000 K входит")

    def test_strict_lower_boundary_is_excluded(self) -> None:
        query = "Лампа выше 4000 K в Таганроге, бюджет 15 000 ₽"
        self.assert_every_item(query, "color_temperature_k", lambda value: value > 4000, "не выше 4000 K")
        self.assertNotIn(4000, self.values(query, "color_temperature_k"))

    def test_exact_value(self) -> None:
        query = "Лампа с температурой 3000 K в Таганроге, бюджет 15 000 ₽"
        self.assert_every_item(query, "color_temperature_k", lambda value: value == 3000, "не 3000 K")

    def test_interval(self) -> None:
        query = "Лампа с температурой от 3000 до 3500 K в Таганроге, бюджет 15 000 ₽"
        self.assert_every_item(query, "color_temperature_k", lambda value: 3000 <= value <= 3500, "вне 3000..3500 K")

    def test_unknown_temperature_never_satisfies_a_constraint(self) -> None:
        unknown = [
            listing_id
            for listing_id, raw in self.by_id.items()
            if raw["category"] == "lamp" and "color_temperature_k" not in raw["attributes"]
        ]
        self.assertTrue(unknown, "в каталоге должны быть лампы без характеристики")
        for query in (
            "Лампа не выше 4000 K в Таганроге, бюджет 15 000 ₽",
            "Лампа с температурой 3000 K в Таганроге, бюджет 15 000 ₽",
        ):
            with self.subTest(query=query):
                for listing_id in unknown:
                    self.assertNotIn(listing_id, selected_ids(query), f"лампа {listing_id} без характеристики попала в подбор")


class NumbersTest(unittest.TestCase):
    """Числа: слова, сокращения, разряды, единицы и граница бюджета."""

    def assert_same_bundle(self, first: str, second: str) -> None:
        first_document = document_for(first)
        second_document = document_for(second)
        self.assertEqual(first_document.get("status"), second_document.get("status"), f"{first} vs {second}")
        if first_document.get("status") != "ok":
            return
        self.assertEqual(
            [frozenset(option_ids(option)) for option in first_document["options"]],
            [frozenset(option_ids(option)) for option in second_document["options"]],
            f"разные записи одного числа дали разные подборы: {first!r} vs {second!r}",
        )
        self.assertEqual(
            [option["total_rub"] for option in first_document["options"]],
            [option["total_rub"] for option in second_document["options"]],
        )

    def test_budget_number_forms_are_equivalent(self) -> None:
        """Один и тот же бюджет, записанный по-разному, обязан давать один подбор."""
        reference = "Стол в Азове до 2300 ₽"
        forms = (
            "Стол в Азове за 2300 ₽",
            "Стол в Азове, бюджет 2300",
            "Стол в Азове, 2 300 ₽",
            "Стол в Азове за две тысячи триста рублей",
            "Стол в Азове до двух тысяч трёхсот рублей",
            "Стол в Азове до 2,3 тысяч рублей",
        )
        for form in forms:
            with self.subTest(query=form):
                self.assert_same_bundle(reference, form)

    def test_large_budget_number_forms_are_equivalent(self) -> None:
        reference = "Стол в Таганроге до 15 000 ₽"
        forms = (
            "Стол в Таганроге до 15000 рублей",
            "Стол в Таганроге до 15 тысяч",
            "Стол в Таганроге за пятнадцать тысяч рублей",
            "Стол в Таганроге до 15к",
        )
        for form in forms:
            with self.subTest(query=form):
                self.assert_same_bundle(reference, form)

    def test_budget_boundary_is_inclusive(self) -> None:
        """Равенство бюджета допустимо (§3): ровно 2 300 ₽ — это ``ok``, 2 299 ₽ — ``no_match``."""
        fits = document_for("Стол в Азове до 2300 ₽")
        short = document_for("Стол в Азове до 2299 ₽")
        self.assertEqual(fits.get("status"), "ok", fits)
        self.assertEqual(short.get("status"), "no_match", short)

    def test_lower_bound_of_budget_is_not_a_cap(self) -> None:
        """«От 5000 ₽» — это не потолок (бюджет по §3 — верхний предел).

        Такая фраза раньше давала бюджет 5 000 ₽, и покупатель, просивший «не
        дешевле», получал самые дешёвые комплекты. Ожидается: бюджет остаётся
        неизвестным, и помощник спрашивает его отдельно.
        """
        document = document_for("Стол в Таганроге, от 5000 ₽")
        self.assertEqual(document.get("status"), "clarify", document)
        fields = {question["field"] for question in document["questions"]}
        self.assertIn("budget_rub", fields, "бюджет неизвестен — его нужно спросить")

    def test_lower_bound_without_currency_is_not_a_cap(self) -> None:
        """То же правило, когда валюта не написана: «бюджет от 5000», «дороже 5000».

        Нижняя граница без единицы попадает в «нейтральные» числа и раньше
        подставлялась как потолок уже на шаге :mod:`app.nlp.extract`.
        """
        for query in ("Стол в Таганроге, бюджет от 5000", "Стол в Таганроге дороже 5000"):
            with self.subTest(query=query):
                document = document_for(query)
                self.assertEqual(document.get("status"), "clarify", document)
                fields = {question["field"] for question in document["questions"]}
                self.assertIn("budget_rub", fields, "бюджет неизвестен — его нужно спросить")

    def test_budget_pair_still_caps_at_the_upper_bound(self) -> None:
        """«От 3000 до 5000 ₽» — потолок по-прежнему 5 000 (нижняя граница его не отменяет)."""
        document = document_for("Стол в Таганроге от 3000 до 5000 ₽")
        self.assertEqual(document.get("status"), "ok", document)
        for option in document["options"]:
            with self.subTest(option=option["listing_ids"]):
                self.assertLessEqual(option["total_rub"], 5000)

    def test_simultaneous_constraints_are_all_applied(self) -> None:
        """§2.2: «поддерживаются одновременные ограничения» — по одной характеристике на категорию."""
        by_id = index_by_id(load_catalog())
        query = (
            "Стол шириной от 80 до 100 см, кресло с регулировкой высоты "
            "и лампа не выше 4000 K в Таганроге до 20 000 ₽"
        )
        document = document_for(query)
        self.assertEqual(document.get("status"), "ok", document)
        checks: dict[str, Callable[[Any], bool]] = {
            "table": lambda value: 80 <= value <= 100,
            "chair": lambda value: value is True,
            "lamp": lambda value: value <= 4000,
        }
        for option in document["options"]:
            ids = option_ids(option)
            self.assertEqual(len(ids), 3, "в запросе три категории — в варианте должно быть три товара")
            for listing_id in ids:
                raw = by_id[listing_id]
                key = ATTRIBUTE_OF[raw["category"]]
                with self.subTest(listing_id=listing_id):
                    value = raw["attributes"].get(key)
                    self.assertIsNotNone(value, f"{listing_id}: характеристика {key} неизвестна")
                    self.assertTrue(checks[raw["category"]](value), f"{listing_id}: {key}={value!r} нарушает условие")


class LowerBoundRegressionTest(AttributeTests):
    """Регрессия БАГ №1: нижняя граница терялась, если «до <бюджет>» стоит далеко.

    Причина была в ``app/nlp/numbers.py::_merge_pairs``: нижняя граница искала пару
    среди последующих верхних границ; если ближайшая из них дальше
    ``MAX_RANGE_GAP_TOKENS`` токенов, цикл делал ``break``, ветка ``else`` у ``for``
    после ``break`` не выполнялась, и одиночная нижняя граница в результат не
    попадала — условие исчезало молча.

    Дефект закрыт: «пара не найдена» проверяется явным флагом (см.
    ``docs/QA-ERRORS.md``, БАГ №1). Тесты описывают требование CONTRACT §2.2.
    """

    def test_lower_inclusive_width_survives_a_distant_budget(self) -> None:
        query = "Стол от 100 см в Таганроге до 15 000 ₽"
        self.assert_every_item(query, "width_cm", lambda value: value >= 100, "уже 100 см")

    def test_lower_strict_width_survives_a_distant_budget(self) -> None:
        query = "Стол шире 100 см в Таганроге до 15 000 ₽"
        self.assert_every_item(query, "width_cm", lambda value: value > 100, "не шире 100 см")

    def test_lower_inclusive_width_in_a_combined_query_survives(self) -> None:
        query = "Стол не уже 90 см в Таганроге до 15 000 ₽"
        self.assert_every_item(query, "width_cm", lambda value: value >= 90, "уже 90 см")

    def test_lower_temperature_bound_survives_a_distant_budget(self) -> None:
        query = "Лампа от 4000 K в Таганроге до 15 000 ₽"
        self.assert_every_item(query, "color_temperature_k", lambda value: value >= 4000, "ниже 4000 K")

    def test_lower_bound_is_not_lost_for_any_phrasing(self) -> None:
        """Все формулировки нижней границы должны переживать бюджет, заданный словом «до»."""
        cases = (
            ("Стол от 100 см в Таганроге до 15 000 ₽", "width_cm", lambda value: value >= 100),
            ("Стол не менее 100 см в Таганроге до 15 000 ₽", "width_cm", lambda value: value >= 100),
            ("Стол шириной минимум 100 см в Таганроге до 15 000 ₽", "width_cm", lambda value: value >= 100),
            ("Стол больше 100 см в Таганроге до 15 000 ₽", "width_cm", lambda value: value > 100),
            ("Стол свыше 100 см в Таганроге до 15 000 ₽", "width_cm", lambda value: value > 100),
        )
        for query, key, check in cases:
            with self.subTest(query=query):
                self.assert_every_item(query, key, check, "нижняя граница не соблюдена")


class LowerBoundControlTest(AttributeTests):
    """Контроль к БАГ №1: без «до» перед бюджетом та же нижняя граница работает.

    Это доказывает, что дефект не в разборе «от»/«шире», а в потере условия при
    слиянии пар, и что каталог/фильтр нижние границы поддерживают.
    """

    def test_lower_bound_works_when_budget_is_named_without_trigger(self) -> None:
        for query, key, check in (
            ("Стол от 100 см в Таганроге, бюджет 15 000 ₽", "width_cm", lambda value: value >= 100),
            ("Стол шире 100 см в Таганроге, бюджет 15 000 ₽", "width_cm", lambda value: value > 100),
            ("Лампа от 4000 K в Таганроге, бюджет 15 000 ₽", "color_temperature_k", lambda value: value >= 4000),
        ):
            with self.subTest(query=query):
                self.assert_every_item(query, key, check, "нижняя граница не соблюдена")

    def test_same_query_with_close_budget_keeps_lower_bound(self) -> None:
        """Та же фраза, но «до» ближе: граница сохраняется — подтверждает причину дефекта."""
        query = "Стол от 100 см Таганрог до 15 000 ₽"
        self.assert_every_item(query, "width_cm", lambda value: value >= 100, "уже 100 см")


if __name__ == "__main__":
    unittest.main()
