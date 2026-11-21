"""Извлечение условий: lexicon-first, кэш разборов, классификация чисел."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.model import (
    FIELD_ADJUSTABLE,
    FIELD_BUDGET,
    FIELD_CATEGORIES,
    FIELD_CITY,
    FIELD_TEMPERATURE,
    FIELD_WIDTH,
)
from app.nlp.cache import LexiconCache, phrase_key
from app.nlp.extract import extract_message, is_complete, missing_fields

EXAMPLE_01 = "Стол не шире метра, кресло и лампа. Таганрог. До 15 000 ₽ за всё."
EXAMPLE_02 = "Стол не шире метра, кресло и лампа. Таганрог."


class ExtractExampleTest(unittest.TestCase):
    """Сквозной пример кейса должен разбираться без модели."""

    def test_example_01_slots(self) -> None:
        slots = extract_message(EXAMPLE_01)
        self.assertEqual(slots.city_id, "taganrog")
        self.assertEqual(slots.budget_rub, 15000)
        self.assertEqual(slots.categories, ("table", "chair", "lamp"))
        self.assertIsNotNone(slots.width)
        assert slots.width is not None and slots.width.upper is not None
        self.assertEqual(slots.width.upper.value, 100)
        self.assertTrue(slots.width.upper.inclusive)
        self.assertTrue(is_complete(slots))
        self.assertEqual(slots.source, "lexicon")

    def test_example_02_misses_only_budget(self) -> None:
        slots = extract_message(EXAMPLE_02)
        self.assertIsNone(slots.budget_rub)
        self.assertEqual(missing_fields(slots), (FIELD_BUDGET,))

    def test_mentioned_fields_are_explicit(self) -> None:
        slots = extract_message(EXAMPLE_01)
        self.assertEqual(
            slots.mentioned,
            frozenset({FIELD_CITY, FIELD_BUDGET, FIELD_CATEGORIES, FIELD_WIDTH}),
        )

    def test_short_answer_to_question(self) -> None:
        """Ответ на уточнение может быть коротким: «15 тысяч»."""
        slots = extract_message("15 тысяч")
        self.assertEqual(slots.budget_rub, 15000)
        self.assertEqual(slots.mentioned, frozenset({FIELD_BUDGET}))

    def test_answer_with_currency_and_word_order(self) -> None:
        for text in ("15000 ₽", "до 15к", "бюджет 15 000 рублей", "пятнадцать тысяч"):
            with self.subTest(text=text):
                self.assertEqual(extract_message(text).budget_rub, 15000)


class ExtractCityAndCategoriesTest(unittest.TestCase):
    """Город и категории: формы, сленг, отсутствие категории."""

    def test_city_changes_with_last_mention(self) -> None:
        slots = extract_message("Стол в Таганроге, но лучше в Азове, до 20000")
        self.assertEqual(slots.city_id, "azov")

    def test_city_adjective_leaves_only_city_to_clarify(self) -> None:
        slots = extract_message("В Азовском, стол до 5000")
        self.assertIsNone(slots.city_id)
        self.assertEqual(missing_fields(slots), (FIELD_CITY,))

    def test_unknown_city_is_flagged(self) -> None:
        slots = extract_message("Нужен стол в Москве до 10000")
        self.assertIsNone(slots.city_id)
        self.assertEqual(slots.city_unknown, "москва")
        self.assertIn(FIELD_CITY, missing_fields(slots))

    def test_unknown_category_is_flagged(self) -> None:
        slots = extract_message("Нужен диван и шкаф в Таганроге до 30000")
        self.assertEqual(slots.categories, ())
        self.assertEqual(slots.categories_unknown, ("диван", "шкаф"))
        self.assertIn(FIELD_CATEGORIES, missing_fields(slots))

    def test_partially_known_categories_keep_known_ones(self) -> None:
        slots = extract_message("Нужен стол и диван в Таганроге до 30000")
        self.assertEqual(slots.categories, ("table",))
        self.assertEqual(slots.categories_unknown, ("диван",))

    def test_only_lamp(self) -> None:
        slots = extract_message("Лампа до 3000 рублей, Таганрог")
        self.assertEqual(slots.categories, ("lamp",))
        self.assertTrue(is_complete(slots))

    def test_quantity_is_not_a_budget(self) -> None:
        for quantity in ("3 кресла", "до 3 кресел", "2 стола", "1500 кресел", "1500 штук кресел"):
            with self.subTest(quantity=quantity):
                slots = extract_message(f"Нужно {quantity} в Азове")
                self.assertIsNone(slots.budget_rub)
                self.assertEqual(missing_fields(slots), (FIELD_BUDGET,))

    def test_budget_before_a_comma_is_not_a_quantity(self) -> None:
        slots = extract_message("Таганрог, 1500, кресло")
        self.assertEqual(slots.budget_rub, 1500)

    def test_explicit_monetary_context_allows_small_amounts(self) -> None:
        self.assertEqual(extract_message("Бюджет до 900").budget_rub, 900)
        self.assertIsNone(extract_message("До 900").budget_rub)

    def test_mat_does_not_break_extraction(self) -> None:
        slots = extract_message("Бля, срочно нужен сраный стол и кресло в Таганроге, до 10000")
        self.assertEqual(slots.categories, ("table", "chair"))
        self.assertEqual(slots.city_id, "taganrog")
        self.assertEqual(slots.budget_rub, 10000)


class ExtractAttributeTest(unittest.TestCase):
    """Характеристики: ширина, температура, регулировка высоты."""

    def test_width_boundaries(self) -> None:
        inclusive = extract_message("стол не шире 100 см, Таганрог, до 20000")
        assert inclusive.width is not None and inclusive.width.upper is not None
        self.assertEqual((inclusive.width.upper.value, inclusive.width.upper.inclusive), (100, True))
        strict = extract_message("стол уже 100 см, Таганрог, до 20000")
        assert strict.width is not None and strict.width.upper is not None
        self.assertEqual((strict.width.upper.value, strict.width.upper.inclusive), (100, False))

    def test_width_range(self) -> None:
        slots = extract_message("стол от 80 до 120 см, Таганрог, до 20000")
        assert slots.width is not None and slots.width.lower is not None and slots.width.upper is not None
        self.assertEqual((slots.width.lower.value, slots.width.lower.inclusive), (80, True))
        self.assertEqual((slots.width.upper.value, slots.width.upper.inclusive), (120, True))

    def test_temperature_from_unit(self) -> None:
        slots = extract_message("лампа 4000 к, Таганрог, до 5000")
        assert slots.color_temperature is not None
        self.assertEqual(slots.color_temperature.lower.value, 4000)
        self.assertEqual(slots.color_temperature.upper.value, 4000)

    def test_explicit_temperature_range_without_unit(self) -> None:
        slots = extract_message("лампа с температурой от 3000 до 5000, Таганрог, до 5000 рублей")
        assert slots.color_temperature is not None
        self.assertEqual(slots.color_temperature.lower.value, 3000)
        self.assertEqual(slots.color_temperature.upper.value, 5000)

    def test_unlabelled_temperature_range_is_ambiguous(self) -> None:
        for suffix, budget in (("", None), (", до 5000 рублей", 5000)):
            with self.subTest(suffix=suffix):
                slots = extract_message(f"Лампа от 3000 до 5000, Таганрог{suffix}")
                self.assertIsNone(slots.color_temperature)
                self.assertEqual(slots.budget_rub, budget)

    def test_explicit_width_is_kept_even_if_catalogue_cannot_match(self) -> None:
        slots = extract_message("стол шириной 100 метров, Таганрог, до 20000")
        assert slots.width is not None
        self.assertEqual(slots.width.lower.value, 10000)
        self.assertEqual(slots.width.upper.value, 10000)

    def test_adjustable_true(self) -> None:
        slots = extract_message("кресло с регулировкой высоты, Таганрог, до 20000")
        self.assertIs(slots.adjustable_height, True)
        self.assertIn(FIELD_ADJUSTABLE, slots.mentioned)

    def test_adjustable_false_is_not_lost(self) -> None:
        slots = extract_message("кресло без регулировки, Таганрог, до 20000")
        self.assertIs(slots.adjustable_height, False)
        self.assertIn(FIELD_ADJUSTABLE, slots.mentioned)

    def test_no_adjustable_requirement(self) -> None:
        self.assertIsNone(extract_message("кресло, Таганрог, до 20000").adjustable_height)


class FieldRemovalTest(unittest.TestCase):
    """Снятие ограничения — это изменение условия, а не новое значение (§2.3)."""

    def test_width_removal_is_detected(self) -> None:
        slots = extract_message("давай расширим допустимую ширину")
        self.assertIsNone(slots.width)
        self.assertEqual(slots.fields_removed, (FIELD_WIDTH,))
        self.assertIn(FIELD_WIDTH, slots.mentioned)

    def test_new_value_wins_over_removal(self) -> None:
        """«расширим ширину до 250» — это НОВАЯ ширина, а не снятие."""
        slots = extract_message("расширим ширину до 250 см")
        self.assertIsNotNone(slots.width)
        self.assertEqual(slots.fields_removed, ())

    def test_adjustable_removal_is_detected(self) -> None:
        slots = extract_message("убери требование к регулировке")
        self.assertIsNone(slots.adjustable_height)
        self.assertEqual(slots.fields_removed, (FIELD_ADJUSTABLE,))

    def test_temperature_removal_is_detected(self) -> None:
        slots = extract_message("без ограничений по температуре")
        self.assertEqual(slots.fields_removed, (FIELD_TEMPERATURE,))

    def test_ordinary_message_removes_nothing(self) -> None:
        slots = extract_message("стол в Таганроге до 20000")
        self.assertEqual(slots.fields_removed, ())


class PhraseCategoryTest(unittest.TestCase):
    """Предмет назван описанием: «на чём сидеть», «что светится» (лексикон)."""

    def test_seat_phrase_is_chair(self) -> None:
        slots = extract_message("на чём мне сидеть в Таганроге до 15000")
        self.assertIn("chair", slots.categories)
        self.assertTrue(slots.categories_additive)

    def test_light_phrase_is_lamp(self) -> None:
        slots = extract_message("что светится в Азове до 10000")
        self.assertIn("lamp", slots.categories)

    def test_phrase_does_not_replace_named_categories(self) -> None:
        slots = extract_message("нужен стол и на чём сидеть, Таганрог, до 20000")
        self.assertEqual(slots.categories, ("table", "chair"))


class CacheTest(unittest.TestCase):
    """Кэш разборов: повтор стоит 0 токенов и не меняет результат."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_second_lookup_is_a_cache_hit(self) -> None:
        cache = LexiconCache(self.directory)
        first = extract_message(EXAMPLE_01, cache=cache)
        self.assertEqual(cache.hits, 0)
        second = extract_message(EXAMPLE_01, cache=cache)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(second.source, "cache")
        self.assertEqual(first.city_id, second.city_id)
        self.assertEqual(first.budget_rub, second.budget_rub)
        self.assertEqual(first.categories, second.categories)
        assert first.width is not None and second.width is not None
        self.assertEqual(first.width.upper, second.width.upper)

    def test_cache_survives_new_instance(self) -> None:
        extract_message(EXAMPLE_01, cache=LexiconCache(self.directory))
        reopened = LexiconCache(self.directory)
        slots = extract_message(EXAMPLE_01, cache=reopened)
        self.assertEqual(slots.source, "cache")
        self.assertEqual(slots.budget_rub, 15000)

    def test_cached_removal_is_not_lost(self) -> None:
        """Снятие ограничения переживает кэш: иначе диалог зависел бы от попадания."""
        text = "давай расширим допустимую ширину"
        first = extract_message(text, cache=LexiconCache(self.directory))
        reopened = LexiconCache(self.directory)
        second = extract_message(text, cache=reopened)
        self.assertEqual(second.source, "cache")
        self.assertEqual(first.fields_removed, (FIELD_WIDTH,))
        self.assertEqual(second.fields_removed, (FIELD_WIDTH,))

    def test_normalized_variants_share_key(self) -> None:
        self.assertEqual(phrase_key("До 15 000 ₽"), phrase_key("до 15000 ₽"))

    def test_broken_cache_file_is_ignored(self) -> None:
        path = self.directory / "lexicon.json"
        path.write_text("{ битый", encoding="utf-8")
        cache = LexiconCache(self.directory)
        self.assertIsNone(cache.get("что угодно"))
        slots = extract_message(EXAMPLE_01, cache=cache)
        self.assertEqual(slots.budget_rub, 15000)
        self.assertEqual(extract_message(EXAMPLE_01, cache=cache).source, "cache")

    def test_disabled_cache_never_hits(self) -> None:
        cache = LexiconCache(self.directory, enabled=False)
        extract_message(EXAMPLE_01, cache=cache)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.hit_rate, 0.0)


class ModelFallbackTest(unittest.TestCase):
    """Модель вызывается только там, где она МОЖЕТ закрыть недостающее условие.

    Блок 9: недостаточно «условия неполны» — нужно, чтобы у модели был материал.
    Бюджет и город из ниоткуда не берутся, поэтому такие вызовы отсекаются
    (:func:`app.nlp.extract.model_can_help`).
    """

    def setUp(self) -> None:
        self.calls: list[str] = []

    def resolver(self, text: str, slots) -> dict[str, object]:
        self.calls.append(text)
        return {"categories": ["table"], "budget_rub": 12000, "city_id": "azov"}

    def test_no_model_call_when_everything_is_known(self) -> None:
        slots = extract_message(EXAMPLE_01, resolver=self.resolver)
        self.assertEqual(self.calls, [])
        self.assertEqual(slots.source, "lexicon")

    def test_one_model_call_when_only_categories_missing(self) -> None:
        """Город и бюджет названы, предмет — описательно: это работа для модели."""
        slots = extract_message(
            "Нужна деревяшка с четырьмя ножками. Таганрог. До 15000.", resolver=self.resolver
        )
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(slots.source, "model")
        self.assertEqual(slots.categories, ("table",))
        self.assertEqual(slots.city_id, "taganrog")
        self.assertEqual(slots.budget_rub, 15000)

    def test_no_model_call_when_budget_is_not_named(self) -> None:
        """В EXAMPLE_02 нет ни одного числа: бюджет взять неоткуда, вызов бесполезен."""
        slots = extract_message(EXAMPLE_02, resolver=self.resolver)
        self.assertEqual(self.calls, [])
        self.assertIsNone(slots.budget_rub)

    def test_no_model_call_when_nothing_is_named(self) -> None:
        slots = extract_message("Хочу мебель", resolver=self.resolver)
        self.assertEqual(self.calls, [])
        self.assertEqual(
            missing_fields(slots), (FIELD_CITY, FIELD_BUDGET, FIELD_CATEGORIES)
        )

    def test_no_model_call_for_unknown_city(self) -> None:
        """Город вне справочника: подставлять другой город нельзя."""
        extract_message("Стол в Москве до 30000", resolver=self.resolver)
        self.assertEqual(self.calls, [])

    def test_no_model_call_for_ambiguous_primary_fields(self) -> None:
        for text, field in (
            ("В Азовском, стол до 5000", FIELD_CITY),
            ("До 1500 кресел в Азове", FIELD_BUDGET),
            ("Нужен диван в Азове до 5000", FIELD_CATEGORIES),
        ):
            with self.subTest(text=text):
                slots = extract_message(text, resolver=self.resolver)
                self.assertEqual(self.calls, [])
                self.assertEqual(missing_fields(slots), (field,))

    def test_no_model_call_for_quantity_without_money(self) -> None:
        """«2 кресла» — количество, а не бюджет: признака денег в тексте нет."""
        extract_message("Нужны 2 кресла", resolver=self.resolver)
        self.assertEqual(self.calls, [])

    def test_model_does_not_override_local_values(self) -> None:
        def hostile(text: str, slots) -> dict[str, object]:
            return {"city_id": "azov", "budget_rub": 1, "categories": ["lamp"]}

        slots = extract_message(EXAMPLE_01, resolver=hostile)
        self.assertEqual(slots.city_id, "taganrog")  # локальное значение сохранено
        self.assertEqual(slots.budget_rub, 15000)
        self.assertEqual(slots.categories, ("table", "chair", "lamp"))

    def test_model_city_is_validated_against_catalog_cities(self) -> None:
        def bad(text: str, slots) -> dict[str, object]:
            return {"city_id": "moscow", "budget_rub": 9000}

        slots = extract_message("кресло и лампа до 20000, заберу в выходные", resolver=bad)
        self.assertIsNone(slots.city_id)

    def test_model_error_is_not_swallowed(self) -> None:
        """Сбой модели — техническая ошибка запуска, а не «нет вариантов»."""
        def broken(text: str, slots) -> dict[str, object]:
            raise TimeoutError("шлюз не ответил")

        with self.assertRaises(TimeoutError):
            extract_message("Нужна деревяшка. Таганрог, до 15000", resolver=broken)

    def test_empty_model_answer_keeps_local_result(self) -> None:
        def silent(text: str, slots) -> dict[str, object] | None:
            return None

        slots = extract_message("Нужна деревяшка. Таганрог, до 15000", resolver=silent)
        self.assertEqual(slots.city_id, "taganrog")
        self.assertEqual(slots.budget_rub, 15000)
        self.assertEqual(slots.categories, ())

    def test_offline_never_calls_model(self) -> None:
        slots = extract_message(EXAMPLE_02, resolver=self.resolver, allow_model=False)
        self.assertEqual(self.calls, [])
        self.assertIsNone(slots.budget_rub)


class ModelGateTest(unittest.TestCase):
    """Три фразы из задачи + защита от переусердствования (блок 9)."""

    def calls_for(self, text: str) -> list[str]:
        calls: list[str] = []

        def resolver(payload: str, slots) -> dict[str, object]:
            calls.append(payload)
            return {"categories": ["table"]}

        extract_message(text, resolver=resolver)
        return calls

    def test_descriptive_phrase_asks_the_model(self) -> None:
        """«Деревяшка с четырьмя ножками»: лексикон таких слов не знает."""
        self.assertEqual(
            len(self.calls_for("Нужна деревяшка с 4 ножками, на которую ставят вещи. Таганрог. До 15000.")),
            1,
        )

    def test_preposition_role_does_not_mean_a_table(self) -> None:
        """«Посветить над столом» — просьба о лампе, стол здесь место."""
        slots = extract_message("Надо чем посветить над столом. Ростов-на-Дону, до 5000.")
        self.assertEqual(slots.categories, ("lamp",))
        self.assertEqual(self.calls_for("Надо чем посветить над столом. Ростов-на-Дону, до 5000."), [])

    def test_work_is_not_a_working_table(self) -> None:
        """«Сидеть за работой»: fuzzy-совпадение с «рабочий» — не товар."""
        slots = extract_message("Купил бы что-нибудь, чтобы сидеть за работой. Таганрог, тысяч пятнадцать.")
        self.assertEqual(slots.categories, ())
        # Город и бюджет известны → модель может назвать предмет по описанию.
        self.assertEqual(len(self.calls_for("Купил бы что-нибудь, чтобы сидеть за работой. Таганрог, тысяч пятнадцать.")), 1)

    def test_typo_is_still_a_category(self) -> None:
        """«Стлл» — опечатка существительного, а не прилагательного: работает локально."""
        slots = extract_message("Нужен стлл. Таганрог, до 15000.")
        self.assertEqual(slots.categories, ("table",))
        self.assertEqual(self.calls_for("Нужен стлл. Таганрог, до 15000."), [])

    def test_exact_phrase_costs_nothing(self) -> None:
        self.assertEqual(self.calls_for(EXAMPLE_01), [])

    def test_table_for_a_laptop_keeps_the_table(self) -> None:
        """Предлог ПОСЛЕ названия товара не отменяет товар."""
        slots = extract_message("Стол под ноутбук. Таганрог, до 15000.")
        self.assertEqual(slots.categories, ("table",))

    def test_amount_is_not_a_place(self) -> None:
        slots = extract_message("Стол и кресло на 15 000. Таганрог.")
        self.assertEqual(slots.categories, ("table", "chair"))
        self.assertEqual(slots.budget_rub, 15000)


if __name__ == "__main__":
    unittest.main()
