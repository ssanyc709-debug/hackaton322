"""Извлечение условий: lexicon-first, кэш разборов, классификация чисел."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.model import FIELD_ADJUSTABLE, FIELD_BUDGET, FIELD_CATEGORIES, FIELD_CITY, FIELD_WIDTH
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
        slots = extract_message("нужны 2 кресла в Азове")
        self.assertIsNone(slots.budget_rub)

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

    def test_temperature_range_without_unit_uses_lamp_context(self) -> None:
        slots = extract_message("лампа от 3000 до 5000, Таганрог, до 5000 рублей")
        assert slots.color_temperature is not None
        self.assertEqual(slots.color_temperature.lower.value, 3000)
        self.assertEqual(slots.color_temperature.upper.value, 5000)

    def test_impossible_width_is_dropped(self) -> None:
        slots = extract_message("стол шириной 100 метров, Таганрог, до 20000")
        self.assertIsNone(slots.width)

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
    """Модель вызывается только когда лексикон не закрыл обязательные условия."""

    def setUp(self) -> None:
        self.calls: list[str] = []

    def resolver(self, text: str, slots) -> dict[str, object]:
        self.calls.append(text)
        return {"budget_rub": 12000}

    def test_no_model_call_when_everything_is_known(self) -> None:
        slots = extract_message(EXAMPLE_01, resolver=self.resolver)
        self.assertEqual(self.calls, [])
        self.assertEqual(slots.source, "lexicon")

    def test_one_model_call_when_budget_missing(self) -> None:
        slots = extract_message(EXAMPLE_02, resolver=self.resolver)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(slots.budget_rub, 12000)
        self.assertEqual(slots.source, "model")

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

        slots = extract_message("кресло и лампа", resolver=bad)
        self.assertIsNone(slots.city_id)

    def test_model_error_is_not_swallowed(self) -> None:
        """Сбой модели — техническая ошибка запуска, а не «нет вариантов»."""
        def broken(text: str, slots) -> dict[str, object]:
            raise TimeoutError("шлюз не ответил")

        with self.assertRaises(TimeoutError):
            extract_message(EXAMPLE_02, resolver=broken)

    def test_empty_model_answer_keeps_local_result(self) -> None:
        def silent(text: str, slots) -> dict[str, object] | None:
            return None

        slots = extract_message(EXAMPLE_02, resolver=silent)
        self.assertEqual(slots.city_id, "taganrog")
        self.assertIsNone(slots.budget_rub)

    def test_offline_never_calls_model(self) -> None:
        slots = extract_message(EXAMPLE_02, resolver=self.resolver, allow_model=False)
        self.assertEqual(self.calls, [])
        self.assertIsNone(slots.budget_rub)


if __name__ == "__main__":
    unittest.main()
