"""Тесты чисел, единиц и сравнительных конструкций (§2.2)."""

from __future__ import annotations

import unittest

from app.nlp.numbers import (
    EXACT,
    LOWER_INC,
    LOWER_STRICT,
    NO_TRIGGER,
    UNIT_CM,
    UNIT_K,
    UNIT_M,
    UNIT_RUB,
    UPPER_INC,
    UPPER_STRICT,
    collect_constraints,
    find_amounts,
    normalize_text,
    parse_amount,
    tokenize,
)


class NormalizeTextTest(unittest.TestCase):
    """«15 тысяч», «1 метр», ₽, неразрывные пробелы."""

    def test_nbsp_and_thin_spaces_become_plain_spaces(self) -> None:
        self.assertEqual(normalize_text("До 15\u00a0000\u202f₽"), "до 15000 ₽")

    def test_digit_groups_are_glued(self) -> None:
        self.assertEqual(normalize_text("15 000"), "15000")
        self.assertEqual(normalize_text("1 500 000"), "1500000")

    def test_yo_is_normalized(self) -> None:
        self.assertEqual(normalize_text("ЕЩЁ ЛАМПУ"), "еще лампу")

    def test_dashes_and_case(self) -> None:
        self.assertEqual(normalize_text("Ростов\u2014на\u2014Дону"), "ростов-на-дону")

    def test_empty_string(self) -> None:
        self.assertEqual(normalize_text("   "), "")

    def test_tokenize_splits_numbers_words_and_ruble_sign(self) -> None:
        self.assertEqual(tokenize(normalize_text("до 15000 ₽ за всё")), ("до", "15000", "₽", "за", "все"))


class AmountParsingTest(unittest.TestCase):
    """Числа словами, множители, единицы."""

    def test_digits_with_currency(self) -> None:
        amounts = find_amounts("До 15 000 ₽ за всё")
        self.assertEqual(len(amounts), 1)
        self.assertEqual((amounts[0].value, amounts[0].unit, amounts[0].kind),
                         (15000, UNIT_RUB, UPPER_INC))

    def test_thousands_word(self) -> None:
        amounts = find_amounts("бюджет 15 тысяч рублей")
        self.assertEqual(amounts[0].value, 15000)
        self.assertEqual(amounts[0].unit, UNIT_RUB)

    def test_word_numerals_sequence(self) -> None:
        amounts = find_amounts("двадцать пять тысяч рублей")
        self.assertEqual(amounts[0].value, 25000)

    def test_half_thousand(self) -> None:
        amounts = find_amounts("полторы тысячи рублей")
        self.assertEqual(amounts[0].value, 1500)

    def test_inverted_multiplier(self) -> None:
        """Разговорный порядок: «тысяч на двенадцать» — это 12 000, а не 1 000."""
        self.assertEqual(parse_amount("тысяч на двенадцать"), 12000)
        self.assertEqual(parse_amount("тысяч пятнадцать"), 15000)
        self.assertEqual(parse_amount("тысячи 5"), 5000)

    def test_inverted_multiplier_is_not_double_counted(self) -> None:
        """Обычный порядок не ломается и не удваивается."""
        self.assertEqual(parse_amount("двенадцать тысяч"), 12000)
        self.assertEqual(parse_amount("15 тысяч"), 15000)

    def test_half_after_number(self) -> None:
        """«двенадцать с половиной тысяч» = 12 500."""
        self.assertEqual(parse_amount("двенадцать с половиной тысяч"), 12500)
        self.assertEqual(parse_amount("двенадцати с половиной тысяч"), 12500)
        self.assertEqual(parse_amount("полтора миллиона"), 1500000)

    def test_glued_k_is_thousands(self) -> None:
        self.assertEqual(parse_amount("15к"), 15000)
        self.assertEqual(parse_amount("15 к"), 15000)

    def test_large_k_is_kelvin(self) -> None:
        amounts = find_amounts("лампа 4000 к")
        self.assertEqual((amounts[0].value, amounts[0].unit), (4000, UNIT_K))

    def test_meter_becomes_100_cm(self) -> None:
        amounts = find_amounts("стол не шире метра")
        self.assertEqual((amounts[0].value, amounts[0].unit, amounts[0].kind),
                         (100, UNIT_CM, UPPER_INC))

    def test_one_and_half_meters(self) -> None:
        """Метры приводятся к сантиметрам на разборе: 1,5 м → 150 см."""
        amounts = find_amounts("стол до 1,5 метра")
        self.assertEqual((amounts[0].value, amounts[0].unit), (150, UNIT_CM))

    def test_centimeters(self) -> None:
        amounts = find_amounts("ширина 95 см")
        self.assertEqual((amounts[0].value, amounts[0].unit), (95, UNIT_CM))

    def test_decimal_comma(self) -> None:
        amounts = find_amounts("0,95 метра")
        self.assertEqual(amounts[0].value, 95)

    def test_width_word_makes_bare_number_width(self) -> None:
        """«ширина до 250» без «см» — всё равно ширина, иначе условие теряется."""
        constraints = collect_constraints(find_amounts("ширина до 250"))
        self.assertIsNotNone(constraints.width)
        assert constraints.width is not None
        self.assertEqual(constraints.width.upper.value, 250)
        self.assertTrue(constraints.width.upper.inclusive)

    def test_width_word_is_stronger_than_money_hint(self) -> None:
        """«бюджет» не перетягивает число, если рядом прямо сказано «ширина»."""
        constraints = collect_constraints(find_amounts("бюджет потом, ширина до 250"))
        self.assertIsNone(constraints.budget_rub)
        self.assertIsNotNone(constraints.width)

    def test_width_word_does_not_swallow_money_number(self) -> None:
        """«ширина 100, 15000»: большое число рядом со словом «ширина» — не ширина."""
        constraints = collect_constraints(find_amounts("ширина 100, 15000"))
        assert constraints.width is not None
        self.assertEqual(constraints.width.upper.value, 100)
        self.assertTrue(constraints.neutral)

    def test_quantity_without_trigger_is_not_money(self) -> None:
        """«2 кресла» — это количество, а не бюджет."""
        constraints = collect_constraints(find_amounts("нужны 2 кресла"))
        self.assertIsNone(constraints.budget_rub)

    def test_instrumental_thousands(self) -> None:
        """«располагаю 15 тысячами» — бюджет 15 000, а не 15 (падежная форма)."""
        constraints = collect_constraints(find_amounts("располагаю 15 тысячами"))
        self.assertEqual(constraints.budget_rub, 15000)

    def test_bare_unit_after_width_word_is_exact(self) -> None:
        """«стол шириной метр» — точное значение 100 см (контракт §2.2)."""
        constraints = collect_constraints(find_amounts("стол шириной метр"))
        self.assertIsNotNone(constraints.width)
        self.assertEqual(constraints.width.lower.value, 100)
        self.assertEqual(constraints.width.upper.value, 100)

    def test_bare_unit_without_width_word_is_ignored(self) -> None:
        """«уложусь в метр» — не условие о ширине: слова «ширина» рядом нет."""
        constraints = collect_constraints(find_amounts("кресло и лампа, уложусь в метр"))
        self.assertIsNone(constraints.width)


class TriggerTest(unittest.TestCase):
    """Включительные и строгие границы §2.2."""

    def assert_kind(self, text: str, expected: str) -> None:
        amounts = find_amounts(text)
        self.assertTrue(amounts, f"не найдено число в {text!r}")
        self.assertEqual(amounts[0].kind, expected, text)

    def test_inclusive_triggers(self) -> None:
        for text in ("до 100 см", "не более 100 см", "не шире 100 см", "максимум 100 см"):
            with self.subTest(text=text):
                self.assert_kind(text, UPPER_INC)

    def test_strict_triggers(self) -> None:
        for text in ("меньше 100 см", "уже 100 см", "ниже 100 см"):
            with self.subTest(text=text):
                self.assert_kind(text, UPPER_STRICT)

    def test_lower_inclusive(self) -> None:
        for text in ("от 80 см", "не менее 80 см", "минимум 80 см"):
            with self.subTest(text=text):
                self.assert_kind(text, LOWER_INC)

    def test_lower_strict(self) -> None:
        for text in ("больше 80 см", "шире 80 см"):
            with self.subTest(text=text):
                self.assert_kind(text, LOWER_STRICT)

    def test_exact(self) -> None:
        self.assert_kind("ровно 95 см", EXACT)

    def test_negated_trigger_wins_over_short_form(self) -> None:
        """«не уже 100» — это >= 100, а не < 100."""
        self.assert_kind("не уже 100 см", LOWER_INC)

    def test_no_trigger(self) -> None:
        self.assert_kind("стол 95 см", NO_TRIGGER)


class ConstraintsTest(unittest.TestCase):
    """Сборка условий из реплики."""

    def test_width_range_from_ot_to(self) -> None:
        constraints = collect_constraints(find_amounts("стол от 80 до 100 см"))
        self.assertIsNotNone(constraints.width)
        assert constraints.width is not None
        self.assertEqual((constraints.width.lower.value, constraints.width.lower.inclusive), (80, True))
        self.assertEqual((constraints.width.upper.value, constraints.width.upper.inclusive), (100, True))

    def test_kelvin_range_from_ot_to(self) -> None:
        constraints = collect_constraints(find_amounts("лампа от 3000 до 5000 кельвинов"))
        assert constraints.temperature is not None
        self.assertEqual((constraints.temperature.lower.value, constraints.temperature.lower.inclusive), (3000, True))
        self.assertEqual((constraints.temperature.upper.value, constraints.temperature.upper.inclusive), (5000, True))

    def test_strict_upper_in_range(self) -> None:
        constraints = collect_constraints(find_amounts("стол до 100 см"))
        assert constraints.width is not None
        self.assertTrue(constraints.width.upper.inclusive)
        strict = collect_constraints(find_amounts("стол уже 100 см"))
        assert strict.width is not None
        self.assertFalse(strict.width.upper.inclusive)

    def test_budget_and_width_in_one_message(self) -> None:
        constraints = collect_constraints(
            find_amounts("Стол не шире метра, кресло и лампа. Таганрог. До 15 000 ₽ за всё.")
        )
        self.assertEqual(constraints.budget_rub, 15000)
        assert constraints.width is not None
        self.assertEqual(constraints.width.upper.value, 100)
        self.assertTrue(constraints.width.upper.inclusive)

    def test_neutral_number_with_money_hint_is_budget(self) -> None:
        constraints = collect_constraints(find_amounts("бюджет 15000"))
        self.assertEqual(constraints.budget_rub, 15000)

    def test_neutral_number_for_lamp_is_left_to_extract(self) -> None:
        """Без единицы «от 3000 до 5000» решает слой категорий, а не догадка."""
        constraints = collect_constraints(find_amounts("лампа от 3000 до 5000"))
        self.assertIsNone(constraints.budget_rub)
        self.assertIsNone(constraints.temperature)
        self.assertTrue(constraints.neutral)

    def test_no_numbers(self) -> None:
        constraints = collect_constraints(find_amounts("подбери стол, кресло и лампу"))
        self.assertIsNone(constraints.budget_rub)
        self.assertIsNone(constraints.width)
        self.assertIsNone(constraints.temperature)


class NicheWidthTest(unittest.TestCase):
    """«Влезало в нишу 142 см» — предел места, а не точная ширина стола.

    Ментор нашёл это на ручном каталоге: покупатель просил «не шире», а ответ
    был ``no_match``, потому что число без сравнительной конструкции читалось
    как «ровно 142». Ниша и глаголы влезания задают верхнюю включительную
    границу, как «до»/«не шире».
    """

    def assert_width(self, text: str, value: int) -> None:
        constraints = collect_constraints(find_amounts(text))
        self.assertIsNotNone(constraints.width, f"{text!r}: ширина не найдена")
        assert constraints.width is not None
        self.assertIsNone(
            constraints.width.lower, f"{text!r}: у предела места нет нижней границы"
        )
        self.assertEqual(constraints.width.upper.value, value, text)
        self.assertTrue(constraints.width.upper.inclusive, f"{text!r}: граница включительная")

    def test_niche_phrase_is_an_upper_bound(self) -> None:
        for text in (
            "влезало в нишу 142 см",
            "влезает в нишу 142 см",
            "в нишу 142 см",
            "ниша 142 см",
            "чтобы влезал в нишу 142 см",
        ):
            with self.subTest(text=text):
                self.assert_width(text, 142)

    def test_niche_without_unit_is_still_width(self) -> None:
        """«ниша 90» без «см» — это ширина, иначе условие терялось молча."""
        for text in ("в нишу 142", "ниша 90", "чтобы влезал в нишу 142"):
            with self.subTest(text=text):
                self.assert_width(text, 142 if "142" in text else 90)

    def test_doorway_and_fit_verbs_are_upper_bounds(self) -> None:
        for text in (
            "должен влезть в проём 142 см",
            "поместится в проём 142 см",
            "впишется в проем 142 см",
            "уместится в нишу 142 см",
        ):
            with self.subTest(text=text):
                self.assert_width(text, 142)

    def test_niche_in_meters(self) -> None:
        """«влезает в нишу метр» — предел 100 см, без «см» и без сравнительной фразы."""
        constraints = collect_constraints(find_amounts("влезает в нишу метр"))
        assert constraints.width is not None
        self.assertEqual(constraints.width.upper.value, 100)
        self.assertTrue(constraints.width.upper.inclusive)

    def test_explicit_construction_wins_over_niche(self) -> None:
        """Явная конструкция сильнее слова «ниша»: «ровно» остаётся точным."""
        exact = collect_constraints(find_amounts("ниша ровно 142 см"))
        assert exact.width is not None
        self.assertEqual(exact.width.lower.value, 142)
        self.assertEqual(exact.width.upper.value, 142)

        strict = collect_constraints(find_amounts("в нишу уже 142 см"))
        assert strict.width is not None
        self.assertFalse(strict.width.upper.inclusive, "«уже» — строгая граница")

    def test_table_width_word_is_still_exact(self) -> None:
        """Регресс: «стол шириной 142 см» — это сам стол, а не место под него."""
        constraints = collect_constraints(find_amounts("стол шириной 142 см"))
        assert constraints.width is not None
        self.assertEqual(constraints.width.lower.value, 142)
        self.assertEqual(constraints.width.upper.value, 142)

    def test_fit_word_near_huge_number_is_not_a_width(self) -> None:
        """«влезет в 5000» — не ширина 5000 см: неправдоподобное число не ширина."""
        constraints = collect_constraints(find_amounts("влезет в 5000"))
        self.assertIsNone(constraints.width)
        self.assertTrue(constraints.neutral, "число должно остаться нейтральным")

    def test_unrelated_meter_is_not_a_width(self) -> None:
        """Регресс: «уложусь в метр» — не условие о ширине (нет ниши и слова «ширина»)."""
        constraints = collect_constraints(find_amounts("кресло и лампа, уложусь в метр"))
        self.assertIsNone(constraints.width)


if __name__ == "__main__":
    unittest.main()
