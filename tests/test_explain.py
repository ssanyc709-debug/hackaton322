"""Тексты ответа: объяснения и ``no_match`` собираются из фактов, а не из описаний.

Проверяется то, что видит покупатель и комиссия: сообщение должно быть
грамотным, называть препятствие и не содержать ни одного ID объявления.
"""

from __future__ import annotations

import json
import re
import unittest

from app.catalog import explain
from app.catalog.index import build_index
from app.catalog.plan import build_plan
from app.contract.request import parse_request
from app.model import Bound, NumericRange, Slots
from app.pipeline import answer
from app.config import Settings
from tests.support import CATALOG_PATH

DOCUMENT = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
BY_ID = {raw["id"]: raw for raw in DOCUMENT}
INDEX = build_index(DOCUMENT)
SETTINGS = Settings.load(environ={"ASSISTANT_OFFLINE": "1"})


def plan(text: str) -> dict:
    """Ответ помощника на одну реплику покупателя."""
    return answer(
        request=parse_request({"messages": [{"role": "user", "text": text}]}),
        index=INDEX,
        settings=SETTINGS,
        deadline=1e18,
    )


class CityLabelTest(unittest.TestCase):
    """Предложный падеж: «в Таганроге», а не «в Таганрог»."""

    def test_three_catalog_cities(self) -> None:
        self.assertEqual(explain.city_label("Таганрог"), "в Таганроге")
        self.assertEqual(explain.city_label("Азов"), "в Азове")
        self.assertEqual(explain.city_label("Ростов-на-Дону"), "в Ростове-на-Дону")

    def test_feminine_and_soft_sign(self) -> None:
        self.assertEqual(explain.city_label("Москва"), "в Москве")
        self.assertEqual(explain.city_label("Казань"), "в Казани")

    def test_empty_city(self) -> None:
        self.assertEqual(explain.city_label(""), "в указанном городе")


class DescribeRangeTest(unittest.TestCase):
    """Числовое условие словами: включительность границы видна в тексте."""

    def test_inclusive_upper(self) -> None:
        self.assertEqual(
            explain.describe_range("см", NumericRange(upper=Bound(100, True))),
            "не более 100 см",
        )

    def test_strict_upper(self) -> None:
        self.assertEqual(
            explain.describe_range("см", NumericRange(upper=Bound(100, False))),
            "строго меньше 100 см",
        )

    def test_lower_bounds(self) -> None:
        self.assertEqual(
            explain.describe_range("K", NumericRange(lower=Bound(3000, True))),
            "не менее 3000 K",
        )
        self.assertEqual(
            explain.describe_range("K", NumericRange(lower=Bound(3000, False))),
            "больше 3000 K",
        )

    def test_range_and_exact_value(self) -> None:
        self.assertEqual(
            explain.describe_range("K", NumericRange(lower=Bound(3000, True), upper=Bound(5000, True))),
            "от 3000 до 5000 K",
        )
        self.assertEqual(
            explain.describe_range("K", NumericRange(lower=Bound(4000, True), upper=Bound(4000, True))),
            "ровно 4000 K",
        )

    def test_no_bounds(self) -> None:
        self.assertEqual(explain.describe_range("см", NumericRange()), "любое значение см")


class DescribeListingTest(unittest.TestCase):
    """Строка о товаре: факты каталога, включая «отрицательные» значения."""

    def test_width_and_price(self) -> None:
        text = explain.describe_listing(INDEX.by_id["t12"])
        self.assertIn("Стол", text)
        self.assertIn("95 см", text)
        self.assertIn("6\u00a0500\u00a0₽", text)

    def test_false_flag_is_described_not_dropped(self) -> None:
        listing = next(
            item for item in INDEX.listings
            if item.category == "chair" and item.flag("adjustable_height") is False
        )
        self.assertIn("без регулировки высоты", explain.describe_listing(listing))

    def test_missing_attribute_says_so(self) -> None:
        listing = next(
            item for item in INDEX.listings
            if item.category == "table" and item.numeric("width_cm") is None
        )
        self.assertIn("ширина не указана", explain.describe_listing(listing))

    def test_lamp_with_temperature(self) -> None:
        text = explain.describe_listing(INDEX.by_id["l09"])
        self.assertIn("Лампа с температурой 4000 K", text)
        self.assertIn("1\u00a0500\u00a0₽", text)

    def test_missing_detail_does_not_glue_to_the_noun(self) -> None:
        """«Лампа, температура не указана, за 650 ₽» — а не «Лампа температура …»."""
        listing = next(
            item for item in INDEX.listings
            if item.category == "lamp" and item.numeric("color_temperature_k") is None
        )
        text = explain.describe_listing(listing)
        self.assertTrue(text.startswith("Лампа, температура не указана, за "), text)
        self.assertNotIn("Лампа температура", text)

    def test_explanation_has_no_ids(self) -> None:
        document = plan("Стол, кресло и лампа. Таганрог. До 15 000 ₽.")
        for option in document["options"]:
            for listing_id in option["listing_ids"]:
                self.assertNotIn(listing_id, option["explanation"])


class NoMatchTextTest(unittest.TestCase):
    """``no_match``: называет препятствие, суммы из каталога, и не содержит ID."""

    def test_budget_shortage_explains_the_gap(self) -> None:
        document = plan("Стол, кресло и лампа. Таганрог. До 900 ₽.")
        self.assertEqual(document["status"], "no_match", document)
        message = document["message"]
        self.assertIn("4\u00a0700\u00a0₽", message)
        self.assertIn("не хватает", message)
        self.assertIn("увеличить бюджет", message)

    def test_width_constraint_names_the_obstacle(self) -> None:
        document = plan("Стол не шире 30 см, кресло и лампа. Таганрог. До 20 000 ₽.")
        self.assertEqual(document["status"], "no_match", document)
        message = document["message"]
        self.assertIn("в Таганроге", message)
        self.assertNotIn("в Таганрог ", message)
        self.assertIn("не более 30 см", message)
        self.assertIn("ближайшая ширина", message)
        self.assertIn("расширить допустимую ширину", message)

    def test_missing_category_names_it(self) -> None:
        """Город без нужной категории: сообщение называет её, а не молчит."""
        without_lamps = [
            raw for raw in DOCUMENT
            if not (raw["city_id"] == "azov" and raw["category"] == "lamp")
        ]
        document = answer(
            request=parse_request({"messages": [{"role": "user", "text": "Стол, кресло и лампа. Азов. До 30 000 ₽."}]}),
            index=build_index(without_lamps),
            settings=SETTINGS,
            deadline=1e18,
        )
        self.assertEqual(document["status"], "no_match", document)
        self.assertIn("в Азове", document["message"])
        self.assertIn("«лампы»", document["message"])

    def test_no_ids_in_no_match(self) -> None:
        for text in (
            "Стол, кресло и лампа. Таганрог. До 900 ₽.",
            "Стол не шире 30 см. Таганрог. До 20 000 ₽.",
            "Лампа 8000 к, стол. Ростов-на-Дону. До 20 000.",
        ):
            with self.subTest(text=text):
                document = plan(text)
                self.assertEqual(document["status"], "no_match", document)
                for listing_id in BY_ID:
                    self.assertNotIn(listing_id, document["message"])
                self.assertIsNone(re.search(r"\b(?:demo|t|c|l)\d", document["message"]))

    def test_message_is_one_paragraph_without_newlines(self) -> None:
        document = plan("Стол, кресло и лампа. Таганрог. До 900 ₽.")
        self.assertNotIn("\n", document["message"])
        self.assertTrue(document["message"].endswith("."))


class BuildPlanGuardTest(unittest.TestCase):
    """``build_plan`` не вызывается с неполными условиями (это был бы баг пайплайна)."""

    def test_incomplete_slots_raise(self) -> None:
        with self.assertRaises(ValueError):
            build_plan(Slots(city_id="taganrog", categories=("table",)), INDEX)


if __name__ == "__main__":
    unittest.main()
