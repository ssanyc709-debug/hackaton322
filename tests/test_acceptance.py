"""Приёмочный стенд по CONTRACT §1–§4 внутри общего прогона ``unittest``.

Здесь два уровня:

1. **Тесты самого Oracle** — он проверяет §4 независимо от решения, поэтому
   его правила (включительные и строгие границы, ``false``/``0``, неизвестная
   характеристика) обязаны быть верными сами по себе.
2. **Короткий прогон стенда** на фиксированном зерне: реальный каталог плюс
   два синтетических, все три статуса и технический вход §1.

Полный прогон (сотни запросов, разные зёрна) запускается отдельно:
``python bench/acceptance.py --cases 40``.
"""

from __future__ import annotations

import json
import random
import unittest
from pathlib import Path

from bench.acceptance import (
    CATALOG_PATH,
    CITY_PHRASES,
    Bound,
    FlagCondition,
    Intent,
    NumericCondition,
    catalog_schema_errors,
    cheapest_bundle,
    complete_bundles,
    expected_status,
    generate_catalog,
    make_dialogue,
    matches,
    run_acceptance,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def listing(
    listing_id: str,
    category: str = "table",
    price: int = 1000,
    city: str = "taganrog",
    **attributes: object,
) -> dict[str, object]:
    """Объявление каталога для проверок Oracle."""
    return {
        "id": listing_id,
        "category": category,
        "title": f"Товар {listing_id}",
        "description": "",
        "price_rub": price,
        "city_id": city,
        "attributes": dict(attributes),
    }


class OracleNumericTest(unittest.TestCase):
    """§2.2: включительные и строгие границы, отсутствующее значение."""

    def test_inclusive_upper_boundary_is_allowed(self) -> None:
        item = listing("t1", width_cm=100)
        self.assertTrue(matches(item, NumericCondition(upper=Bound(100, True))))
        self.assertTrue(matches(item, NumericCondition(upper=Bound(101, True))))

    def test_strict_upper_boundary_is_excluded(self) -> None:
        item = listing("t1", width_cm=100)
        self.assertFalse(matches(item, NumericCondition(upper=Bound(100, False))))
        self.assertTrue(matches(item, NumericCondition(upper=Bound(101, False))))

    def test_inclusive_lower_boundary_is_allowed(self) -> None:
        item = listing("t1", width_cm=100)
        self.assertTrue(matches(item, NumericCondition(lower=Bound(100, True))))
        self.assertFalse(matches(item, NumericCondition(lower=Bound(100, False))))

    def test_interval_includes_both_ends(self) -> None:
        condition = NumericCondition(lower=Bound(90, True), upper=Bound(110, True))
        self.assertTrue(matches(listing("t1", width_cm=90), condition))
        self.assertTrue(matches(listing("t2", width_cm=110), condition))
        self.assertFalse(matches(listing("t3", width_cm=89), condition))
        self.assertFalse(matches(listing("t4", width_cm=111), condition))

    def test_unknown_attribute_never_satisfies_a_constraint(self) -> None:
        item = listing("t1")
        self.assertFalse(matches(item, NumericCondition(upper=Bound(100, True))))
        self.assertTrue(matches(item, None), "без ограничения объявление участвует в подборе")

    def test_zero_is_a_value_not_absence(self) -> None:
        item = listing("t1", width_cm=0)
        self.assertTrue(matches(item, NumericCondition(upper=Bound(10, True))))
        self.assertTrue(matches(item, NumericCondition(lower=Bound(0, True))))

    def test_boolean_is_not_a_number(self) -> None:
        item = listing("t1", width_cm=True)
        self.assertFalse(matches(item, NumericCondition(upper=Bound(10, True))))


class OracleFlagTest(unittest.TestCase):
    """§2.2: ``false`` — полноценное значение, а не «нет данных»."""

    def test_false_matches_false_and_not_true(self) -> None:
        item = listing("c1", category="chair", adjustable_height=False)
        self.assertTrue(matches(item, FlagCondition(False)))
        self.assertFalse(matches(item, FlagCondition(True)))

    def test_unknown_flag_matches_neither(self) -> None:
        item = listing("c1", category="chair")
        self.assertFalse(matches(item, FlagCondition(True)))
        self.assertFalse(matches(item, FlagCondition(False)))


class OracleStatusTest(unittest.TestCase):
    """§4: clarify → ok/no_match; равенство бюджета допустимо (§3)."""

    def setUp(self) -> None:
        self.catalog = [
            listing("t1", price=1000),
            listing("c1", category="chair", price=500, adjustable_height=True),
            listing("l1", category="lamp", price=300, color_temperature_k=4000),
        ]

    def intent(self, **changes: object) -> Intent:
        base: dict[str, object] = {
            "text": "Стол, кресло и лампа в Таганроге до 1 800 ₽",
            "city_id": "taganrog",
            "budget_rub": 1800,
            "categories": ("table", "chair", "lamp"),
            "conditions": {},
        }
        base.update(changes)
        return Intent(**base)  # type: ignore[arg-type]

    def test_budget_equality_is_ok(self) -> None:
        self.assertEqual(expected_status(self.intent(), self.catalog), "ok")
        self.assertEqual(cheapest_bundle(self.intent(), self.catalog), 1800)

    def test_one_ruble_short_is_no_match(self) -> None:
        shortage = self.intent(budget_rub=1799)
        self.assertEqual(expected_status(shortage, self.catalog), "no_match")
        self.assertEqual(complete_bundles(shortage, self.catalog), [])

    def test_missing_budget_is_clarify_even_when_bundle_exists(self) -> None:
        status = expected_status(self.intent(budget_rub=None), self.catalog)
        self.assertEqual(status, "clarify")

    def test_missing_city_and_categories_are_clarify(self) -> None:
        self.assertIn("city_id", self.intent(city_id=None).missing_fields())
        self.assertIn("categories", self.intent(categories=()).missing_fields())

    def test_constraint_excludes_unknown_and_wrong_values(self) -> None:
        catalog = self.catalog + [listing("t2", price=100)]
        condition = {"table": NumericCondition(upper=Bound(90, True))}
        intent = self.intent(categories=("table",), budget_rub=10_000, conditions=condition)
        bundles = complete_bundles(intent, catalog)
        self.assertEqual(bundles, [], "стол без width_cm не подтверждает ограничение")

    def test_bundle_uses_listings_of_the_buyers_city_only(self) -> None:
        catalog = [listing("t-other", price=10, city="azov")]
        intent = self.intent(categories=("table",), budget_rub=10_000)
        self.assertEqual(expected_status(intent, catalog), "no_match")


class ContractRehearsalTest(unittest.TestCase):
    """Короткий прогон стенда: самопроверка ответа по §1–§4.

    Полный прогон — ``python bench/acceptance.py``; здесь фиксированное зерно
    и малое число запросов, чтобы тест оставался быстрым и воспроизводимым.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_acceptance(
            case_count=8,
            seed=2026_11_21,
            catalog_path=CATALOG_PATH,
            synthetic_catalogs=2,
            include_inputs=True,
            quiet=True,
        )

    def test_generated_catalogs_follow_the_schema(self) -> None:
        """Стенд не имеет права проверять решение на невалидном каталоге (§2)."""
        self.assertEqual(self.report["schema_problems"], [])

    def test_no_contract_violations(self) -> None:
        """Главный тест: каждый ответ §4 соответствует каталогу и запросу."""
        failures = self.report["failures"]
        message = json.dumps(failures[:3], ensure_ascii=False, indent=1)
        self.assertEqual(failures, [], f"нарушения §4:\n{message}")

    def test_technical_failures_are_clean(self) -> None:
        """§1: ненулевой код, пустой stdout, без трассировки в stderr."""
        failed = [item for item in self.report["technical"] if not item["ok"]]
        self.assertEqual(failed, [], f"технические входы §1:\n{failed}")

    def test_output_is_deterministic(self) -> None:
        """§1: тот же вход — побайтово тот же ответ."""
        self.assertTrue(self.report["determinism"], "прогон детерминизма не выполнялся")
        failed = [item for item in self.report["determinism"] if not item["ok"]]
        self.assertEqual(failed, [])

    def test_all_three_statuses_are_exercised(self) -> None:
        """Защита от «пустого» зелёного прогона: проверены все три статуса."""
        statuses = self.report["by_status"]
        for status in ("ok", "no_match", "clarify"):
            with self.subTest(status=status):
                self.assertIn(status, statuses, f"статус {status} не встретился")

    def test_dialogs_keep_conditions(self) -> None:
        """§3: после ответа на уточнение условия не теряются и ответ допустим."""
        dialogs = self.report["dialogs"]
        self.assertEqual(len(dialogs), 3, "в коротком прогоне ожидались три диалога")
        failed = [item for item in dialogs if not item["ok"]]
        self.assertEqual(failed, [], f"диалоги §3:\n{failed}")
        asked = {item["name"].rsplit(" ", 1)[-1] for item in dialogs}
        self.assertEqual(asked, {"budget_rub", "city_id", "categories"})


class DialogueScenarioTest(unittest.TestCase):
    """Замысел диалога: ровно одно недостающее условие, ответ его закрывает."""

    def setUp(self) -> None:
        self.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    def test_first_message_misses_exactly_one_field(self) -> None:
        rng = random.Random(2026_11_21)
        for missing in ("budget_rub", "city_id", "categories"):
            with self.subTest(missing=missing):
                dialogue = make_dialogue(rng, self.catalog, missing)
                self.assertEqual(dialogue.first.missing_fields(), (missing,))
                self.assertEqual(dialogue.final.missing_fields(), ())

    def test_answer_fills_the_missing_field(self) -> None:
        rng = random.Random(5)
        dialogue = make_dialogue(rng, self.catalog, "city_id")
        self.assertIsNone(dialogue.first.city_id)
        self.assertIsNotNone(dialogue.final.city_id)
        self.assertIn(dialogue.answer, CITY_PHRASES.values())

    def test_budget_dialogue_keeps_city_and_categories(self) -> None:
        rng = random.Random(11)
        dialogue = make_dialogue(rng, self.catalog, "budget_rub")
        self.assertEqual(dialogue.first.city_id, dialogue.final.city_id)
        self.assertEqual(dialogue.first.categories, dialogue.final.categories)
        self.assertIsNotNone(dialogue.final.budget_rub)


class CatalogGeneratorTest(unittest.TestCase):
    """Сгенерированный каталог — валиден по §2 при разных зёрнах."""

    def test_generated_catalog_is_schema_valid(self) -> None:
        for seed in (1, 2, 3, 2026_11_21):
            with self.subTest(seed=seed):
                catalog = generate_catalog(random.Random(seed), 30)
                self.assertEqual(len(catalog), 30)
                self.assertEqual(catalog_schema_errors(catalog), [])

    def test_attributes_do_not_use_nested_values(self) -> None:
        """§2: в attributes нет вложенных объектов, массивов и null."""
        catalog = generate_catalog(random.Random(9), 60)
        for item in catalog:
            for value in item["attributes"].values():  # type: ignore[union-attr]
                with self.subTest(listing=item["id"]):
                    self.assertIsInstance(value, (str, int, float, bool))
                    self.assertNotIsInstance(value, (list, dict))


if __name__ == "__main__":
    unittest.main()
