"""Подбор комплектов: DFS с отсечением, топ-3, no_match."""

from __future__ import annotations

import itertools
import json
import random
import unittest

from app.catalog.index import build_index, bucket_for, filter_flag, filter_numeric
from app.catalog.plan import build_plan
from app.catalog.search import (
    CategoryState,
    choose_options,
    collect_states,
    enumerate_combinations,
    select_candidates,
    suffix_minimums,
)
from app.model import Bound, NumericRange, Slots
from tests.support import CATALOG_PATH

DOCUMENT = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
INDEX = build_index(DOCUMENT)
BY_ID = {listing["id"]: listing for listing in DOCUMENT}


def slots(**overrides) -> Slots:
    """Слоты с известными условиями по умолчанию (Таганрог, все три категории)."""
    base = dict(
        city_id="taganrog",
        budget_rub=15000,
        categories=("table", "chair", "lamp"),
    )
    base.update(overrides)
    return Slots(**base)


def validate_option(document: dict, slots_value: Slots) -> None:
    """Проверяет вариант на соответствие контракту и данным каталога."""
    listings = [BY_ID[listing_id] for listing_id in document["listing_ids"]]
    categories = [listing["category"] for listing in listings]
    assert sorted(categories) == sorted(slots_value.categories), categories
    assert len(set(document["listing_ids"])) == len(document["listing_ids"])
    for listing in listings:
        assert listing["city_id"] == slots_value.city_id
    assert document["total_rub"] == sum(listing["price_rub"] for listing in listings)
    assert document["total_rub"] <= slots_value.budget_rub
    assert document["explanation"].strip()


class CombinationsTest(unittest.TestCase):
    """DFS с отсечением: полнота и корректность отсечения."""

    def test_suffix_minimums(self) -> None:
        states = collect_states(slots(), INDEX)
        suffix = suffix_minimums(states)
        self.assertEqual(suffix[-1], 0)
        for position, state in enumerate(states):
            cheapest = state.cheapest.price_rub
            self.assertLessEqual(suffix[position], cheapest)

    def test_matches_naive_brute_force_on_small_budget(self) -> None:
        """Отсечение не теряет допустимые наборы: сверка с полным перебором."""
        constraints = [
            dict(),
            dict(width=NumericRange(upper=Bound(100, True))),
            dict(width=NumericRange(upper=Bound(80, True))),
            dict(adjustable_height=True),
            dict(adjustable_height=False),
            dict(color_temperature=NumericRange(lower=Bound(3000, True), upper=Bound(5000, True))),
        ]
        for constraint in constraints:
            for budget in (3000, 5000, 8000, 12200, 15000):
                current = slots(budget_rub=budget, **constraint)
                states = collect_states(current, INDEX)
                if any(state.empty for state in states):
                    self.assertEqual(enumerate_combinations(states, budget), [])
                    continue
                found = enumerate_combinations(states, budget)
                naive = [
                    combo
                    for combo in itertools.product(*(state.candidates for state in states))
                    if sum(listing.price_rub for listing in combo) <= budget
                ]
                with self.subTest(constraint=constraint, budget=budget):
                    self.assertEqual(len(found), min(len(naive), 200))
                    for combination in found:
                        self.assertLessEqual(sum(listing.price_rub for listing in combination), budget)

    def test_combinations_are_unique_and_sorted_by_total(self) -> None:
        combinations = enumerate_combinations(collect_states(slots(), INDEX), 15000)
        totals = [sum(listing.price_rub for listing in combination) for combination in combinations]
        self.assertEqual(totals, sorted(totals))
        compositions = {frozenset(listing.id for listing in combination) for combination in combinations}
        self.assertEqual(len(compositions), len(combinations))

    def test_budget_below_minimum_gives_nothing(self) -> None:
        self.assertEqual(enumerate_combinations(collect_states(slots(budget_rub=100), INDEX), 100), [])

    def test_exact_budget_boundary_is_inclusive(self) -> None:
        """Равенство бюджета допустимо: минимальный комплект проходит."""
        states = collect_states(slots(), INDEX)
        minimum = sum(state.cheapest.price_rub for state in states)
        combinations = enumerate_combinations(states, minimum)
        self.assertTrue(combinations)
        self.assertEqual(sum(listing.price_rub for listing in combinations[0]), minimum)

    def test_budget_one_ruble_less_gives_nothing(self) -> None:
        states = collect_states(slots(), INDEX)
        minimum = sum(state.cheapest.price_rub for state in states)
        self.assertEqual(enumerate_combinations(states, minimum - 1), [])

    def test_single_category_listing_is_a_valid_set(self) -> None:
        single = slots(categories=("lamp",))
        combinations = enumerate_combinations(collect_states(single, INDEX), 15000)
        self.assertTrue(combinations)
        self.assertEqual(len(combinations[0]), 1)


class ChooseOptionsTest(unittest.TestCase):
    """До трёх вариантов, различающихся составом."""

    def test_options_differ_in_composition(self) -> None:
        combinations = enumerate_combinations(collect_states(slots(), INDEX), 15000)
        chosen = choose_options(combinations)
        self.assertTrue(1 <= len(chosen) <= 3)
        compositions = [frozenset(listing.id for listing in option) for option in chosen]
        self.assertEqual(len(set(compositions)), len(compositions))

    def test_first_option_is_the_cheapest(self) -> None:
        combinations = enumerate_combinations(collect_states(slots(), INDEX), 15000)
        chosen = choose_options(combinations)
        self.assertEqual(
            sum(listing.price_rub for listing in chosen[0]),
            sum(listing.price_rub for listing in combinations[0]),
        )

    def test_single_combination_gives_single_option(self) -> None:
        combinations = enumerate_combinations(collect_states(slots(categories=("lamp",)), INDEX), 501)
        self.assertTrue(combinations)
        self.assertEqual(len(choose_options(combinations)), 1)


class PlanTest(unittest.TestCase):
    """Сквозной подбор: ok и no_match."""

    def test_reference_combination_is_reachable(self) -> None:
        """t12 + c04 + l09 = 12 200 ₽ ≤ 15 000 — такой комплект существует.

        Выдача начинается с самых дешёвых наборов (контракт разрешает любой порядок),
        поэтому проверяем именно допустимость эталонного состава.
        """
        states = collect_states(slots(), INDEX)
        exhaustive = enumerate_combinations(states, 15000, limit=10_000)
        ids = {frozenset(listing.id for listing in combination) for combination in exhaustive}
        self.assertIn(frozenset({"t12", "c04", "l09"}), ids)

    def test_ok_document_validates_against_catalog(self) -> None:
        document = build_plan(slots(), INDEX)
        self.assertEqual(document["status"], "ok")
        self.assertTrue(1 <= len(document["options"]) <= 3)
        for option in document["options"]:
            validate_option(option, slots())

    def test_option_ids_are_unique_across_options(self) -> None:
        document = build_plan(slots(), INDEX)
        compositions = [frozenset(option["listing_ids"]) for option in document["options"]]
        self.assertEqual(len(set(compositions)), len(compositions))

    def test_width_constraint_is_respected(self) -> None:
        current = slots(width=NumericRange(upper=Bound(100, True)))
        document = build_plan(current, INDEX)
        self.assertEqual(document["status"], "ok")
        for option in document["options"]:
            validate_option(option, current)
            for listing_id in option["listing_ids"]:
                if BY_ID[listing_id]["category"] == "table":
                    self.assertLessEqual(BY_ID[listing_id]["attributes"]["width_cm"], 100)

    def test_no_match_when_budget_too_small(self) -> None:
        document = build_plan(slots(budget_rub=1000), INDEX)
        self.assertEqual(document["status"], "no_match")
        self.assertNotIn("listing_ids", document)
        self.assertIn("бюджет", document["message"])

    def test_no_match_when_constraint_impossible(self) -> None:
        current = slots(width=NumericRange(upper=Bound(50, True)), budget_rub=50000)
        document = build_plan(current, INDEX)
        self.assertEqual(document["status"], "no_match")
        self.assertIn("ширин", document["message"])
        self.assertNotIn("listing_ids", document)

    def test_no_match_when_city_has_no_listings_of_category(self) -> None:
        """Город без нужной категории: сообщение называет категорию."""
        empty_index = build_index([listing for listing in DOCUMENT if listing["category"] != "lamp"])
        document = build_plan(slots(), empty_index)
        self.assertEqual(document["status"], "no_match")
        self.assertIn("ламп", document["message"])

    def test_no_match_mentions_what_to_change(self) -> None:
        document = build_plan(slots(budget_rub=1000), INDEX)
        self.assertIn("Можно", document["message"])
        self.assertTrue(document["message"].strip().endswith("."))

    def test_explanation_uses_catalog_facts(self) -> None:
        document = build_plan(slots(), INDEX)
        explanation = document["options"][0]["explanation"]
        self.assertIn("₽", explanation)
        self.assertIn("Итого", explanation)

    def test_plan_is_deterministic(self) -> None:
        first = build_plan(slots(), INDEX)
        second = build_plan(slots(), INDEX)
        self.assertEqual(first, second)

    def test_every_option_within_budget_for_random_dialogues(self) -> None:
        generator = random.Random(20261123)
        for _ in range(60):
            budget = generator.choice([1000, 3000, 5000, 8000, 12000, 15000, 20000])
            city = generator.choice(["taganrog", "azov", "rostov-na-donu"])
            categories = generator.choice([
                ("table",), ("chair",), ("lamp",),
                ("table", "chair"), ("table", "lamp"), ("table", "chair", "lamp"),
            ])
            current = slots(city_id=city, budget_rub=budget, categories=categories)
            document = build_plan(current, INDEX)
            with self.subTest(budget=budget, city=city, categories=categories):
                if document["status"] == "ok":
                    for option in document["options"]:
                        validate_option(option, current)
                else:
                    self.assertEqual(document["status"], "no_match")
                    self.assertNotIn("listing_ids", document)


class CandidateSelectionTest(unittest.TestCase):
    """Отбор кандидатов: ограничение применяется только к своей категории."""

    def test_without_constraint_all_listings_participate(self) -> None:
        states = {state.category: state for state in collect_states(slots(), INDEX)}
        bucket = bucket_for(INDEX, "taganrog", "table")
        self.assertEqual(len(states["table"].candidates), len(bucket.priced_items))

    def test_constraint_filters_only_its_category(self) -> None:
        states = {state.category: state for state in collect_states(slots(width=NumericRange(upper=Bound(100, True))), INDEX)}
        self.assertTrue(states["table"].constraint_applied)
        self.assertFalse(states["chair"].constraint_applied)
        self.assertFalse(states["lamp"].constraint_applied)

    def test_missing_attribute_does_not_pass_constraint(self) -> None:
        bucket = bucket_for(INDEX, "taganrog", "table")
        filtered = select_candidates(bucket, "table", slots(width=NumericRange(upper=Bound(100, True))))
        self.assertTrue(all(listing.numeric("width_cm") is not None for listing in filtered))

    def test_flag_false_selection(self) -> None:
        bucket = bucket_for(INDEX, "taganrog", "chair")
        filtered = select_candidates(bucket, "chair", slots(adjustable_height=False))
        self.assertTrue(filtered)
        self.assertTrue(all(listing.flag("adjustable_height") is False for listing in filtered))


if __name__ == "__main__":
    unittest.main()
