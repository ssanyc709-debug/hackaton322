"""Индекс каталога, бинарный поиск по границам и фильтры по характеристикам."""

from __future__ import annotations

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

from app.catalog.index import (
    EMPTY_BUCKET,
    binary_search_first_ge,
    binary_search_first_strictly_greater,
    binary_search_last_le,
    bucket_for,
    build_index,
    catalog_digest,
    filter_flag,
    filter_numeric,
    forget_index,
    get_index,
)
from app.model import Bound, CatalogError, NumericRange
from tests.support import CATALOG_PATH

DOCUMENT = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def range_of(lower: int | None = None, upper: int | None = None, *, lower_inc: bool = True,
             upper_inc: bool = True) -> NumericRange:
    """Удобная сборка условия: границы включительны по умолчанию («до»/«от»)."""
    return NumericRange(
        lower=Bound(lower, lower_inc) if lower is not None else None,
        upper=Bound(upper, upper_inc) if upper is not None else None,
    )


class BinarySearchTest(unittest.TestCase):
    """CORE-функции границ: свойства, крайние случаи, сверка с перебором."""

    @staticmethod
    def brute_first_ge(values: list[int], bound: int) -> int:
        for index, value in enumerate(values):
            if value >= bound:
                return index
        return len(values)

    @staticmethod
    def brute_last_le(values: list[int], bound: int) -> int:
        result = -1
        for index, value in enumerate(values):
            if value <= bound:
                result = index
        return result

    @staticmethod
    def brute_first_gt(values: list[int], bound: int) -> int:
        for index, value in enumerate(values):
            if value > bound:
                return index
        return len(values)

    def test_first_ge_matches_brute_force_on_random_arrays(self) -> None:
        generator = random.Random(20261120)
        for _ in range(200):
            values = sorted(generator.randint(1, 200) for _ in range(generator.randint(0, 30)))
            for bound in range(0, 205, 5):
                with self.subTest(values=values, bound=bound):
                    self.assertEqual(binary_search_first_ge(values, bound), self.brute_first_ge(values, bound))

    def test_last_le_matches_brute_force_on_random_arrays(self) -> None:
        generator = random.Random(20261121)
        for _ in range(200):
            values = sorted(generator.randint(1, 200) for _ in range(generator.randint(0, 30)))
            for bound in range(0, 205, 5):
                with self.subTest(values=values, bound=bound):
                    self.assertEqual(binary_search_last_le(values, bound), self.brute_last_le(values, bound))

    def test_edge_cases(self) -> None:
        self.assertEqual(binary_search_first_ge([], 5), 0)
        self.assertEqual(binary_search_last_le([], 5), -1)
        self.assertEqual(binary_search_first_ge([10, 20], 5), 0)
        self.assertEqual(binary_search_first_ge([10, 20], 25), 2)
        self.assertEqual(binary_search_last_le([10, 20], 5), -1)
        self.assertEqual(binary_search_last_le([10, 20], 25), 1)
        # дубликаты: самый левый / самый правый
        self.assertEqual(binary_search_first_ge([10, 10, 10, 20], 10), 0)
        self.assertEqual(binary_search_last_le([10, 10, 10, 20], 10), 2)

    def test_first_strictly_greater_matches_brute_force(self) -> None:
        generator = random.Random(20261122)
        for _ in range(120):
            values = sorted(generator.randint(1, 120) for _ in range(generator.randint(0, 25)))
            for bound in range(0, 125, 5):
                with self.subTest(values=values, bound=bound):
                    self.assertEqual(
                        binary_search_first_strictly_greater(values, bound),
                        self.brute_first_gt(values, bound),
                    )

    def test_strict_boundary_with_duplicates(self) -> None:
        values = [95, 100, 100, 100, 120]
        self.assertEqual(binary_search_first_strictly_greater(values, 100), 4)
        self.assertEqual(binary_search_first_ge(values, 100), 1)
        self.assertEqual(binary_search_last_le(values, 100), 3)


class IndexBuildTest(unittest.TestCase):
    """Построение индекса, валидация каталога, кэш по дайджесту."""

    @classmethod
    def setUpClass(cls) -> None:
        forget_index()
        cls.index = build_index(DOCUMENT)

    def test_index_covers_whole_catalog(self) -> None:
        self.assertEqual(self.index.size, 111)
        self.assertEqual(len(self.index.buckets), 9)  # 3 города × 3 категории
        self.assertEqual(len(self.index.by_id), 111)
        self.assertEqual(self.index.cities(), ("azov", "rostov-na-donu", "taganrog"))
        self.assertEqual(self.index.categories(), ("chair", "lamp", "table"))

    def test_taganrog_table_bucket_is_ready_for_bisect(self) -> None:
        bucket = bucket_for(self.index, "taganrog", "table")
        self.assertEqual(len(bucket.priced_items), 13)
        self.assertEqual(bucket.unknown_count, 3)
        self.assertEqual(bucket.numeric_values, tuple(sorted(bucket.numeric_values)))
        self.assertEqual(len(bucket.numeric_values), 10)
        self.assertIn("t12", {listing.id for listing in bucket.numeric_items})

    def test_unknown_attribute_is_counted_not_dropped(self) -> None:
        """9 объявлений каждой категории без характеристики видны, но отдельно."""
        total_unknown = sum(
            bucket.unknown_count for (_, category), bucket in self.index.buckets.items() if category == "table"
        )
        self.assertEqual(total_unknown, 9)
        self.assertTrue(any(bucket.unknown_count for bucket in self.index.buckets.values()))

    def test_cheapest_listing_of_bucket(self) -> None:
        bucket = bucket_for(self.index, "taganrog", "lamp")
        self.assertEqual(bucket.cheapest.id, "demo-1-3-04")  # 500 ₽, 5000 K
        self.assertIsNone(EMPTY_BUCKET.cheapest)

    def test_missing_bucket_returns_empty(self) -> None:
        self.assertIs(bucket_for(self.index, "moscow", "table"), EMPTY_BUCKET)
        self.assertEqual(bucket_for(self.index, "taganrog", "sofa"), EMPTY_BUCKET)

    def test_digest_is_content_based(self) -> None:
        same = copy.deepcopy(DOCUMENT)
        self.assertEqual(catalog_digest(DOCUMENT), catalog_digest(same))
        changed = copy.deepcopy(DOCUMENT)
        changed[0]["price_rub"] = changed[0]["price_rub"] + 1
        self.assertNotEqual(catalog_digest(DOCUMENT), catalog_digest(changed))

    def test_memo_reuses_index_until_content_changes(self) -> None:
        forget_index()
        first = get_index(DOCUMENT)
        self.assertIs(get_index(DOCUMENT), first)
        changed = copy.deepcopy(DOCUMENT)
        changed[0]["price_rub"] = changed[0]["price_rub"] + 1
        second = get_index(changed)
        self.assertIsNot(second, first)
        self.assertNotEqual(second.digest, first.digest)


class AttributeTypeTest(unittest.TestCase):
    """§2: типы ``attributes`` — часть контракта, а не пожелание.

    Битый тип характеристики — технический сбой (:class:`CatalogError`), а не
    «характеристика неизвестна»: иначе ``width_cm: "90"`` молча выключил бы
    ограничение покупателя, и ответ выглядел бы законным (та же логика, что
    уже действует для ``price_rub``).
    """

    def build_with(self, attributes: object, category: str = "table") -> None:
        """Пытается построить индекс из одного объявления с такими характеристиками."""
        build_index([dict(DOCUMENT[0], category=category, attributes=attributes)])

    def test_valid_scalars_are_accepted(self) -> None:
        """Строка, число и boolean в дополнительных ключах допустимы (§2)."""
        item = dict(
            DOCUMENT[0],
            attributes={
                "width_cm": 90,
                "material": "дуб",
                "height_cm": 75,
                "folding": True,
                "weight_kg": 12.5,
            },
        )
        listing = build_index([item]).listings[0]
        self.assertEqual(listing.numeric("width_cm"), 90)
        self.assertEqual(listing.attributes["material"], "дуб")
        self.assertIs(listing.flag("folding"), True)

    def test_width_as_string_is_rejected(self) -> None:
        with self.assertRaises(CatalogError):
            self.build_with({"width_cm": "90"})

    def test_width_zero_is_rejected(self) -> None:
        with self.assertRaises(CatalogError):
            self.build_with({"width_cm": 0})

    def test_width_boolean_is_rejected(self) -> None:
        """``True`` — не «ширина 1 см»."""
        with self.assertRaises(CatalogError):
            self.build_with({"width_cm": True})

    def test_fractional_width_is_rejected(self) -> None:
        with self.assertRaises(CatalogError):
            self.build_with({"width_cm": 90.5})

    def test_integral_float_width_is_normalised(self) -> None:
        """``90.0`` математически целое и по схеме валидно, поэтому это ширина 90."""
        listing = build_index([dict(DOCUMENT[0], attributes={"width_cm": 90.0})]).listings[0]
        self.assertEqual(listing.numeric("width_cm"), 90)

    def test_temperature_as_string_is_rejected(self) -> None:
        with self.assertRaises(CatalogError):
            self.build_with({"color_temperature_k": "4000"}, category="lamp")

    def test_flag_as_number_is_rejected(self) -> None:
        """``1`` вместо ``true`` — нарушение схемы: ``false`` и ``0`` не путаются."""
        with self.assertRaises(CatalogError):
            self.build_with({"adjustable_height": 1}, category="chair")

    def test_nested_or_null_value_is_rejected(self) -> None:
        for value in ({"cm": 90}, [90], None):
            with self.subTest(value=value):
                with self.assertRaises(CatalogError):
                    self.build_with({"extra": value})

    def test_unknown_key_with_zero_and_false_is_allowed(self) -> None:
        """Дополнительные ключи §2 не ограничивает: ``0`` и ``false`` там валидны."""
        listing = build_index(
            [dict(DOCUMENT[0], attributes={"height_cm": 0, "folding": False})]
        ).listings[0]
        self.assertEqual(listing.attributes["height_cm"], 0)
        self.assertIs(listing.flag("folding"), False)


class DiskCacheTest(unittest.TestCase):
    """Кэш индекса: ключ — содержимое каталога, перестройка при изменении."""

    def setUp(self) -> None:
        forget_index()
        self.temporary = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temporary.name)

    def tearDown(self) -> None:
        forget_index()
        self.temporary.cleanup()

    def test_second_process_gets_same_layout_from_cache(self) -> None:
        fresh = get_index(DOCUMENT, cache_dir=self.cache_dir)
        self.assertFalse(fresh.from_cache)
        files = list(self.cache_dir.glob("index-*.json"))
        self.assertEqual(len(files), 1, "файл кэша должен появиться после первого построения")

        forget_index()  # имитируем новый процесс
        cached = get_index(DOCUMENT, cache_dir=self.cache_dir)
        self.assertTrue(cached.from_cache)
        self.assertEqual(cached.digest, fresh.digest)
        for key, bucket in fresh.buckets.items():
            with self.subTest(bucket=key):
                self.assertEqual(cached.buckets[key].numeric_values, bucket.numeric_values)
                self.assertEqual(
                    tuple(listing.id for listing in cached.buckets[key].numeric_items),
                    tuple(listing.id for listing in bucket.numeric_items),
                )
                self.assertEqual(
                    tuple(listing.id for listing in cached.buckets[key].flags_false),
                    tuple(listing.id for listing in bucket.flags_false),
                )
                self.assertEqual(cached.buckets[key].unknown_count, bucket.unknown_count)

    def test_cache_is_rebuilt_when_catalog_content_changes(self) -> None:
        get_index(DOCUMENT, cache_dir=self.cache_dir)
        changed = copy.deepcopy(DOCUMENT)
        changed[0]["price_rub"] = changed[0]["price_rub"] + 500
        forget_index()

        index = get_index(changed, cache_dir=self.cache_dir)
        self.assertFalse(index.from_cache, "изменённый каталог не должен попадать в старый кэш")
        self.assertEqual(index.by_id[changed[0]["id"]].price_rub, changed[0]["price_rub"])

    def test_broken_cache_file_falls_back_to_rebuild(self) -> None:
        fresh = get_index(DOCUMENT, cache_dir=self.cache_dir)
        path = next(self.cache_dir.glob("index-*.json"))
        path.write_text("{ битый json", encoding="utf-8")
        forget_index()

        rebuilt = get_index(DOCUMENT, cache_dir=self.cache_dir)
        self.assertFalse(rebuilt.from_cache)
        self.assertEqual(rebuilt.digest, fresh.digest)

    def test_cache_with_foreign_digest_is_ignored(self) -> None:
        get_index(DOCUMENT, cache_dir=self.cache_dir)
        path = next(self.cache_dir.glob("index-*.json"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["digest"] = "0" * 64
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        forget_index()

        index = get_index(DOCUMENT, cache_dir=self.cache_dir)
        self.assertFalse(index.from_cache)

    def test_unwritable_cache_dir_does_not_break_index(self) -> None:
        blocker = self.cache_dir / "not-a-dir"
        blocker.write_text("x", encoding="utf-8")
        index = get_index(DOCUMENT, cache_dir=blocker / "nested")
        self.assertFalse(index.from_cache)
        self.assertEqual(index.size, 111)

    def test_duplicate_ids_are_rejected(self) -> None:
        duplicated = [dict(DOCUMENT[0]), dict(DOCUMENT[1])]
        duplicated[1]["id"] = duplicated[0]["id"]
        with self.assertRaises(CatalogError) as ctx:
            build_index(duplicated)
        self.assertIn("повторяющийся id", str(ctx.exception))

    def test_invalid_listing_fields_are_rejected(self) -> None:
        cases = {
            "price_zero": {"price_rub": 0},
            "price_bool": {"price_rub": True},
            "price_str": {"price_rub": "6500"},
            "category": {"category": "sofa"},
            "title": {"title": "  "},
            "description": {"description": None},
            "attributes": {"attributes": []},
            "id": {"id": ""},
        }
        for name, patch in cases.items():
            with self.subTest(case=name):
                broken = [dict(DOCUMENT[0], **patch)]
                with self.assertRaises(CatalogError):
                    build_index(broken)

    def test_extra_or_missing_fields_are_rejected(self) -> None:
        with self.assertRaises(CatalogError):
            build_index([dict(DOCUMENT[0], rating=5)])
        without = dict(DOCUMENT[0])
        without.pop("attributes")
        with self.assertRaises(CatalogError):
            build_index([without])

    def test_build_is_deterministic(self) -> None:
        first = build_index(DOCUMENT)
        second = build_index(DOCUMENT)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(
            tuple(listing.id for listing in first.listings),
            tuple(listing.id for listing in second.listings),
        )


class NumericFilterTest(unittest.TestCase):
    """Границы §2.2: включительные «до/не более/не шире/от», строгие «уже/шире/больше»."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = build_index(DOCUMENT)
        cls.tables = bucket_for(cls.index, "taganrog", "table")
        cls.lamps = bucket_for(cls.index, "taganrog", "lamp")

    def widths(self, constraint: NumericRange) -> list[int]:
        return [listing.numeric("width_cm") for listing in filter_numeric(self.tables, constraint)]  # type: ignore[misc]

    def temperatures(self, constraint: NumericRange) -> list[int]:
        return [listing.numeric("color_temperature_k") for listing in filter_numeric(self.lamps, constraint)]  # type: ignore[misc]

    def test_inclusive_upper_bound_keeps_exactly_100(self) -> None:
        widths = self.widths(range_of(upper=100))
        self.assertEqual(sorted(widths), [60, 80, 80, 95, 95, 95, 100, 100])
        self.assertNotIn(110, widths)
        self.assertNotIn(120, widths)

    def test_strict_upper_bound_drops_100(self) -> None:
        widths = self.widths(range_of(upper=100, upper_inc=False))
        self.assertEqual(sorted(widths), [60, 80, 80, 95, 95, 95])
        self.assertNotIn(100, widths)

    def test_strict_lower_bound_drops_the_boundary_value(self) -> None:
        widths = self.widths(range_of(lower=80, lower_inc=False))
        self.assertEqual(sorted(widths), [95, 95, 95, 100, 100, 110, 120])

    def test_inclusive_lower_bound_keeps_the_boundary_value(self) -> None:
        widths = self.widths(range_of(lower=80))
        self.assertEqual(sorted(widths), [80, 80, 95, 95, 95, 100, 100, 110, 120])

    def test_range_includes_both_ends(self) -> None:
        temperatures = self.temperatures(range_of(lower=3000, upper=5000))
        self.assertEqual(sorted(temperatures), [3000, 3000, 3000, 4000, 4000, 4000, 5000, 5000])
        self.assertNotIn(6500, temperatures)

    def test_exact_value(self) -> None:
        self.assertEqual(self.widths(range_of(lower=95, upper=95)), [95, 95, 95])

    def test_missing_attribute_never_passes_a_constraint(self) -> None:
        """Неизвестная ширина не подтверждает «не шире метра» (3 объявления без ключа)."""
        everything = len(self.widths(range_of(lower=1, upper=10_000)))
        self.assertEqual(everything, 10)
        self.assertEqual(everything + self.tables.unknown_count, len(self.tables.priced_items))

    def test_without_constraint_all_listings_participate(self) -> None:
        """Без ограничения объявления без характеристики участвуют (их видно в priced_items)."""
        self.assertEqual(len(self.tables.priced_items), 13)
        self.assertEqual(len({listing.id for listing in self.tables.priced_items}), 13)

    def test_impossible_range_is_empty(self) -> None:
        self.assertEqual(filter_numeric(self.tables, range_of(lower=200, upper=300)), ())
        self.assertEqual(filter_numeric(self.tables, range_of(lower=120, upper=60)), ())


class FlagFilterTest(unittest.TestCase):
    """``adjustable_height``: два хеш-множества, ``false`` — валидные данные."""

    @classmethod
    def setUpClass(cls) -> None:
        index = build_index(DOCUMENT)
        cls.chairs = bucket_for(index, "taganrog", "chair")

    def test_false_is_returned_not_dropped_by_truthiness(self) -> None:
        without_adjustment = filter_flag(self.chairs, False)
        self.assertEqual(len(without_adjustment), 5)
        for listing in without_adjustment:
            self.assertIs(listing.attributes["adjustable_height"], False)

    def test_true_returns_only_true(self) -> None:
        with_adjustment = filter_flag(self.chairs, True)
        self.assertEqual(len(with_adjustment), 5)
        for listing in with_adjustment:
            self.assertIs(listing.attributes["adjustable_height"], True)
        self.assertIn("c04", {listing.id for listing in with_adjustment})

    def test_missing_flag_is_not_counted_as_false(self) -> None:
        self.assertEqual(self.chairs.unknown_count, 3)
        self.assertEqual(len(filter_flag(self.chairs, False)) + len(filter_flag(self.chairs, True)),
                         len(self.chairs.priced_items) - self.chairs.unknown_count)


if __name__ == "__main__":
    unittest.main()
