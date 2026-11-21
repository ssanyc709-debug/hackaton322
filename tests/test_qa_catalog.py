"""Блок 7 — устойчивость к проверочному каталогу (CONTRACT §2).

Проверочный каталог отличается от учебного: другие ID, цены, наборы объявлений,
до 250 записей, путь передаётся аргументом ``--catalog`` (§1). Поэтому тесты
берут каталог из аргумента, а не из ``data/catalog.json``, и проверяют:

* ответ собран именно из переданного каталога (ID, цены, суммы), а не из
  зашитых констант;
* ``--catalog`` работает с другим путём: относительным, с пробелами и кириллицей,
  с ``~``, с BOM и CRLF;
* до 250 объявлений включительно; больше — техническая ошибка (предел схемы);
* объявление с любой валидной формой ``attributes`` участвует в подборе,
  дополнительные ключи не мешают;
* ``false`` и ``0`` — рабочие значения, а не «пусто» (никакого truthiness);
* отсутствующий ключ характеристики не подтверждает соответствие;
* кэш индекса считается по содержимому: подмена файла по тому же пути не даёт
  устаревший ответ;
* битый каталог — технический сбой: ненулевой код и пустой stdout
  (``no_match`` для этого запрещён, §4).

Все каталоги для тестов создаются в ``tempfile`` копированием реального, поэтому
``data/`` не изменяется.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tests.test_qa_support import (
    EXIT_OK,
    ask,
    index_by_id,
    load_catalog,
    run_cli,
    write_catalog,
)

_RESPONSES: dict[tuple, dict[str, Any]] = {}
_WORKDIR: tempfile.TemporaryDirectory[str] | None = None
_CACHE: Path | None = None


def setUpModule() -> None:
    global _WORKDIR, _CACHE
    _WORKDIR = tempfile.TemporaryDirectory(prefix="qa-catalog-")
    _CACHE = Path(_WORKDIR.name) / "cache"


def tearDownModule() -> None:
    if _WORKDIR is not None:
        _WORKDIR.cleanup()
    _RESPONSES.clear()


def workdir(name: str) -> Path:
    """Подкаталог модуля для файлов каталога/кэша (создаётся по требованию)."""
    assert _WORKDIR is not None
    path = Path(_WORKDIR.name) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def document_for(
    query: str,
    *,
    catalog: Path | str | None = None,
    cwd: Path | str | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Ответ на запрос с указанным каталогом; одинаковый вызов идёт один раз."""
    key = (query, str(catalog or ""), str(cwd or ""), tuple(sorted((extra_env or {}).items())))
    if key not in _RESPONSES:
        result = run_cli(
            ask(query),
            catalog=catalog,
            cache_dir=_CACHE,
            cwd=cwd,
            extra_env=extra_env,
        )
        if result.code != EXIT_OK:
            raise AssertionError(f"запрос {query!r}: код {result.code}, stderr: {result.stderr[-400:]}")
        _RESPONSES[key] = result.one_json_object()
    return _RESPONSES[key]


def option_ids_of(document: Mapping[str, Any]) -> list[str]:
    """Все ID из всех вариантов ответа (для ``ok``)."""
    return [str(listing_id) for option in document["options"] for listing_id in option["listing_ids"]]


def slug(name: str) -> str:
    """Имя подкаталога из названия случая: без пробелов и знаков препинания."""
    return "".join(character if character.isalnum() else "-" for character in name.lower()).strip("-")


# --- Готовые каталоги ----------------------------------------------------


def renamed(catalog: Sequence[Mapping[str, Any]], prefix: str) -> list[dict[str, Any]]:
    """Тот же каталог с непрозрачными ID: помощник не может опираться на их вид."""
    result: list[dict[str, Any]] = []
    for position, item in enumerate(catalog):
        copy = dict(item)
        copy["id"] = f"{prefix}-{position:03d}"
        copy["attributes"] = dict(item.get("attributes") or {})
        result.append(copy)
    return result


def repriced(catalog: Sequence[Mapping[str, Any]], factor: int) -> list[dict[str, Any]]:
    """Тот же каталог с другими ценами (целые, положительные)."""
    result: list[dict[str, Any]] = []
    for item in catalog:
        copy = dict(item)
        copy["price_rub"] = max(1, int(item["price_rub"]) * factor)
        copy["attributes"] = dict(item.get("attributes") or {})
        result.append(copy)
    return result


def listing(
    listing_id: str,
    category: str = "table",
    price: int = 1000,
    city_id: str = "azov",
    attributes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Синтетическое объявление, проходящее схему §2."""
    return {
        "id": listing_id,
        "category": category,
        "title": f"Товар {listing_id}",
        "description": "",
        "price_rub": price,
        "city_id": city_id,
        "attributes": dict(attributes or {}),
    }


class CatalogCase(unittest.TestCase):
    """Общие данные: реальный каталог и корень временного каталога."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.real = load_catalog()
        cls.real_by_id = index_by_id(cls.real)

    def catalog_file(self, name: str, document: Any, *, raw: str | bytes | None = None) -> Path:
        """Кладёт каталог во временный подкаталог модуля.

        ``raw`` пишется байт в байт, ``document`` — как JSON: список объявлений
        или любой другой JSON-значение (для проверки битых каталогов).
        """
        folder = workdir(name)
        path = folder / "catalog.json"
        if raw is not None:
            if isinstance(raw, bytes):
                path.write_bytes(raw)
            else:
                path.write_text(raw, encoding="utf-8", newline="")
            return path
        if isinstance(document, list):
            return write_catalog(folder, document)
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        return path

    def assert_ok(self, document: Mapping[str, Any], *, context: str = "") -> None:
        self.assertEqual(document.get("status"), "ok", f"{context}: {document}")


class CatalogIsTakenFromArgumentTest(CatalogCase):
    """§1: путь каталога приходит аргументом; данные берутся из него."""

    def test_ids_belong_to_the_given_catalog(self) -> None:
        catalog = renamed(self.real, "prov")
        path = self.catalog_file("arg-renamed", catalog)
        document = document_for("Стол, кресло и лампа в Таганроге до 12 000 ₽", catalog=path)
        self.assert_ok(document)
        returned = set(option_ids_of(document))
        self.assertTrue(returned, "ответ без ID")
        self.assertTrue(
            returned <= {item["id"] for item in catalog},
            f"в ответе есть ID не из переданного каталога: {returned - {item['id'] for item in catalog}}",
        )

    def test_totals_are_summed_from_the_given_catalog(self) -> None:
        catalog = repriced(self.real, 3)
        path = self.catalog_file("arg-repriced", catalog)
        by_id = index_by_id(catalog)
        document = document_for("Стол и кресло в Азове до 20 000 ₽", catalog=path)
        self.assert_ok(document)
        for index, option in enumerate(document["options"]):
            with self.subTest(option=index):
                expected = sum(int(by_id[listing_id]["price_rub"]) for listing_id in option["listing_ids"])
                self.assertEqual(
                    option["total_rub"],
                    expected,
                    "total_rub должен считаться по ценам переданного каталога, а не учебного",
                )

    def test_same_query_in_two_catalogs_gives_different_ids(self) -> None:
        """Доказательство, что ID не зашиты: та же фраза, другой каталог — другие ID."""
        first = self.catalog_file("arg-a", renamed(self.real, "alpha"))
        second = self.catalog_file("arg-b", renamed(self.real, "beta"))
        query = "Стол в Таганроге до 5000 ₽"
        ids_first = set(option_ids_of(document_for(query, catalog=first)))
        ids_second = set(option_ids_of(document_for(query, catalog=second)))
        self.assertTrue(ids_first and ids_second)
        self.assertFalse(
            ids_first & ids_second,
            f"ID совпали между разными каталогами: {ids_first & ids_second}",
        )

    def test_relative_catalog_path_is_accepted(self) -> None:
        root = workdir("relative")
        (root / "data").mkdir(exist_ok=True)
        shutil.copy(self.catalog_file("relative-src", renamed(self.real, "rel")), root / "data" / "catalog.json")
        document = document_for("Стол в Азове до 5000 ₽", catalog="data/catalog.json", cwd=root)
        self.assert_ok(document)

    def test_catalog_path_with_spaces_and_cyrillic(self) -> None:
        folder = workdir("путь с пробелами") / "вложенная папка"
        folder.mkdir(parents=True, exist_ok=True)
        catalog = renamed(self.real, "space")
        path = folder / "каталог 250.json"
        path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
        document = document_for("Стол в Азове до 5000 ₽", catalog=path)
        self.assert_ok(document)

    def test_tilde_in_catalog_path_is_expanded(self) -> None:
        home = workdir("home")
        shutil.copy(self.catalog_file("tilde-src", renamed(self.real, "tilde")), home / "catalog.json")
        document = document_for(
            "Стол в Азове до 5000 ₽",
            catalog="~/catalog.json",
            extra_env={"HOME": str(home)},
        )
        self.assert_ok(document)

    def test_bom_and_crlf_in_catalog_file_are_accepted(self) -> None:
        text = json.dumps(renamed(self.real, "bom"), ensure_ascii=False, indent=2)
        for name, raw in (
            ("bom", "\ufeff" + text),
            ("crlf", text.replace("\n", "\r\n")),
            ("one-line", json.dumps(renamed(self.real, "bom"), ensure_ascii=False, separators=(",", ":"))),
        ):
            with self.subTest(case=name):
                path = self.catalog_file(f"encoding-{name}", None, raw=raw)
                self.assert_ok(document_for("Стол в Азове до 5000 ₽", catalog=path), context=name)


class CatalogSizeTest(CatalogCase):
    """§2: до 250 объявлений; предел схемы — тоже проверяемое поведение."""

    def test_catalog_at_the_250_limit_is_supported(self) -> None:
        catalog = [listing(f"limit-{index:03d}", price=100 + index * 10) for index in range(250)]
        path = self.catalog_file("limit-250", catalog)
        document = document_for("Стол в Азове до 5000 ₽", catalog=path)
        self.assert_ok(document)
        self.assertLessEqual(len(document["options"]), 3)
        for option in document["options"]:
            self.assertLessEqual(int(option["total_rub"]), 5000)

    def test_catalog_above_the_limit_is_a_technical_error(self) -> None:
        catalog = [listing(f"over-{index:03d}", price=100 + index) for index in range(251)]
        path = self.catalog_file("limit-251", catalog)
        result = run_cli(ask("Стол в Азове до 5000 ₽"), catalog=path, cache_dir=_CACHE)
        self.assertNotEqual(result.code, 0, "каталог сверх лимита схемы — технический сбой")
        self.assertEqual(result.stdout, b"", "при техническом сбое stdout обязан быть пустым")

    def test_single_listing_catalog_answers_ok(self) -> None:
        path = self.catalog_file("one-listing", [listing("only-1", price=1500)])
        document = document_for("Стол в Азове до 5000 ₽", catalog=path)
        self.assert_ok(document)
        self.assertEqual(option_ids_of(document), ["only-1"])


class CatalogCompositionTest(CatalogCase):
    """Набор объявлений может быть любым: другой город, нет категории, пустые поля."""

    def test_city_without_the_category_gives_no_match_not_a_crash(self) -> None:
        path = self.catalog_file(
            "missing-category",
            [item for item in self.real if not (item["city_id"] == "azov" and item["category"] == "lamp")],
        )
        document = document_for("Лампа в Азове до 15 000 ₽", catalog=path)
        self.assertEqual(document.get("status"), "no_match", document)
        self.assertNotIn("options", document)
        self.assertTrue(document["message"].strip())

    def test_catalog_without_a_city_answers_no_match_for_it(self) -> None:
        """Каталог, где есть только Азов: Таганрог — no_match, а не ошибка."""
        path = self.catalog_file("only-azov", [item for item in self.real if item["city_id"] == "azov"])
        document = document_for("Стол в Таганроге до 15 000 ₽", catalog=path)
        self.assertEqual(document.get("status"), "no_match", document)

    def test_other_city_still_works_when_one_city_is_absent(self) -> None:
        path = self.catalog_file("only-azov-2", [item for item in self.real if item["city_id"] == "azov"])
        document = document_for("Стол в Азове до 15 000 ₽", catalog=path)
        self.assert_ok(document)
        for option in document["options"]:
            for listing_id in option["listing_ids"]:
                self.assertTrue(listing_id, "пустой ID")

    def test_listing_order_does_not_break_the_answer(self) -> None:
        """Обратный порядок объявлений: ответ обязан остаться допустимым."""
        catalog = renamed(list(reversed(self.real)), "rev")
        path = self.catalog_file("reversed", catalog)
        by_id = index_by_id(catalog)
        document = document_for("Стол, кресло и лампа в Азове до 9000 ₽", catalog=path)
        self.assert_ok(document)
        for index, option in enumerate(document["options"]):
            with self.subTest(option=index):
                expected = sum(int(by_id[listing_id]["price_rub"]) for listing_id in option["listing_ids"])
                self.assertEqual(option["total_rub"], expected)
                self.assertLessEqual(int(option["total_rub"]), 9000)

    def test_extreme_prices_are_supported(self) -> None:
        """Цены вне учебного диапазона (350–12 500 ₽) — обычные данные."""
        catalog = [
            listing("cheap-1", price=1),
            listing("rich-1", price=9_999_999),
        ]
        path = self.catalog_file("extreme-prices", catalog)
        cheap = document_for("Стол в Азове до 5 ₽", catalog=path)
        self.assert_ok(cheap)
        self.assertEqual(option_ids_of(cheap), ["cheap-1"])
        rich = document_for("Стол в Азове до 9 999 999 ₽", catalog=path)
        self.assert_ok(rich)
        self.assertIn("rich-1", option_ids_of(rich))

    def test_city_outside_the_directory_does_not_break_other_results(self) -> None:
        """Каталог с незнакомым city_id: он не ломает подбор в известных городах."""
        catalog = [item for item in self.real if item["city_id"] == "azov"]
        catalog.append(listing("alien-1", price=100, city_id="unknown-city"))
        path = self.catalog_file("unknown-city", catalog)
        document = document_for("Стол в Азове до 5000 ₽", catalog=path)
        self.assert_ok(document)
        self.assertNotIn("alien-1", option_ids_of(document), "город не из справочника не может попасть в ответ")


class AttributesTest(CatalogCase):
    """§2.3: характеристики, дополнительные ключи, ``false``/``0``, неизвестное значение."""

    def test_extra_attribute_keys_are_allowed(self) -> None:
        catalog = [
            listing(
                "extra-1",
                price=1000,
                attributes={"width_cm": 90, "material": "дуб", "height_cm": 75, "folding": True},
            )
        ]
        path = self.catalog_file("extra-keys", catalog)
        document = document_for("Стол в Азове до 5000 ₽", catalog=path)
        self.assert_ok(document)
        self.assertEqual(option_ids_of(document), ["extra-1"])

    def test_empty_attributes_object_is_valid(self) -> None:
        path = self.catalog_file("empty-attrs", [listing("plain-1", price=700, attributes={})])
        self.assert_ok(document_for("Стол в Азове до 5000 ₽", catalog=path))

    def test_false_flag_is_respected_without_truthiness(self) -> None:
        catalog = [
            listing("flag-false", "chair", 500, attributes={"adjustable_height": False}),
            listing("flag-true", "chair", 700, attributes={"adjustable_height": True}),
            listing("flag-none", "chair", 600, attributes={}),
        ]
        path = self.catalog_file("flags", catalog)
        without = document_for("Кресло без регулировки высоты в Азове до 5000 ₽", catalog=path)
        self.assert_ok(without)
        self.assertEqual(
            option_ids_of(without),
            ["flag-false"],
            "«без регулировки» — это adjustable_height == false, а не «поле отсутствует»",
        )
        with_flag = document_for("Кресло с регулировкой высоты в Азове до 5000 ₽", catalog=path)
        self.assert_ok(with_flag)
        self.assertEqual(option_ids_of(with_flag), ["flag-true"])

    def test_false_and_zero_values_are_not_filtered_out_without_a_constraint(self) -> None:
        """§2.3: без ограничения объявление с ``false``/``0`` участвует в подборе.

        Проверка через truthiness отбросила бы такие объявления как «пустые»,
        хотя значения валидны.
        """
        catalog = [
            listing("flag-false", "chair", 500, attributes={"adjustable_height": False, "height_cm": 0}),
            listing("flag-true", "chair", 700, attributes={"adjustable_height": True}),
        ]
        path = self.catalog_file("false-zero", catalog)
        document = document_for("Кресло в Азове до 5000 ₽", catalog=path)
        self.assert_ok(document)
        self.assertEqual(
            set(option_ids_of(document)),
            {"flag-false", "flag-true"},
            "объявление с false/0 выпало из подбора, хотя ограничения по характеристике нет",
        )

    def test_missing_attribute_does_not_satisfy_a_constraint(self) -> None:
        catalog = [
            listing("known-80", price=1000, attributes={"width_cm": 80}),
            listing("known-120", price=1200, attributes={"width_cm": 120}),
            listing("unknown-width", price=900, attributes={}),
        ]
        path = self.catalog_file("unknown-attribute", catalog)
        document = document_for("Стол не шире 90 см в Азове до 5000 ₽", catalog=path)
        self.assert_ok(document)
        self.assertEqual(
            option_ids_of(document),
            ["known-80"],
            "объявление без width_cm не подтверждает ограничение (§2.3)",
        )

    def test_missing_attribute_is_mentioned_in_no_match(self) -> None:
        catalog = [
            listing("no-width-1", price=1000, attributes={}),
            listing("no-width-2", price=1200, attributes={"width_cm": 200}),
        ]
        path = self.catalog_file("unknown-attribute-no-match", catalog)
        document = document_for("Стол не шире 90 см в Азове до 5000 ₽", catalog=path)
        self.assertEqual(document.get("status"), "no_match", document)
        message = document["message"].lower()
        self.assertTrue(
            "ширин" in message,
            f"в объяснении нет препятствия по ширине: {document['message']!r}",
        )


class CatalogCacheTest(CatalogCase):
    """Кэш индекса учитывает содержимое, а не путь файла."""

    def test_replacing_the_file_at_the_same_path_gives_fresh_results(self) -> None:
        folder = workdir("cache-replace")
        path = folder / "catalog.json"
        query = ask("Стол в Азове до 5000 ₽")
        path.write_text(json.dumps([listing("before-1", price=1000)], ensure_ascii=False), encoding="utf-8")
        before = run_cli(query, catalog=path, cache_dir=_CACHE)
        self.assertEqual(before.code, EXIT_OK, before.stderr[-300:])
        self.assertEqual(
            [listing_id for option in before.one_json_object()["options"] for listing_id in option["listing_ids"]],
            ["before-1"],
        )

        path.write_text(json.dumps([listing("after-1", price=2000)], ensure_ascii=False), encoding="utf-8")
        after = run_cli(query, catalog=path, cache_dir=_CACHE)
        self.assertEqual(after.code, EXIT_OK, after.stderr[-300:])
        self.assertEqual(
            [listing_id for option in after.one_json_object()["options"] for listing_id in option["listing_ids"]],
            ["after-1"],
            "ответ построен по устаревшему кэшу: содержимое файла изменилось",
        )

    def test_cache_lives_in_the_given_directory(self) -> None:
        folder = workdir("cache-location")
        catalog = folder / "catalog.json"
        catalog.write_text(json.dumps([listing("cache-1", price=1000)], ensure_ascii=False), encoding="utf-8")
        cache = folder / "own-cache"
        result = run_cli(ask("Стол в Азове до 5000 ₽"), catalog=catalog, cache_dir=cache)
        self.assertEqual(result.code, EXIT_OK, result.stderr[-300:])
        self.assertTrue(
            any(cache.rglob("index-*.json")),
            f"кэш индекса не создан в заданном каталоге: {sorted(cache.rglob('*')) if cache.exists() else 'нет каталога'}",
        )

    def test_same_content_in_another_path_gives_the_same_answer(self) -> None:
        catalog = renamed(self.real, "same")
        first = self.catalog_file("identical-a", catalog)
        second = self.catalog_file("identical-b", catalog)
        query = "Стол, кресло и лампа в Таганроге до 12 000 ₽"
        ids_first = option_ids_of(document_for(query, catalog=first))
        ids_second = option_ids_of(document_for(query, catalog=second))
        self.assertEqual(
            ids_first,
            ids_second,
            "одинаковое содержимое каталога должно давать одинаковый ответ независимо от пути",
        )


#: Битые каталоги: каждый обязан дать ненулевой код и пустой stdout (§4).
BROKEN_CATALOGS: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("пустой файл", lambda: ""),
    ("не JSON", lambda: "{не json"),
    ("объект вместо массива", lambda: {"listings": []}),
    ("число вместо массива", lambda: 42),
    ("нет обязательного поля", lambda: [{k: v for k, v in listing("broken-1").items() if k != "price_rub"}]),
    ("лишнее поле", lambda: [dict(listing("broken-2"), note="x")]),
    ("пустой id", lambda: [listing("")]),
    ("категория вне справочника", lambda: [listing("broken-3", category="sofa")]),
    ("цена 0", lambda: [dict(listing("broken-4"), price_rub=0)]),
    ("цена строкой", lambda: [dict(listing("broken-5"), price_rub="1000")]),
    ("цена boolean", lambda: [dict(listing("broken-6"), price_rub=True)]),
    ("attributes не объект", lambda: [dict(listing("broken-7"), attributes=None)]),
    ("дублирующиеся id", lambda: [listing("dup-1", price=1000), listing("dup-1", price=2000)]),
    (
        "каталог не в UTF-8",
        lambda: json.dumps([listing("utf16-1")], ensure_ascii=False).encode("utf-16"),
    ),
)


class BrokenCatalogTest(CatalogCase):
    """§4: битый каталог — технический сбой, а не ``no_match`` и не ``ok``."""

    def test_every_broken_catalog_exits_nonzero_with_empty_stdout(self) -> None:
        """Один прогон на случай: код, пустой stdout, причина в stderr и не ``no_match``."""
        for name, factory in BROKEN_CATALOGS:
            with self.subTest(case=name):
                document = factory()
                raw = document if isinstance(document, (str, bytes)) else None
                path = self.catalog_file(f"broken-{slug(name)}", document, raw=raw)
                result = run_cli(ask("Стол в Азове до 5000 ₽"), catalog=path, cache_dir=_CACHE)
                self.assertNotEqual(result.code, 0, f"{name}: ожидался ненулевой код")
                self.assertEqual(result.stdout, b"", f"{name}: stdout при сбое обязан быть пустым")
                self.assertTrue(result.stderr.strip(), f"{name}: причина должна быть в stderr")
                self.assertNotIn(
                    b"no_match",
                    result.stdout,
                    f"{name}: битый каталог не может давать статус no_match",
                )

    def test_missing_catalog_file_exits_nonzero(self) -> None:
        missing = workdir("missing") / "нет-такого.json"
        result = run_cli(ask("Стол в Азове до 5000 ₽"), catalog=missing, cache_dir=_CACHE)
        self.assertNotEqual(result.code, 0)
        self.assertEqual(result.stdout, b"")


if __name__ == "__main__":
    unittest.main()
