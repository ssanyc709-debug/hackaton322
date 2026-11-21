"""Индекс каталога ``(city_id, category)``, бинарный поиск и фильтры.

Требования §2.1 и §9:
* индекс строится от СОДЕРЖИМОГО каталога (``sha256``), а не от пути файла,
  поэтому подмена каталога организаторами всегда вызывает перестройку;
* кэш индекса лежит на диске (``data/cache``) и проверяется по дайджесту;
* числовые ограничения применяются бинарным поиском O(log n), булевы — двумя
  готовыми выборками O(1), без проверки truthiness.

Токенов модели этот слой не тратит вообще: всё считается локально.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.model import CATEGORIES, CATEGORY_ATTRIBUTE, Bound, CatalogError, Listing, NumericRange

REQUIRED_LISTING_FIELDS: frozenset[str] = frozenset(
    {"id", "category", "title", "description", "price_rub", "city_id", "attributes"}
)

#: Числовые характеристики §2 — целое >= 1; булевы — только boolean.
#: ``bool`` в Python — подкласс ``int``, поэтому проверяется раньше: иначе
#: ``True`` превратился бы в ширину «1 см», а ``0`` — в «ширина неизвестна».
NUMERIC_ATTRIBUTES: frozenset[str] = frozenset(
    {CATEGORY_ATTRIBUTE["table"], CATEGORY_ATTRIBUTE["lamp"]}
)
FLAG_ATTRIBUTES: frozenset[str] = frozenset({CATEGORY_ATTRIBUTE["chair"]})

#: Версия формата файла кэша: при смене структуры старые файлы игнорируются.
CACHE_FORMAT_VERSION = 1
#: Длина префикса дайджеста в имени файла кэша (полный дайджест хранится внутри).
CACHE_NAME_PREFIX_LEN = 16

BucketKey = tuple[str, str]  # (city_id, category)


@dataclass(frozen=True, slots=True)
class Bucket:
    """Кандидаты одного города и одной категории, подготовленные для фильтров."""

    category: str
    # Числовая характеристика: значения по возрастанию, ``numeric_items`` параллелен им.
    numeric_values: tuple[int, ...]
    numeric_items: tuple[Listing, ...]
    # Булева характеристика: готовые выборки (False — валидные данные, не «пусто»).
    flags_true: tuple[Listing, ...]
    flags_false: tuple[Listing, ...]
    # Объявления без характеристики: не проходят заданное ограничение, но видны в no_match.
    unknown_count: int
    # Все объявления бакета по возрастанию (цена, id) — минимумы и подбор.
    priced_items: tuple[Listing, ...]

    @property
    def cheapest(self) -> Listing | None:
        """Самое дешёвое объявление бакета (``None`` для пустого бакета)."""
        return self.priced_items[0] if self.priced_items else None


EMPTY_BUCKET = Bucket(
    category="",
    numeric_values=(),
    numeric_items=(),
    flags_true=(),
    flags_false=(),
    unknown_count=0,
    priced_items=(),
)


@dataclass(frozen=True, slots=True)
class CatalogIndex:
    """Готовый индекс: объявления, бакеты по (город, категория) и доступ по ID."""

    digest: str
    listings: tuple[Listing, ...]
    buckets: Mapping[BucketKey, Bucket]
    by_id: Mapping[str, Listing]
    from_cache: bool = False

    @property
    def size(self) -> int:
        """Число объявлений в каталоге."""
        return len(self.listings)

    def cities(self) -> tuple[str, ...]:
        """Города, реально присутствующие в каталоге (по возрастанию)."""
        return tuple(sorted({city for city, _ in self.buckets}))

    def categories(self) -> tuple[str, ...]:
        """Категории, реально присутствующие в каталоге (по возрастанию)."""
        return tuple(sorted({category for _, category in self.buckets}))


def catalog_digest(document: Sequence[Mapping[str, Any]]) -> str:
    """CORE. Считает ``sha256`` содержимого каталога — ключ кэша индекса.

    Зачем: контракт разрешает кэш индекса только если он учитывает содержимое
    переданного каталога. Организаторы подменяют данные по тому же пути, поэтому
    ключом не может быть путь или mtime — только сам документ.

    Вход:  document — JSON-массив объявлений (уже прочитанный).
    Выход: 64-символьная hex-строка.

    Алгоритм: каноническая сериализация (ключи отсортированы, компактные
    разделители) → sha256 по UTF-8 байтам.
    Сложность: время O(total_bytes) однократно, память O(total_bytes) (<= ~60 КБ).
    Почему именно он: сравнение структур «в лоб» дало бы O(n·k) на каждый запуск,
    а sha256 — один проход и нулевая вероятность ложного попадания; mtime
    ненадёжен (файл может быть перезаписан с тем же временем).
    Крайние случаи: другой порядок объявлений даёт другой дайджест — порядок
    в каталоге является частью данных (нужен детерминизм выдачи).
    Вызывается из: app/catalog/index.py::build_index, get_index.
    """
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def binary_search_first_ge(values: Sequence[int], bound: int) -> int:
    """CORE. Находит индекс первого элемента values[i] >= bound.

    Зачем: фильтр «не шире метра» — это поиск нижней границы по
    отсортированному массиву ширин, чтобы не сканировать все объявления.

    Вход:  values — отсортированный неубывающий список int (ширины, см).
           bound  — целочисленная граница из запроса покупателя.
    Выход: индекс i, 0 <= i <= len(values); len(values) означает «ничего не подходит».

    Алгоритм: бинарный поиск (деление пополам), вариант lower bound.
    Сложность: время O(log n), память O(1).
    Почему именно он: массив отсортирован, сравнение целочисленное и монотонное,
              значит бинарный поиск корректен и быстрее линейного O(n).
    Крайние случаи: пустой список -> 0; bound <= values[0] -> 0;
              bound > values[-1] -> len(values); дубликаты -> самый левый индекс.
    Вызывается из: app/catalog/index.py::filter_numeric.
    """
    low, high = 0, len(values)
    while low < high:
        middle = (low + high) // 2
        if values[middle] < bound:
            low = middle + 1
        else:
            high = middle
    return low


def binary_search_last_le(values: Sequence[int], bound: int) -> int:
    """CORE. Находит индекс последнего элемента values[i] <= bound.

    Зачем: верхняя граница «до 100 см» — правый край среза по отсортированному
    массиву; нужен ровно последний подходящий элемент, чтобы не терять дубликаты.

    Вход:  values — отсортированный неубывающий список int (ширины или кельвины).
           bound  — целочисленная граница из запроса покупателя.
    Выход: индекс i, -1 <= i < len(values); -1 означает «ни один не подходит».

    Алгоритм: бинарный поиск, вариант upper bound минус один.
    Сложность: время O(log n), память O(1).
    Почему именно он: включительная верхняя граница — это именно последний
              элемент <= bound; линейный проход не даёт выигрыша и теряет дубликаты
              при неаккуратной реализации.
    Крайние случаи: пустой список -> -1; bound < values[0] -> -1;
              bound >= values[-1] -> len(values) - 1; дубликаты -> самый правый.
    Вызывается из: app/catalog/index.py::filter_numeric.
    """
    low, high = 0, len(values) - 1
    result = -1
    while low <= high:
        middle = (low + high) // 2
        if values[middle] <= bound:
            result = middle
            low = middle + 1
        else:
            high = middle - 1
    return result


def normalize_attributes(value: Any, where: str) -> dict[str, Any]:
    """Проверяет типы значений ``attributes`` по §2 и нормализует целые float.

    Каталог, нарушающий схему, — технический сбой (``CatalogError``), а не
    ``no_match``: иначе «ширина строкой» тихо превратилась бы в «ширина
    неизвестна» и ограничение покупателя потерялось бы без следа.

    Допустимо (§2): строка, число, boolean; известные ключи — строго своих
    типов. ``90.0`` в JSON — математически целое и по схеме валидно, поэтому
    приводится к ``90``; ``90.5`` и ``0`` — уже нарушение схемы.

    Сложность: O(k) по числу характеристик (k — мало), память O(k).
    """
    if not isinstance(value, dict):
        raise CatalogError(f"{where}: attributes должен быть объектом")

    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise CatalogError(
                f"{where}: имя характеристики должно быть строкой, "
                f"получено {type(key).__name__}"
            )
        if isinstance(item, bool):
            if key in NUMERIC_ATTRIBUTES:
                raise CatalogError(
                    f"{where}: {key} должен быть целым >= 1, получено {item!r}"
                )
            normalized[key] = item
        elif isinstance(item, int):
            if key in FLAG_ATTRIBUTES:
                raise CatalogError(
                    f"{where}: {key} должен быть boolean, получено {item!r}"
                )
            if key in NUMERIC_ATTRIBUTES and item < 1:
                raise CatalogError(
                    f"{where}: {key} должен быть целым >= 1, получено {item!r}"
                )
            normalized[key] = item
        elif isinstance(item, float):
            if key in FLAG_ATTRIBUTES:
                raise CatalogError(
                    f"{where}: {key} должен быть boolean, получено {item!r}"
                )
            if key in NUMERIC_ATTRIBUTES:
                if not item.is_integer() or item < 1:
                    raise CatalogError(
                        f"{where}: {key} должен быть целым >= 1, получено {item!r}"
                    )
                normalized[key] = int(item)
            else:
                normalized[key] = item
        elif isinstance(item, str):
            if key in NUMERIC_ATTRIBUTES or key in FLAG_ATTRIBUTES:
                raise CatalogError(
                    f"{where}: {key} должен иметь тип по схеме, получено строкой {item!r}"
                )
            normalized[key] = item
        else:
            raise CatalogError(
                f"{where}: значение характеристики {key!r} недопустимого типа "
                f"{type(item).__name__} (разрешены строка, число, boolean)"
            )
    return normalized


def parse_listing(raw: Mapping[str, Any], position: int) -> Listing:
    """Превращает сырой объект каталога в типизированный :class:`Listing`.

    Границы §2.2: ``price_rub`` — положительное целое (bool не число),
    ``attributes`` — объект с типами по §2 (см. :func:`normalize_attributes`),
    остальные поля — непустые строки где положено.
    Ошибки каталога — технический сбой (exit != 0), а не ``no_match``.

    Сложность: O(1) на объявление (характеристик всегда единицы).
    """
    where = f"объявление #{position}"
    if not isinstance(raw, dict):
        raise CatalogError(f"{where}: ожидался объект, получено {type(raw).__name__}")

    extra = set(raw) - REQUIRED_LISTING_FIELDS
    if extra:
        raise CatalogError(f"{where}: лишние поля {sorted(extra)}")
    missing = REQUIRED_LISTING_FIELDS - set(raw)
    if missing:
        raise CatalogError(f"{where}: отсутствуют поля {sorted(missing)}")

    listing_id = raw["id"]
    if not isinstance(listing_id, str) or not listing_id:
        raise CatalogError(f"{where}: id должен быть непустой строкой")

    category = raw["category"]
    if category not in CATEGORIES:
        raise CatalogError(f"{where} ({listing_id}): недопустимая категория {category!r}")

    title = raw["title"]
    if not isinstance(title, str) or not title.strip():
        raise CatalogError(f"{where} ({listing_id}): title должен быть непустой строкой")

    description = raw["description"]
    if not isinstance(description, str):
        raise CatalogError(f"{where} ({listing_id}): description должен быть строкой")

    price = raw["price_rub"]
    if isinstance(price, bool) or not isinstance(price, int) or price < 1:
        raise CatalogError(f"{where} ({listing_id}): price_rub должен быть целым > 0, получено {price!r}")

    city_id = raw["city_id"]
    if not isinstance(city_id, str) or not city_id:
        raise CatalogError(f"{where} ({listing_id}): city_id должен быть непустой строкой")

    attributes = raw["attributes"]
    if not isinstance(attributes, dict):
        raise CatalogError(f"{where} ({listing_id}): attributes должен быть объектом")

    return Listing(
        id=listing_id,
        category=category,
        title=title,
        description=description,
        price_rub=price,
        city_id=city_id,
        attributes=normalize_attributes(attributes, f"{where} ({listing_id})"),
    )


def build_index(document: Sequence[Mapping[str, Any]]) -> CatalogIndex:
    """CORE. Строит индекс (город, категория) → подготовленные кандидаты.

    Зачем: подбору нужны «все столы Таганрога», а не весь каталог. Группировка
    отбрасывает чужие города и категории до любых фильтров, а внутри бакета
    заранее сортируются значения под бинарный поиск и цены под отсечение.

    Вход:  document — JSON-массив объявлений (n <= 250).
    Выход: CatalogIndex (с дайджестом) либо CatalogError при нарушении схемы.

    Алгоритм: разбор в Listing O(n) → проверка уникальности ID хеш-множеством O(n)
    → группировка в словарь O(n) → сортировка значений и цен внутри бакетов
    суммарно O(n log n). Память O(n).
    Почему именно он: сортировка выполняется один раз на каталог, а не на каждый
    запрос; фильтры потом работают за O(log n), подбор — по сужающимся спискам.
    Крайние случаи: дубль ID — ошибка; категория вне справочника — ошибка;
    price_rub = 0/True/строка — ошибка; объявление без характеристики попадает
    в ``unknown_count`` и не участвует в ограничениях, но видно в ``priced_items``.
    Вызывается из: app/catalog/index.py::get_index, тесты.
    """
    digest = catalog_digest(document)
    listings = [parse_listing(raw, position) for position, raw in enumerate(document, start=1)]

    by_id: dict[str, Listing] = {}
    for listing in listings:
        if listing.id in by_id:
            raise CatalogError(f"повторяющийся id объявления: {listing.id!r}")
        by_id[listing.id] = listing

    grouped: dict[BucketKey, list[Listing]] = {}
    for listing in listings:
        grouped.setdefault((listing.city_id, listing.category), []).append(listing)

    buckets = {key: _build_bucket(items, key[1]) for key, items in grouped.items()}
    return CatalogIndex(digest=digest, listings=tuple(listings), buckets=buckets, by_id=by_id)


def _build_bucket(items: Iterable[Listing], category: str) -> Bucket:
    """Готовит один бакет: массивы для бинарного поиска и выборки по флагу."""
    attribute = CATEGORY_ATTRIBUTE[category]
    numeric_pairs: list[tuple[int, Listing]] = []
    flags_true: list[Listing] = []
    flags_false: list[Listing] = []
    unknown = 0

    for item in items:
        number = item.numeric(attribute)
        if number is not None:
            numeric_pairs.append((number, item))
            continue
        flag = item.flag(attribute)
        if flag is True:
            flags_true.append(item)
        elif flag is False:  # явная проверка: False — валидное значение, а не «нет данных»
            flags_false.append(item)
        else:
            unknown += 1

    # Детерминированный порядок: (значение, id) и (цена, id).
    numeric_pairs.sort(key=lambda pair: (pair[0], pair[1].id))
    price_key = lambda listing: (listing.price_rub, listing.id)  # noqa: E731 — локальный ключ сортировки
    return Bucket(
        category=category,
        numeric_values=tuple(pair[0] for pair in numeric_pairs),
        numeric_items=tuple(pair[1] for pair in numeric_pairs),
        flags_true=tuple(sorted(flags_true, key=price_key)),
        flags_false=tuple(sorted(flags_false, key=price_key)),
        unknown_count=unknown,
        priced_items=tuple(sorted(items, key=price_key)),
    )


# --- Кэш индекса: память процесса + файл на диске, ключ — sha256 -----------

_INDEX_MEMO: tuple[str, CatalogIndex] | None = None


def get_index(document: Sequence[Mapping[str, Any]], cache_dir: Path | None = None) -> CatalogIndex:
    """CORE. Отдаёт индекс каталога, перестраивая его при изменении содержимого.

    Зачем: контракт разрешает кэш индекса, но требует учитывать содержимое
    каталога. Ключ — ``sha256`` документа, поэтому подмена каталога по тому же
    пути автоматически даёт промах кэша и полную перестройку.

    Вход:  document  — прочитанный JSON-массив объявлений;
           cache_dir — каталог для файлового кэша (``None`` — только память).
    Выход: CatalogIndex; поле ``from_cache`` показывает, попали ли в кэш.

    Алгоритм: дайджест → попадание в память процесса → попытка чтения файла
    ``index-<digest[:16]>.json`` с проверкой версии формата, дайджеста и целостности
    ссылок на ID → иначе полная перестройка и атомарная запись файла.
    Сложность: попадание O(n) на разбор каталога + O(b) на восстановление бакетов;
    промах дополнительно O(n log n) на сортировки и O(b) на запись.
    Почему именно он: парсинг JSON неизбежен (данные нужны для проверки фактов),
    а кэш убирает повторные сортировки и группировку; файл читается один раз
    и только если каталог не менялся.
    Крайние случаи: битый файл кэша, чужой дайджест, ссылка на отсутствующий ID,
    недоступный каталог кэша — всё это логируется и приводит к перестройке,
    а не к падению запуска.
    Вызывается из: app/main.py::main, тесты, bench/load.py.
    """
    global _INDEX_MEMO
    digest = catalog_digest(document)
    if _INDEX_MEMO is not None and _INDEX_MEMO[0] == digest:
        return _INDEX_MEMO[1]

    fresh = build_index(document)
    if cache_dir is not None:
        cached = _load_cached_index(document, fresh, cache_dir)
        if cached is not None:
            _INDEX_MEMO = (digest, cached)
            return cached
        _store_cached_index(fresh, cache_dir)
    _INDEX_MEMO = (digest, fresh)
    return fresh


def forget_index() -> None:
    """Сбрасывает кэш индекса в памяти (тесты, смена каталога в одном процессе)."""
    global _INDEX_MEMO
    _INDEX_MEMO = None


def _cache_path(cache_dir: Path, digest: str) -> Path:
    """Путь файла кэша: короткий префикс дайджеста в имени, полный — внутри."""
    return cache_dir / f"index-{digest[:CACHE_NAME_PREFIX_LEN]}.json"


def _load_cached_index(
    document: Sequence[Mapping[str, Any]],
    fresh: CatalogIndex,
    cache_dir: Path,
) -> CatalogIndex | None:
    """Восстанавливает бакеты из файла кэша, если он валиден для этого каталога.

    Сами объявления берутся из только что разобранного документа (факты о цене,
    городе и характеристиках всегда свежие); кэш хранит только раскладку:
    порядок значений под бинарный поиск и порядок цен. Так исключается класс
    ошибок «в кэше старая цена».
    """
    path = _cache_path(cache_dir, fresh.digest)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("ожидался объект")
        if payload.get("version") != CACHE_FORMAT_VERSION:
            raise ValueError(f"версия формата {payload.get('version')!r} не поддерживается")
        if payload.get("digest") != fresh.digest:
            raise ValueError("дайджест каталога не совпадает")
        buckets = _decode_buckets(payload["buckets"], fresh.by_id)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[buyer-assistant] кэш индекса игнорируется ({path.name}): {exc}", file=sys.stderr, flush=True)
        return None

    return CatalogIndex(
        digest=fresh.digest,
        listings=fresh.listings,
        buckets=buckets,
        by_id=fresh.by_id,
        from_cache=True,
    )


def _decode_buckets(raw_buckets: Any, by_id: Mapping[str, Listing]) -> dict[BucketKey, Bucket]:
    """Собирает бакеты из раскладки кэша, проверяя каждую ссылку на ID."""
    if not isinstance(raw_buckets, dict):
        raise ValueError("buckets должен быть объектом")
    buckets: dict[BucketKey, Bucket] = {}
    for key, raw in raw_buckets.items():
        city_id, separator, category = str(key).partition("|")
        if not separator or category not in CATEGORIES:
            raise ValueError(f"некорректный ключ бакета {key!r}")
        if not isinstance(raw, dict):
            raise ValueError(f"бакет {key!r} должен быть объектом")

        numeric_pairs = raw["numeric"]
        if not isinstance(numeric_pairs, list):
            raise ValueError(f"бакет {key!r}: numeric должен быть массивом")
        values: list[int] = []
        numeric_items: list[Listing] = []
        for pair in numeric_pairs:
            value, listing_id = pair
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"бакет {key!r}: некорректное значение {value!r}")
            listing = by_id.get(listing_id)
            if listing is None or listing.category != category or listing.city_id != city_id:
                raise ValueError(f"бакет {key!r}: ссылка на чужой ID {listing_id!r}")
            values.append(value)
            numeric_items.append(listing)

        def listing_list(field: str) -> tuple[Listing, ...]:
            ids = raw[field]
            if not isinstance(ids, list):
                raise ValueError(f"бакет {key!r}: {field} должен быть массивом")
            items = tuple(by_id.get(listing_id) for listing_id in ids)
            if any(item is None for item in items):
                raise ValueError(f"бакет {key!r}: {field} содержит неизвестный ID")
            return items  # type: ignore[return-value]

        unknown = raw["unknown"]
        if isinstance(unknown, bool) or not isinstance(unknown, int) or unknown < 0:
            raise ValueError(f"бакет {key!r}: некорректное unknown_count {unknown!r}")
        priced = listing_list("priced")
        if len(priced) != len(values) + len(listing_list("true")) + len(listing_list("false")) + unknown:
            raise ValueError(f"бакет {key!r}: раскладка не сходится по количеству")

        buckets[(city_id, category)] = Bucket(
            category=category,
            numeric_values=tuple(values),
            numeric_items=tuple(numeric_items),
            flags_true=listing_list("true"),
            flags_false=listing_list("false"),
            unknown_count=unknown,
            priced_items=priced,
        )
    return buckets


def _store_cached_index(index: CatalogIndex, cache_dir: Path) -> None:
    """Атомарно записывает раскладку индекса; ошибки записи не ломают запрос."""
    path = _cache_path(cache_dir, index.digest)
    payload = {
        "version": CACHE_FORMAT_VERSION,
        "digest": index.digest,
        "buckets": {
            f"{city_id}|{category}": {
                "numeric": [[value, listing.id] for value, listing in zip(bucket.numeric_values, bucket.numeric_items)],
                "true": [listing.id for listing in bucket.flags_true],
                "false": [listing.id for listing in bucket.flags_false],
                "unknown": bucket.unknown_count,
                "priced": [listing.id for listing in bucket.priced_items],
            }
            for (city_id, category), bucket in index.buckets.items()
        },
    }
    temporary_path: str | None = None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Уникальный временный файл: параллельные запуски (фронт шлёт запросы
        # подряд) не должны писать в один и тот же `.tmp` — иначе один процесс
        # может опубликовать недописанный файл другого и кэш станет битым.
        descriptor, temporary_path = tempfile.mkstemp(
            dir=str(cache_dir), prefix=path.name + ".", suffix=".tmp"
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError as exc:
        print(f"[buyer-assistant] не удалось записать кэш индекса ({path}): {exc}", file=sys.stderr, flush=True)
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


# --- Фильтры ---------------------------------------------------------------


def bucket_for(index: CatalogIndex, city_id: str, category: str) -> Bucket:
    """Бакет города и категории; :data:`EMPTY_BUCKET`, если пары в каталоге нет."""
    return index.buckets.get((city_id, category), EMPTY_BUCKET)


def _numeric_slice(values: Sequence[int], constraint: NumericRange) -> tuple[int, int]:
    """Границы среза ``[start, stop)`` для числового условия.

    Включительность реализована выбором варианта поиска, а не проверкой после
    среза: включительная нижняя → ``first_ge``, строгая → первый строго больше;
    включительная верхняя → ``last_le + 1``, строгая → первый неубывающий элемент
    (он уже не входит в срез).
    """
    lower, upper = constraint.lower, constraint.upper
    if lower is None:
        start = 0
    elif lower.inclusive:
        start = binary_search_first_ge(values, lower.value)
    else:
        start = binary_search_first_strictly_greater(values, lower.value)

    if upper is None:
        stop = len(values)
    elif upper.inclusive:
        stop = binary_search_last_le(values, upper.value) + 1
    else:
        stop = binary_search_first_ge(values, upper.value)
    return start, max(start, stop)


def binary_search_first_strictly_greater(values: Sequence[int], bound: int) -> int:
    """CORE. Индекс первого элемента values[i] > bound (строгая нижняя граница).

    Зачем: «уже 100 см» и «больше 5000 K» отбрасывают само значение границы,
    поэтому нужен первый элемент строго больше.

    Вход:  values — отсортированный неубывающий массив int; bound — целая граница.
    Выход: индекс ``0..len(values)``; ``len(values)`` — «ничего не подходит».

    Алгоритм: бинарный поиск, вариант upper bound (как first_ge, но сравнение строгое).
    Сложность: время O(log n), память O(1).
    Почему именно он: без отдельной функции пришлось бы сдвигать индекс на 1
    после включительного поиска, что ломается на дубликатах границы.
    Крайние случаи: пустой массив → 0; все элементы <= bound → ``len(values)``;
    дубликаты bound → индекс сразу после последнего дубликата.
    Вызывается из: app/catalog/index.py::_numeric_slice.
    """
    low, high = 0, len(values)
    while low < high:
        middle = (low + high) // 2
        if values[middle] <= bound:
            low = middle + 1
        else:
            high = middle
    return low


def filter_numeric(bucket: Bucket, constraint: NumericRange) -> tuple[Listing, ...]:
    """CORE. Кандидаты бакета, удовлетворяющие числовому ограничению.

    Зачем: «стол не шире метра», «лампа от 3000 до 5000» — основная часть
    ограничений каталога; нужен точный срез за O(log n) без перебора.

    Вход:  bucket     — кандидаты одного города и категории;
           constraint — нижняя и/или верхняя граница с признаком включительности.
    Выход: кортеж объявлений в порядке возрастания характеристики; пустой, если
    не подошло ни одно. Объявления без характеристики не попадают НИКОГДА:
    неизвестное значение не подтверждает требование покупателя.

    Алгоритм: два бинарных поиска по ``numeric_values`` → срез ``numeric_items``.
    Сложность: время O(log n + k) на выдачу k подходящих, память O(k).
    Почему именно он: значения отсортированы при индексации, поэтому границы —
    это индексы; перебор сравнивал бы каждое объявление (O(n)) и на 250 записях
    давал бы тысячи лишних сравнений в горячем пути.
    Крайние случаи: обе границы None → все объявления с характеристикой;
    lower > upper → пусто; равные границы → точное значение.
    Вызывается из: app/catalog/search.py, app/catalog/explain.py.
    """
    start, stop = _numeric_slice(bucket.numeric_values, constraint)
    return bucket.numeric_items[start:stop] if stop > start else ()


def filter_flag(bucket: Bucket, value: bool) -> tuple[Listing, ...]:
    """CORE. Кандидаты с ``adjustable_height == value``.

    Зачем: булева характеристика кресла. ``False`` — полноценные данные
    («регулировки нет»), поэтому проверять его truthiness нельзя: фильтр
    «без регулировки» обязан вернуть выборку ``false``, а не пустоту.
    Отсутствующий ключ означает «неизвестно»: при заданном ограничении такое
    объявление не подходит (оно попадает только в ``unknown_count``).

    Вход:  bucket — кандидаты города и категории; value — требуемое значение bool.
    Выход: кортеж объявлений, где ключ присутствует и строго равен ``value``.

    Алгоритм: выбор готовой выборки по ``value is True`` (два кортежа построены
    при индексации, поэтому сравнение с ``False`` корректно).
    Сложность: время O(1) на выбор + O(k) на выдачу k объявлений, память O(k).
    Почему именно он: разложить по двум выборкам один раз дешевле, чем проверять
    значение на каждом запросе, и это исключает класс ошибок ``if attr:``.
    Крайние случаи: ``value=False`` → ``flags_false`` (не пустая проверка!);
    бакет без таких объявлений → пустой кортеж.
    Вызывается из: app/catalog/search.py.
    """
    return bucket.flags_true if value is True else bucket.flags_false


def sorted_cheapest(items: Iterable[Listing], limit: int) -> tuple[Listing, ...]:
    """Первые ``limit`` объявлений по возрастанию (цена, id) — детерминированно."""
    return tuple(sorted(items, key=lambda listing: (listing.price_rub, listing.id))[:limit])
