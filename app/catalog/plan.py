"""Справочник городов и сборка ответа: ``ok`` / ``no_match``.

Слой связывает слоты (условия покупателя), индекс каталога (кандидаты) и
шаблонные тексты. Модель здесь не вызывается: подбор, проверка цены и
объяснения — код.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.catalog import explain
from app.catalog.index import CatalogIndex
from app.catalog.search import CategoryState, Shortage, choose_options, collect_states, enumerate_combinations
from app.config import DATA_DIR
from app.contract import response
from app.model import NumericRange, Slots

CITIES_FILE = DATA_DIR / "cities.json"


@lru_cache(maxsize=1)
def city_names() -> dict[str, str]:
    """Названия городов из ``data/cities.json`` (справочник общий с организаторами).

    При недоступном файле возвращается пустой словарь: подставится ``city_id``,
    и это не сломает ответ (текст не сравнивается побайтово).
    """
    try:
        raw = json.loads(Path(CITIES_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, list):
        return {}
    names: dict[str, str] = {}
    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and isinstance(item.get("name"), str):
            names[item["id"]] = item["name"]
    return names


def city_name(city_id: str | None) -> str:
    """Человеческое название города или сам ``city_id``, если справочник недоступен."""
    if not city_id:
        return "в указанном городе"
    return city_names().get(city_id, city_id)


def build_plan(slots: Slots, index: CatalogIndex) -> dict[str, Any]:
    """CORE. Собирает ответ для известных условий: ``ok`` с комплектами или ``no_match``.

    Зачем: это точка, где условия покупателя превращаются в конкретные комплекты,
    и где выполняется главное требование контракта — «ровно одно объявление
    каждой запрошенной категории, все в городе покупателя, сумма ≤ бюджета».

    Вход:  slots — итоговые условия (город, бюджет, категории, ограничения);
           index — индекс текущего каталога (путь задан через ``--catalog``).
    Выход: документ ответа ``ok`` (1..3 варианта) либо ``no_match`` с объяснением;
    при неполных условиях вызывающий слой обязан вернуть ``clarify`` ДО вызова.

    Алгоритм: отбор кандидатов по категориям (бинарный поиск по числовым условиям,
    выборка по флагу) → проверка «какая категория пуста и почему» → перечисление
    допустимых комплектов DFS с отсечением по суффиксным минимумам → отбор до трёх
    различных по составу → шаблонные ``explanation``; иначе анализ минимумов.
    Сложность: время O(k·log n + k·C), где k <= 3, C <= 200 найденных наборов;
    память O(k·C). Модель не вызывается: 0 токенов.
    Почему именно он: связывает готовые чистые функции без собственной логики
    фильтрации, поэтому любое правило проверяется в одном месте (модули index/search).
    Крайние случаи: пустая категория в городе → ``no_match`` с названием категории;
    жёсткое ограничение отсекло всё → ``no_match`` с ближайшим доступным значением;
    все кандидаты есть, но сумма минимумов больше бюджета → ``no_match`` с нехваткой;
    ровно один вариант в бюджете → ``ok`` с одним вариантом (контракт допускает).
    Вызывается из: app/pipeline.py::answer, тесты.
    """
    states = collect_states(slots, index)
    if not states or slots.budget_rub is None:
        raise ValueError("build_plan вызван с неполными условиями")

    shortages: list[Shortage] = []
    for state in states:
        if state.empty:
            reason = "constraint" if state.constraint_applied else "no_listings"
            shortages.append(
                Shortage(category=state.category, reason=reason, details=_describe_constraint(state, slots))
            )

    if shortages:
        return response.no_match(explain.explain_no_match(slots, states, shortages, city_name(slots.city_id)))

    combinations = enumerate_combinations(states, slots.budget_rub)
    if not combinations:
        shortages.append(Shortage(category="", reason="budget", details=""))
        return response.no_match(explain.explain_no_match(slots, states, shortages, city_name(slots.city_id)))

    options: list[dict[str, Any]] = []
    for combination in choose_options(combinations):
        total = sum(listing.price_rub for listing in combination)
        options.append(
            response.option(
                listing_ids=[listing.id for listing in combination],
                total_rub=total,
                explanation=explain.explain_option(combination, total, slots.budget_rub),
            )
        )
    return response.ok(options)


def _describe_constraint(state: CategoryState, slots: Slots) -> str:
    """Формулирует, какое условие отсекло все объявления категории.

    Показывает ближайшее доступное значение из каталога, чтобы покупателю было
    понятно, что менять. Отсутствие характеристики в объявлении — тоже причина,
    и она называется явно (неизвестное значение не подтверждает требование).
    """
    if not state.bucket.priced_items:
        return f"нет предложений в категории «{explain.CATEGORY_PLURAL[state.category]}»"

    if state.category == "table" and slots.width is not None:
        bounds = explain.describe_range("см", slots.width)
        nearest = _nearest_numeric(state, slots.width)
        suffix = f" (ближайшая ширина — {nearest} см)" if nearest is not None else ""
        unknown = state.bucket.unknown_count
        hint = f"; у {unknown} объявлений ширина не указана" if unknown else ""
        return f"нет столов, у которых ширина {bounds}{suffix}{hint}"

    if state.category == "lamp" and slots.color_temperature is not None:
        bounds = explain.describe_range("K", slots.color_temperature)
        nearest = _nearest_numeric(state, slots.color_temperature)
        suffix = f" (ближайшая температура — {nearest} K)" if nearest is not None else ""
        return f"нет ламп, у которых температура {bounds}{suffix}"

    if state.category == "chair" and slots.adjustable_height is not None:
        wanted = "с регулировкой высоты" if slots.adjustable_height else "без регулировки высоты"
        alternative = "без регулировки" if slots.adjustable_height else "с регулировкой"
        return f"нет кресел {wanted} (есть {alternative})"

    return f"нет подходящих объявлений в категории «{explain.CATEGORY_PLURAL[state.category]}»"


def _nearest_numeric(state: CategoryState, constraint: NumericRange) -> int | None:
    """Ближайшее к границе условия доступное значение характеристики в бакете.

    Стоимость: O(n) по значениям бакета (<= 37) — вызывается только на пути
    ``no_match``, где время не критично; при равенстве расстояний выбирается
    меньшее значение (детерминизм).
    """
    target: int | None = None
    if constraint.upper is not None:
        target = constraint.upper.value
    elif constraint.lower is not None:
        target = constraint.lower.value
    if target is None or not state.bucket.numeric_values:
        return None
    return min(state.bucket.numeric_values, key=lambda value: (abs(value - target), value))
