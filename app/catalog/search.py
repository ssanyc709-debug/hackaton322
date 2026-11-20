"""Подбор: DFS с отсечением по суффиксным минимумам цен (CONTRACT.md §2.4).

Задача: из отсортированных по цене списков кандидатов одной категории каждый
собрать 1..3 комплекта (ровно по одному товару каждой запрошенной категории),
у которых суммарная цена не превышает бюджет.

Наивный перебор — O(n^k) на k категорий (до 13³ = 2197 наборов на городскую
пару, и это только один город). Отсечение по суффиксным минимумам цен
превращает перебор в «жадный по возрастанию цены»: как только текущий кандидат
вместе с минимально возможным довеском превышает бюджет, все следующие
кандидаты (они не дешевле) можно не рассматривать.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.catalog.index import Bucket, CatalogIndex, bucket_for, filter_flag, filter_numeric
from app.model import CATEGORIES, FIELD_ADJUSTABLE, Listing, Slots

#: Сколько допустимых наборов перебираем максимум (для выбора разнообразных вариантов).
MAX_COMBINATIONS = 200
#: Сколько вариантов отдаём покупателю (контракт: 1..3).
MAX_OPTIONS = 3


@dataclass(frozen=True, slots=True)
class CategoryState:
    """Состояние одной категории: сколько подходящих объявлений и что мешает."""

    category: str
    candidates: tuple[Listing, ...]
    bucket: Bucket
    constraint_applied: bool

    @property
    def empty(self) -> bool:
        return not self.candidates

    @property
    def cheapest(self) -> Listing | None:
        return self.candidates[0] if self.candidates else None


@dataclass(frozen=True, slots=True)
class Shortage:
    """Описание препятствия для ``no_match``."""

    category: str
    reason: str  # no_listings | constraint | budget
    details: str


def select_candidates(bucket: Bucket, category: str, slots: Slots) -> tuple[Listing, ...]:
    """Кандидаты категории с учётом ограничений покупателя (или все, если их нет).

    Без ограничения участвуют ВСЕ объявления бакета, включая те, у которых
    характеристика не указана: «отсутствующий ключ при отсутствии ограничения
    участвует». С ограничением — только подтверждённые характеристикой.
    """
    if category == "table" and slots.width is not None:
        return filter_numeric(bucket, slots.width)
    if category == "lamp" and slots.color_temperature is not None:
        return filter_numeric(bucket, slots.color_temperature)
    if category == "chair" and slots.adjustable_height is not None:
        return filter_flag(bucket, slots.adjustable_height)
    return bucket.priced_items


def collect_states(slots: Slots, index: CatalogIndex) -> list[CategoryState]:
    """Готовит по одному состоянию на каждую запрошенную категорию.

    Категории упорядочены канонически (стол → кресло → лампа), кандидаты —
    по возрастанию (цена, id): от этого зависит и отсечение, и детерминизм выдачи.
    """
    states: list[CategoryState] = []
    for category in CATEGORIES:
        if category not in slots.categories:
            continue
        bucket = bucket_for(index, slots.city_id or "", category)
        candidates = select_candidates(bucket, category, slots)
        constraint_applied = (
            (category == "table" and slots.width is not None)
            or (category == "lamp" and slots.color_temperature is not None)
            or (category == "chair" and slots.adjustable_height is not None)
        )
        states.append(
            CategoryState(
                category=category,
                candidates=tuple(sorted(candidates, key=lambda listing: (listing.price_rub, listing.id))),
                bucket=bucket,
                constraint_applied=constraint_applied,
            )
        )
    return states


def suffix_minimums(states: Sequence[CategoryState]) -> list[int]:
    """CORE. Суффиксные минимумы цен: ``result[i]`` — минимальная цена довеска с i-й категории.

    Зачем: это ключ к отсечению в DFS. При выборе товара в категории ``i``
    достаточно знать, сколько минимум придётся добавить на оставшиеся категории.

    Вход:  states — состояния категорий, кандидаты отсортированы по возрастанию цены.
    Выход: список длины ``len(states) + 1``; последний элемент 0 (за категориями
    ничего не остаётся). ``result[i] = min(цена первого кандидата i, result[i+1])``.

    Алгоритм: один проход справа налево.
    Сложность: время O(k) по числу категорий (k <= 3), память O(k).
    Почему именно он: суффиксный минимум считается за O(k) один раз и дальше
    сравнивается за O(1) на каждом шаге DFS — это дешевле, чем считать минимум
    остатка заново на каждом узле (O(k²), а при 250×3 — заметно).
    Крайние случаи: пустой список → [0]; пустая категория в середине → минимум
    становится 0, то есть отсечение по бюджету не сработает, и нехватку обнаружит
    отдельная проверка пустых списков (иначе получился бы ложный no_match).
    Вызывается из: app/catalog/search.py::enumerate_combinations.
    """
    result = [0] * (len(states) + 1)
    for position in range(len(states) - 1, -1, -1):
        candidates = states[position].candidates
        cheapest = candidates[0].price_rub if candidates else 0
        result[position] = min(cheapest, result[position + 1])
    return result


def enumerate_combinations(
    states: Sequence[CategoryState],
    budget_rub: int,
    limit: int = MAX_COMBINATIONS,
) -> list[tuple[Listing, ...]]:
    """CORE. Перечисляет допустимые комплекты в порядке возрастания суммы.

    Зачем: нужно найти до трёх разных комплектов в бюджете, не перебирая
    все ``n^k`` наборов. Отсечение по суффиксным минимумам делает перебор
    пропорциональным числу реально допустимых наборов.

    Вход:  states     — состояния категорий (кандидаты отсортированы по цене);
           budget_rub — общий верхний предел стоимости комплекта (граница включается);
           limit      — максимум найденных наборов (для последующего выбора).
    Выход: список кортежей объявлений (по одному на категорию), суммарная цена
    каждого <= ``budget_rub``, порядок — по возрастанию суммы.

    Алгоритм: DFS по категориям с суффиксными минимумами. На шаге ``i`` перебираем
    кандидатов по возрастанию цены; если ``сумма + цена + suffix_min[i+1] > budget``,
    прерываем цикл (``break``), потому что все следующие кандидаты не дешевле.
    Доказательство полноты: пусть есть допустимый набор с текущим кандидатом
    ``c``. Тогда ``сумма + price(c) + suffix_min[i+1] <= сумма набора <= budget``,
    значит условие отсечения ложно и ветка не обрезается. Если же оно истинно, то
    любой набор с ``c`` и любым доведением стоит >= этой величины, то есть
    недопустим. Следовательно, ни один допустимый набор не теряется.
    Сложность: время O(k · допустимых наборов) в худшем случае, на практике
    O(k · n) при малом бюджете; память O(k) на стек рекурсии.
    Почему именно он: наивные вложенные циклы без отсечения дают O(n^k), а полный
    перебор с сортировкой результатов — лишние аллокации; DFS с отсечением
    останавливается на первом же превышении бюджета и не строит недопустимые наборы.
    Крайние случаи: пустая категория → пустой результат (вызывающий слой отдаёт
    ``no_match``); бюджет меньше минимума → пустой результат; одна категория →
    просто отфильтрованные кандидаты; ``limit`` достигнут → поиск прекращается.
    Вызывается из: app/catalog/search.py::build_plan.
    """
    if not states or any(state.empty for state in states):
        return []
    suffix = suffix_minimums(states)
    results: list[tuple[Listing, ...]] = []
    chosen: list[Listing] = []

    def walk(position: int, total: int) -> None:
        if len(results) >= limit:
            return
        if position == len(states):
            results.append(tuple(chosen))
            return
        last = position == len(states) - 1
        for listing in states[position].candidates:
            new_total = total + listing.price_rub
            if last:
                if new_total > budget_rub:
                    break  # кандидаты отсортированы: дальше только дороже
            elif new_total + suffix[position + 1] > budget_rub:
                break
            chosen.append(listing)
            walk(position + 1, new_total)
            chosen.pop()
            if len(results) >= limit:
                return

    walk(0, 0)
    # Суммы возрастают не гарантированно: DFS идёт по возрастанию цены внутри
    # категории, но при нескольких категориях порядок наборов перемешивается.
    results.sort(key=lambda combination: (sum(listing.price_rub for listing in combination),
                                          tuple(listing.id for listing in combination)))
    return results


def choose_options(
    combinations: Sequence[tuple[Listing, ...]],
    limit: int = MAX_OPTIONS,
) -> list[tuple[Listing, ...]]:
    """CORE. Отбирает до ``limit`` вариантов, максимально различающихся составом.

    Зачем: контракт требует, чтобы варианты отличались составом ID. Показывать
    три комплекта, отличающиеся одной лампой, бессмысленно для покупателя.

    Вход:  combinations — допустимые наборы в порядке возрастания суммы;
           limit        — сколько вариантов нужно (<= 3).
    Выход: список наборов; первый — самый дешёвый; остальные выбраны так, чтобы
    отличаться от уже выбранных максимумом числа позиций (при равном числе —
    более дешёвый раньше).

    Алгоритм: жадный отбор. Первый набор — самый дешёвый. Далее в два прохода:
    сначала берём кандидатов, отличающихся во всех позициях (расстояние Хэмминга
    по составу = k), затем — отличающихся хотя бы в одной позиции. Дедупликация
    по составу ID через ``frozenset`` (порядок категорий фиксирован, поэтому
    сравнение кортежей тоже корректно).
    Сложность: время O(C · limit · k) при C найденных наборах (C <= 200, k <= 3),
    память O(limit · k).
    Почему именно он: перебор всех троек дал бы O(C³); жадный отбор за два прохода
    даёт тот же практический результат («разные комплекты»), но линейно.
    Крайние случаи: один допустимый набор → один вариант (контракт допускает 1);
    наборы совпадают по составу (невозможно при уникальных ID, но проверяется) →
    дубликат отбрасывается; меньше ``limit`` наборов → отдаём сколько есть.
    Вызывается из: app/catalog/search.py::build_plan.
    """
    if not combinations:
        return []
    selected: list[tuple[Listing, ...]] = [combinations[0]]
    seen: set[frozenset[str]] = {frozenset(listing.id for listing in combinations[0])}

    for required_distance in (len(combinations[0]), max(1, len(combinations[0]) - 1)):
        for combination in combinations[1:]:
            if len(selected) >= limit:
                return selected
            composition = frozenset(listing.id for listing in combination)
            if composition in seen:
                continue
            distance = sum(
                1 for left, right in zip(selected[0], combination) if left.id != right.id
            )
            if distance < required_distance:
                continue
            same_as_selected = any(
                sum(1 for left, right in zip(option, combination) if left.id != right.id) < required_distance
                for option in selected
            )
            if same_as_selected:
                continue
            selected.append(combination)
            seen.add(composition)
    return selected[:limit]
