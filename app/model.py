"""Доменные типы: объявление каталога, граница условия, диапазон.

Отдельный модуль нужен, чтобы ``app.nlp`` (лексикон, разбор чисел) и
``app.catalog`` (индекс, подбор) говорили об одних и тех же типах и не
импортировали друг друга по кругу.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

#: Поддерживаемые категории — ровно как в data/categories.json.
CATEGORIES: tuple[str, ...] = ("table", "chair", "lamp")

#: Характеристика, по которой проверяется ограничение каждой категории.
CATEGORY_ATTRIBUTE: dict[str, str] = {
    "table": "width_cm",
    "chair": "adjustable_height",
    "lamp": "color_temperature_k",
}


class CatalogError(ValueError):
    """Каталог не соответствует контракту (технический сбой, а не ``no_match``)."""


@dataclass(frozen=True, slots=True)
class Bound:
    """Одна граница числового условия.

    ``inclusive=True`` для «до», «не более», «не шире», «от», «не менее»;
    ``inclusive=False`` для «меньше», «больше», «уже», «шире».
    """

    value: int
    inclusive: bool = True


@dataclass(frozen=True, slots=True)
class NumericRange:
    """Условие на числовую характеристику: нижняя и/или верхняя граница."""

    lower: Bound | None = None
    upper: Bound | None = None


@dataclass(frozen=True, slots=True)
class Listing:
    """Объявление каталога — только структурированные поля как источник фактов."""

    id: str
    category: str
    title: str
    description: str
    price_rub: int
    city_id: str
    attributes: Mapping[str, Any]

    def numeric(self, key: str) -> int | None:
        """Числовое значение характеристики или ``None``, если ключа нет.

        ``bool`` намеренно не считается числом (в Python ``True`` — это ``1``),
        а отсутствующий ключ — это «неизвестно», а не «ноль».
        """
        if key not in self.attributes:
            return None
        value = self.attributes[key]
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

    def flag(self, key: str) -> bool | None:
        """Булево значение характеристики или ``None``, если ключа нет/тип иной."""
        if key not in self.attributes:
            return None
        value = self.attributes[key]
        return value if isinstance(value, bool) else None


#: Имена полей условий — используются и в слиянии истории, и в трассировке.
FIELD_CITY = "city_id"
FIELD_BUDGET = "budget_rub"
FIELD_CATEGORIES = "categories"
FIELD_WIDTH = "width_cm"
FIELD_TEMPERATURE = "color_temperature_k"
FIELD_ADJUSTABLE = "adjustable_height"

ALL_FIELDS: tuple[str, ...] = (
    FIELD_CITY,
    FIELD_BUDGET,
    FIELD_CATEGORIES,
    FIELD_WIDTH,
    FIELD_TEMPERATURE,
    FIELD_ADJUSTABLE,
)


@dataclass(frozen=True, slots=True)
class Slots:
    """Условия покупателя, извлечённые из реплики.

    ``mentioned`` хранит поля, которые были явно названы в ЭТОМ сообщении:
    только они заменяют прежние значения при слиянии истории (§2.3), остальные
    сохраняются как есть. ``categories_removed`` — категории, от которых покупатель
    отказался («без лампы», «стол не нужен»), ``categories_additive`` — признак
    «добавь ещё», а не «замени всё». ``fields_removed`` — ограничения, которые
    покупатель снял («расширим ширину», «убери требование к регулировке»):
    при слиянии такое поле обнуляется, как и отказанная категория.
    """

    city_id: str | None = None
    budget_rub: int | None = None
    categories: tuple[str, ...] = ()
    width: NumericRange | None = None
    color_temperature: NumericRange | None = None
    adjustable_height: bool | None = None
    city_unknown: str | None = None
    categories_unknown: tuple[str, ...] = ()
    categories_removed: tuple[str, ...] = ()
    fields_removed: tuple[str, ...] = ()
    categories_explicit: bool = False
    categories_additive: bool = False
    mentioned: frozenset[str] = frozenset()
    source: str = "lexicon"
    categories_uncertain: bool = False


