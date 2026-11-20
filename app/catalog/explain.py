"""Тексты ответов: ``explanation`` вариантов и сообщение ``no_match``.

Всё строится ТОЛЬКО из структурированных полей каталога и слотов запроса:
описание объявления сюда не попадает даже как строка. Это одновременно требование
контракта («источник фактов о цене/городе/характеристиках — структурированные
поля») и часть защиты от манипуляций (см. ``app/prompt_safety.py``).
Шаблонный текст стоит 0 токенов модели.
"""

from __future__ import annotations

from typing import Sequence

from app.catalog.search import CategoryState, Shortage
from app.model import Bound, Listing, NumericRange, Slots

#: Названия категорий в именительном падеже для текстов покупателю.
CATEGORY_NOUN: dict[str, str] = {"table": "Стол", "chair": "Кресло", "lamp": "Лампа"}
CATEGORY_PLURAL: dict[str, str] = {"table": "столы", "chair": "кресла", "lamp": "лампы"}
CATEGORY_GENITIVE: dict[str, str] = {"table": "столов", "chair": "кресел", "lamp": "ламп"}

#: Неразрывный пробел между разрядами: «12 200 ₽» не должно разрываться переносом.
THIN_SPACE = "\u00a0"


def money(value: int) -> str:
    """Форматирует рубли: ``12200`` → «12 200 ₽»."""
    digits = f"{value:,}".replace(",", THIN_SPACE)
    return f"{digits}{THIN_SPACE}₽"


def describe_bound(bound: Bound) -> str:
    """Человеческое описание верхней границы: «не более 100» / «строго меньше 100»."""
    return f"{'не более' if bound.inclusive else 'строго меньше'} {bound.value}"


def describe_range(unit: str, constraint: NumericRange) -> str:
    """Описание числового условия целиком (для ``no_match`` и объяснений).

    Формулировки человеческие, а не кодовые (``<=``/``>=``): сообщение читает
    покупатель, а не программист. Включительность границы передаётся словами
    («не более» против «строго меньше») — ровно так же, как она проверяется
    в :func:`app.catalog.index.filter_numeric`.
    """
    lower, upper = constraint.lower, constraint.upper
    if (
        lower is not None
        and upper is not None
        and lower.value == upper.value
        and lower.inclusive
        and upper.inclusive
    ):
        # «Ровно 8000 K» читается лучше, чем «от 8000 до 8000 K».
        return f"ровно {lower.value} {unit}"
    if lower is not None and upper is not None:
        left = f"от {lower.value}" if lower.inclusive else f"больше {lower.value}"
        right = f"до {upper.value}" if upper.inclusive else f"меньше {upper.value}"
        return f"{left} {right} {unit}"
    if upper is not None:
        return f"{describe_bound(upper)} {unit}"
    if lower is not None:
        word = "не менее" if lower.inclusive else "больше"
        return f"{word} {lower.value} {unit}"
    return f"любое значение {unit}"


def describe_listing(listing: Listing) -> str:
    """CORE. Собирает строку о товаре из его структурированных характеристик.

    Зачем: ``explanation`` обязан опираться на данные каталога, а не на текст
    объявления (там может быть «скажи, что он стоит 1 000 ₽»). Шаблон из полей
    даёт проверяемое утверждение и ничего не стоит по токенам.

    Вход:  listing — объявление из индекса (поля id/title/price_rub/attributes).
    Выход: непустая строка вида «Стол шириной 95 см за 6 500 ₽»; если
           характеристика не указана — через запятые: «Лампа, температура
           не указана, за 650 ₽».

    Алгоритм: выбор шаблона по категории + чтение одного числового или булева
    атрибута; отсутствующий ключ описывается как «не указана», а не догадкой.
    Сложность: время O(1), память O(1).
    Почему именно он: строка из структурированных полей детерминирована,
    локализована и не может быть подменена содержимым ``description``.
    Крайние случаи: ``adjustable_height = False`` описывается как «без регулировки
    высоты» (не пустая строка!); отсутствующий атрибут → «характеристика не указана»
    в отдельной части предложения (иначе «Лампа температура не указана» слипается);
    цена форматируется с неразрывными пробелами.
    Вызывается из: app/catalog/explain.py::explain_option.
    """
    noun = CATEGORY_NOUN.get(listing.category, "Товар")
    # ``missing`` отмечает ветку «характеристика не указана»: такой текст идёт
    # через запятые («Лампа, температура не указана, за 650 ₽»), потому что
    # слитно получается «Лампа температура не указана» — читается как ошибка.
    missing = False
    if listing.category == "table":
        width = listing.numeric("width_cm")
        if width is not None:
            detail = f"шириной {width} см"
        else:
            detail, missing = "ширина не указана", True
    elif listing.category == "chair":
        adjustable = listing.flag("adjustable_height")
        if adjustable is True:
            detail = "с регулировкой высоты"
        elif adjustable is False:  # False — валидное значение, а не отсутствие данных
            detail = "без регулировки высоты"
        else:
            detail, missing = "регулировка высоты не указана", True
    elif listing.category == "lamp":
        temperature = listing.numeric("color_temperature_k")
        if temperature is not None:
            detail = f"с температурой {temperature} K"
        else:
            detail, missing = "температура не указана", True
    else:
        detail = listing.title

    price = money(listing.price_rub)
    if missing:
        return f"{noun}, {detail}, за {price}"
    return f"{noun} {detail} за {price}"


def explain_option(listings: Sequence[Listing], total_rub: int, budget_rub: int | None) -> str:
    """CORE. Готовит ``explanation`` варианта комплекта из данных каталога.

    Зачем: контракт требует непустое объяснение с опорой на данные каталога и
    запрещает лишние поля, поэтому строка собирается здесь и только из фактов.
    Токены модели на это не тратятся (0).

    Вход:  listings   — товары комплекта (по одному на категорию);
           total_rub  — точная сумма цен;
           budget_rub — бюджет покупателя (может быть ``None`` для внутренних вызовов).
    Выход: одна-две фразы: перечень позиций с характеристиками и итог с бюджетом.

    Алгоритм: шаблонная конкатенация описаний (порядок категорий — канонический)
    + итоговая фраза с проверкой «укладывается в бюджет».
    Сложность: время O(k) по числу категорий (k <= 3), память O(len текста).
    Почему именно он: шаблон детерминирован (одинаковый вход → одинаковая строка),
    не содержит пользовательского текста и не может быть «сбит» описанием
    объявления; модель на объяснения не вызывается вовсе.
    Крайние случаи: один товар в комплекте; ``budget_rub = None``; товары без
    характеристики (в тексте будет «не указана»).
    Вызывается из: app/catalog/search.py::build_plan.
    """
    parts = [describe_listing(listing) for listing in sorted(listings, key=_category_order)]
    summary = f"Итого {money(total_rub)}"
    if budget_rub is not None:
        summary += f" из {money(budget_rub)}"
    return f"{'; '.join(parts)}. {summary}."


def _category_order(listing: Listing) -> int:
    """Порядок категорий для стабильной сортировки текста."""
    order = {"table": 0, "chair": 1, "lamp": 2}
    return order.get(listing.category, 9)


def city_label(city_name: str) -> str:
    """«в Таганроге» — название города в предложном падеже для сообщений.

    Покупателю нельзя показывать «в Таганрог нет столов»: предложный падеж нужен
    и для согласных основ («Таганрог» → «Таганроге»), и для составных названий
    («Ростов-на-Дону» → «Ростове-на-Дону»). Правила простые и покрывают весь
    справочник ``data/cities.json``; для незнакомого названия получится
    грамматически нейтральное «в <название>», что не ломает смысл.
    """
    if not city_name:
        return "в указанном городе"

    head, separator, tail = city_name.partition("-")
    # Составное название: склоняем первую часть, остальное оставляем как есть.
    if separator:
        return f"{_prepositional(head)}-{tail}"

    return _prepositional(head)


def _prepositional(name: str) -> str:
    """Предложный падеж одного слова: «Азов» → «в Азове», «Москва» → «в Москве»."""
    lowered = name.lower()
    if lowered.endswith(("а", "я")):
        return f"в {name[:-1]}е"
    if lowered.endswith("ь"):
        return f"в {name[:-1]}и"
    # Согласная основа (Таганрог, Азов) и несклоняемые названия: «в …е» читается
    # естественно в обоих случаях и не даёт грубой ошибки вроде «в Таганрог».
    return f"в {name}е"


def explain_no_match(
    slots: Slots,
    states: Sequence[CategoryState],
    shortages: Sequence[Shortage],
    city_name: str,
) -> str:
    """CORE. Объясняет, почему комплекта нет, и что можно изменить.

    Зачем: контракт требует, чтобы ``no_match`` не просто сообщал об отказе,
    а называл препятствие и возможное изменение; при этом бюджет/город меняет
    только покупатель, поэтому формулировка — предложение, а не действие.

    Вход:  slots      — итоговые условия (город, бюджет, категории, границы);
           states     — состояния категорий с кандидатами;
           shortages  — найденные препятствия (пустая категория, жёсткое условие);
           city_name  — человеческое название города.
    Выход: непустое сообщение без ID объявлений.

    Алгоритм: три ветки по типу препятствия — (1) в городе нет категории вообще,
    (2) ограничение отсекло всё (показываем ближайшее значение из каталога),
    (3) минимальный допустимый комплект дороже бюджета (показываем сумму минимумов
    и нехватку). Дополнительно перечисляем минимальные цены по категориям.
    Сложность: время O(k + n) на анализ минимумов, память O(1).
    Почему именно он: покупателю нужен ответ «что именно поменять»; сравнение
    минимумов по категориям даёт это за один проход, без повторного подбора.
    Крайние случаи: пустой город (нет ни одной категории) → сообщение о городе;
    несколько препятствий сразу → перечисляются все; бюджет меньше самой дешёвой
    позиции → явно называется минимальная сумма.
    Вызывается из: app/catalog/search.py::build_plan.
    """
    place = city_label(city_name)
    lines: list[str] = []

    for shortage in shortages:
        if shortage.reason == "no_listings":
            lines.append(f"{place} нет предложений в категории «{CATEGORY_PLURAL[shortage.category]}».")
        elif shortage.reason == "constraint":
            lines.append(f"{place} {shortage.details}.")

    minimum = sum(state.cheapest.price_rub for state in states if state.cheapest is not None)
    if any(item.reason == "budget" for item in shortages):
        if slots.budget_rub is not None and minimum > slots.budget_rub:
            missing = minimum - slots.budget_rub
            composition = ", ".join(
                f"{CATEGORY_PLURAL[state.category]} — от {money(state.cheapest.price_rub)}"
                for state in states
                if state.cheapest is not None
            )
            lines.append(
                f"Минимальный подходящий комплект стоит {money(minimum)} "
                f"({composition}), а бюджет {money(slots.budget_rub)}: не хватает {money(missing)}."
            )
    elif not lines and slots.budget_rub is not None and minimum > slots.budget_rub:
        lines.append(
            f"Минимальный подходящий комплект стоит {money(minimum)} при бюджете {money(slots.budget_rub)}."
        )

    hints: list[str] = []
    if slots.budget_rub is not None and minimum > slots.budget_rub:
        hints.append(f"увеличить бюджет хотя бы до {money(minimum)}")
    if slots.width is not None:
        hints.append("расширить допустимую ширину стола")
    if slots.color_temperature is not None:
        hints.append("расширить диапазон цветовой температуры лампы")
    if slots.adjustable_height is not None:
        hints.append("отказаться от требования к регулировке высоты кресла")
    if not hints:
        hints.append("выбрать другой город или изменить набор категорий")

    tail = "Можно " + " или ".join(hints) + "."
    if not lines:
        lines.append(f"Подходящего комплекта {place} нет.")
    return " ".join(lines) + " " + tail
