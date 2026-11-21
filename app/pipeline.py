"""Оркестратор: запрос покупателя → один ответ контракта.

Пайплайн линейный и чистый: история → слоты → проверка обязательных условий →
подбор комплектов. Ни один шаг не читает ``description`` объявления, поэтому
текст не может изменить правила (см. ``app/prompt_safety.py``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.catalog.index import CatalogIndex
from app.catalog.plan import build_plan
from app.config import Settings
from app.contract import response
from app.contract.request import Request
from app.nlp.cache import LexiconCache
from app.nlp.extract import Resolver, is_unintelligible, missing_fields
from app.nlp.merge import merge_history


def answer(
    request: Request,
    index: CatalogIndex,
    settings: Settings,
    deadline: float,
    *,
    cache: LexiconCache | None = None,
    resolver: Resolver | None = None,
) -> dict[str, Any]:
    """CORE. Возвращает документ ответа (``ok`` / ``clarify`` / ``no_match``).

    Зачем: это единственное место, где зафиксирован порядок правил контракта:
    сначала обязательные условия (иначе ``clarify``), только потом подбор.

    Вход:  request  — провалидированный запрос диалога (вся история);
           index    — индекс текущего каталога;
           settings — настройки процесса (модель, кэш, офлайн-режим);
           deadline — монотонное время, после которого запуск обязан завершиться
                      (используется клиентом модели, чтобы не выйти за 120 с);
           cache    — кэш лексиконных разборов (0 токенов на повторах);
           resolver — вызов модели для незакрытых слотов (``None`` в офлайн-режиме).
    Выход: словарь ровно одного статуса; самопроверка выполняется в ``app.main``.

    Алгоритм: :func:`merge_history` по репликам покупателя → :func:`missing_fields` →
    ``clarify`` при нехватке/неизвестности → :func:`build_plan` (``ok``/``no_match``).
    Сложность: O(сумма длин реплик + k·log n + k·C); модель — только если лексикон
    не закрыл обязательные условия.
    Почему именно он: порядок «сначала уточнить, потом подбирать» прямо записан
    в контракте (§2.4: при нескольких неизвестных условиях сначала ``clarify``),
    и он же избавляет от подбора по догадкам.
    Крайние случаи: пустая история (невозможна после ``parse_request``); все
    условия известны → подбор; город вне справочника → ``clarify`` по ``city_id``;
    бюджет не назван → ``clarify`` только с ``budget_rub``.
    Вызывается из: app/main.py::main, тесты, bench/load.py.
    """
    slots = merge_history(
        request.user_texts,
        cache=cache,
        resolver=resolver,
        allow_model=settings.llm_available and resolver is not None,
    )

    # Непонятная последняя реплика не подтверждает прежний набор категорий
    # (CONTRACT.md §3): «ыаллфдылааввы» — это не «продолжай». Сбрасываем ровно
    # категории, чтобы missing_fields спросил их заново; город и бюджет,
    # названные ранее, повторно не спрашиваются.
    if request.user_texts and is_unintelligible(request.user_texts[-1]):
        slots = replace(
            slots,
            categories=(),
            categories_unknown=(),
            categories_removed=(),
            categories_explicit=False,
            categories_additive=False,
        )

    missing = missing_fields(slots)
    if missing:
        return response.clarify(missing)

    return build_plan(slots, index)
