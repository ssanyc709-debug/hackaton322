"""Точка входа ``python -m app.main`` (вызывается из ``./run``).

Контракт §2.1: один JSON-объект из stdin до EOF → ровно один JSON-объект в stdout
(допустим завершающий перевод строки). Всё остальное — только в stderr.
Все три статуса (``ok``/``clarify``/``no_match``) — успешная обработка, exit 0.
Нарушение контракта входа, битый каталог, сбой внешнего сервиса — stderr
и ненулевой код; ``no_match`` для технических сбоев не используется.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

from app import config
from app.catalog import CatalogError, load_json_data, parse_catalog_source, validate_catalog_document
from app.catalog.index import CatalogIndex, get_index
from app.catalog.text_search import TextCatalog, is_text_catalog
from app.contract import response
from app.contract.request import RequestError, parse_request
from app.contract.response import ResponseError
from app.jsonio import JsonError, loads_strict
from app.llm.client import LlmError, make_resolver
from app.llm.dialogue import DialogueAnalysis
from app.llm.product import make_preference_checker, make_product_resolver
from app.nlp.cache import LexiconCache
from app.pipeline import answer
from app.prompt_safety import audit_catalog

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_INPUT = 2
EXIT_CATALOG = 3
EXIT_CONFIG = 4
EXIT_INTERNAL = 5

_LOG_PREFIX = "[buyer-assistant]"


def log(message: str) -> None:
    """Единственный канал диагностики: stderr, с префиксом и немедленным flush."""
    print(f"{_LOG_PREFIX} {message}", file=sys.stderr, flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    """Аргументы запуска. Обязателен только ``--catalog`` (путь не хардкодится)."""
    parser = argparse.ArgumentParser(
        prog="run",
        description="ИИ-помощник покупателя: один JSON-запрос из stdin, один JSON-ответ в stdout.",
    )
    parser.add_argument("--catalog", required=True, help="путь или HTTP(S)-URL файла catalog.json")
    parser.add_argument("--text-search", action="store_true", help="искать по названиям товаров без таксономии")
    parser.add_argument("--env-file", default=None, help="альтернативный .env (по умолчанию <repo>/.env)")
    parser.add_argument("--debug", action="store_true", help="подробная трассировка этапов в stderr")
    parser.add_argument("--dialogue-analysis-dir", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CORE. Один запрос = один процесс: разбор входа → пайплайн → ответ → код выхода.

    Зачем: автопроверка вызывает ``./run`` отдельным процессом на каждый запрос
    и читает stdout как единственный JSON. Любая строчка лога в stdout = провал,
    поэтому вывод собирается целиком, проверяется и пишется одной операцией.

    Вход:  argv — аргументы командной строки (по умолчанию ``sys.argv[1:]``).
    Выход: код процесса: 0 — ответ отдан (любой из трёх статусов),
           2 — нарушение контракта входа, 3 — проблема каталога,
           4 — ошибка конфигурации, 5 — внутренний сбой/недоступность сервиса.

    Алгоритм: линейный конвейер с ранними выходами; общий дедлайн
    ``RUN_BUDGET_SECONDS - RUN_BUDGET_SAFETY_SECONDS`` передаётся в пайплайн,
    чтобы вызовы модели не вышли за 120 с.
    Сложность: время O(n + m) на разбор каталога и запроса, память O(n).
    Почему именно он: единственная точка, где известны и контракт, и каталог,
    и бюджет времени; всё остальное — чистые функции.
    Крайние случаи: пустой stdin, не-UTF-8, два объекта в stdin, отсутствующий
    файл каталога, падение внутри пайплайна — во всех случаях stdout остаётся пустым.
    Вызывается из: ``python -m app.main`` (``run``).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    started = time.monotonic()
    deadline = started + config.RUN_BUDGET_SECONDS - config.RUN_BUDGET_SAFETY_SECONDS

    _force_utf8_streams()

    catalog_source = parse_catalog_source(args.catalog)
    if isinstance(catalog_source, Path) and not catalog_source.is_file():
        log(f"каталог не найден: {catalog_source}")
        return EXIT_CATALOG

    try:
        settings = config.Settings.load(
            env_file=Path(args.env_file).expanduser() if args.env_file else None
        )
    except config.ConfigError as exc:
        log(f"конфигурация: {exc}")
        return EXIT_CONFIG
    log(f"настройки: {settings.llm.redacted()} offline={settings.offline}")

    stage = time.monotonic()
    try:
        raw_catalog = load_json_data(catalog_source)
        if args.text_search or is_text_catalog(raw_catalog):
            index = TextCatalog.from_document(raw_catalog)
        else:
            catalog = validate_catalog_document(raw_catalog, catalog_source)
            index = get_index(catalog, cache_dir=settings.cache_dir)
    except CatalogError as exc:
        log(f"каталог: {exc}")
        return EXIT_CATALOG
    if isinstance(index, TextCatalog):
        log(f"каталог: {len(index.products)} товаров, текстовый индекс за {_ms(stage)} мс")
    else:
        log(
            f"каталог: {index.size} объявлений, {len(index.buckets)} пар (город, категория), "
            f"дайджест {index.digest[:12]}, кэш={'hit' if index.from_cache else 'miss'}, "
            f"индекс за {_ms(stage)} мс"
        )
        _report_injections(index)

    try:
        raw = sys.stdin.buffer.read()
    except OSError as exc:
        log(f"stdin: не удалось прочитать вход: {exc}")
        return EXIT_INPUT
    if not raw.strip():
        log("stdin: пустой запрос (ожидался один JSON-объект)")
        return EXIT_INPUT
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        log(f"stdin: не UTF-8 ({exc.reason} на байте {exc.start})")
        return EXIT_INPUT

    stage = time.monotonic()
    try:
        payload = loads_strict(text, source="stdin")
    except JsonError as exc:
        log(f"{exc}")
        return EXIT_INPUT
    try:
        request = parse_request(payload)
    except RequestError as exc:
        log(f"запрос: {exc}")
        return EXIT_INPUT
    log(f"запрос: {len(request.messages)} сообщений, разобран за {_ms(stage)} мс")

    stage = time.monotonic()
    if not settings.llm_available:
        log("модель: недоступна (офлайн-режим или нет ключа), разбор только лексиконом")
    try:
        analysis = DialogueAnalysis(args.dialogue_analysis_dir) if args.dialogue_analysis_dir else None
        if isinstance(index, TextCatalog):
            product_resolver = make_product_resolver(settings.llm, deadline) if settings.llm_available else None
            preference_checker = make_preference_checker(settings.llm, deadline) if settings.llm_available else None
            if analysis is not None:
                if product_resolver is not None:
                    product_resolver = analysis.wrap(product_resolver)
                if preference_checker is not None:
                    preference_checker = analysis.wrap(preference_checker)
            document = index.answer(request.user_texts, resolver=product_resolver, preference_checker=preference_checker)
        else:
            resolver = make_resolver(settings.llm, deadline) if settings.llm_available else None
            if resolver is not None and analysis is not None:
                resolver = analysis.wrap(resolver)
            document = answer(
                request=request,
                index=index,
                settings=settings,
                deadline=deadline,
                cache=LexiconCache(settings.cache_dir, enabled=not settings.offline_cache_disabled),
                resolver=resolver,
            )
        rendered = response.finalize(document)
    except LlmError as exc:
        log(f"модель: {exc.kind}: {exc}")
        return EXIT_INTERNAL
    except ResponseError as exc:
        log(f"самопроверка ответа не пройдена: {exc}")
        return EXIT_INTERNAL
    except Exception:  # noqa: BLE001 — любая ошибка ядра обязана стать кодом 5, а не no_match
        import traceback

        log("внутренняя ошибка пайплайна:\n" + traceback.format_exc())
        return EXIT_INTERNAL

    _write_stdout(rendered)
    log(
        f"готово: status={document['status']} за {_ms(stage)} мс, "
        f"всего {_ms(started)} мс"
    )
    return EXIT_OK


def _ms(since: float) -> int:
    """Миллисекунды с момента ``since`` (монотонные часы)."""
    return int((time.monotonic() - since) * 1000)


def _report_injections(index: CatalogIndex) -> None:
    """Сообщает в stderr, сколько объявлений пытаются управлять помощником (★).

    Выводится только количество, ID и виды находок — текст описания в лог не
    попадает (он может содержать поддельные строки лога). Строка нужна, чтобы
    попытка манипуляции была видна и не выглядела «повезло».
    """
    suspicious = audit_catalog(index.listings)
    if not suspicious:
        return
    shown = ", ".join(f"{listing_id} ({'/'.join(kinds)})" for listing_id, kinds in suspicious[:5])
    tail = "" if len(suspicious) <= 5 else f" и ещё {len(suspicious) - 5}"
    log(
        f"★ описания с попыткой манипуляции: {len(suspicious)} — {shown}{tail}; "
        f"в подборе описание не участвует, цена берётся только из price_rub"
    )


def _force_utf8_streams() -> None:
    """UTF-8 для логов даже на консоли Windows с cp1251."""
    for stream in (sys.stderr,):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # Поток уже обёрнут или перенаправлен — писать как есть.
                continue


def _write_stdout(text: str) -> None:
    """Пишет ровно один JSON-объект и перевод строки, минуя текстовую кодировку."""
    data = (text + "\n").encode("utf-8")
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None:
        buffer.write(data)
        buffer.flush()
    else:  # stdout уже текстовый (например, перехвачен в тестах)
        sys.stdout.write(text + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
