"""Нагрузочный прогон помощника: задержки, пропускная способность, расход токенов.

Стенд отвечает на четыре вопроса, которые задают на защите:

1. **Сколько стоит один запрос по времени.** Перцентили p50/p95/p99 считаются
   по замерам ``answer()`` в процессе, отдельно для холодного и прогретого кэшей.
2. **Какая пропускная способность.** Запросов в секунду на одном процессе —
   подбор и разбор локальные, поэтому узкое место только сеть модели.
3. **Какая доля запросов обходится без модели (§5: цель ≥ 60 %).** Считается
   по числу фактических обращений к ``resolver``, а не по «мы так думаем».
4. **Сколько токенов это экономит.** Каждое обращение к модели заменяется на
   константу измеренного обмена (см. ``docs/TOKEN_BUDGET.md``); в конце
   печатается «потрачено / сэкономлено» относительно стратегии «каждую реплику
   отдаём модели».

Прогон по умолчанию полностью офлайновый: модель подменяется функцией, которая
ничего не возвращает (эквивалент «модель ответила пустым JSON»). Сеть, ключ и
квота организаторов не расходуются — стенд можно запускать сколько угодно раз.

Запуск::

    python bench/load.py                     # 200 запросов в процессе + 5 CLI-запусков
    python bench/load.py --requests 500      # больше замеров
    python bench/load.py --process 20        # больше запусков ./run целиком
    python bench/load.py --json              # метрики в JSON (для вставки в отчёт)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.catalog.index import build_index, get_index  # noqa: E402
from app.config import LLM_MAX_OUTPUT_TOKENS, Settings  # noqa: E402
from app.contract.request import parse_request  # noqa: E402
from app.jsonio import loads_strict  # noqa: E402
from app.nlp.cache import LexiconCache  # noqa: E402
from app.pipeline import answer  # noqa: E402

CATALOG_PATH = REPO_ROOT / "data" / "catalog.json"

#: Измеренный обмен с моделью на одну реплику (live-прогон, см. docs/TOKEN_BUDGET.md).
#: prompt=197 (сжатый системный промпт + схема + реплика), completion=54.
TOKENS_PROMPT_PER_CALL = 197
TOKENS_COMPLETION_PER_CALL = 54
TOKENS_PER_CALL = TOKENS_PROMPT_PER_CALL + TOKENS_COMPLETION_PER_CALL

#: Смесь запросов, повторяющая демонстрацию и автопроверку.
#: Метка → реплики покупателя (assistant-реплики добавляет :func:`make_request`).
WORKLOAD: dict[str, tuple[str, ...]] = {
    # Лексикон закрывает все обязательные условия: 0 токенов.
    "выбор-01": (
        "Нужен стол не шире метра, кресло и лампа. Таганрог. До 15 000 ₽ за всё.",
    ),
    "выбор-02": (
        "Хочу письменный стол, офисное кресло с регулировкой высоты и лампу "
        "на 4000 K. Ростов-на-Дону, бюджет 20 тысяч.",
    ),
    "выбор-03": (
        "Стол и кресло в Азове, тысяч на двенадцать.",
    ),
    # Уточнение: не хватает обязательного условия.
    "уточнение-город": ("Нужен стол и кресло, бюджет 15 000 ₽.",),
    "уточнение-бюджет": ("Стол, кресло и лампа в Таганроге.",),
    # Нет вариантов: бюджет ниже минимального комплекта.
    "нет-вариантов": ("Стол, кресло и лампа. Таганрог. До 900 ₽.",),
    # Диалог из трёх реплик: условия приходят по частям.
    "диалог": (
        "Стол и кресло в Таганроге.",
        "Бюджет — двенадцать с половиной тысяч.",
        "И ещё лампу добавь, пожалуйста.",
    ),
    # Рецепт, где модель действительно может закрыть условие: город и бюджет
    # названы, а предмет — нет («деревяшка с четырьмя ножками»).
    "нужна-модель": ("Хочу обставить кабинет в Таганроге, тысяч на двенадцать.",),
    # Гейт блока 9: ни города, ни бюджета, ни предмета — модель закрыть нечего.
    "пустая-реплика": ("Хочу обставить кабинет, помогите",),
}


def make_request(texts: Sequence[str]) -> dict[str, Any]:
    """Собирает запрос контракта из реплик покупателя (между ними — ответ помощника)."""
    messages: list[dict[str, str]] = []
    for index, text in enumerate(texts):
        if index:
            messages.append({"role": "assistant", "text": "Уточняю."})
        messages.append({"role": "user", "text": text})
    return {"messages": messages}


def percentile(values: Sequence[float], fraction: float) -> float:
    """Перцентиль по методу ближайшего ранга (без внешних зависимостей).

    ``fraction=0.95`` при 200 замерах даёт 190-й элемент отсортированного ряда —
    то же определение ближайшего ранга, что и ``numpy.percentile(method="lower")``,
    но без установки numpy: стенд обязан работать на чистом stdlib.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def summarize(samples: Sequence[float]) -> dict[str, float]:
    """Компактная сводка задержек в миллисекундах: min/mean/p50/p95/p99/max."""
    if not samples:
        return {key: 0.0 for key in ("min", "mean", "p50", "p95", "p99", "max")}
    return {
        "min": min(samples),
        "mean": statistics.fmean(samples),
        "p50": percentile(samples, 0.50),
        "p95": percentile(samples, 0.95),
        "p99": percentile(samples, 0.99),
        "max": max(samples),
    }


class CountingResolver:
    """Подмена модели: считает обращения и не тратит ни одного токена.

    Возвращает ``{}`` — ровно то, что вернула бы модель, не нашедшая условий.
    Такой resolver позволяет измерить долю запросов, которым модель НУЖНА,
    не отправляя ни одного сетевого запроса.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str, slots: Any) -> dict[str, Any]:  # noqa: ARG002 — подпись Resolver
        self.calls += 1
        return {}


def run_phase(
    count: int,
    index: Any,
    settings: Settings,
    cache: LexiconCache,
    resolver: CountingResolver,
) -> dict[str, Any]:
    """Один проход нагрузки: ``count`` запросов подряд по кругу из :data:`WORKLOAD`."""
    names = list(WORKLOAD)
    requests = {name: parse_request(make_request(WORKLOAD[name])) for name in names}
    latencies: list[float] = []
    statuses: dict[str, int] = {}
    calls_by_recipe: dict[str, int] = {name: 0 for name in names}
    recipe_runs: dict[str, int] = {name: 0 for name in names}
    started = time.perf_counter()

    for position in range(count):
        name = names[position % len(names)]
        request = requests[name]
        before = resolver.calls
        began = time.perf_counter()
        document = answer(
            request=request,
            index=index,
            settings=settings,
            deadline=time.monotonic() + 120.0,
            cache=cache,
            resolver=resolver,
        )
        latencies.append((time.perf_counter() - began) * 1000.0)
        status = str(document.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
        calls_by_recipe[name] += resolver.calls - before
        recipe_runs[name] += 1

    elapsed = time.perf_counter() - started
    return {
        "requests": count,
        "elapsed_seconds": elapsed,
        "latency_ms": summarize(latencies),
        "throughput_per_second": count / elapsed if elapsed > 0 else 0.0,
        "statuses": statuses,
        "model_calls": resolver.calls,
        "calls_per_recipe": {
            name: calls_by_recipe[name] / recipe_runs[name] for name in names
        },
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "cache_hit_rate": cache.hit_rate,
    }


def measure_index(document: Sequence[Any], cache_dir: Path, catalog_path: Path) -> dict[str, Any]:
    """Замер построения индекса: холодная сборка и попадание в файловый кэш.

    Чтение из кэша замеряется в отдельном процессе: у ``get_index`` есть ещё
    и memo внутри процесса, поэтому повторный вызов в том же процессе ответил бы
    из памяти и ничего не сказал бы про файл. В подпроцессе замер идёт только
    вокруг ``get_index`` (чтение каталога — вне таймера), поэтому число честное.
    """
    began = time.perf_counter()
    cold = build_index(document)
    cold_ms = (time.perf_counter() - began) * 1000.0

    get_index(document, cache_dir)  # первый вызов пишет файл кэша

    code = (
        "import json, sys, time\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from app.catalog.index import get_index\n"
        "document = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))\n"
        "began = time.perf_counter()\n"
        "index = get_index(document, Path(sys.argv[3]))\n"
        "elapsed = (time.perf_counter() - began) * 1000.0\n"
        "print(elapsed, index.from_cache)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(REPO_ROOT), str(catalog_path), str(cache_dir)],
        capture_output=True,
        timeout=120,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
        cwd=str(REPO_ROOT),
    )
    warm_ms, from_cache = 0.0, False
    if completed.returncode == 0:
        measured, flag = completed.stdout.decode("utf-8").split()[:2]
        warm_ms, from_cache = float(measured), flag == "True"

    return {
        "listings": cold.size,
        "buckets": len(cold.buckets),
        "digest": cold.digest[:12],
        "build_ms": cold_ms,
        "cached_get_ms": warm_ms,
        "from_cache": from_cache,
    }


def measure_process(count: int, texts: Sequence[str]) -> dict[str, Any]:
    """Замер полного ``./run`` (интерпретатор + чтение каталога + ответ) на процесс."""
    payload = json.dumps(make_request(texts), ensure_ascii=False).encode("utf-8")
    timings: list[float] = []
    codes: list[int] = []
    for _ in range(count):
        began = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, "-m", "app.main", "--catalog", str(CATALOG_PATH)],
            input=payload,
            cwd=str(REPO_ROOT),
            capture_output=True,
            timeout=150,
            env={
                **__import__("os").environ,
                "ASSISTANT_OFFLINE": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONPATH": str(REPO_ROOT),
            },
        )
        timings.append((time.perf_counter() - began) * 1000.0)
        codes.append(completed.returncode)
    return {
        "runs": count,
        "wall_ms": summarize(timings),
        "exit_codes": sorted(set(codes)),
    }


def token_report(requests: int, model_calls: int) -> dict[str, Any]:
    """Сравнение с «наивной» стратегией: каждая реплика уходит в модель.

    Считается в терминах §5: сколько токенов потратили мы и сколько потратил бы
    подход без lexicon-first и без кэша.
    """
    naive = requests * TOKENS_PER_CALL
    spent = model_calls * TOKENS_PER_CALL
    return {
        "tokens_per_call": TOKENS_PER_CALL,
        "requests": requests,
        "model_calls": model_calls,
        "zero_token_requests": requests - model_calls,
        "zero_token_share": (requests - model_calls) / requests if requests else 0.0,
        "spent_tokens": spent,
        "naive_tokens": naive,
        "saved_tokens": naive - spent,
        "saved_share": (naive - spent) / naive if naive else 0.0,
    }


def format_summary(label: str, phase: dict[str, Any]) -> str:
    """Одна строка отчёта по проходу: задержки, пропускная способность, кэш."""
    latency = phase["latency_ms"]
    return (
        f"{label}: n={phase['requests']}, "
        f"p50={latency['p50']:.2f} мс, p95={latency['p95']:.2f} мс, "
        f"p99={latency['p99']:.2f} мс, max={latency['max']:.2f} мс, "
        f"{phase['throughput_per_second']:.0f} запр/с, "
        f"модель={phase['model_calls']}, кэш разборов={phase['cache_hit_rate']:.0%}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа стенда: печатает отчёт и возвращает код 0."""
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--requests", type=int, default=200, help="запросов в каждом проходе")
    parser.add_argument("--process", type=int, default=5, help="запусков ./run целиком")
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH, help="путь к каталогу")
    parser.add_argument("--json", action="store_true", help="вывести метрики в JSON")
    args = parser.parse_args(argv)

    if args.requests < len(WORKLOAD):
        parser.error(f"--requests должен быть >= {len(WORKLOAD)} (по одному на рецепт)")

    document = loads_strict(args.catalog.read_text(encoding="utf-8"), source="каталог")
    # Модель НЕ вызывается, но и не «выключена»: офлайн-флаг отключил бы сам
    # fallback, и стенд не смог бы измерить долю запросов, которым модель нужна.
    # Вместо сети работает CountingResolver, поэтому ключ фиктивный.
    settings = Settings.load(
        environ={
            "LLM_BASE_URL": "https://bench.invalid/v1",
            "LLM_MODEL": "google/gemma-4-31B-it",
            "LLM_API_KEY": "bench-no-network",
            "ASSISTANT_OFFLINE": "0",
        }
    )

    with tempfile.TemporaryDirectory(prefix="bench-index-") as folder:
        index_metrics = measure_index(document, Path(folder), args.catalog)

    index = build_index(document)
    cold_dir = tempfile.TemporaryDirectory(prefix="bench-cold-")
    warm_dir = tempfile.TemporaryDirectory(prefix="bench-warm-")

    cold_resolver = CountingResolver()
    cold = run_phase(
        args.requests,
        index,
        settings,
        LexiconCache(Path(cold_dir.name), enabled=True),
        cold_resolver,
    )
    warm_resolver = CountingResolver()
    warm_cache = LexiconCache(Path(warm_dir.name), enabled=True)
    run_phase(args.requests, index, settings, warm_cache, CountingResolver())  # прогрев файла
    warm_cache.hits = 0
    warm_cache.misses = 0
    warm = run_phase(args.requests, index, settings, warm_cache, warm_resolver)

    process_metrics = measure_process(args.process, WORKLOAD["выбор-01"]) if args.process > 0 else None
    tokens_cold = token_report(cold["requests"], cold["model_calls"])
    tokens_warm = token_report(warm["requests"], warm["model_calls"])

    cold_dir.cleanup()
    warm_dir.cleanup()

    if args.json:
        print(
            json.dumps(
                {
                    "index": index_metrics,
                    "cold": cold,
                    "warm": warm,
                    "process": process_metrics,
                    "tokens_cold": tokens_cold,
                    "tokens_warm": tokens_warm,
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        return 0

    print("=" * 78)
    print("Нагрузочный прогон помощника покупателя (офлайн, модель подменена)")
    print("=" * 78)
    print(
        f"каталог: {index_metrics['listings']} объявлений, "
        f"{index_metrics['buckets']} пар (город, категория), дайджест {index_metrics['digest']}"
    )
    print(
        f"индекс: холодная сборка {index_metrics['build_ms']:.2f} мс, "
        f"чтение из кэша {index_metrics['cached_get_ms']:.2f} мс "
        f"(from_cache={index_metrics['from_cache']})"
    )
    print()
    print(format_summary("холодный кэш разборов", cold))
    print(f"    статусы: {cold['statuses']}, попаданий в кэш: {cold['cache_hits']}")
    print(format_summary("прогретый кэш разборов", warm))
    print(f"    статусы: {warm['statuses']}, попаданий в кэш: {warm['cache_hits']}")
    print("    вызовов модели на один запрос по рецептам:")
    for name, calls in warm["calls_per_recipe"].items():
        if calls == 0:
            mark = "0 токенов (лексикон)"
        else:
            mark = f"{calls:g} вызов{'а' if calls != 1 else ''} модели"
        print(f"      {name:<22} {mark}")
    if process_metrics is not None:
        wall = process_metrics["wall_ms"]
        print(
            f"полный ./run (интерпретатор + каталог + ответ), n={process_metrics['runs']}: "
            f"p50={wall['p50']:.0f} мс, p95={wall['p95']:.0f} мс, max={wall['max']:.0f} мс, "
            f"коды выхода {process_metrics['exit_codes']}"
        )
    print()
    print("─" * 78)
    print("Бюджет токенов §5 (измеренный обмен: prompt=197, completion=54)")
    print("─" * 78)
    for label, report in (("холодный кэш", tokens_cold), ("прогретый кэш", tokens_warm)):
        print(
            f"{label}: запросов {report['requests']}, "
            f"без модели {report['zero_token_requests']} "
            f"({report['zero_token_share']:.0%}), "
            f"потрачено ≈{report['spent_tokens']} ток., "
            f"сэкономлено ≈{report['saved_tokens']} ({report['saved_share']:.0%})"
        )
    print(
        f"потолок одного вызова: max_tokens={LLM_MAX_OUTPUT_TOKENS} "
        f"(контракт разрешает 2000), фактический ответ ≈{TOKENS_COMPLETION_PER_CALL} токена — "
        f"запас ×{2000 // TOKENS_COMPLETION_PER_CALL} до предела контракта"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
