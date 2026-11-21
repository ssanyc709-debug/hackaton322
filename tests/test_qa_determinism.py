"""Блок 9 — детерминизм и повторяемость.

Один и тот же запрос при одних и тех же данных обязан давать побайтово тот же
ответ: иначе проверяющая сторона не сможет воспроизвести результат, а ручная
демонстрация — повторить показ (§1, §5).

Проверяется, что ответ не зависит от того, что не является условием задачи:

* от повторного запуска и от состояния кэша (холодный/прогретый);
* от переменных окружения: ``PYTHONHASHSEED`` (порядок элементов множеств),
  локали, часового пояса;
* от форматирования запроса: порядок ключей, отступы, BOM, CRLF, завершающий
  перевод строки;
* от одновременного запуска нескольких процессов с общим кэшем.

Порядок вариантов внутри ответа тоже часть детерминизма: контракт разрешает любой
порядок (§4), но конкретный запуск обязан быть воспроизводимым.
"""

from __future__ import annotations

import concurrent.futures
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.test_qa_support import (
    EXIT_OK,
    ask,
    assistant,
    request,
    run_cli,
    user,
)

_WORKDIR: tempfile.TemporaryDirectory[str] | None = None

#: Запросы трёх статусов: детерминизм нужен всем, а не только ``ok``.
QUERIES: tuple[str, ...] = (
    "Стол, кресло и лампа в Таганроге до 12 000 ₽",
    "Стол в Азове до 100 ₽",
    "Стол",
)


def setUpModule() -> None:
    global _WORKDIR
    _WORKDIR = tempfile.TemporaryDirectory(prefix="qa-determinism-")


def tearDownModule() -> None:
    if _WORKDIR is not None:
        _WORKDIR.cleanup()


def workdir(name: str) -> Path:
    assert _WORKDIR is not None
    path = Path(_WORKDIR.name) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def stdout_of(
    stdin: bytes | str | object,
    *,
    cache: Path | str | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> bytes:
    """Сырой stdout одного запуска (сравнение побайтово)."""
    result = run_cli(stdin, cache_dir=cache, extra_env=extra_env)
    if result.code != EXIT_OK:
        raise AssertionError(f"код {result.code}, stderr: {result.stderr[-400:]}")
    return result.stdout


class RepeatabilityTest(unittest.TestCase):
    """Повторный запуск и состояние кэша не влияют на ответ."""

    def test_repeated_runs_are_byte_identical(self) -> None:
        cache = workdir("repeat-cache")
        for query in QUERIES:
            with self.subTest(query=query):
                first = stdout_of(ask(query), cache=cache)
                for attempt in range(1, 3):
                    self.assertEqual(
                        stdout_of(ask(query), cache=cache),
                        first,
                        f"попытка {attempt + 1}: ответ изменился при повторном запуске",
                    )

    def test_cold_and_warm_cache_give_the_same_answer(self) -> None:
        cache = workdir("warm-cache")
        for query in QUERIES:
            with self.subTest(query=query):
                warm = stdout_of(ask(query), cache=cache)  # уже прогретый к предыдущим тестам
                cold = stdout_of(ask(query), cache=workdir(f"cold-{abs(hash(query))}"))
                self.assertEqual(cold, warm, "ответ зависит от того, был ли кэш уже заполнен")

    def test_shared_cache_between_different_queries_does_not_leak(self) -> None:
        """Общий кэш на разные запросы: каждый получает свой ответ."""
        cache = workdir("shared-cache")
        responses = {query: stdout_of(ask(query), cache=cache) for query in QUERIES}
        self.assertEqual(len(set(responses.values())), len(QUERIES), "разные запросы дали одинаковый ответ")
        repeated = QUERIES[0]
        self.assertEqual(
            stdout_of(ask(repeated), cache=cache),
            responses[repeated],
            "после соседних запросов ответ изменился",
        )


class EnvironmentIndependenceTest(unittest.TestCase):
    """Окружение не влияет на ответ: хеш-сид, локаль, часовой пояс."""

    def test_hash_seed_locale_and_timezone_do_not_change_the_answer(self) -> None:
        query = "Стол, кресло и лампа в Таганроге до 12 000 ₽"
        cache = workdir("env-cache")
        baseline = stdout_of(ask(query), cache=cache)
        variants: tuple[tuple[str, Mapping[str, str]], ...] = (
            ("другой PYTHONHASHSEED", {"PYTHONHASHSEED": "12345"}),
            ("локаль C", {"LC_ALL": "C", "LANG": "C"}),
            ("часовой пояс Камчатка", {"TZ": "Asia/Kamchatka"}),
        )
        for name, env in variants:
            with self.subTest(case=name):
                self.assertEqual(
                    stdout_of(ask(query), cache=cache, extra_env=env),
                    baseline,
                    f"ответ изменился при условии «{name}»",
                )


class RequestFormattingTest(unittest.TestCase):
    """Форматирование JSON не является условием задачи и не влияет на ответ."""

    def test_equivalent_request_formats_give_the_same_answer(self) -> None:
        query = "Стол в Таганроге до 5000 ₽"
        cache = workdir("format-cache")
        baseline = stdout_of(ask(query), cache=cache)
        message = {"role": "user", "text": query}
        variants: tuple[tuple[str, str], ...] = (
            ("ключи в другом порядке", json.dumps({"messages": [{"text": query, "role": "user"}]}, ensure_ascii=False)),
            (
                "отступы и переводы строк",
                json.dumps({"messages": [message]}, ensure_ascii=False, indent=4),
            ),
            ("BOM перед объектом", "\ufeff" + json.dumps({"messages": [message]}, ensure_ascii=False)),
            ("CRLF и хвостовые пробелы", json.dumps({"messages": [message]}, ensure_ascii=False) + "\r\n "),
        )
        for name, stdin in variants:
            with self.subTest(case=name):
                self.assertEqual(
                    stdout_of(stdin, cache=cache),
                    baseline,
                    f"ответ изменился при форматировании «{name}»",
                )


class StableDocumentsTest(unittest.TestCase):
    """``clarify`` и ``no_match`` тоже воспроизводимы: те же поля, тот же текст."""

    def test_clarify_questions_have_a_stable_order(self) -> None:
        cache = workdir("clarify-cache")
        for query, expected in (
            ("Стол", {"budget_rub", "city_id"}),
            ("Хочу мебель", {"budget_rub", "city_id", "categories"}),
        ):
            with self.subTest(query=query):
                orders: set[tuple[str, ...]] = set()
                for _ in range(2):
                    document = json.loads(stdout_of(ask(query), cache=cache))
                    self.assertEqual(document.get("status"), "clarify", document)
                    orders.add(tuple(question["field"] for question in document["questions"]))
                    self.assertEqual(
                        {question["field"] for question in document["questions"]},
                        expected,
                    )
                self.assertEqual(len(orders), 1, f"порядок вопросов меняется между запусками: {orders}")

    def test_no_match_message_is_stable(self) -> None:
        cache = workdir("no-match-cache")
        query = "Стол, кресло и лампа в Азове до 1500 ₽"
        messages = set()
        for _ in range(2):
            document = json.loads(stdout_of(ask(query), cache=cache))
            self.assertEqual(document.get("status"), "no_match", document)
            messages.add(document["message"])
        self.assertEqual(len(messages), 1, f"сообщение no_match меняется: {messages}")

    def test_long_dialogue_is_stable(self) -> None:
        cache = workdir("dialogue-cache")
        conversation = request(
            user("Нужен стол, кресло и лампа в Таганроге"),
            assistant("Какой у вас бюджет на весь комплект?"),
            user("15 тысяч"),
            assistant("Вот варианты."),
            user("Без лампы"),
        )
        first = stdout_of(conversation, cache=cache)
        self.assertEqual(stdout_of(conversation, cache=cache), first, "ответ на диалог не воспроизводится")


class ConcurrencyTest(unittest.TestCase):
    """Несколько процессов с общим кэшем не мешают друг другу."""

    def test_parallel_runs_share_one_cache_without_conflict(self) -> None:
        cache = workdir("parallel-cache")
        stdin = ask("Стол, кресло и лампа в Таганроге до 12 000 ₽")

        def run_once(_: int) -> bytes:
            return stdout_of(stdin, cache=cache)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(run_once, range(4)))

        self.assertEqual(len(set(outputs)), 1, "параллельные запуски дали разные ответы")
        document = json.loads(outputs[0])
        self.assertEqual(document.get("status"), "ok", document)


class TimeBudgetTest(unittest.TestCase):
    """§1: один запрос укладывается в бюджет времени с большим запасом."""

    def test_each_request_finishes_quickly(self) -> None:
        cache = workdir("time-cache")
        for query in QUERIES:
            with self.subTest(query=query):
                started = time.monotonic()
                stdout_of(ask(query), cache=cache)
                elapsed = time.monotonic() - started
                self.assertLess(
                    elapsed,
                    20.0,
                    f"{query!r} занял {elapsed:.1f} с — это сигнал о проблеме, а не о нагрузке",
                )


if __name__ == "__main__":
    unittest.main()
