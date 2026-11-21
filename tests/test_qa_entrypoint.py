"""Точка входа: статическая проверка ``run`` и боевой путь ``python -m app.main``.

``run`` — bash-скрипт, поэтому сам он не запускается (в окружении без bash это
дало бы ложное падение). Внутри него стоит ровно ``exec "$PYTHON_BIN" -m app.main
"$@"``, значит функционально это тот же код, что и ``python -m app.main``, и все
поведенческие проверки делаются через него. У ``run`` проверяется только
содержимое: файл на месте и пробрасывает модуль и ``--catalog``.
"""

from __future__ import annotations

import json
import unittest

from tests.test_qa_support import (
    CATALOG_PATH,
    EXIT_OK,
    RUN_SCRIPT,
    UPSTREAM_EXAMPLES,
    ask,
    assistant,
    cache_dir,
    load_catalog,
    make_temp_dir,
    option_ids,
    request,
    run_cli,
    user,
    write_catalog,
)

#: Запрос заведомо без комплекта в бюджете: 100 ₽ на стол, кресло и лампу.
NO_MATCH_QUERY = "Стол, кресло и лампа. Таганрог. До 100 ₽."


def example_request(name: str) -> bytes:
    """Байты запроса из открытого примера кейса."""
    return (UPSTREAM_EXAMPLES / name / "request.json").read_bytes()


class RunScriptContentTest(unittest.TestCase):
    """``run`` существует и вызывает модуль ``app.main`` с аргументами."""

    def test_run_script_exists_and_is_not_empty(self) -> None:
        self.assertTrue(RUN_SCRIPT.is_file(), "в корне репозитория нет файла run (контракт §1)")
        self.assertGreater(RUN_SCRIPT.stat().st_size, 0, "run пустой")

    def test_run_script_forwards_module_and_catalog_argument(self) -> None:
        text = RUN_SCRIPT.read_text(encoding="utf-8", errors="replace")
        with self.subTest(fragment="app.main"):
            self.assertIn("app.main", text, "run должен запускать модуль app.main")
        with self.subTest(fragment="--catalog"):
            self.assertIn("--catalog", text, "run должен документировать аргумент --catalog")
        with self.subTest(fragment="args"):
            self.assertTrue(
                '"$@"' in text or "$@" in text,
                "run обязан пробрасывать аргументы в app.main, иначе --catalog потеряется",
            )


class MainModuleContractTest(unittest.TestCase):
    """Боевой путь проверяющего: ``python -m app.main --catalog …``."""

    def test_example_01_returns_one_json_object_with_exit_zero(self) -> None:
        result = run_cli(example_request("01-selection"), cache_dir=cache_dir(self))
        self.assertEqual(result.code, EXIT_OK, result.stderr)
        document = result.one_json_object()
        self.assertIn(document.get("status"), {"ok", "clarify", "no_match"})

    def test_example_02_returns_clarify_with_exit_zero(self) -> None:
        result = run_cli(example_request("02-clarification"), cache_dir=cache_dir(self))
        self.assertEqual(result.code, EXIT_OK, result.stderr)
        self.assertEqual(result.one_json_object().get("status"), "clarify")

    def test_every_status_exits_zero(self) -> None:
        """Успешная обработка любого из трёх статусов — код 0 (контракт §1)."""
        cases = {
            "ok": ask("Стол, кресло и лампа. Таганрог. До 15 000 ₽."),
            "clarify": ask("Стол не шире метра, кресло и лампа. Таганрог."),
            "no_match": ask(NO_MATCH_QUERY),
        }
        seen: set[str] = set()
        for expected, payload in cases.items():
            with self.subTest(status=expected):
                result = run_cli(payload, cache_dir=cache_dir(self))
                self.assertEqual(result.code, EXIT_OK, f"{expected}: {result.stderr[-400:]}")
                document = result.one_json_object()
                self.assertEqual(document.get("status"), expected, document)
                seen.add(expected)
        self.assertEqual(seen, {"ok", "clarify", "no_match"}, "не все статусы достижимы")

    def test_stdout_has_no_logs_or_markdown(self) -> None:
        result = run_cli(example_request("01-selection"), cache_dir=cache_dir(self))
        self.assertEqual(result.code, EXIT_OK, result.stderr)
        text = result.stdout_text
        for forbidden in ("[buyer-assistant]", "Traceback", "```", "status="):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text, "в stdout попал лог или разметка")

    def test_catalog_argument_is_used_instead_of_hardcoded_ids(self) -> None:
        """Подменённый каталог по другому пути должен давать свои ID."""
        document = load_catalog()
        renamed = [dict(raw, id=f"qa-{index}") for index, raw in enumerate(document)]
        catalog = write_catalog(make_temp_dir(self), renamed)
        result = run_cli(example_request("01-selection"), catalog=catalog, cache_dir=cache_dir(self))
        self.assertEqual(result.code, EXIT_OK, result.stderr)
        response = result.one_json_object()
        self.assertEqual(response.get("status"), "ok", response)
        returned = {listing_id for option in response["options"] for listing_id in option_ids(option)}
        self.assertTrue(returned, "ответ ok без ID")
        for listing_id in returned:
            with self.subTest(listing_id=listing_id):
                self.assertTrue(listing_id.startswith("qa-"), f"ID {listing_id} не из переданного каталога")

    def test_broken_input_exits_nonzero_with_empty_stdout(self) -> None:
        """Битый вход: ненулевой код и пустой stdout (конкретное число не фиксируем)."""
        payloads = {
            "not_json": b'{"messages": [}',
            "not_object": b"[]",
            "missing_messages": b"{}",
            "empty_messages": b'{"messages": []}',
            "extra_field": json.dumps({"messages": [user("стол")], "city": "taganrog"}).encode(),
            "roles_do_not_alternate": json.dumps(request(user("а"), user("б"))).encode(),
            "last_is_assistant": json.dumps(request(user("а"), assistant("?"))).encode(),
            "not_utf8": b'{"messages": [{"role": "user", "text": "\xff\xfe"}]}',
            "empty_stdin": b"   \n",
        }
        for name, payload in payloads.items():
            with self.subTest(case=name):
                result = run_cli(payload, cache_dir=cache_dir(self))
                self.assertNotEqual(result.code, EXIT_OK, f"{name}: битый вход не должен давать код 0")
                self.assertEqual(result.stdout, b"", f"{name}: stdout должен быть пустым")
                self.assertTrue(result.stderr.strip(), f"{name}: причина должна быть в stderr")


if __name__ == "__main__":
    unittest.main()
