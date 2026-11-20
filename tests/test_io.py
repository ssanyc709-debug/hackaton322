"""Контурные проверки CLI: stdin/stdout, коды выхода, отсутствие мусора в stdout."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.support import CATALOG_PATH, read_example_request, run_cli

VALID_STATUSES = {"ok", "clarify", "no_match"}


class CliContractTest(unittest.TestCase):
    """Один процесс = один запрос = ровно один JSON-объект в stdout."""

    def assert_single_json_object(self, stdout: str) -> dict:
        self.assertTrue(stdout.endswith("\n"), "ответ должен завершаться переводом строки")
        self.assertEqual(stdout.count("\n"), 1, "в stdout должна быть ровно одна строка")
        self.assertFalse(stdout.lstrip().startswith("```"), "Markdown в stdout запрещён")
        document = json.loads(stdout)
        self.assertIsInstance(document, dict)
        return document

    def test_example_01_returns_contract_response(self) -> None:
        result = run_cli(read_example_request("01-selection"))
        self.assertEqual(result.code, 0, result.stderr)
        document = self.assert_single_json_object(result.stdout)
        self.assertIn(document["status"], VALID_STATUSES)
        self.assertIn("buyer-assistant", result.stderr)

    def test_example_02_returns_contract_response(self) -> None:
        result = run_cli(read_example_request("02-clarification"))
        self.assertEqual(result.code, 0, result.stderr)
        document = self.assert_single_json_object(result.stdout)
        self.assertIn(document["status"], VALID_STATUSES)

    def test_broken_json_gives_nonzero_and_empty_stdout(self) -> None:
        result = run_cli('{"messages": [}')
        self.assertNotEqual(result.code, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("stdin", result.stderr)

    def test_empty_stdin_is_an_error(self) -> None:
        result = run_cli("   \n")
        self.assertNotEqual(result.code, 0)
        self.assertEqual(result.stdout, "")

    def test_nan_is_rejected(self) -> None:
        result = run_cli('{"messages": [{"role": "user", "text": "x", "n": NaN}]}')
        self.assertNotEqual(result.code, 0)
        self.assertEqual(result.stdout, "")

    def test_duplicate_keys_are_rejected(self) -> None:
        result = run_cli('{"messages": [{"role": "user", "text": "a", "text": "b"}]}')
        self.assertNotEqual(result.code, 0)
        self.assertEqual(result.stdout, "")

    def test_extra_field_is_rejected(self) -> None:
        result = run_cli({"messages": [{"role": "user", "text": "стол"}], "city": "taganrog"})
        self.assertNotEqual(result.code, 0)
        self.assertEqual(result.stdout, "")

    def test_roles_must_alternate(self) -> None:
        result = run_cli({"messages": [{"role": "user", "text": "a"}, {"role": "user", "text": "b"}]})
        self.assertNotEqual(result.code, 0)
        self.assertEqual(result.stdout, "")

    def test_missing_catalog_file_is_a_technical_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "nope.json"
            result = run_cli({"messages": [{"role": "user", "text": "стол"}]}, args=["--catalog", str(missing)])
        self.assertNotEqual(result.code, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("каталог", result.stderr)

    def test_broken_catalog_is_a_technical_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "catalog.json"
            broken.write_text("{\"not\": \"a list\"}", encoding="utf-8")
            result = run_cli({"messages": [{"role": "user", "text": "стол"}]}, args=["--catalog", str(broken)])
        self.assertNotEqual(result.code, 0)
        self.assertEqual(result.stdout, "")

    def test_catalog_is_used_from_argument_not_hardcoded(self) -> None:
        """Подменённый каталог другого пути всё равно принимается."""
        payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))[:2]
        with tempfile.TemporaryDirectory() as directory:
            tiny = Path(directory) / "tiny.json"
            tiny.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            result = run_cli({"messages": [{"role": "user", "text": "стол"}]}, args=["--catalog", str(tiny)])
        self.assertEqual(result.code, 0, result.stderr)
        self.assertIn(json.loads(result.stdout)["status"], VALID_STATUSES)


if __name__ == "__main__":
    unittest.main()
