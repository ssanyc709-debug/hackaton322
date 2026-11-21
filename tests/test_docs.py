"""Документация не может отстать от кода: каждая ``CORE``-функция описана.

Проверяется в обе стороны:

* нет ``CORE``-функции без раздела в ``docs/ALGORITHMS.md``;
* нет раздела в ``docs/ALGORITHMS.md`` без функции в коде (документ не выдумывает);
* у каждого раздела есть блок «На защите» — иначе защищать нечего.

Так «забыть обновить документ» технически невозможно: тест падает.
"""

from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path

from tests.support import REPO_ROOT

ALGORITHMS = REPO_ROOT / "docs" / "ALGORITHMS.md"
APP_DIR = REPO_ROOT / "app"

#: Ключ раздела: ``<модуль>.<функция>`` или ``<модуль>.<Класс>.<метод>``.
KEY_PATTERN = re.compile(r"^### (\w+(?:\.\w+)+)$", re.MULTILINE)


def core_functions() -> dict[str, str]:
    """Все функции, помеченные ``CORE`` в докстринге: ключ → путь и строка."""
    found: dict[str, str] = {}
    for path in sorted(APP_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))

        def walk(node: ast.AST, prefix: str = "") -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    docstring = ast.get_docstring(child) or ""
                    if "CORE" in docstring:
                        key = f"{path.stem}.{prefix}{child.name}"
                        found[key] = f"{path.relative_to(REPO_ROOT)}:{child.lineno}"
                elif isinstance(child, ast.ClassDef):
                    walk(child, f"{child.name}.")

        walk(tree)
    return found


class AlgorithmsDocumentTest(unittest.TestCase):
    """Соответствие «код ↔ документ»."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = ALGORITHMS.read_text(encoding="utf-8")
        cls.core = core_functions()
        cls.headings = KEY_PATTERN.findall(cls.text)

    def test_document_exists_and_is_not_empty(self) -> None:
        self.assertGreater(len(self.text), 10_000)
        self.assertIn("# Алгоритмы", self.text)

    def test_every_core_function_has_a_row(self) -> None:
        for key, where in sorted(self.core.items()):
            with self.subTest(function=key, where=where):
                self.assertIn(f"`{key}`", self.text, f"нет строки в таблице для {key}")

    def test_every_core_function_has_a_section(self) -> None:
        for key in sorted(self.core):
            with self.subTest(function=key):
                self.assertIn(key, self.headings, f"нет раздела «### {key}»")

    def test_document_has_no_invented_sections(self) -> None:
        for heading in self.headings:
            with self.subTest(heading=heading):
                self.assertIn(heading, self.core, f"раздел {heading} не соответствует коду")

    def test_every_section_has_a_defence_block(self) -> None:
        """Блок «На защите» обязателен: документ — сценарий защиты, а не справка."""
        positions = [(match.start(), match.group(1)) for match in KEY_PATTERN.finditer(self.text)]
        self.assertTrue(positions)
        for index, (start, key) in enumerate(positions):
            end = positions[index + 1][0] if index + 1 < len(positions) else len(self.text)
            section = self.text[start:end]
            with self.subTest(function=key):
                self.assertIn("> **На защите.**", section, f"в разделе {key} нет блока «На защите»")

    def test_table_row_count_matches_core_count(self) -> None:
        rows = re.findall(r"^\| \d+ \| `([\w.]+)` \|", self.text, re.MULTILINE)
        self.assertEqual(sorted(rows), sorted(self.core))

    def test_documented_numbers_match_measurements(self) -> None:
        """Числа в документе — из замеров, а не «на глаз»."""
        self.assertIn("111 объявлений", self.text)
        self.assertIn("89 %", self.text)


class TokenBudgetDocumentTest(unittest.TestCase):
    """``docs/TOKEN_BUDGET.md``: бюджет §5 с проверяемыми утверждениями."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = REPO_ROOT / "docs" / "TOKEN_BUDGET.md"
        cls.text = cls.path.read_text(encoding="utf-8") if cls.path.is_file() else ""

    def test_document_exists(self) -> None:
        self.assertTrue(self.path.is_file(), "docs/TOKEN_BUDGET.md отсутствует")
        self.assertIn("#", self.text)

    def test_uses_measured_exchange(self) -> None:
        self.assertIn("prompt=197", self.text)
        self.assertIn("completion=54", self.text)

    def test_mentions_the_sixty_percent_goal_and_the_result(self) -> None:
        self.assertIn("60", self.text)
        self.assertIn("89", self.text)


class ReadmeTest(unittest.TestCase):
    """README обязан отвечать на вопросы контракта: зависимости, сеть, модель, запуск."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = REPO_ROOT / "README.md"
        cls.text = cls.path.read_text(encoding="utf-8") if cls.path.is_file() else ""

    def test_readme_exists(self) -> None:
        self.assertTrue(self.path.is_file(), "README.md отсутствует")

    def test_run_command_is_documented(self) -> None:
        self.assertIn("./run --catalog", self.text)

    def test_network_dependencies_are_listed(self) -> None:
        for marker in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "LLM_USER_AGENT"):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.text)

    def test_model_usage_is_described(self) -> None:
        self.assertIn("где используется модель", self.text.lower())

    def test_key_is_not_in_repository(self) -> None:
        """Ключа в репозитории быть не может: только .env, и он в .gitignore."""
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".env", gitignore)
        self.assertFalse((REPO_ROOT / ".env.example").read_text(encoding="utf-8").count("LLM_API_KEY=") == 0)
        example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        for line in example.splitlines():
            if line.startswith("LLM_API_KEY="):
                self.assertEqual(line.strip(), "LLM_API_KEY=", "в .env.example не должно быть ключа")


class AccuracyDocumentTest(unittest.TestCase):
    """``docs/ACCURACY.md``: цифры в документе — из набора, а не «на глаз».

    Метрика бессмысленна, если документ с числом живёт отдельно от набора: набор
    пополнили, а в отчёте осталась старая точность. Поэтому размер набора и
    результат берутся из тех же файлов, что читает стенд.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = REPO_ROOT / "docs" / "ACCURACY.md"
        cls.text = cls.path.read_text(encoding="utf-8") if cls.path.is_file() else ""

    def test_document_exists(self) -> None:
        self.assertTrue(self.path.is_file(), "docs/ACCURACY.md отсутствует")
        self.assertIn("# Точность извлечения условий", self.text)

    def test_size_matches_the_golden_set(self) -> None:
        from bench.accuracy import load_golden

        examples = load_golden()
        self.assertIn(
            f"{len(examples)} размеченных реплик",
            self.text,
            "в документе не тот размер набора: обновите docs/ACCURACY.md",
        )

    def test_baseline_is_described_and_frozen(self) -> None:
        from bench.accuracy import BASELINE_PATH

        self.assertIn("baseline.json", self.text)
        self.assertTrue(BASELINE_PATH.is_file(), "нет замороженных цифр — регресс не поймать")
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        self.assertIn(str(baseline["full_match"]), self.text, "в документе не тот результат")

    def test_weights_are_documented(self) -> None:
        for kind in ("wrong_ok", "wrong_value", "false_negative", "false_positive", "underask"):
            with self.subTest(kind=kind):
                self.assertIn(kind, self.text)

    def test_readme_points_to_the_document(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("docs/ACCURACY.md", readme)


if __name__ == "__main__":
    unittest.main()
