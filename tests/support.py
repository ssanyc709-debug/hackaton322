"""Общая обвязка тестов: запуск CLI как отдельного процесса, пути репозитория."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "data" / "catalog.json"
SCHEMAS_DIR = REPO_ROOT / "schemas"
EXAMPLES_DIR = REPO_ROOT / "examples"


@dataclass(frozen=True)
class CliResult:
    """Результат прогона ``python -m app.main``."""

    code: int
    stdout: str
    stderr: str

    def json(self) -> object:
        """Разбирает stdout ровно как JSON (контракт: один объект)."""
        return json.loads(self.stdout)


def run_cli(
    stdin: bytes | str | object,
    args: list[str] | None = None,
    timeout: float = 150.0,
) -> CliResult:
    """Запускает CLI так же, как автопроверка: отдельный процесс, stdin → stdout.

    ``stdin`` может быть готовым JSON-текстом, байтами или Python-объектом
    (тогда он сериализуется) — так удобно подавать битые и «почти валидные» входы.
    """
    command = [sys.executable, "-m", "app.main"]
    command.extend(args if args is not None else ["--catalog", str(CATALOG_PATH)])
    env = dict(os.environ)
    env["ASSISTANT_OFFLINE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(REPO_ROOT)
    if isinstance(stdin, bytes):
        data = stdin
    elif isinstance(stdin, str):
        data = stdin.encode("utf-8")
    else:
        data = json.dumps(stdin, ensure_ascii=False).encode("utf-8")
    completed = subprocess.run(
        command,
        input=data,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        timeout=timeout,
    )
    return CliResult(
        code=completed.returncode,
        stdout=completed.stdout.decode("utf-8"),
        stderr=completed.stderr.decode("utf-8"),
    )


def read_example_request(name: str) -> str:
    """Текст запроса из ``examples/<name>/request.json``."""
    return (EXAMPLES_DIR / name / "request.json").read_text(encoding="utf-8")


def read_json(path: Path) -> object:
    """Читает JSON-файл репозитория в UTF-8."""
    return json.loads(path.read_text(encoding="utf-8"))
