"""Сквозной замер ./run по открытому контракту. По умолчанию работает офлайн."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from bench.acceptance import (
    Bound, CATALOG_PATH, Intent, NumericCondition, REPO_ROOT, check_document, expected_status,
)
from app.jsonio import loads_strict


def cases():
    bundle = ("table", "chair", "lamp")
    for directory, budget in [("01-selection", 15000), ("02-clarification", None)]:
        path = REPO_ROOT / "examples" / directory / "request.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        intent = Intent("", "taganrog", budget, bundle, {"table": NumericCondition(upper=Bound(100))})
        yield directory, payload, intent

    for name, text, city, budget, category in [
        ("price-limit", "Стол в Таганроге до 100 ₽", "taganrog", 100, "table"),
        ("lamp", "Настольная лампа в Таганроге до 1000 рублей", "taganrog", 1000, "lamp"),
        ("describe-chair", "Нужно сиденье со спинкой. Таганрог, до 10000.", "taganrog", 10000, "chair"),
        ("describe-table", "Нужна деревяшка с четырьмя ножками. Таганрог. До 15000.", "taganrog", 15000, "table"),
        ("describe-writing", "Нужен предмет, за которым пишут. Азов, до 8000.", "azov", 8000, "table"),
    ]:
        yield name, {"messages": [{"role": "user", "text": text}]}, Intent(text, city, budget, (category,))

    messages = [
        {"role": "user", "text": "Настольная лампа в Таганроге"},
        {"role": "assistant", "text": "Какой у вас бюджет?"},
        {"role": "user", "text": "ну давай до 1000р"},
    ]
    yield "budget-followup", {"messages": messages}, Intent("", "taganrog", 1000, ("lamp",))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="разрешить вызовы настроенной модели")
    parser.add_argument("--text-search", action="store_true")
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--case", action="append", choices=[name for name, _, _ in cases()])
    args = parser.parse_args()
    path = args.catalog.resolve()
    catalog = loads_strict(path.read_text(encoding="utf-8-sig"))
    command = ["./run", "--catalog", str(path)]
    if os.name == "nt":
        bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        command.insert(0, str(bash) if bash.exists() else shutil.which("bash") or "bash")
    if args.text_search:
        command.append("--text-search")

    passed = True
    with tempfile.TemporaryDirectory() as cache:
        env = dict(os.environ, PYTHON=Path(sys.executable).as_posix(), ASSISTANT_CACHE_DIR=cache,
                   ASSISTANT_OFFLINE="0" if args.live else "1")
        for name, payload, intent in cases():
            if args.case and name not in args.case:
                continue
            expected = expected_status(intent, catalog)
            record = {"case": name, "expected": expected, "status": None, "tokens": None}
            started = time.perf_counter()
            try:
                result = subprocess.run(command, cwd=REPO_ROOT, env=env, timeout=120,
                                        input=json.dumps(payload).encode(), capture_output=True)
                record["wall_ms"] = round((time.perf_counter() - started) * 1000)
                stderr = result.stderr.decode("utf-8", errors="replace")
                usage = re.findall(r"модель: prompt=(\d+) completion=(\d+) total=(\d+)", stderr)
                record["usage_records"] = len(usage)
                if usage or not args.live:
                    record["tokens"] = sum(int(total) for _, _, total in usage)
                if result.returncode:
                    record["errors"] = [f"exit {result.returncode}", *stderr.strip().splitlines()[-1:]]
                else:
                    document = loads_strict(result.stdout.decode("utf-8"))
                    record["status"] = document.get("status") if isinstance(document, dict) else None
                    record["errors"] = check_document(document, intent, catalog, expected)
                allowed_calls = 1 if args.live and name.startswith("describe-") else 0
                if len(usage) > allowed_calls:
                    record["errors"].append(f"ответов шлюза с usage: {len(usage)}, ожидалось ≤ {allowed_calls}")
            except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                record["wall_ms"] = round((time.perf_counter() - started) * 1000)
                record["errors"] = [str(exc)]
            passed &= not record["errors"]
            print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
