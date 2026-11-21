"""Общие помощники QA-набора: запуск процесса, каталог, проверка ответа.

Файл попадает под ``unittest discover`` (имя начинается с ``test_``), поэтому на
уровне модуля НЕТ побочных эффектов: только константы и определения функций.
Ни один файл не читается и ни один процесс не запускается при импорте.

Правила, зафиксированные здесь и не дублируемые в каждом тесте:

* запуск — отдельным процессом из корня репозитория, ровно как боевой путь
  ``python -m app.main --catalog …``; ``input=`` всегда ``bytes``, вывод
  декодируется явно как UTF-8 (консольная кодировка Windows не участвует);
* ``ASSISTANT_OFFLINE=1`` — модель не вызывается, сеть тестам не нужна;
* ``ASSISTANT_CACHE_DIR`` — свой каталог во временной папке, поэтому тесты не
  зависят от ``data/cache`` и от порядка запуска;
* коды выхода: успешная обработка любого статуса — ровно 0; битый вход или
  технический сбой — ЛЮБОЙ ненулевой код (контракт §1 не фиксирует конкретное
  число, поэтому ``assertEqual(code, 2)`` в наборе не используется);
* файлы проекта (``data/``, ``schemas/``, ``_upstream/``) только читаются.

Источник истины: ``_upstream/CONTRACT.md`` (v1.1) и ``_upstream/schemas/*.json``.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "data" / "catalog.json"
CITIES_PATH = REPO_ROOT / "data" / "cities.json"
CATEGORIES_PATH = REPO_ROOT / "data" / "categories.json"
RUN_SCRIPT = REPO_ROOT / "run"
UPSTREAM_DIR = REPO_ROOT / "_upstream"
UPSTREAM_SCHEMAS = UPSTREAM_DIR / "schemas"
UPSTREAM_EXAMPLES = UPSTREAM_DIR / "examples"
SCHEMAS_DIR = REPO_ROOT / "schemas"

VALID_STATUSES = frozenset({"ok", "clarify", "no_match"})
QUESTION_FIELDS = frozenset({"budget_rub", "city_id", "categories"})
SUPPORTED_CATEGORIES = ("table", "chair", "lamp")

#: Успешная обработка любого статуса — единственный код, который контракт
#: фиксирует числом. Для ошибок проверяется только «ненулевой + пустой stdout».
EXIT_OK = 0

#: Порог на один процесс. Контракт даёт 120 с, офлайн-прогон занимает ~0,2 с,
#: поэтому 60 с — уже аварийный сигнал, а не рабочий лимит.
DEFAULT_TIMEOUT = 60.0


@dataclass(frozen=True)
class QaResult:
    """Результат одного запуска решения."""

    code: int
    stdout: bytes
    stderr: str

    @property
    def stdout_text(self) -> str:
        """stdout как текст UTF-8 (декодирование только явное)."""
        return self.stdout.decode("utf-8")

    def json(self) -> dict[str, Any]:
        """Разбирает stdout как JSON-объект ответа."""
        document = json.loads(self.stdout_text)
        if not isinstance(document, dict):
            raise AssertionError(f"stdout должен быть JSON-объектом, получено {type(document).__name__}")
        return document

    def one_json_object(self) -> dict[str, Any]:
        """Проверяет «один объект и один завершающий перевод строки» (§1)."""
        if not self.stdout.endswith(b"\n"):
            raise AssertionError(f"ответ должен завершаться переводом строки, хвост: {self.stdout[-20:]!r}")
        newlines = self.stdout.count(b"\n")
        if newlines != 1:
            raise AssertionError(f"в stdout должна быть ровно одна строка, найдено {newlines}")
        return self.json()


def qa_env(cache_dir: Path | str | None = None) -> dict[str, str]:
    """Окружение прогона: офлайн-модель, свой кэш, UTF-8 без сюрпризов консоли."""
    env = dict(os.environ)
    env["ASSISTANT_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("LLM_API_KEY", None)
    if cache_dir is not None:
        env["ASSISTANT_CACHE_DIR"] = str(cache_dir)
    return env


def run_cli(
    stdin: bytes | str | object,
    *,
    catalog: Path | str | None = None,
    cache_dir: Path | str | None = None,
    extra_env: Mapping[str, str] | None = None,
    extra_args: Sequence[str] = (),
    timeout: float = DEFAULT_TIMEOUT,
    cwd: Path | str | None = None,
) -> QaResult:
    """Запускает ``python -m app.main --catalog …`` — боевой путь проверяющего.

    ``stdin`` может быть байтами (для битых входов и не-UTF-8), текстом или
    Python-объектом (тогда он сериализуется в UTF-8).
    """
    command = [sys.executable, "-m", "app.main", "--catalog", str(catalog or CATALOG_PATH)]
    command.extend(extra_args)
    env = qa_env(cache_dir)
    if extra_env:
        env.update({key: str(value) for key, value in extra_env.items()})
    completed = subprocess.run(
        command,
        input=to_bytes(stdin),
        cwd=str(cwd or REPO_ROOT),
        env=env,
        capture_output=True,
        timeout=timeout,
    )
    return QaResult(
        code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr.decode("utf-8", errors="replace"),
    )


def to_bytes(stdin: bytes | str | object) -> bytes:
    """Приводит вход к байтам: контракт читает stdin как UTF-8."""
    if isinstance(stdin, bytes):
        return stdin
    if isinstance(stdin, str):
        return stdin.encode("utf-8")
    return json.dumps(stdin, ensure_ascii=False).encode("utf-8")


# --- Общий кэш ответов на весь прогон -----------------------------------

_SHARED_RESPONSES: dict[tuple[str, bytes], QaResult] = {}
_SHARED_HOLDER: tempfile.TemporaryDirectory[str] | None = None
_SHARED_LOCK = threading.Lock()


def run_shared(
    stdin: bytes | str | object,
    *,
    catalog: Path | str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> QaResult:
    """Запускает процесс ровно один раз на одинаковый (каталог, вход).

    Зачем: разные блоки проверяют разные свойства одного и того же ответа.
    Без общего кэша один и тот же запрос запускался бы десятки раз, и время
    прогона росло бы без пользы для проверки.

    Условия безопасности: решение детерминировано (это отдельно проверяет блок
    про детерминизм), режим офлайн, у прогона собственный ``ASSISTANT_CACHE_DIR``.
    Тесты, которым нужен именно свежий процесс (повторяемость, кэш каталога,
    параллельный запуск), вызывают :func:`run_cli` напрямую и обходят кэш.
    """
    global _SHARED_HOLDER
    key = (str(catalog or CATALOG_PATH), to_bytes(stdin))
    with _SHARED_LOCK:
        cached = _SHARED_RESPONSES.get(key)
        if cached is not None:
            return cached
        if _SHARED_HOLDER is None:
            _SHARED_HOLDER = tempfile.TemporaryDirectory(prefix="qa-shared-cache-")
            atexit.register(_SHARED_HOLDER.cleanup)
        cache_dir = Path(_SHARED_HOLDER.name)
    result = run_cli(stdin, catalog=catalog, cache_dir=cache_dir, timeout=timeout)
    with _SHARED_LOCK:
        _SHARED_RESPONSES[key] = result
    return result


# --- Запросы -------------------------------------------------------------


def user(text: str) -> dict[str, str]:
    """Реплика покупателя."""
    return {"role": "user", "text": text}


def assistant(text: str) -> dict[str, str]:
    """Реплика помощника: по контракту это обычный текст в истории."""
    return {"role": "assistant", "text": text}


def request(*messages: Mapping[str, str]) -> dict[str, list[dict[str, str]]]:
    """Запрос из готовых реплик."""
    return {"messages": [dict(message) for message in messages]}


def ask(text: str) -> dict[str, list[dict[str, str]]]:
    """Запрос с единственной репликой покупателя."""
    return request(user(text))


# --- Каталог и справочники ----------------------------------------------


def load_catalog(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Читает каталог (только чтение)."""
    raw = json.loads(Path(path or CATALOG_PATH).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise AssertionError("каталог должен быть JSON-массивом")
    return raw


def index_by_id(document: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Объявления по ID — чтобы проверять ответ по данным каталога, а не по коду."""
    return {str(raw["id"]): dict(raw) for raw in document}


def load_cities() -> dict[str, str]:
    """Справочник городов: ``id`` → название."""
    raw = json.loads(CITIES_PATH.read_text(encoding="utf-8"))
    return {str(item["id"]): str(item["name"]) for item in raw}


def city_ids() -> tuple[str, ...]:
    """Идентификаторы городов из справочника (общий для учебных и проверочных данных)."""
    return tuple(load_cities())


def write_catalog(directory: Path | str, document: Sequence[Mapping[str, Any]], name: str = "catalog.json") -> Path:
    """Кладёт каталог во временный каталог и возвращает путь."""
    path = Path(directory) / name
    path.write_text(json.dumps(list(document), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def make_temp_dir(case: unittest.TestCase) -> Path:
    """Временный каталог, который удалится после теста (вне ``data/``)."""
    holder = tempfile.TemporaryDirectory(prefix="qa-work-")
    case.addCleanup(holder.cleanup)
    return Path(holder.name)


def cache_dir(case: unittest.TestCase) -> Path:
    """Свой ``ASSISTANT_CACHE_DIR`` на тест: кэш не влияет на соседние тесты."""
    return make_temp_dir(case) / "cache"


# --- Проверка ответа по каталогу ----------------------------------------


def option_ids(option: Mapping[str, Any]) -> list[str]:
    """ID варианта как список строк."""
    ids = option.get("listing_ids")
    if not isinstance(ids, list):
        raise AssertionError(f"listing_ids должен быть массивом, получено {type(ids).__name__}")
    return [str(item) for item in ids]


def categories_of(ids: Iterable[str], by_id: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Категории выбранных объявлений в порядке ID."""
    result: list[str] = []
    for listing_id in ids:
        if listing_id not in by_id:
            raise AssertionError(f"ID {listing_id!r} отсутствует в каталоге")
        result.append(str(by_id[listing_id]["category"]))
    return result


def sum_price(ids: Iterable[str], by_id: Mapping[str, Mapping[str, Any]]) -> int:
    """Сумма ``price_rub`` по каталогу для набора ID."""
    total = 0
    for listing_id in ids:
        if listing_id not in by_id:
            raise AssertionError(f"ID {listing_id!r} отсутствует в каталоге")
        total += int(by_id[listing_id]["price_rub"])
    return total


def assert_option_matches_catalog(
    case: unittest.TestCase,
    option: Mapping[str, Any],
    *,
    by_id: Mapping[str, Mapping[str, Any]],
    categories: Sequence[str],
    city_id: str,
    budget_rub: int,
) -> None:
    """Сверяет вариант ``ok`` с каталогом и условиями запроса (CONTRACT §4).

    Жёстко проверяется только то, что контракт требует: состав категорий,
    город, цена по каталогу, бюджет, уникальность ID. Порядок ID внутри
    варианта и текст ``explanation`` свободные и не сравниваются.
    """
    ids = option_ids(option)
    case.assertTrue(ids, "вариант не может быть пустым")
    case.assertEqual(len(ids), len(set(ids)), f"ID внутри варианта повторяются: {ids}")

    case.assertEqual(
        sorted(categories_of(ids, by_id)),
        sorted(categories),
        f"состав категорий не совпадает с запрошенным: {ids}",
    )

    for listing_id in ids:
        case.assertEqual(
            by_id[listing_id]["city_id"],
            city_id,
            f"объявление {listing_id} не в городе покупателя ({city_id})",
        )

    case.assertEqual(
        option.get("total_rub"),
        sum_price(ids, by_id),
        "total_rub должен точно равняться сумме price_rub по каталогу",
    )
    case.assertLessEqual(
        int(option["total_rub"]),
        budget_rub,
        "итоговая цена не может превышать бюджет покупателя",
    )


def assert_documents_differ_by_composition(case: unittest.TestCase, options: Sequence[Mapping[str, Any]]) -> None:
    """Разные варианты обязаны отличаться составом ID (CONTRACT §4)."""
    compositions = [frozenset(option_ids(option)) for option in options]
    case.assertEqual(len(compositions), len(set(compositions)), "варианты обязаны отличаться составом ID")


def assert_clarify_covers(case: unittest.TestCase, document: Mapping[str, Any], expected_fields: Iterable[str]) -> None:
    """``clarify`` спрашивает про все недостающие условия, каждое по одному разу."""
    case.assertEqual(document.get("status"), "clarify")
    questions = document.get("questions")
    case.assertIsInstance(questions, list, "questions должен быть массивом")
    case.assertTrue(questions, "questions не может быть пустым")
    fields = [question.get("field") for question in questions]
    case.assertEqual(len(fields), len(set(fields)), f"поле спрошено повторно: {fields}")
    for name in fields:
        case.assertIn(name, QUESTION_FIELDS, f"недопустимое поле вопроса: {name!r}")
    case.assertEqual(set(fields), set(expected_fields), "спрошены не те поля, которые отсутствуют")


# --- Схемы из _upstream (источник истины) --------------------------------


def read_schema(name: str) -> dict[str, Any]:
    """Читает схему из ``_upstream/schemas`` по имени файла без расширения."""
    return json.loads((UPSTREAM_SCHEMAS / f"{name}.schema.json").read_text(encoding="utf-8"))


def response_schema() -> dict[str, Any]:
    """Схема ответа (``_upstream/schemas/response.schema.json``)."""
    return read_schema("response")


def request_schema() -> dict[str, Any]:
    """Схема запроса (``_upstream/schemas/request.schema.json``)."""
    return read_schema("request")


def catalog_schema() -> dict[str, Any]:
    """Схема каталога (``_upstream/schemas/catalog.schema.json``)."""
    return read_schema("catalog")


#: Ключи JSON Schema, которые этот набор реально встречает в ``_upstream/schemas``.
#: Всё, что не входит в набор, считается аннотацией (``title``, ``description``…).
_SUPPORTED_KEYWORDS = frozenset(
    {
        "type",
        "const",
        "enum",
        "required",
        "properties",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "oneOf",
        "anyOf",
        "not",
        "pattern",
    }
)

#: Служебные ключи схемы: на валидацию не влияют.
_ANNOTATION_KEYWORDS = frozenset({"$schema", "$id", "$comment", "title", "description", "default", "examples"})


def schema_errors(document: Any, schema: Mapping[str, Any] | None = None) -> list[str]:
    """Нарушения схемы как список строк (пустой список — документ валиден).

    Проверка написана на stdlib: ``jsonschema`` в требованиях проекта есть, но в
    проверяющем окружении его может не быть, а тест не должен зависеть от
    установленных пакетов. Поддерживаются ровно те ключи, которые встречаются в
    ``_upstream/schemas/*.json``; если схема начнёт использовать новый ключ,
    проверка вернёт ошибку, а не промолчит.
    """
    target = schema if schema is not None else response_schema()
    errors: list[str] = []
    _collect_errors(document, target, "", errors)
    return errors


def _matches_type(value: Any, name: str) -> bool:
    """Проверка типа по правилам JSON, а не Python.

    В Python ``isinstance(True, int)`` истинно, а JSON различает ``true`` и
    ``1``, поэтому ``boolean`` не проходит как ``integer``/``number``.
    """
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    if name == "integer":
        return not isinstance(value, bool) and isinstance(value, int)
    if name == "number":
        return not isinstance(value, bool) and isinstance(value, (int, float))
    raise AssertionError(f"валидатор не знает тип {name!r} из схемы")


def _collect_errors(value: Any, schema: Mapping[str, Any], path: str, errors: list[str]) -> None:
    """Рекурсивно собирает нарушения одного значения по одной подсхеме."""
    if not isinstance(schema, dict):  # bool-схемы (true/false) в _upstream не встречаются
        raise AssertionError(f"валидатор ожидает объект-схему, получено {schema!r}")
    where = path or "<root>"

    unknown = [key for key in schema if key not in _SUPPORTED_KEYWORDS and key not in _ANNOTATION_KEYWORDS]
    if unknown:
        errors.append(f"{where}: валидатор не поддерживает ключи схемы {unknown}")
        return

    if "type" in schema:
        expected = schema["type"]
        names = [expected] if isinstance(expected, str) else list(expected)
        if not any(_matches_type(value, name) for name in names):
            errors.append(f"{where}: ожидался тип {'|'.join(names)}, получено {type(value).__name__}")
            return  # при неверном типе остальные проверки бессмысленны

    if "const" in schema and value != schema["const"]:
        errors.append(f"{where}: ожидалось значение {schema['const']!r}, получено {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{where}: значение {value!r} не из {schema['enum']}")

    if "oneOf" in schema:
        matched = [index for index, sub in enumerate(schema["oneOf"]) if not _sub_errors(value, sub)]
        if len(matched) != 1:
            errors.append(f"{where}: подходит под {len(matched)} ветвей oneOf из {len(schema['oneOf'])}, нужна ровно одна")
    if "anyOf" in schema and not any(not _sub_errors(value, sub) for sub in schema["anyOf"]):
        errors.append(f"{where}: не подходит ни под одну ветвь anyOf")
    if "not" in schema and not _sub_errors(value, schema["not"]):
        errors.append(f"{where}: значение попадает под запрещающую схему not")

    if isinstance(value, dict):
        _collect_object_errors(value, schema, path, errors)
    elif isinstance(value, list):
        _collect_array_errors(value, schema, path, errors)
    elif isinstance(value, str):
        _collect_string_errors(value, schema, path, errors)
    elif _matches_type(value, "number"):
        _collect_number_errors(value, schema, path, errors)


def _sub_errors(value: Any, schema: Mapping[str, Any]) -> list[str]:
    """Ошибки значения по подсхеме (для oneOf/anyOf/not)."""
    local: list[str] = []
    _collect_errors(value, schema, "", local)
    return local


def _collect_object_errors(value: Mapping[str, Any], schema: Mapping[str, Any], path: str, errors: list[str]) -> None:
    where = path or "<root>"
    properties = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in value:
            errors.append(f"{where}: отсутствует обязательное поле {name!r}")

    additional = schema.get("additionalProperties", True)
    for name, item in value.items():
        child = f"{path}/{name}"
        if name in properties:
            _collect_errors(item, properties[name], child, errors)
        elif additional is False:
            errors.append(f"{child}: дополнительное поле запрещено схемой")
        elif isinstance(additional, dict):
            _collect_errors(item, additional, child, errors)


def _collect_array_errors(value: list[Any], schema: Mapping[str, Any], path: str, errors: list[str]) -> None:
    where = path or "<root>"
    if "minItems" in schema and len(value) < schema["minItems"]:
        errors.append(f"{where}: элементов {len(value)}, минимум {schema['minItems']}")
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        errors.append(f"{where}: элементов {len(value)}, максимум {schema['maxItems']}")
    if schema.get("uniqueItems") is True:
        keys = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in value]
        if len(keys) != len(set(keys)):
            errors.append(f"{where}: элементы повторяются, но uniqueItems=true")
    if "items" in schema:
        for index, item in enumerate(value):
            _collect_errors(item, schema["items"], f"{path}/{index}", errors)


def _collect_string_errors(value: str, schema: Mapping[str, Any], path: str, errors: list[str]) -> None:
    where = path or "<root>"
    if "minLength" in schema and len(value) < schema["minLength"]:
        errors.append(f"{where}: длина {len(value)}, минимум {schema['minLength']}")
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        errors.append(f"{where}: длина {len(value)}, максимум {schema['maxLength']}")
    if "pattern" in schema and re.search(schema["pattern"], value) is None:
        errors.append(f"{where}: строка не соответствует шаблону {schema['pattern']!r}")


def _collect_number_errors(value: Any, schema: Mapping[str, Any], path: str, errors: list[str]) -> None:
    where = path or "<root>"
    if "minimum" in schema and value < schema["minimum"]:
        errors.append(f"{where}: значение {value} меньше минимума {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        errors.append(f"{where}: значение {value} больше максимума {schema['maximum']}")
