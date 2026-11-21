"""Метрика точности извлечения условий и метрика безопасности ответов.

Зачем этот файл
---------------
Юнит-тесты доказывают, что правила работают на **известных** примерах. Они не
отвечают на вопрос жюри «какая у вас точность». Этот стенд отвечает: есть
эталонная разметка реплик (``bench/golden/extraction.jsonl``), разметка
сравнивается с тем, что достал лексикон, и по каждому полю считаются точность,
лишние и потерянные срабатывания.

Метрика считает не только «угадали ли значение», но и **цену ошибки**. Ментор
сформулировал политику продукта: «лучше переспросить, чем понять неверно».
Поэтому ошибки взвешены:

=========== ===================================================== ======
Вес         Ошибка                                              Что значит
=========== ===================================================== ======
5           ``wrong_ok``                                        дали ``ok``, хотя условие не понято
3           ``wrong_value``                                     поняли условие неверно (город, число)
3           ``false_negative``                                  потеряли названное условие
2           ``wrong_no_match``                                  отсекли всё при существующих вариантах
2           ``underask``                                        не спросили то, без чего нельзя подбирать
1           ``false_positive``                                  выдумали условие, которого не было
1           ``false_clarify``                                   переспросили, хотя всё было известно
1           ``overask``                                         спросили лишнее при ``clarify``
=========== ===================================================== ======

``safety = 1 - сумма(весов) / (5 · N)`` — доля «штрафа» от максимума, где N —
число реплик. Идеал — 1.0, любая опасная ошибка видна сразу.

Запуск::

    python bench/accuracy.py                 # отчёт по эталонному набору
    python bench/accuracy.py --errors 40     # сколько ошибок печатать
    python bench/accuracy.py --block width   # только один блок реплик
    python bench/accuracy.py --json          # машинный отчёт
    python bench/accuracy.py --update-baseline   # заморозить текущие цифры

Модель не вызывается: ``bench`` проверяет лексикон, то есть 0 токенов.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GOLDEN_PATH = REPO_ROOT / "bench" / "golden" / "extraction.jsonl"
BASELINE_PATH = REPO_ROOT / "bench" / "golden" / "baseline.json"
CATALOG_PATH = REPO_ROOT / "data" / "catalog.json"

#: Категории и города — ровно как в §2 контракта.
CATEGORIES: tuple[str, ...] = ("table", "chair", "lamp")
CITIES: tuple[str, ...] = ("taganrog", "rostov-na-donu", "azov")
STATUSES: tuple[str, ...] = ("ok", "clarify", "no_match")
FIELD_NAMES: tuple[str, ...] = (
    "budget_rub", "city_id", "categories",
)

#: Вес ошибки: чем дороже покупателю и жюри, тем больше.
WEIGHT_WRONG_OK = 5
WEIGHT_FALSE_NEGATIVE = 3
WEIGHT_WRONG_VALUE = 3
WEIGHT_WRONG_NO_MATCH = 2
WEIGHT_UNDERASK = 2
WEIGHT_FALSE_POSITIVE = 1
WEIGHT_FALSE_CLARIFY = 1
WEIGHT_OVERASK = 1

#: Максимальный вес ошибки на одну реплику — нормировка ``safety``.
MAX_WEIGHT = WEIGHT_WRONG_OK

#: Поля, которые сравниваются в эталонном наборе. ``width`` и ``temperature`` —
#: числовые условия, ``adjustable`` — трёхзначный флаг (да/нет/неизвестно).
SLOT_FIELDS: tuple[str, ...] = (
    "city", "budget", "categories", "width", "temperature", "adjustable",
)

#: Обязательные ключи строки набора: «забыть поле» невозможно — набор не загрузится.
REQUIRED_KEYS: tuple[str, ...] = ("id", "block", "text") + SLOT_FIELDS

#: Белый список блоков: опечатка в имени блока ломает отчёт по блокам.
BLOCKS: tuple[str, ...] = (
    "width", "budget", "city", "categories", "temperature", "adjustable",
    "mixed", "hard", "noop",
)


class GoldenError(ValueError):
    """Эталонный набор размечен неверно — это ошибка стенда, а не продукта."""


@dataclass(frozen=True, slots=True)
class Example:
    """Одна размеченная реплика."""

    id: str
    block: str
    text: str
    city: str | None
    budget: int | None
    categories: tuple[str, ...]
    width: Mapping[str, Any] | None
    temperature: Mapping[str, Any] | None
    adjustable: bool | None
    status: str | None = None
    missing: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class FieldError:
    """Расхождение по одному полю одной реплики."""

    example_id: str
    block: str
    text: str
    slot: str
    kind: str  # false_positive | false_negative | wrong_value
    expected: Any
    actual: Any

    @property
    def weight(self) -> int:
        return {
            "wrong_value": WEIGHT_WRONG_VALUE,
            "false_negative": WEIGHT_FALSE_NEGATIVE,
            "false_positive": WEIGHT_FALSE_POSITIVE,
        }[self.kind]


@dataclass(frozen=True, slots=True)
class StatusError:
    """Расхождение по статусу ответа или составу вопросов."""

    example_id: str
    text: str
    kind: str  # wrong_ok | wrong_no_match | false_clarify | underask | overask
    expected: Any
    actual: Any

    @property
    def weight(self) -> int:
        return {
            "wrong_ok": WEIGHT_WRONG_OK,
            "wrong_no_match": WEIGHT_WRONG_NO_MATCH,
            "false_clarify": WEIGHT_FALSE_CLARIFY,
            "underask": WEIGHT_UNDERASK,
            "overask": WEIGHT_OVERASK,
        }[self.kind]


@dataclass
class FieldStats:
    """Счётчики по одному полю."""

    total: int = 0
    correct: int = 0
    false_positive: int = 0
    false_negative: int = 0
    wrong_value: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "wrong_value": self.wrong_value,
        }


@dataclass
class BlockStats:
    """Счётчики по одному блоку реплик."""

    total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
        }


@dataclass
class Result:
    """Полный результат прогона: поля, блоки, статусы, ошибки."""

    total: int = 0
    fields: dict[str, FieldStats] = field(default_factory=dict)
    blocks: dict[str, BlockStats] = field(default_factory=dict)
    statuses: dict[str, int] = field(default_factory=dict)
    field_errors: list[FieldError] = field(default_factory=list)
    status_errors: list[StatusError] = field(default_factory=list)
    full_match: int = 0
    checked_statuses: int = 0

    @property
    def full_match_rate(self) -> float:
        return self.full_match / self.total if self.total else 1.0

    @property
    def penalty(self) -> int:
        return sum(item.weight for item in self.field_errors) + sum(
            item.weight for item in self.status_errors
        )

    @property
    def safety(self) -> float:
        """1 − штраф за ошибки; 1.0 — ни одной ошибки ни в одной реплике."""
        if not self.total:
            return 1.0
        return max(0.0, 1.0 - self.penalty / (MAX_WEIGHT * self.total))

    @property
    def dangerous(self) -> list[Any]:
        """Ошибки, из-за которых покупатель получил неверный ответ."""
        return [
            item
            for item in [*self.field_errors, *self.status_errors]
            if item.weight >= WEIGHT_WRONG_VALUE
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "full_match": self.full_match,
            "full_match_rate": round(self.full_match_rate, 4),
            "safety": round(self.safety, 4),
            "penalty": self.penalty,
            "dangerous": len(self.dangerous),
            "fields": {name: stats.as_dict() for name, stats in sorted(self.fields.items())},
            "blocks": {name: stats.as_dict() for name, stats in sorted(self.blocks.items())},
            "statuses": dict(sorted(self.statuses.items())),
            "checked_statuses": self.checked_statuses,
            "errors": [render_error(item) for item in [*self.field_errors, *self.status_errors]],
        }


def render_error(item: FieldError | StatusError) -> dict[str, Any]:
    """Ошибка в виде, удобном для JSON и для печати."""
    if isinstance(item, FieldError):
        return {
            "id": item.example_id,
            "block": item.block,
            "slot": item.slot,
            "kind": item.kind,
            "text": item.text,
            "expected": item.expected,
            "actual": item.actual,
            "weight": item.weight,
        }
    return {
        "id": item.example_id,
        "block": "status",
        "slot": "status",
        "kind": item.kind,
        "text": item.text,
        "expected": item.expected,
        "actual": item.actual,
        "weight": item.weight,
    }


def load_golden(path: Path | None = None) -> list[Example]:
    """Читает и валидирует эталонный набор.

    Валидация намеренно строгая: если поле забыто или значение вне домена, это
    ошибка стенда (``GoldenError``), а не «ноль в метрике». Иначе опечатка в
    разметке тихо занижает точность и её никто не заметит.
    """
    source = path or GOLDEN_PATH
    if not source.exists():
        raise GoldenError(f"нет эталонного набора: {source}")
    examples: list[Example] = []
    seen: set[str] = set()
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise GoldenError(f"строка {number}: не JSON ({error})") from error
        where = f"строка {number} ({raw.get('id', '?')})"
        missing = [key for key in REQUIRED_KEYS if key not in raw]
        if missing:
            raise GoldenError(f"{where}: нет полей {missing}")
        example = _validate_example(raw, where)
        if example.id in seen:
            raise GoldenError(f"{where}: повтор id {example.id!r}")
        seen.add(example.id)
        examples.append(example)
    if not examples:
        raise GoldenError("эталонный набор пуст")
    return examples


def _validate_example(raw: Mapping[str, Any], where: str) -> Example:
    block = raw["block"]
    if block not in BLOCKS:
        raise GoldenError(f"{where}: неизвестный блок {block!r}")
    city = raw["city"]
    if city is not None and city not in CITIES:
        raise GoldenError(f"{where}: город {city!r} вне §2")
    budget = raw["budget"]
    if budget is not None and (not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0):
        raise GoldenError(f"{where}: бюджет {budget!r} должен быть положительным целым")
    categories = raw["categories"]
    if not isinstance(categories, list) or any(item not in CATEGORIES for item in categories):
        raise GoldenError(f"{where}: категории {categories!r} вне §2")
    if len(set(categories)) != len(categories):
        raise GoldenError(f"{where}: повтор категории")
    for name in ("width", "temperature"):
        _validate_range(raw[name], where, name)
    adjustable = raw["adjustable"]
    if adjustable is not None and not isinstance(adjustable, bool):
        raise GoldenError(f"{where}: adjustable должен быть true/false/null")
    status = raw.get("status")
    if status is not None and status not in STATUSES:
        raise GoldenError(f"{where}: статус {status!r} вне §4")
    missing_fields = raw.get("missing")
    if missing_fields is not None:
        if not isinstance(missing_fields, list) or any(
            item not in FIELD_NAMES for item in missing_fields
        ):
            raise GoldenError(f"{where}: missing {missing_fields!r} вне §2.4")
        if status != "clarify":
            raise GoldenError(f"{where}: missing имеет смысл только при status=clarify")
    return Example(
        id=raw["id"],
        block=block,
        text=raw["text"],
        city=city,
        budget=budget,
        categories=tuple(categories),
        width=raw["width"],
        temperature=raw["temperature"],
        adjustable=adjustable,
        status=status,
        missing=None if missing_fields is None else tuple(missing_fields),
    )


def _validate_range(value: Any, where: str, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise GoldenError(f"{where}: {name} должен быть объектом или null")
    keys = {"lo", "hi", "lo_inc", "hi_inc"}
    if set(value) != keys:
        raise GoldenError(f"{where}: у {name} ожидались ключи {sorted(keys)}")
    if value["lo"] is None and value["hi"] is None:
        raise GoldenError(f"{where}: у {name} нет ни одной границы — это null, а не диапазон")
    for key in ("lo", "hi"):
        bound = value[key]
        if bound is not None and (not isinstance(bound, int) or isinstance(bound, bool)):
            raise GoldenError(f"{where}: {name}.{key} должен быть целым или null")
    for key in ("lo_inc", "hi_inc"):
        if not isinstance(value[key], bool):
            raise GoldenError(f"{where}: {name}.{key} должен быть bool")


def range_to_dict(value: Any) -> dict[str, Any] | None:
    """``NumericRange`` → сравнимый словарь (``None`` — условия нет)."""
    if value is None:
        return None
    lower = getattr(value, "lower", None)
    upper = getattr(value, "upper", None)
    if lower is None and upper is None:
        return None
    return {
        "lo": None if lower is None else lower.value,
        "hi": None if upper is None else upper.value,
        "lo_inc": bool(lower is not None and lower.inclusive),
        "hi_inc": bool(upper is not None and upper.inclusive),
    }


def compare_field(slot: str, expected: Any, actual: Any) -> str | None:
    """Вид расхождения по полю или ``None``, если поле извлечено верно."""
    if slot == "categories":
        expected_set, actual_set = set(expected), set(actual)
        if expected_set == actual_set:
            return None
        if actual_set - expected_set:
            return "false_positive"
        return "false_negative"
    if expected is None and actual is None:
        return None
    if expected is None:
        return "false_positive"
    if actual is None:
        return "false_negative"
    return None if expected == actual else "wrong_value"


def actual_slots(text: str) -> dict[str, Any]:
    """Слоты реплики по лексикону, в виде, пригодном для сравнения.

    Категории сравниваются **эффективные**: «стол без лампы» лексикон видит как
    «названы стол и лампа, лампа снята» (это нужно слиянию реплик, «добавь ещё»),
    а покупателю в итоге достаётся только стол. Поэтому из набора вычитаются
    ``categories_removed`` — иначе метрика наказывала бы код за верное поведение.
    """
    from app.nlp.extract import extract_message

    slots = extract_message(text)
    effective = tuple(
        name for name in slots.categories if name not in set(slots.categories_removed)
    )
    return {
        "city": slots.city_id,
        "budget": slots.budget_rub,
        "categories": effective,
        "width": range_to_dict(slots.width),
        "temperature": range_to_dict(slots.color_temperature),
        "adjustable": slots.adjustable_height,
    }


def expected_slots(example: Example) -> dict[str, Any]:
    """Ожидаемые слоты из разметки (те же имена, что у :func:`actual_slots`)."""
    return {
        "city": example.city,
        "budget": example.budget,
        "categories": example.categories,
        "width": example.width,
        "temperature": example.temperature,
        "adjustable": example.adjustable,
    }


def _status_errors(example: Example, document: Mapping[str, Any]) -> list[StatusError]:
    """Сравнивает фактический ответ со статусом и составом вопросов."""
    if example.status is None:
        return []
    actual = str(document.get("status"))
    if actual != example.status:
        kind = "wrong_ok" if actual == "ok" else "false_clarify" if actual == "clarify" else "wrong_no_match"
        if example.status == "clarify" and actual == "no_match":
            kind = "wrong_no_match"
        return [StatusError(example.id, example.text, kind, example.status, actual)]
    if example.missing is None:
        return []
    asked = [question["field"] for question in document.get("questions", [])]
    errors: list[StatusError] = []
    missed = sorted(set(example.missing) - set(asked))
    extra = sorted(set(asked) - set(example.missing))
    if missed:
        errors.append(StatusError(example.id, example.text, "underask", list(example.missing), asked))
    if extra:
        errors.append(StatusError(example.id, example.text, "overask", list(example.missing), asked))
    return errors


def evaluate(
    examples: Sequence[Example],
    *,
    check_statuses: bool = True,
) -> Result:
    """Считает метрику по набору. Модель не вызывается — только лексикон."""
    result = Result(total=len(examples))
    for name in SLOT_FIELDS:
        result.fields[name] = FieldStats()

    index = settings = None
    if check_statuses and any(example.status is not None for example in examples):
        from app.catalog.index import build_index
        from app.config import Settings

        document = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        index = build_index(document)
        settings = Settings.load(environ={"ASSISTANT_OFFLINE": "1"})

    for example in examples:
        expected = expected_slots(example)
        actual = actual_slots(example.text)
        clean = True
        for slot in SLOT_FIELDS:
            stats = result.fields[slot]
            stats.total += 1
            kind = compare_field(slot, expected[slot], actual[slot])
            if kind is None:
                stats.correct += 1
                continue
            clean = False
            setattr(stats, kind, getattr(stats, kind) + 1)
            result.field_errors.append(
                FieldError(example.id, example.block, example.text, slot, kind,
                           expected[slot], actual[slot])
            )

        block_stats = result.blocks.setdefault(example.block, BlockStats())
        block_stats.total += 1

        if example.status is not None and index is not None and settings is not None:
            from app.contract.request import parse_request
            from app.pipeline import answer

            request = parse_request({"messages": [{"role": "user", "text": example.text}]})
            document = answer(request=request, index=index, settings=settings, deadline=1e18)
            result.checked_statuses += 1
            actual_status = str(document.get("status"))
            result.statuses[actual_status] = result.statuses.get(actual_status, 0) + 1
            status_errors = _status_errors(example, document)
            if status_errors:
                clean = False
                result.status_errors.extend(status_errors)

        if clean:
            result.full_match += 1
            block_stats.correct += 1
    return result


def baseline_payload(result: Result) -> dict[str, Any]:
    """Агрегаты для заморозки: без списка ошибок, чтобы baseline не разрастался."""
    return {
        "total": result.total,
        "full_match": result.full_match,
        "full_match_rate": round(result.full_match_rate, 4),
        "safety": round(result.safety, 4),
        "penalty": result.penalty,
        "dangerous": len(result.dangerous),
        "fields": {name: stats.as_dict() for name, stats in sorted(result.fields.items())},
        "blocks": {name: stats.as_dict() for name, stats in sorted(result.blocks.items())},
        "statuses": dict(sorted(result.statuses.items())),
    }


def regressions(result: Result, baseline: Mapping[str, Any]) -> list[str]:
    """Список ухудшений относительно замороженных цифр (пусто — всё не хуже)."""
    problems: list[str] = []
    known_total = int(baseline.get("total", 0))
    if known_total and result.total != known_total:
        problems.append(
            f"набор изменился: было {known_total} реплик, стало {result.total} "
            "(обновите baseline осознанно)"
        )
    if result.full_match < int(baseline.get("full_match", 0)):
        problems.append(
            f"полностью верных реплик стало меньше: {result.full_match} "
            f"< {baseline['full_match']}"
        )
    if round(result.safety, 4) < float(baseline.get("safety", 0.0)):
        problems.append(f"безопасность упала: {result.safety:.4f} < {baseline['safety']}")
    if len(result.dangerous) > int(baseline.get("dangerous", 0)):
        problems.append(
            f"опасных ошибок стало больше: {len(result.dangerous)} "
            f"> {baseline['dangerous']}"
        )
    for name, stats in sorted(result.fields.items()):
        expected = baseline.get("fields", {}).get(name)
        if not expected:
            continue
        for key in ("correct", "false_positive", "false_negative", "wrong_value"):
            got = getattr(stats, key)
            was = int(expected.get(key, 0))
            if key == "correct" and got < was:
                problems.append(f"{name}: верных меньше ({got} < {was})")
            if key != "correct" and got > was:
                problems.append(f"{name}: {key} больше ({got} > {was})")
    return problems


def format_report(result: Result, *, errors: int = 20) -> str:
    """Человеческий отчёт: таблица полей, блоков, статусов и список ошибок."""
    lines: list[str] = []
    lines.append(
        f"эталон: {result.total} реплик, полностью верных {result.full_match} "
        f"({result.full_match_rate * 100:.1f} %)"
    )
    lines.append(
        f"безопасность: {result.safety:.4f} (штраф {result.penalty}, "
        f"опасных ошибок {len(result.dangerous)})"
    )
    lines.append("")
    header = f"{'поле':<14}{'всего':>6}{'верно':>7}{'точность':>10}{'лишних':>8}{'потеряно':>10}{'неверно':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for name in SLOT_FIELDS:
        stats = result.fields[name]
        lines.append(
            f"{name:<14}{stats.total:>6}{stats.correct:>7}"
            f"{stats.accuracy * 100:>9.1f}%{stats.false_positive:>8}"
            f"{stats.false_negative:>10}{stats.wrong_value:>9}"
        )
    lines.append("")
    lines.append(f"{'блок':<14}{'всего':>6}{'верно':>7}{'точность':>10}")
    lines.append("-" * 37)
    for name, stats in sorted(result.blocks.items()):
        lines.append(
            f"{name:<14}{stats.total:>6}{stats.correct:>7}{stats.accuracy * 100:>9.1f}%"
        )
    if result.checked_statuses:
        lines.append("")
        lines.append(
            f"статусы (проверено {result.checked_statuses}): "
            + ", ".join(f"{key}={value}" for key, value in sorted(result.statuses.items()))
        )
    all_errors = sorted(
        [*result.field_errors, *result.status_errors],
        key=lambda item: (-item.weight, item.example_id),
    )
    if all_errors:
        lines.append("")
        lines.append(f"ошибки (показано {min(errors, len(all_errors))} из {len(all_errors)}):")
        for item in all_errors[:errors]:
            if isinstance(item, FieldError):
                lines.append(
                    f"  [{item.weight}] {item.example_id} {item.slot}: {item.kind} "
                    f"ожидалось {item.expected!r}, получено {item.actual!r}  ← {item.text!r}"
                )
            else:
                lines.append(
                    f"  [{item.weight}] {item.example_id} status: {item.kind} "
                    f"ожидалось {item.expected!r}, получено {item.actual!r}  ← {item.text!r}"
                )
    else:
        lines.append("")
        lines.append("ошибок нет")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH, help="эталонный набор (JSONL)")
    parser.add_argument("--errors", type=int, default=20, help="сколько ошибок печатать")
    parser.add_argument("--block", action="append", default=None, help="только эти блоки")
    parser.add_argument("--json", action="store_true", help="машинный отчёт")
    parser.add_argument("--no-statuses", action="store_true", help="не проверять статусы ответов")
    parser.add_argument("--update-baseline", action="store_true", help="заморозить текущие цифры")
    parser.add_argument("--check", action="store_true", help="упасть, если хуже baseline")
    args = parser.parse_args(argv)

    try:
        examples = load_golden(args.golden)
    except GoldenError as error:
        print(f"набор размечен неверно: {error}", file=sys.stderr)
        return 2
    if args.block:
        examples = [example for example in examples if example.block in set(args.block)]
        if not examples:
            print("в выбранных блоках нет реплик", file=sys.stderr)
            return 2

    result = evaluate(examples, check_statuses=not args.no_statuses)

    if args.update_baseline:
        BASELINE_PATH.write_text(
            json.dumps(baseline_payload(result), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"baseline обновлён: {BASELINE_PATH}")
        return 0

    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_report(result, errors=args.errors))

    if args.check:
        if not BASELINE_PATH.exists():
            print(f"нет baseline: {BASELINE_PATH}", file=sys.stderr)
            return 2
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        problems = regressions(result, baseline)
        if problems:
            print("", file=sys.stderr)
            for problem in problems:
                print(f"регресс: {problem}", file=sys.stderr)
            return 1
        print("", file=sys.stderr)
        print("регрессов нет: метрика не хуже baseline", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
