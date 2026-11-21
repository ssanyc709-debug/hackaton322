"""Метрика точности как тест: эталонный набор не деградирует.

Зачем отдельный файл, если есть ``bench/accuracy.py``: стенд без теста
запускают «когда вспомнят». Здесь три вещи:

1. **Эталонный набор валиден.** Каждая строка размечена по §2, повтор id,
   забытое поле и статус вне §4 — ошибка стенда, а не «ноль в метрике».
2. **Метрика считает то, что заявлено.** Арифметика штрафа, сравнение
   диапазонов и категорий проверяются на маленьких ручных примерах: если
   метрика сама врёт, её цифрам нельзя верить.
3. **Нет регресса относительно замороженных цифр.** ``bench/golden/baseline.json``
   обновляется осознанно (``python bench/accuracy.py --update-baseline``), а
   случайное ухудшение точности роняет тест с указанием поля и веса ошибки.

Запуск только этого файла::

    python -m unittest tests.test_accuracy
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from bench.accuracy import (
    BASELINE_PATH,
    GOLDEN_PATH,
    FieldError,
    GoldenError,
    Result,
    StatusError,
    compare_field,
    evaluate,
    expected_slots,
    load_golden,
    range_to_dict,
    regressions,
)
from app.model import Bound, NumericRange


def range_dict(lo: int | None, hi: int | None, lo_inc: bool, hi_inc: bool) -> dict:
    """Диапазон в том же виде, в каком его сравнивает метрика."""
    return {"lo": lo, "hi": hi, "lo_inc": lo_inc, "hi_inc": hi_inc}


class GoldenSetTest(unittest.TestCase):
    """Эталонный набор: разметка и её границы."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.examples = load_golden()

    def test_set_is_big_enough_to_mean_something(self) -> None:
        self.assertGreaterEqual(len(self.examples), 200, "набор меньше 200 реплик — метрика шумная")

    def test_every_block_is_present(self) -> None:
        blocks = {example.block for example in self.examples}
        for expected in (
            "width", "budget", "city", "categories", "temperature",
            "adjustable", "mixed", "hard", "noop",
        ):
            with self.subTest(block=expected):
                self.assertIn(expected, blocks, "блок пропал из набора")

    def test_ids_are_unique(self) -> None:
        ids = [example.id for example in self.examples]
        self.assertEqual(len(ids), len(set(ids)), "повтор id делает отчёт бессмысленным")

    def test_missing_key_or_unknown_block_is_a_stand_error(self) -> None:
        """Опечатка в разметке обязана падать, а не тихо занижать точность."""
        import tempfile
        from pathlib import Path

        broken = (
            '{"id": "x1", "block": "width", "text": "стол", "city": null}\n',
            '{"id": "x2", "block": "ширина", "text": "стол", "city": null, "budget": null,'
            ' "categories": [], "width": null, "temperature": null, "adjustable": null}\n',
        )
        for line in broken:
            with self.subTest(line=line[:40]):
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "golden.jsonl"
                    path.write_text(line, encoding="utf-8")
                    with self.assertRaises(GoldenError):
                        load_golden(path)

    def test_empty_range_is_rejected(self) -> None:
        """Объект без единой границы — это ``null``; иначе он ничего не проверяет."""
        import tempfile
        from pathlib import Path

        line = (
            '{"id": "x3", "block": "width", "text": "стол", "city": null, "budget": null,'
            ' "categories": [], "width": {"lo": null, "hi": null, "lo_inc": false,'
            ' "hi_inc": false}, "temperature": null, "adjustable": null}\n'
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "golden.jsonl"
            path.write_text(line, encoding="utf-8")
            with self.assertRaises(GoldenError):
                load_golden(path)


class MetricMathTest(unittest.TestCase):
    """Арифметика метрики на ручных примерах."""

    def test_categories_compare_as_sets(self) -> None:
        self.assertIsNone(compare_field("categories", ("table", "chair"), ("chair", "table")))
        self.assertEqual(compare_field("categories", ("table",), ("table", "lamp")), "false_positive")
        self.assertEqual(compare_field("categories", ("table", "lamp"), ("table",)), "false_negative")

    def test_absent_condition_versus_wrong_condition(self) -> None:
        self.assertIsNone(compare_field("width", None, None))
        self.assertEqual(compare_field("width", None, range_dict(None, 100, False, True)), "false_positive")
        self.assertEqual(compare_field("width", range_dict(None, 100, False, True), None), "false_negative")
        self.assertEqual(
            compare_field(
                "width",
                range_dict(None, 100, False, True),
                range_dict(None, 100, False, False),
            ),
            "wrong_value",
            "включительность границы — часть условия, а не оформление",
        )

    def test_range_to_dict_matches_golden_shape(self) -> None:
        value = NumericRange(lower=Bound(80, True), upper=Bound(100, False))
        self.assertEqual(range_to_dict(value), range_dict(80, 100, True, False))
        self.assertEqual(range_to_dict(NumericRange(lower=None, upper=None)), None)

    def test_safety_is_one_minus_weighted_penalty(self) -> None:
        result = Result(total=2)
        self.assertEqual(result.safety, 1.0)
        result.status_errors.append(StatusError("a", "текст", "wrong_ok", "clarify", "ok"))
        self.assertAlmostEqual(result.safety, 1 - 5 / (5 * 2), places=4)
        self.assertEqual(len(result.dangerous), 1, "wrong_ok обязан попасть в опасные")

    def test_dangerous_is_wider_than_wrong_ok(self) -> None:
        """Неверное значение тоже опасно: покупатель получит не то, что просил."""
        result = Result(total=1)
        result.field_errors.append(
            FieldError(
                "a", "width", "стол", "width", "wrong_value",
                None, range_dict(None, 90, False, True),
            )
        )
        self.assertEqual(len(result.dangerous), 1)
        self.assertAlmostEqual(result.safety, 1 - 3 / 5, places=4)


class NoRegressionTest(unittest.TestCase):
    """Главный тест: текущая точность не хуже замороженной."""

    @classmethod
    def setUpClass(cls) -> None:
        if not BASELINE_PATH.exists():
            raise unittest.SkipTest(f"нет замороженных цифр: {BASELINE_PATH}")
        cls.baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        cls.examples = load_golden()
        cls.result = evaluate(cls.examples)

    def test_set_size_is_the_same(self) -> None:
        self.assertEqual(
            self.result.total,
            int(self.baseline.get("total", 0)),
            "набор изменился: обновите baseline осознанно (--update-baseline)",
        )

    def test_no_field_gets_worse(self) -> None:
        problems = regressions(self.result, self.baseline)
        self.assertEqual(problems, [], "метрика ухудшилась: " + "; ".join(problems))

    def test_no_dangerous_errors(self) -> None:
        self.assertEqual(
            len(self.result.dangerous),
            0,
            "опасные ошибки: "
            + "; ".join(f"{item.example_id} {item.kind} ← {item.text!r}" for item in self.result.dangerous),
        )

    def test_full_match_is_complete(self) -> None:
        self.assertEqual(self.result.full_match, self.result.total, "не все реплики разобраны целиком")

    def test_broken_extraction_is_detected(self) -> None:
        """Проверка самой проверки: испорченный лексикон обязан ронять метрику."""
        original = expected_slots
        subset = [example for example in self.examples if example.city is not None][:20]
        self.assertTrue(subset, "в наборе нет реплик с городом — испортить нечего")

        def broken(example):
            slots = dict(original(example))
            slots["city"] = None  # «город перестал извлекаться»
            return slots

        with mock.patch("bench.accuracy.expected_slots", side_effect=broken):
            result = evaluate(subset, check_statuses=False)
        self.assertLess(result.safety, 1.0, "метрика не заметила потерю города")
        self.assertGreater(len(result.field_errors), 0)


if __name__ == "__main__":
    unittest.main()
