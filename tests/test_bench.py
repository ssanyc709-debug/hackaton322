"""Стенд нагрузки: проверяются сами метрики, а не скорость машины.

Тесты намеренно не сравнивают миллисекунды (они разные на каждом ноутбуке и
в облаке автопроверки). Проверяется то, что должно быть верно всегда:
перцентили, сводка, учёт токенов и то, что стенд не ходит в сеть.
"""

from __future__ import annotations

import unittest

from app.nlp.merge import dialogue_text
from bench.load import (
    TOKENS_PER_CALL,
    CountingResolver,
    percentile,
    summarize,
    token_report,
)


class PercentileTest(unittest.TestCase):
    """Перцентиль по ближайшему рангу: без numpy и без интерполяции."""

    def test_extremes(self) -> None:
        values = list(range(1, 101))
        self.assertEqual(percentile(values, 0.0), 1)
        self.assertEqual(percentile(values, 1.0), 100)

    def test_p50_of_even_sample(self) -> None:
        self.assertEqual(percentile([4, 1, 3, 2], 0.5), 2)

    def test_p95_is_the_190th_of_200(self) -> None:
        values = list(range(1, 201))
        self.assertEqual(percentile(values, 0.95), 190)

    def test_empty_sample_is_zero(self) -> None:
        self.assertEqual(percentile([], 0.95), 0.0)

    def test_single_sample(self) -> None:
        self.assertEqual(percentile([7.5], 0.99), 7.5)


class SummarizeTest(unittest.TestCase):
    """Сводка задержек: порядок величин и согласованность полей."""

    def test_keys_and_ordering(self) -> None:
        summary = summarize([3.0, 1.0, 2.0])
        self.assertEqual(
            set(summary), {"min", "mean", "p50", "p95", "p99", "max"}
        )
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 3.0)
        self.assertLessEqual(summary["p50"], summary["p95"])
        self.assertLessEqual(summary["p95"], summary["p99"])

    def test_empty_is_all_zeros(self) -> None:
        self.assertTrue(all(value == 0.0 for value in summarize([]).values()))


class TokenReportTest(unittest.TestCase):
    """Учёт §5: потрачено / сэкономлено относительно «всё в модель»."""

    def test_nothing_spent_when_model_is_not_needed(self) -> None:
        report = token_report(200, 0)
        self.assertEqual(report["spent_tokens"], 0)
        self.assertEqual(report["naive_tokens"], 200 * TOKENS_PER_CALL)
        self.assertEqual(report["zero_token_share"], 1.0)
        self.assertEqual(report["saved_share"], 1.0)

    def test_every_call_spends_exactly_one_exchange(self) -> None:
        report = token_report(10, 10)
        self.assertEqual(report["spent_tokens"], 10 * TOKENS_PER_CALL)
        self.assertEqual(report["saved_tokens"], 0)
        self.assertEqual(report["zero_token_share"], 0.0)

    def test_half_of_requests_is_half_of_savings(self) -> None:
        report = token_report(200, 100)
        self.assertAlmostEqual(report["zero_token_share"], 0.5)
        self.assertAlmostEqual(report["saved_share"], 0.5)
        self.assertEqual(report["spent_tokens"] + report["saved_tokens"], report["naive_tokens"])

    def test_zero_requests_do_not_divide_by_zero(self) -> None:
        report = token_report(0, 0)
        self.assertEqual(report["zero_token_share"], 0.0)
        self.assertEqual(report["saved_share"], 0.0)


class CountingResolverTest(unittest.TestCase):
    """Подмена модели: считает вызовы и не тратит токены (возвращает пустой ответ)."""

    def test_counts_and_returns_empty(self) -> None:
        resolver = CountingResolver()
        self.assertEqual(resolver("Стол в Таганроге", None), {})
        self.assertEqual(resolver("И кресло", None), {})
        self.assertEqual(resolver.calls, 2)


class DialogueTextTest(unittest.TestCase):
    """Текст для единственного вызова модели: диалог целиком, а не обрывок."""

    def test_messages_are_joined(self) -> None:
        self.assertEqual(
            dialogue_text(["Стол и кресло", "12 тысяч"]),
            "Стол и кресло / 12 тысяч",
        )

    def test_empty_history_gives_empty_text(self) -> None:
        self.assertEqual(dialogue_text([]), "")
        self.assertEqual(dialogue_text(["   ", ""]), "")


if __name__ == "__main__":
    unittest.main()
