"""Снятие обёртки с ответа модели: ``json_from`` (блок 9).

Разбор обязан быть терпимым к шуму вокруг JSON (markdown-забор, проза,
служебные токены) и одновременно строгим к самому JSON: NaN, Infinity,
повторяющиеся ключи и массив вместо объекта остаются ошибкой. Иначе «починить»
разбор значило бы ослабить контракт.
"""

from __future__ import annotations

import unittest

from app.jsonio import MAX_CANDIDATES, JsonError, json_from

NEEDLE = '{"city_id":"taganrog","budget_rub":15000,"categories":["table"]}'


class WrappedContentTest(unittest.TestCase):
    """Восемь видов шума: каждый сегодня стоил бы потерянного запроса."""

    def test_markdown_fence_with_language(self) -> None:
        self.assertEqual(json_from("```json\n" + NEEDLE + "\n```")["city_id"], "taganrog")

    def test_markdown_fence_without_language(self) -> None:
        self.assertEqual(json_from("```\n" + NEEDLE + "\n```")["budget_rub"], 15000)

    def test_prose_before_json(self) -> None:
        self.assertEqual(json_from("Вот результат: " + NEEDLE)["city_id"], "taganrog")

    def test_prose_after_json(self) -> None:
        self.assertEqual(json_from(NEEDLE + "\nНадеюсь, помог!")["city_id"], "taganrog")

    def test_useless_braces_before_json(self) -> None:
        self.assertEqual(json_from("Пример {a} и ответ: " + NEEDLE)["budget_rub"], 15000)

    def test_service_tokens_around_json(self) -> None:
        self.assertEqual(json_from("<|assistant|>" + NEEDLE + "<|end|>")["city_id"], "taganrog")

    def test_leading_word_json(self) -> None:
        self.assertEqual(json_from("json\n" + NEEDLE)["city_id"], "taganrog")

    def test_byte_order_mark(self) -> None:
        self.assertEqual(json_from("\ufeff" + NEEDLE)["city_id"], "taganrog")

    def test_brace_inside_string_is_not_a_boundary(self) -> None:
        text = '{"note":"см. {пример}","budget_rub":15000}'
        self.assertEqual(json_from(text)["note"], "см. {пример}")


class StrictnessTest(unittest.TestCase):
    """Шум снимается, строгость JSON — нет."""

    def test_duplicate_keys_are_rejected(self) -> None:
        with self.assertRaises(JsonError) as caught:
            json_from('{"a":1,"a":2}')
        self.assertIn("повторяющийся ключ", str(caught.exception))

    def test_nan_is_rejected(self) -> None:
        with self.assertRaises(JsonError) as caught:
            json_from('{"budget_rub":NaN}')
        self.assertIn("запрещённое значение", str(caught.exception))

    def test_array_instead_of_object_is_rejected(self) -> None:
        with self.assertRaises(JsonError) as caught:
            json_from("[1, 2, 3]")
        self.assertIn("массив", str(caught.exception))

    def test_text_without_json_is_rejected(self) -> None:
        with self.assertRaises(JsonError) as caught:
            json_from("Извините, я не могу помочь.")
        self.assertIn("нет объекта JSON", str(caught.exception))

    def test_empty_text_is_rejected(self) -> None:
        with self.assertRaises(JsonError):
            json_from("   ")

    def test_truncated_object_is_rejected(self) -> None:
        """Обрезка по max_tokens: объекта нет целиком, догадываться нельзя."""
        with self.assertRaises(JsonError):
            json_from(NEEDLE[:40])

    def test_source_is_included_in_message(self) -> None:
        with self.assertRaises(JsonError) as caught:
            json_from("мусор", source="choices[0].message.content")
        self.assertIn("choices[0].message.content", str(caught.exception))


class CandidateSelectionTest(unittest.TestCase):
    """Лимиты перебора и выбор пригодного объекта из нескольких."""

    def test_attempt_limit_is_enforced(self) -> None:
        text = "{" * (MAX_CANDIDATES + 10)
        with self.assertRaises(JsonError):
            json_from(text)

    def test_scan_is_limited_by_size(self) -> None:
        text = "я" * 70_000 + NEEDLE
        with self.assertRaises(JsonError):
            json_from(text)

    def test_accept_prefers_usable_object(self) -> None:
        text = '{"error":"x"}' + NEEDLE
        value = json_from(text, accept=lambda raw: bool(raw.get("city_id")))
        self.assertEqual(value["city_id"], "taganrog")

    def test_first_found_object_is_returned_when_nothing_is_acceptable(self) -> None:
        value = json_from('{"a":1}{"b":2}', accept=lambda raw: False)
        self.assertEqual(value, {"a": 1})

    def test_strictness_wins_over_scanning(self) -> None:
        """Даже если пригодный объект есть позже, NaN в найденном — ошибка."""
        with self.assertRaises(JsonError):
            json_from('{"a":NaN}' + NEEDLE, accept=lambda raw: bool(raw.get("city_id")))


if __name__ == "__main__":
    unittest.main()
