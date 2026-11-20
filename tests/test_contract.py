"""Контракт входа и выхода: структура запроса, сборка и самопроверка ответа."""

from __future__ import annotations

import json
import unittest

from app.contract import response
from app.contract.request import RequestError, parse_request
from app.contract.response import ResponseError
from tests.support import EXAMPLES_DIR, SCHEMAS_DIR, read_json

try:  # схема — эталон, но не горячий путь
    import jsonschema
except ImportError:  # pragma: no cover - окружение без зависимости
    jsonschema = None


def _user(text: str = "Стол, кресло и лампа. Таганрог. До 15 000 ₽.") -> dict[str, str]:
    return {"role": "user", "text": text}


class RequestParsingTest(unittest.TestCase):
    """Проверки §2.3: только messages, роли, чередование, 1..20 сообщений."""

    def test_example_request_is_parsed(self) -> None:
        payload = read_json(EXAMPLES_DIR / "01-selection" / "request.json")
        request = parse_request(payload)
        self.assertEqual(len(request.messages), 1)
        self.assertIn("Таганрог", request.last_user_text)

    def test_dialogue_with_assistant_is_accepted(self) -> None:
        request = parse_request(
            {"messages": [_user(), {"role": "assistant", "text": "Какой бюджет?"}, _user("15 тысяч")]}
        )
        self.assertEqual(len(request.messages), 3)
        self.assertEqual(request.user_texts, (_user()["text"], "15 тысяч"))

    def test_not_an_object_is_rejected(self) -> None:
        for payload in ([], "text", 5, None):
            with self.subTest(payload=payload):
                with self.assertRaises(RequestError):
                    parse_request(payload)

    def test_extra_top_level_field_is_rejected(self) -> None:
        with self.assertRaises(RequestError):
            parse_request({"messages": [_user()], "city": "taganrog"})

    def test_missing_messages_is_rejected(self) -> None:
        with self.assertRaises(RequestError):
            parse_request({})

    def test_empty_messages_is_rejected(self) -> None:
        with self.assertRaises(RequestError):
            parse_request({"messages": []})

    def test_more_than_twenty_messages_is_rejected(self) -> None:
        messages = []
        for index in range(21):
            role = "user" if index % 2 == 0 else "assistant"
            messages.append({"role": role, "text": f"реплика {index}"})
        messages.append(_user("хвост"))  # 22-я реплика, и первая/последняя снова user
        with self.assertRaises(RequestError):
            parse_request({"messages": messages})

    def test_twenty_messages_are_accepted_only_on_user_turn(self) -> None:
        """20 реплик с чередованием заканчиваются репликой assistant — это нарушение."""
        messages = [
            {"role": "user" if index % 2 == 0 else "assistant", "text": f"реплика {index}"}
            for index in range(20)
        ]
        with self.assertRaises(RequestError):
            parse_request({"messages": messages})
        messages.pop()  # 19 реплик: последняя снова от покупателя
        request = parse_request({"messages": messages})
        self.assertEqual(len(request.messages), 19)

    def test_extra_field_in_message_is_rejected(self) -> None:
        with self.assertRaises(RequestError):
            parse_request({"messages": [{"role": "user", "text": "привет", "id": 1}]})

    def test_missing_text_is_rejected(self) -> None:
        with self.assertRaises(RequestError):
            parse_request({"messages": [{"role": "user"}]})

    def test_blank_text_is_rejected(self) -> None:
        for text in ("", "   ", "\n\t"):
            with self.subTest(text=text):
                with self.assertRaises(RequestError):
                    parse_request({"messages": [_user(text)]})

    def test_non_string_text_is_rejected(self) -> None:
        with self.assertRaises(RequestError):
            parse_request({"messages": [{"role": "user", "text": 42}]})

    def test_unknown_role_is_rejected(self) -> None:
        with self.assertRaises(RequestError):
            parse_request({"messages": [{"role": "system", "text": "ты помощник"}]})

    def test_first_message_must_be_user(self) -> None:
        with self.assertRaises(RequestError):
            parse_request({"messages": [{"role": "assistant", "text": "чем помочь?"}, _user()]})

    def test_last_message_must_be_user(self) -> None:
        with self.assertRaises(RequestError):
            parse_request({"messages": [_user(), {"role": "assistant", "text": "уточните бюджет"}]})

    def test_roles_must_alternate(self) -> None:
        with self.assertRaises(RequestError):
            parse_request({"messages": [_user("раз"), _user("два")]})

    @unittest.skipIf(jsonschema is None, "jsonschema не установлен")
    def test_structural_check_matches_json_schema(self) -> None:
        """По общим правилам наш разбор не расходится с schemas/request.schema.json.

        Чередование ролей и «первое/последнее — user» схема выразить не может
        (это запрет на уровне §2.3, а не JSON Schema), поэтому такие случаи
        проверяются отдельными тестами выше.
        """
        schema = read_json(SCHEMAS_DIR / "request.schema.json")
        validator = jsonschema.Draft202012Validator(schema)
        cases = [
            {"messages": [_user()]},
            {"messages": [_user(), {"role": "assistant", "text": "?"}, _user("15 тысяч")]},
            {"messages": []},
            {"messages": [{"role": "system", "text": "ты помощник"}]},
            {"messages": [_user("")]},
            {"messages": [_user()], "extra": 1},
            {"messages": [{"role": "user", "text": "привет", "id": 1}]},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                schema_ok = not list(validator.iter_errors(payload))
                try:
                    parse_request(payload)
                    code_ok = True
                except RequestError:
                    code_ok = False
                self.assertEqual(code_ok, schema_ok)


class ResponseBuildingTest(unittest.TestCase):
    """Проверки §2.4: ровно один статус, лишние поля запрещены."""

    def test_clarify_keeps_canonical_order_and_dedupes(self) -> None:
        doc = response.clarify(["categories", "budget_rub", "budget_rub"])
        response.validate(doc)
        self.assertEqual([q["field"] for q in doc["questions"]], ["budget_rub", "categories"])

    def test_clarify_without_fields_is_impossible(self) -> None:
        with self.assertRaises(ResponseError):
            response.clarify([])

    def test_ok_document_is_valid(self) -> None:
        doc = response.ok(
            [
                response.option(["t12", "c04", "l09"], 12200, "Стол шириной 95 см"),
                response.option(["t12", "c04", "l10"], 13100, "Другой свет"),
            ]
        )
        response.validate(doc)

    def test_empty_explanation_is_rejected(self) -> None:
        doc = response.ok([response.option(["t12", "c04", "l09"], 12200, "   ")])
        with self.assertRaises(ResponseError):
            response.validate(doc)

    def test_boolean_total_is_rejected(self) -> None:
        doc = response.ok([response.option(["t12"], True, "цена")])
        with self.assertRaises(ResponseError):
            response.validate(doc)

    def test_zero_total_is_rejected(self) -> None:
        doc = response.ok([response.option(["t12"], 0, "цена")])
        with self.assertRaises(ResponseError):
            response.validate(doc)

    def test_duplicate_ids_inside_option_are_rejected(self) -> None:
        doc = response.ok([response.option(["t12", "t12"], 6500, "дубль")])
        with self.assertRaises(ResponseError):
            response.validate(doc)

    def test_duplicate_compositions_are_rejected(self) -> None:
        doc = response.ok(
            [
                response.option(["t12", "c04"], 10700, "первый"),
                response.option(["c04", "t12"], 10700, "второй"),
            ]
        )
        with self.assertRaises(ResponseError):
            response.validate(doc)

    def test_four_options_are_rejected(self) -> None:
        doc = response.ok([response.option([f"t{index}"], 100 + index, "ок") for index in range(4)])
        with self.assertRaises(ResponseError):
            response.validate(doc)

    def test_extra_field_is_rejected(self) -> None:
        doc = response.ok([response.option(["t12"], 6500, "стол")])
        doc["options"][0]["price"] = 6500
        with self.assertRaises(ResponseError):
            response.validate(doc)

    def test_no_match_with_ids_is_rejected(self) -> None:
        doc = response.no_match("нет подходящих объявлений")
        doc["listing_ids"] = ["t12"]
        with self.assertRaises(ResponseError):
            response.validate(doc)

    def test_unknown_status_is_rejected(self) -> None:
        with self.assertRaises(ResponseError):
            response.validate({"status": "error", "message": "нет"})

    @unittest.skipIf(jsonschema is None, "jsonschema не установлен")
    def test_documents_pass_json_schema(self) -> None:
        documents = [
            response.clarify(["budget_rub"]),
            response.ok([response.option(["t12", "c04", "l09"], 12200, "Комплект в бюджете")]),
            response.no_match("В Таганроге нет столов уже 60 см."),
        ]
        for doc in documents:
            with self.subTest(doc=doc["status"]):
                response.validate_with_schema(doc)

    def test_serialization_is_one_line_without_ascii_escapes(self) -> None:
        text = response.finalize(response.clarify(["city_id"]))
        self.assertNotIn("\n", text)
        self.assertIn("В каком городе", text)

    def test_serialization_rejects_nan(self) -> None:
        with self.assertRaises(ResponseError):
            response.dumps({"status": "no_match", "message": float("nan")})  # type: ignore[arg-type]
        with self.assertRaises(ResponseError):
            response.dumps({"status": "no_match", "message": json.loads("1e400")})  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
