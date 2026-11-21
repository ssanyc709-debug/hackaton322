"""Регрессии текстового поиска, сетевого порога и запуска с новым JSON."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.catalog.text_search import TextCatalog
from app.config import LlmConfig
from app.contract.response import validate_with_schema
from app.llm.client import build_payload
from app.llm.product import make_product_resolver
from tests.test_qa_support import ask, run_cli
from web.serve import build_api


ROWS = [
    {"id": "gl-1", "name": "Железная решётка, оцинкованная", "price": 900, "city": "Томск"},
    {"id": "gl-2", "name": "Железная решетка усиленная", "price": 1500, "city": "Томск"},
    {"id": "gl-3", "name": "Металлическая решётка железная", "price": 2000, "city": "Томск"},
    {"id": "gl-4", "name": "Железная решётка большая", "price": 7000, "city": "Томск"},
    {"id": "gl-other", "name": "Железная решётка", "price": 100, "city": "Кемерово"},
    {"id": "br", "name": "Железная метёлка", "price": 100, "city": "Томск"},
    {"id": "br-for", "name": "Железная метёлка для решёток", "price": 110, "city": "Томск"},
    {"id": "beam", "name": "Железная балка", "price": 500, "city": "Томск"},
    {"id": "stick", "name": "Железная палка", "price": 150, "city": "Томск"},
    {"id": "wood", "name": "Деревянная решётка", "price": 100, "city": "Томск"},
]


def ids(document):
    """Возвращает ID в порядке, который показан покупателю."""
    return [value for option in document["options"] for value in option["listing_ids"]]


class TextSearchTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "new-catalog.json"
        self.write({"products": ROWS})
        self.index = TextCatalog.load(self.path)

    def write(self, document) -> None:
        """Записывает произвольный тестовый каталог в отдельный файл."""
        self.path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    def test_three_alternatives_ignore_accessories_wrong_material_city_and_price(self) -> None:
        resolver = Mock()
        result = self.index.answer(
            ["Покажи три варианта железных решёток в Томске до 3 тысяч рублей"],
            resolver=resolver,
        )
        validate_with_schema(result)
        self.assertEqual(ids(result), ["gl-1", "gl-2", "gl-3"])
        resolver.assert_not_called()

    def test_all_variants_still_respect_contract_limit(self) -> None:
        result = self.index.answer(["Покажи все варианты железных решёток в г. Томск, цена до 8000"])
        self.assertEqual(ids(result), ["gl-1", "gl-2", "gl-3"])

    def test_beam_and_stick_never_replace_each_other(self) -> None:
        resolver = Mock()
        beam = self.index.answer(["Два варианта железных балок в Томске до 1000 ₽"], resolver=resolver)
        self.assertEqual(ids(beam), ["beam"])
        self.assertEqual(ids(self.index.answer(["Железная палка, Томск, до 200 ₽"])), ["stick"])
        without_beam = [row for row in ROWS if row["id"] != "beam"]
        self.write(without_beam)
        self.assertEqual(
            TextCatalog.load(self.path).answer(["Железная балка, Томск, до 1000 ₽"], resolver=resolver)["status"],
            "no_match",
        )
        resolver.assert_not_called()

    def test_accessory_is_found_when_it_is_the_request(self) -> None:
        self.assertEqual(
            ids(self.index.answer(["Метёлка для решёток, Томск, 200 ₽"])), ["br-for"]
        )
        self.assertEqual(
            self.index.answer(["Решётка для метёлки, Томск, 200 ₽"])["status"], "no_match"
        )

    def test_history_keeps_requirements_and_updates_price_and_product(self) -> None:
        self.assertEqual(
            ids(self.index.answer(["Железная решётка в Томске", "До 1000 ₽", "Спасибо"])),
            ["gl-1"],
        )
        self.assertEqual(
            ids(self.index.answer(["Железная балка в Томске до 1000 ₽", "Лучше железная палка"])),
            ["stick"],
        )

    def test_description_uses_one_call_when_city_and_budget_are_known(self) -> None:
        resolver = Mock(return_value="железная решётка")
        text = "Нужно что-нибудь чтобы закрыть отверстие, в Томске"
        missing = self.index.answer([text], resolver=resolver)
        self.assertEqual([question["field"] for question in missing["questions"]], ["budget_rub"])
        resolver.assert_not_called()
        result = self.index.answer([text, "До 1000 ₽"], resolver=resolver)
        self.assertEqual(ids(result), ["gl-1"])
        resolver.assert_called_once()
        self.assertIn("чтобы закрыть отверстие", resolver.call_args.args[0])
        self.assertNotIn("томске", resolver.call_args.args[0])
        self.assertNotIn("1000", resolver.call_args.args[0])

    def test_negation_is_never_discarded_from_the_product_name(self) -> None:
        resolver = Mock()
        result = self.index.answer(["Нужна не железная балка, Томск, до 1000 ₽"], resolver=resolver)
        self.assertEqual([question["field"] for question in result["questions"]], ["categories"])
        resolver.assert_not_called()

    def test_model_cannot_drop_a_word_already_understood_locally(self) -> None:
        text = "Нужно что-нибудь железное чтобы закрыть отверстие, в Томске до 1000 ₽"
        resolver = Mock(return_value="деревянная решётка")
        result = self.index.answer([text], resolver=resolver)
        self.assertEqual(result["status"], "clarify")
        resolver.assert_called_once()
        resolver.return_value = "решётка"
        self.assertEqual(ids(self.index.answer([text], resolver=resolver)), ["gl-1"])

    def test_city_id_and_russian_name_come_from_the_catalog(self) -> None:
        self.write([{"id": "x", "title": "Железная балка", "price_rub": 900, "city_id": "t42", "city_name": "Москва"}])
        index = TextCatalog.load(self.path)
        self.assertEqual(ids(index.answer(["Железная балка в Москве до 1000 ₽"])), ["x"])
        self.assertEqual(ids(index.answer(["Железная балка в t42 до 1000 ₽"])), ["x"])

    def test_hyphenated_city_is_removed_from_the_product_phrase(self) -> None:
        self.write([{**ROWS[0], "city": "rostov-na-donu"}])
        index = TextCatalog.load(self.path)
        self.assertEqual(ids(index.answer(["Железная решётка в Ростове-на-Дону до 1000 ₽"])), ["gl-1"])

    def test_current_file_controls_cli_and_web_cards(self) -> None:
        query = "Железная балка, Томск, до 1000 ₽"
        result = run_cli(ask(query), catalog=self.path, extra_args=("--text-search",))
        self.assertEqual(result.code, 0, result.stderr)
        self.assertEqual(ids(result.one_json_object()), ["beam"])
        api = build_api(self.path, offline=True, text_search=True)
        status, body, _ = api.answer(json.dumps(ask(query)).encode())
        self.assertEqual(status, 200, body)
        code, cards = api.listings({"ids": ids(body)})
        self.assertEqual(code, 200)
        self.assertEqual(cards["items"][0]["title"], "Железная балка")
        self.write([{**ROWS[-3], "id": "new-id"}])
        updated = run_cli(ask(query), catalog=self.path, extra_args=("--text-search",))
        self.assertEqual(ids(updated.one_json_object()), ["new-id"])

    def test_expensive_items_match_response_schema(self) -> None:
        self.write([{"id": "machine", "title": "Промышленный станок", "price_rub": 12_000_000, "city": "Томск"}])
        result = run_cli(
            ask("Промышленный станок в Томске до 15 000 000 рублей"),
            catalog=self.path, extra_args=("--text-search",),
        )
        self.assertEqual(result.code, 0, result.stderr)
        doc = result.one_json_object()
        validate_with_schema(doc)
        self.assertEqual(doc["options"][0]["total_rub"], 12_000_000)

    def test_invalid_catalog_has_technical_exit(self) -> None:
        self.write(ROWS[:1] * 2)
        result = run_cli(ask("Решётка"), catalog=self.path, extra_args=("--text-search",))
        self.assertEqual(result.code, 3)
        self.assertEqual(result.stdout, b"")


class ProductResolverTest(unittest.TestCase):
    def test_payload_uses_small_schema_without_catalog_data(self) -> None:
        config = LlmConfig(base_url="https://example.invalid", model="test", api_key="fake")
        captured = []

        def complete_locally(config, messages, deadline, *, max_tokens, schema, schema_name, sanitize):
            payload = build_payload(config, messages, max_tokens, schema=schema, schema_name=schema_name)
            captured.append(json.loads(payload))
            return sanitize({"product": "железная решётка", "budget_rub": 5})

        resolve = make_product_resolver(config, 60.0, complete_fn=complete_locally)
        self.assertEqual(resolve("что-то закрыть отверстие"), "железная решётка")
        self.assertEqual(captured[0]["messages"][1]["content"], "что-то закрыть отверстие")
        self.assertEqual(captured[0]["max_tokens"], 64)
        schema = captured[0]["response_format"]["json_schema"]["schema"]
        self.assertEqual(list(schema["properties"]), ["product"])

    def test_shared_client_parses_product_response(self) -> None:
        config = LlmConfig(base_url="https://example.invalid", model="test", api_key="fake")
        content = json.dumps({"product": "железная решётка"}, ensure_ascii=False)
        envelope = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
        resolve = make_product_resolver(config, time.monotonic() + 5)

        with patch("app.llm.client._http_post", return_value=(200, {}, envelope)) as post:
            self.assertEqual(resolve("чем закрыть отверстие"), "железная решётка")
        post.assert_called_once()
        payload = json.loads(post.call_args.args[1])
        self.assertEqual(payload["max_tokens"], 64)
        self.assertEqual(payload["messages"][1]["content"], "чем закрыть отверстие")
