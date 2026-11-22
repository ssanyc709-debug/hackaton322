"""Шлюз демо-сайта: 8 HTTP-проверок по своему же серверу (перенесены из автономной самопроверки).

Тесты общаются с сервером так же, как это делает браузер, поэтому проверяют
именно шлюз: маршруты, коды ответов, белый список полей и то, что сбой бэкенда
не превращается в «нет вариантов». Импортов из ``app/`` нет — это граница демо.

Никаких ``print``: стандартный ``unittest`` сам печатает результат, а вывод
сервера заглушается на время тестов, чтобы не шуметь. Запуск:

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import http.client
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_llm_client import API_KEY, MODEL, FakeGateway, envelope
from web import serve
from web.serve import ROOT, build_api, make_server

CATALOG = ROOT / "data" / "catalog.json"
HOST = "127.0.0.1"


class RunningServer:
    """Сервер на свободном порту в отдельном потоке; логи глушим на время теста."""

    def __init__(self, catalog: Path, *, offline: bool = True) -> None:
        self._quiet = io.StringIO()
        self._stderr = sys.stderr
        sys.stderr = self._quiet  # логи сборки не мешают выводу проверок
        try:
            api = build_api(catalog, offline=offline)
            self.server = make_server(api, HOST, 0)
        except BaseException:
            sys.stderr = self._stderr
            raise
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.05}
        )
        self.thread.daemon = True

    def __enter__(self) -> "RunningServer":
        self.thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        sys.stderr = self._stderr
        if os.environ.get("SELFTEST_VERBOSE"):
            sys.stderr.write(self._quiet.getvalue())


def call(
    port: int,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """Один HTTP-запрос; возвращает (код, заголовки, тело в байтах)."""
    connection = http.client.HTTPConnection(HOST, port, timeout=60)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        payload = response.read()
        return response.status, dict(response.getheaders()), payload
    finally:
        connection.close()


def as_json(payload: bytes) -> object:
    """Тело как JSON или ``None`` (для проверки, что ответ вообще разбирается)."""
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def two_real_ids() -> list[str]:
    """Первые два настоящих id из каталога (для /api/listings)."""
    document = json.loads(CATALOG.read_text(encoding="utf-8-sig"))
    ids = [
        item["id"]
        for item in document
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    return ids[:2]


class WebGatewayTest(unittest.TestCase):
    """8 проверок HTTP-шлюза на штатном каталоге плюс один сервер с битым."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = RunningServer(CATALOG)
        cls.server.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.__exit__(None, None, None)

    def test_1_static_page(self) -> None:
        """GET / — 200, text/html, маркер страницы."""
        status, headers, body = call(self.server.port, "GET", "/")
        content_type = headers.get("Content-Type", "")
        marker = "buyer-assistant-demo" in body.decode("utf-8", "replace")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertTrue(marker, "на странице нет маркера buyer-assistant-demo")

    def test_2_static_assets(self) -> None:
        """GET /app.js и /style.css — 200 и верные типы."""
        js_status, js_headers, _ = call(self.server.port, "GET", "/app.js")
        css_status, css_headers, _ = call(self.server.port, "GET", "/style.css")
        self.assertEqual(js_status, 200)
        self.assertEqual(css_status, 200)
        self.assertIn("javascript", js_headers.get("Content-Type", ""))
        self.assertIn("css", css_headers.get("Content-Type", ""))

    def test_3_answer_ok(self) -> None:
        """POST /api/answer — 200 и статус контракта."""
        body = json.dumps(
            {"messages": [{"role": "user", "text": "Стол, кресло и лампа. Таганрог. До 15000."}]},
            ensure_ascii=False,
        ).encode("utf-8")
        status, _, payload = call(
            self.server.port, "POST", "/api/answer", body, {"Content-Type": "application/json"}
        )
        document = as_json(payload)
        self.assertEqual(status, 200)
        self.assertIsInstance(document, dict)
        assert isinstance(document, dict)
        self.assertIn(document.get("status"), {"ok", "clarify", "no_match"})

    def test_4_broken_body(self) -> None:
        """POST /api/answer с битым телом — 400 и error."""
        status, _, payload = call(
            self.server.port,
            "POST",
            "/api/answer",
            "{ это не JSON".encode("utf-8"),
            {"Content-Type": "application/json"},
        )
        document = as_json(payload)
        self.assertEqual(status, 400)
        self.assertIsInstance(document, dict)
        assert isinstance(document, dict)
        self.assertTrue(document.get("error"))

    def test_5_backend_failure(self) -> None:
        """Сломанный каталог: POST /api/answer — 502 и error, а не no_match."""
        bad_catalog = ROOT / "data" / "no_such_catalog_for_web_test.json"
        with RunningServer(bad_catalog) as server:
            body = json.dumps(
                {"messages": [{"role": "user", "text": "Стол в Таганроге до 15000"}]},
                ensure_ascii=False,
            ).encode("utf-8")
            status, _, payload = call(
                server.port, "POST", "/api/answer", body, {"Content-Type": "application/json"}
            )
        document = as_json(payload)
        self.assertEqual(status, 502)
        self.assertIsInstance(document, dict)
        assert isinstance(document, dict)
        self.assertTrue(document.get("error"))

    def test_6_listings(self) -> None:
        """GET /api/listings — карточки без description."""
        ids = two_real_ids()
        status, _, payload = call(self.server.port, "GET", "/api/listings?ids=" + ",".join(ids))
        document = as_json(payload)
        self.assertEqual(status, 200)
        assert isinstance(document, dict)
        items = document.get("items") or []
        required = ("title", "price_rub", "category", "city_id", "attributes")
        self.assertEqual(len(items), 2)
        for item in items:
            for key in required:
                self.assertIn(key, item)
            self.assertNotIn("description", item)

    def test_7_listings_empty(self) -> None:
        """GET /api/listings без ids — 400, а не 500."""
        status, _, payload = call(self.server.port, "GET", "/api/listings?ids=")
        document = as_json(payload)
        self.assertEqual(status, 400)
        self.assertIsInstance(document, dict)
        assert isinstance(document, dict)
        self.assertTrue(document.get("error"))

    def test_8_no_traversal(self) -> None:
        """Обход путей закрыт — файлы репозитория не отдаются."""
        probes = ["/../app/main.py", "/../../.env", "/%2e%2e%2fapp%2fmain.py"]
        for path in probes:
            with self.subTest(path=path):
                status, _, payload = call(self.server.port, "GET", path)
                leaked = b"def main" in payload or b"ASSISTANT" in payload or b"base_url" in payload
                self.assertIn(status, (400, 403, 404))
                self.assertFalse(leaked, f"утечка содержимого по {path}")


class CatalogSnapshotTest(unittest.TestCase):
    def test_answer_and_cards_use_one_version_even_when_source_changes(self) -> None:
        first_row = {"id": "same", "title": "Термос", "price": 900, "city": "Псков"}
        later_row = {**first_row, "price": 1500, "city": "Калуга"}

        def payload(text):
            return json.dumps({"messages": [{"role": "user", "text": text}]}).encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "catalog.json"
            source.write_text(json.dumps([first_row]), encoding="utf-8")
            api = build_api(source, offline=True, text_search=True)
            call_backend = serve.call_backend

            def change_source(*args, **kwargs):
                source.write_text(json.dumps([later_row]), encoding="utf-8")
                return call_backend(*args, **kwargs)

            with patch.object(serve, "call_backend", side_effect=change_source):
                code, first, _ = api.answer(payload("Термос в Пскове до 1000 ₽"), include_cards=True)
            self.assertEqual(code, 200, first)
            self.assertEqual(first["answer"]["options"][0]["total_rub"], 900)
            self.assertEqual(first["items"][0]["price_rub"], 900)
            self.assertEqual(first["items"][0]["city_name"], "Псков")

            code, later, _ = api.answer(payload("Термос в Калуге до 2000 ₽"), include_cards=True)
            self.assertEqual(code, 200, later)
            self.assertEqual(later["answer"]["options"][0]["total_rub"], 1500)
            self.assertEqual(later["items"][0]["price_rub"], 1500)
            self.assertEqual(later["items"][0]["city_name"], "Калуга")


class ModelReportingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gateway = FakeGateway().start()
        self.addCleanup(self.gateway.stop)
        with patch.dict(os.environ, {
            "LLM_BASE_URL": self.gateway.base_url,
            "LLM_MODEL": MODEL,
            "LLM_API_KEY": API_KEY,
            "ASSISTANT_OFFLINE": "0",
        }):
            self.server = self.enterContext(RunningServer(CATALOG, offline=False))

    def ask(self, text: str = "Найди то, на чём можно писать"):
        body = json.dumps({"messages": [{"role": "user", "text": text}]}).encode("utf-8")
        return call(self.server.port, "POST", "/api/answer", body, {"Content-Type": "application/json"})

    def test_reports_every_request_even_if_the_gateway_omits_usage(self) -> None:
        categories = {"categories": ["table"]}
        no_usage = json.loads(envelope(categories))
        del no_usage["usage"]
        examples = [
            (envelope(categories), "228"),
            (json.dumps(no_usage).encode("utf-8"), None),
            (envelope(categories, usage={"total_tokens": 91}), "91"),
        ]
        for number, (response, tokens) in enumerate(examples, start=1):
            with self.subTest(request=number):
                self.gateway.enqueue(200, response)
                status, headers, body = self.ask()
                self.assertEqual(status, 200, body)
                self.assertEqual(len(self.gateway.requests), number)
                self.assertEqual(headers["X-Model-Calls"], "1")
                self.assertEqual(headers.get("X-Model-Tokens"), tokens)
        status, headers, body = self.ask("Нужен стол в Таганроге")
        self.assertEqual(status, 200, body)
        self.assertEqual(headers["X-Model-Calls"], "0")
        self.assertEqual(headers["X-Model-Tokens"], "0")
        self.assertEqual(len(self.gateway.requests), 3)

    def test_rate_limit_retry_counts_as_one_model_request(self) -> None:
        self.gateway.enqueue(429, b'{"error":{"message":"busy"}}', {"Retry-After": "0"})
        self.gateway.enqueue(200, envelope({"categories": ["table"]}))
        status, headers, body = self.ask()
        self.assertEqual(status, 200, body)
        self.assertEqual(len(self.gateway.requests), 2)
        self.assertEqual(headers["X-Model-Calls"], "1")

    def test_gateway_error_still_reports_attempt(self) -> None:
        self.gateway.enqueue(503, b'{"error":{"message":"gateway offline"}}')
        status, headers, body = self.ask()
        self.assertEqual(status, 502, body)
        self.assertEqual(headers["X-Model-Calls"], "1")
        self.assertNotIn("X-Model-Tokens", headers)

    def test_web_reuses_recognized_product_when_user_answers_clarification(self) -> None:
        self.gateway.enqueue(200, envelope({"categories": ["table"]}))
        first = {"role": "user", "text": "Найди то, на чём можно писать"}

        def send(messages, token=""):
            return call(
                self.server.port, "POST", "/api/answer?include=cards",
                json.dumps({"messages": messages}).encode("utf-8"),
                {"Content-Type": "application/json", "X-Dialogue-Token": token},
            )

        code, headers, body = send([first])
        self.assertEqual(code, 200, body)
        self.assertEqual(headers["X-Model-Calls"], "1")
        token = headers["X-Dialogue-Token"]
        reply = {"role": "user", "text": "Таганрог, до 5000 ₽"}
        question = {"role": "assistant", "text": "Уточните город и бюджет"}
        code, headers, body = send([first, question, reply], token)
        self.assertEqual(code, 200, body)
        self.assertEqual(headers["X-Model-Calls"], "0")
        self.assertEqual(headers["X-Model-Reused"], "1")
        self.assertEqual(len(self.gateway.requests), 1)
        result = json.loads(body)
        self.assertEqual(result["answer"]["status"], "ok")
        self.assertTrue(result["items"])
        self.assertTrue(all(item["category"] == "table" for item in result["items"]))

        self.gateway.enqueue(200, envelope({"categories": ["lamp"]}))
        code, headers, body = send([{"role": "user", "text": "Найди деревяшку на четырёх ножках"}, question, reply], token)
        self.assertEqual(code, 200, body)
        self.assertEqual(headers["X-Model-Calls"], "1")
        self.assertEqual(len(self.gateway.requests), 2)


if __name__ == "__main__":
    unittest.main()
