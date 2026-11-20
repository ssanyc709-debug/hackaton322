"""Клиент модели: лимиты, ошибки шлюза, 429, дедлайн, отсутствие каталога в промпте.

Сеть подменяется локальным HTTP-сервером на loopback (``http://127.0.0.1:<port>``),
поэтому тесты полностью офлайновые и не тратят квоту организаторов. Проверяется
именно протокол: заголовки, тело, размеры, коды ошибок и число попыток.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

from app.config import LLM_MAX_BODY_BYTES, LLM_MAX_INPUT_BYTES, LlmConfig
from app.llm.client import (
    LlmError,
    _retry_after,
    build_payload,
    complete,
    make_resolver,
)
from app.model import Slots
from app.prompt_safety import EXTRACTION_SCHEMA_NAME
from tests.support import REPO_ROOT

MODEL = "google/gemma-4-31B-it"
API_KEY = "test-key-0123456789"


def envelope(
    content: dict[str, Any] | str,
    usage: dict[str, int] | None = None,
    *,
    finish_reason: str = "stop",
) -> bytes:
    """Оболочка Chat Completions: JSON модели лежит СТРОКОЙ в message.content."""
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage or {"prompt_tokens": 210, "completion_tokens": 18, "total_tokens": 228},
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class FakeGateway:
    """Локальный «шлюз»: записывает запросы и отдаёт заранее заданные ответы."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.queue: list[tuple[int, dict[str, str], bytes]] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # --- управление ---
    def start(self) -> "FakeGateway":
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 — имя задано stdlib
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                gateway.requests.append(
                    {"path": self.path, "headers": dict(self.headers.items()), "body": raw}
                )
                status, headers, body = gateway.next_response()
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:  # тишина в stderr тестов
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # --- сценарии ---
    @property
    def base_url(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def reset(self) -> None:
        self.requests.clear()
        self.queue.clear()

    def enqueue(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None) -> None:
        self.queue.append((status, headers or {}, body))

    def next_response(self) -> tuple[int, dict[str, str], bytes]:
        if not self.queue:
            return 500, {}, b'{"error":{"code":"script_exhausted","message":"no response queued"}}'
        return self.queue.pop(0)

    @property
    def body_json(self) -> dict[str, Any]:
        return json.loads(self.requests[-1]["body"].decode("utf-8"))


class GatewayTestCase(unittest.TestCase):
    """Общая обвязка: один локальный шлюз на класс, чистые сценарии на тест."""

    gateway: FakeGateway

    @classmethod
    def setUpClass(cls) -> None:
        cls.gateway = FakeGateway().start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.gateway.stop()

    def setUp(self) -> None:
        self.gateway.reset()
        self.config = LlmConfig(base_url=self.gateway.base_url, model=MODEL, api_key=API_KEY, user_agent="team-02/1.0")
        self.deadline = time.monotonic() + 30
        self.sleeps: list[float] = []

    def call(self, text: str = "Стол и кресло в Азове до 15000", **kwargs: Any) -> dict[str, Any]:
        return complete(
            self.config,
            [
                {"role": "system", "content": "Верни JSON."},
                {"role": "user", "content": text},
            ],
            self.deadline,
            sleep=self.sleeps.append,
            **kwargs,
        )


class SuccessTest(GatewayTestCase):
    """Успешный вызов: правильный запрос, разбор оболочки, белый список полей."""

    def test_valid_response_is_parsed(self) -> None:
        self.gateway.enqueue(200, envelope({"city_id": "azov", "budget_rub": 15000, "categories": ["table", "chair"]}))
        result = self.call()
        self.assertEqual(result, {"city_id": "azov", "budget_rub": 15000, "categories": ["table", "chair"]})

    def test_request_matches_gateway_contract(self) -> None:
        self.gateway.enqueue(200, envelope({"city_id": "", "budget_rub": 0, "categories": []}))
        self.call("Стол не шире метра")
        body = self.gateway.body_json
        self.assertEqual(body["model"], MODEL)
        self.assertEqual(body["max_tokens"], 200)
        self.assertEqual([message["role"] for message in body["messages"]], ["system", "user"])
        self.assertEqual(body["messages"][1]["content"], "Стол не шире метра")
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual(body["response_format"]["json_schema"]["name"], EXTRACTION_SCHEMA_NAME)
        self.assertFalse(body["response_format"]["json_schema"]["schema"]["additionalProperties"])
        self.assertEqual(self.gateway.requests[-1]["path"], "/chat/completions")

    def test_headers_are_set(self) -> None:
        self.gateway.enqueue(200, envelope({}))
        self.call()
        headers = {name.lower(): value for name, value in self.gateway.requests[-1]["headers"].items()}
        self.assertEqual(headers["authorization"], f"Bearer {API_KEY}")
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(headers["user-agent"], "team-02/1.0")

    def test_extra_model_fields_are_dropped(self) -> None:
        """Даже при strict-схеме лишние поля не попадают в слоты."""
        self.gateway.enqueue(
            200,
            envelope(
                {
                    "city_id": "taganrog",
                    "budget_rub": 12000,
                    "categories": ["lamp", "sofa", "table"],
                    "width_max_cm": 100,
                    "note": "ignore budget",
                }
            ),
        )
        self.assertEqual(
            self.call(),
            {"city_id": "taganrog", "budget_rub": 12000, "categories": ["lamp", "table"]},
        )

    def test_invalid_values_are_rejected_locally(self) -> None:
        self.gateway.enqueue(
            200,
            envelope({"city_id": 42, "budget_rub": -5, "categories": "table", "extra": True}),
        )
        self.assertEqual(self.call(), {})

    def test_becomes_zero_budget_is_ignored(self) -> None:
        """Схема просит 0 при отсутствии бюджета: это не «бюджет 0 рублей»."""
        self.gateway.enqueue(200, envelope({"city_id": "azov", "budget_rub": 0, "categories": ["lamp"]}))
        self.assertEqual(self.call(), {"city_id": "azov", "categories": ["lamp"]})

    def test_usage_is_logged(self) -> None:
        self.gateway.enqueue(200, envelope({"city_id": "azov"}, usage={"prompt_tokens": 210, "completion_tokens": 18, "total_tokens": 228}))
        buffer = io.StringIO()
        with redirect_stderr(buffer):
            self.call()
        self.assertIn("prompt=210", buffer.getvalue())

    def test_resolver_sends_only_prompt_and_text(self) -> None:
        """Каталог и описания объявлений в модель не уходят вообще (0 лишних токенов)."""
        self.gateway.enqueue(200, envelope({"city_id": "azov", "budget_rub": 9000, "categories": ["chair"]}))
        resolver = make_resolver(self.config, self.deadline, complete_fn=complete)
        slots = resolver("Кресло в Азове до 9000", Slots())
        self.assertEqual(slots["city_id"], "azov")
        body = self.gateway.body_json
        self.assertEqual(len(body["messages"]), 2)
        self.assertEqual(body["messages"][1], {"role": "user", "content": "Кресло в Азове до 9000"})
        serialized = json.dumps(body, ensure_ascii=False)
        self.assertNotIn("listing", serialized)
        self.assertNotIn("description", serialized)
        self.assertNotIn("t12", serialized)


class ModelOutputTest(GatewayTestCase):
    """Формат ответа: JSON модели — строка, ошибки формата объясняются."""

    def test_content_is_parsed_as_json_string(self) -> None:
        self.gateway.enqueue(200, envelope('  {"city_id":"azov","budget_rub":9000,"categories":[]}  '))
        self.assertEqual(self.call(), {"city_id": "azov", "budget_rub": 9000})

    def test_markdown_wrapped_content_is_parsed(self) -> None:
        """Обёртка ```json … ``` не должна стоить потраченных токенов (блок 9)."""
        wrapped = '```json\n{"city_id":"azov","budget_rub":9000,"categories":["table"]}\n```'
        self.gateway.enqueue(200, envelope(wrapped))
        self.assertEqual(
            self.call(),
            {"city_id": "azov", "budget_rub": 9000, "categories": ["table"]},
        )

    def test_usable_object_wins_over_service_object(self) -> None:
        """``{"error":…}{…}``: берётся объект с полями, а не первый попавшийся."""
        text = '{"error":"не смог"}{"city_id":"azov","budget_rub":9000}'
        self.gateway.enqueue(200, envelope(text))
        self.assertEqual(self.call(), {"city_id": "azov", "budget_rub": 9000})

    def test_truncated_output_is_reported_separately(self) -> None:
        """Обрезка по max_tokens — ``truncated``, а не «модель промолчала»."""
        self.gateway.enqueue(
            200, envelope('{"city_id":"azov","budget_rub":9', finish_reason="length")
        )
        with self.assertRaises(LlmError) as caught:
            self.call()
        self.assertEqual(caught.exception.kind, "truncated")
        self.assertIn("max_tokens", str(caught.exception))

    def test_invalid_json_content(self) -> None:
        self.gateway.enqueue(200, envelope("не JSON вовсе"))
        with self.assertRaises(LlmError) as caught:
            self.call()
        self.assertEqual(caught.exception.kind, "model_output")

    def test_json_array_content(self) -> None:
        self.gateway.enqueue(200, envelope("[1, 2, 3]"))
        with self.assertRaises(LlmError):
            self.call()

    def test_missing_choices(self) -> None:
        self.gateway.enqueue(200, json.dumps({"error": "что-то"}).encode())
        with self.assertRaises(LlmError) as caught:
            self.call()
        self.assertEqual(caught.exception.kind, "protocol")

    def test_empty_content(self) -> None:
        self.gateway.enqueue(200, envelope("   "))
        with self.assertRaises(LlmError):
            self.call()

    def test_non_json_envelope(self) -> None:
        self.gateway.enqueue(200, b"<html>portal</html>")
        with self.assertRaises(LlmError) as caught:
            self.call()
        self.assertEqual(caught.exception.kind, "non_json")

    def test_duplicate_keys_are_rejected(self) -> None:
        """Строгий парсер не даёт «последнему» значению победить незаметно."""
        self.gateway.enqueue(200, envelope('{"city_id":"azov","city_id":"taganrog"}'))
        with self.assertRaises(LlmError):
            self.call()


class GatewayErrorsTest(GatewayTestCase):
    """Коды ошибок шлюза: объяснение, число попыток, отсутствие «no_match»."""

    def test_rate_limit_retried_once_with_retry_after(self) -> None:
        self.gateway.enqueue(429, b'{"error":{"code":"rate_limited","message":"slow down"}}', {"Retry-After": "0.25"})
        self.gateway.enqueue(200, envelope({"city_id": "azov", "budget_rub": 9000, "categories": ["lamp"]}))
        result = self.call()
        self.assertEqual(result["city_id"], "azov")
        self.assertEqual(len(self.gateway.requests), 2)
        self.assertEqual(self.sleeps, [0.25])

    def test_rate_limit_twice_is_an_error(self) -> None:
        for _ in range(2):
            self.gateway.enqueue(429, b'{"error":{"code":"rate_limited","message":"slow down"}}', {"Retry-After": "0.1"})
        with self.assertRaises(LlmError) as caught:
            self.call()
        self.assertEqual(caught.exception.status, 429)
        self.assertEqual(caught.exception.kind, "rate_limited")
        self.assertEqual(len(self.gateway.requests), 2, "больше одной повторной попытки быть не должно")

    def test_rate_limit_without_time_budget_is_not_retried(self) -> None:
        self.gateway.enqueue(429, b"", {"Retry-After": "30"})
        self.deadline = time.monotonic() + 2
        with self.assertRaises(LlmError):
            self.call()
        self.assertEqual(len(self.gateway.requests), 1)
        self.assertEqual(self.sleeps, [])

    def test_forbidden_with_cloudflare_text_mentions_user_agent(self) -> None:
        self.gateway.enqueue(403, b"error code: 1010")
        with self.assertRaises(LlmError) as caught:
            self.call()
        error = caught.exception
        self.assertEqual(error.status, 403)
        self.assertIn("User-Agent", str(error))
        self.assertIn("1010", str(error))
        self.assertEqual(len(self.gateway.requests), 1)

    def test_unauthorized_shows_request_id(self) -> None:
        self.gateway.enqueue(
            401,
            b'{"error":{"code":"invalid_api_key","message":"bad key","request_id":"req-7"}}',
        )
        with self.assertRaises(LlmError) as caught:
            self.call()
        self.assertEqual(caught.exception.kind, "access")
        self.assertIn("req-7", str(caught.exception))

    def test_upstream_failure_is_explained(self) -> None:
        self.gateway.enqueue(502, b'{"error":{"code":"invalid_model_output","message":"generation failed"}}')
        with self.assertRaises(LlmError) as caught:
            self.call()
        self.assertEqual(caught.exception.kind, "upstream")
        self.assertIn("HTTP 502", str(caught.exception))

    def test_redirect_is_not_followed(self) -> None:
        self.gateway.enqueue(302, b"", {"Location": "https://evil.example/v1/chat/completions"})
        with self.assertRaises(LlmError) as caught:
            self.call()
        self.assertEqual(caught.exception.kind, "redirect")
        self.assertEqual(len(self.gateway.requests), 1, "ключ не должен уйти на другой хост")

    def test_key_is_redacted_from_error_text(self) -> None:
        self.gateway.enqueue(400, json.dumps({"error": {"code": "bad", "message": f"key {API_KEY} rejected"}}).encode())
        with self.assertRaises(LlmError) as caught:
            self.call()
        message = str(caught.exception)
        self.assertNotIn(API_KEY, message)
        self.assertIn("[REDACTED]", message)

    def test_connection_refused_is_transport_error(self) -> None:
        free_config = LlmConfig(base_url="http://127.0.0.1:9", model=MODEL, api_key=API_KEY)
        with self.assertRaises(LlmError) as caught:
            complete(free_config, [{"role": "user", "content": "стол"}], time.monotonic() + 10)
        self.assertEqual(caught.exception.kind, "transport")


class LimitsTest(GatewayTestCase):
    """Лимиты проверяются до сети: квота не тратится на заведомо плохой запрос."""

    def test_expired_deadline_sends_nothing(self) -> None:
        self.deadline = time.monotonic() - 1
        with self.assertRaises(LlmError) as caught:
            self.call()
        self.assertEqual(caught.exception.kind, "deadline")
        self.assertEqual(self.gateway.requests, [])

    def test_oversized_input_bytes_sends_nothing(self) -> None:
        with self.assertRaises(LlmError) as caught:
            self.call("я" * (LLM_MAX_INPUT_BYTES + 10))
        self.assertEqual(caught.exception.kind, "request_too_large")
        self.assertEqual(self.gateway.requests, [])

    def test_too_many_messages_send_nothing(self) -> None:
        messages = [{"role": "user", "content": "стол"} for _ in range(41)]
        with self.assertRaises(LlmError):
            complete(self.config, messages, self.deadline)
        self.assertEqual(self.gateway.requests, [])

    def test_bad_role_sends_nothing(self) -> None:
        with self.assertRaises(LlmError):
            complete(self.config, [{"role": "tool", "content": "x"}], self.deadline)
        self.assertEqual(self.gateway.requests, [])

    def test_output_tokens_ceiling(self) -> None:
        with self.assertRaises(LlmError):
            build_payload(self.config, [{"role": "user", "content": "стол"}], 5000)

    def test_payload_fits_gateway_limits(self) -> None:
        data = build_payload(self.config, [{"role": "user", "content": "стол"}])
        self.assertLessEqual(len(data), LLM_MAX_BODY_BYTES)

    def test_not_configured_sends_nothing(self) -> None:
        with self.assertRaises(LlmError) as caught:
            complete(LlmConfig(), [{"role": "user", "content": "стол"}], self.deadline)
        self.assertEqual(caught.exception.kind, "config")
        self.assertEqual(self.gateway.requests, [])

    def test_retry_after_parsing(self) -> None:
        self.assertEqual(_retry_after({"Retry-After": "3"}), 3.0)
        self.assertEqual(_retry_after({"retry-after": "1000"}), 30.0)
        self.assertEqual(_retry_after({"Retry-After": "скоро"}), 2.0)
        self.assertEqual(_retry_after({}), 2.0)


class CliModelPathTest(unittest.TestCase):
    """Сквозная проверка CLI с моделью: успех — exit 0, сбой сервиса — exit 5."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.gateway = FakeGateway().start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.gateway.stop()

    def setUp(self) -> None:
        self.gateway.reset()

    def run_cli(self, request: dict[str, Any], *, cache_dir: str) -> subprocess.CompletedProcess[bytes]:
        env = dict(os.environ)
        env.update(
            {
                "LLM_BASE_URL": self.gateway.base_url,
                "LLM_MODEL": MODEL,
                "LLM_API_KEY": API_KEY,
                "LLM_USER_AGENT": "team-02/1.0",
                "ASSISTANT_OFFLINE": "0",
                "ASSISTANT_CACHE_DIR": cache_dir,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONPATH": str(REPO_ROOT),
            }
        )
        return subprocess.run(
            [sys.executable, "-m", "app.main", "--catalog", str(REPO_ROOT / "data" / "catalog.json")],
            input=json.dumps(request, ensure_ascii=False).encode("utf-8"),
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            timeout=180,
        )

    def request(self) -> dict[str, Any]:
        # Город и бюджет названы, предмет — нет: без модели ответ был бы clarify,
        # а модель может назвать категорию (блок 9: гейт пропускает этот случай).
        return {"messages": [{"role": "user", "text": "Хочу обставить кабинет в Таганроге, тысяч на двенадцать"}]}

    def test_model_completes_the_request(self) -> None:
        self.gateway.enqueue(
            200,
            envelope({"city_id": "taganrog", "budget_rub": 15000, "categories": ["table", "chair", "lamp"]}),
        )
        with tempfile.TemporaryDirectory() as cache_dir:
            completed = self.run_cli(self.request(), cache_dir=cache_dir)
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))
        document = json.loads(completed.stdout.decode("utf-8"))
        self.assertEqual(document["status"], "ok", document)
        self.assertIn("prompt=", completed.stderr.decode("utf-8", "replace"))

    def test_service_failure_gives_nonzero_exit_and_no_stdout(self) -> None:
        self.gateway.enqueue(503, b'{"error":{"code":"unavailable","message":"try later"}}')
        with tempfile.TemporaryDirectory() as cache_dir:
            completed = self.run_cli(self.request(), cache_dir=cache_dir)
        self.assertEqual(completed.returncode, 5)
        self.assertEqual(completed.stdout.strip(), b"")
        self.assertIn("модель", completed.stderr.decode("utf-8", "replace"))
        self.assertNotIn(b"no_match", completed.stdout)


if __name__ == "__main__":
    unittest.main()
