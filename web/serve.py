"""Локальный сервер демо-сайта «ИИ-помощник покупателя».

Тонкая обёртка: логику подбора НЕ повторяет. Каждый запрос диалога уходит в
настоящий бэкенд отдельным процессом (`python -m app.main --catalog ...`,
stdin → stdout), а карточки товаров читаются из заданного каталога.

Запуск (из корня репозитория или откуда угодно):

    py -3.14 web/serve.py                  # http://127.0.0.1:8000
    py -3.14 web/serve.py --offline         # аварийный режим: модель не вызывается
    py -3.14 web/serve.py --port 0          # любой свободный порт

Зависимостей нет: только стандартная библиотека.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qs, urlsplit

WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
DEFAULT_CATALOG = ROOT / "data" / "catalog.json"

if not __package__:
    sys.path.insert(0, str(ROOT))  # запуск напрямую через python web/serve.py

from app.catalog import CatalogError, load_json_data, parse_catalog_source

#: Белый список статики: имя файла → Content-Type. Путь из URL никогда не
#: превращается в путь файловой системы, поэтому `..` и NUL-байты бесполезны.
STATIC_FILES: dict[str, str] = {
    "index.html": "text/html; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
}


def static_name(path: str) -> str:
    """Имя файла статики по URL.

    URL никогда не превращается в путь файловой системы напрямую: результат
    ищется в :data:`STATIC_FILES`, поэтому ``..``, абсолютные пути и NUL-байты
    просто не находят совпадения.
    """
    return "index.html" if path in ("/", "/index.html") else path.lstrip("/")


#: Демо не предназначено для поисковой индексации — закрываем краулерам.
ROBOTS_TXT = "User-agent: *\nDisallow: /\n"

MAX_BODY_BYTES = 65_536
BACKEND_TIMEOUT_SECONDS = 125
ROLES = ("user", "assistant")
#: Ровно как в контракте (``app/config.py::MAX_MESSAGES`` и
#: ``app/contract/request.py``). Дублируем значением, а не импортом: шлюз не
#: должен зависеть от внутренних модулей ядра, но обязан отвергать ровно то же.
MAX_MESSAGES = 20
REQUEST_KEYS = frozenset({"messages"})
MESSAGE_KEYS = frozenset({"role", "text"})


class BodyError(Exception):
    """Тело запроса неприемлемо: HTTP-код и причина для клиента.

    411 — нет ``Content-Length``; 413 — тело слишком большое; 400 — заголовок
    разобран, но некорректен. Отдельный тип нужен, чтобы различать эти случаи
    в :meth:`Handler.do_POST`, а не сваливать всё в один 400.
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message

#: Один вызов бэкенда за раз: у шлюза модели лимит 20 запросов в минуту,
#: и два одновременных клика не должны порождать два вызова.
BACKEND_LOCK = threading.Lock()


def log(message: str) -> None:
    """Логи сервера идут только в stderr, в браузер не попадают."""
    print(f"[web] {message}", file=sys.stderr, flush=True)


def header_safe(value: str) -> str:
    """HTTP-заголовки кодируются latin-1: кириллицу и прочее заменяем вопросами."""
    return str(value).encode("latin-1", "replace").decode("latin-1")


def server_headers(info: dict[str, Any]) -> dict[str, str]:
    """Общие заголовки безопасности и состояния, единые для всех ответов."""
    headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Content-Security-Policy": "default-src 'self'; object-src 'none'; frame-ancestors 'none'",
        "X-Catalog": "ok" if info.get("catalog_ok") else "bad",
    }
    if info.get("catalog_error"):
        headers["X-Catalog-Error"] = header_safe(str(info["catalog_error"])[:200])
    return headers


# --- каталог: только чтение, только белый список полей -----------------------


class Catalog:
    """Чтение карточек из файла или URL каталога.

    Отдаёт наружу ТОЛЬКО безопасные для показа поля. ``description`` сюда не
    входит сознательно: по условиям кейса описания не участвуют в подборе и в
    браузер не отдаются вовсе (даже если они есть в файле).
    """

    def __init__(self, source: Path | str) -> None:
        self.source = source
        self._lock = threading.Lock()
        self._stamp: tuple[int, int] | None = None
        self._by_id: dict[str, dict[str, Any]] = {}
        self._cities: dict[str, str] = {}
        self._categories: dict[str, str] = {}

    def _reload_if_changed(self) -> None:
        stamp = None
        if isinstance(self.source, Path):
            try:
                stat = self.source.stat()
            except OSError as exc:
                raise RuntimeError(f"каталог недоступен: {self.source} ({exc})") from exc
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp == self._stamp:
                return

        try:
            document = load_json_data(self.source)
        except CatalogError as exc:
            raise RuntimeError(f"каталог не читается: {exc}") from exc
        if not isinstance(document, list):
            if isinstance(document, dict):
                document = document.get("listings", document.get("products"))
        if not isinstance(document, list):
            raise RuntimeError("каталог должен быть JSON-массивом объявлений")

        by_id: dict[str, dict[str, Any]] = {}
        for raw in document:
            card = self._card(raw)
            if card is not None:
                by_id[card["id"]] = card
        self._by_id = by_id
        self._cities = self._named(ROOT / "data" / "cities.json")
        self._categories = self._named(ROOT / "data" / "categories.json")
        self._stamp = stamp

    @staticmethod
    def _named(path: Path) -> dict[str, str]:
        """Справочник id → русское название; при отсутствии файла — пустой."""
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, list):
            return {}
        result: dict[str, str] = {}
        for item in raw:
            if isinstance(item, dict) and isinstance(item.get("id"), str) and isinstance(item.get("name"), str):
                result[item["id"]] = item["name"]
        return result

    @staticmethod
    def _card(raw: Any) -> dict[str, Any] | None:
        """Карточка для браузера: поля из белого списка, без ``description``."""
        if not isinstance(raw, dict):
            return None
        listing_id = raw.get("id")
        if not isinstance(listing_id, str) or not listing_id:
            return None
        attributes = raw.get("attributes")
        title = raw.get("title", raw.get("name"))
        price = raw.get("price_rub", raw.get("price"))
        city = raw.get("city_id", raw.get("city"))
        city_name = raw.get("city_name", raw.get("city"))
        return {
            "id": listing_id,
            "category": raw.get("category") if isinstance(raw.get("category"), str) else "",
            "title": title if isinstance(title, str) else "",
            "price_rub": price if type(price) is int else 0,
            "city_id": city if isinstance(city, str) else "",
            "city_name": city_name if isinstance(city_name, str) else "",
            "attributes": attributes if isinstance(attributes, dict) else {},
        }

    def find(self, ids: Sequence[str]) -> tuple[list[dict[str, Any]], list[str]]:
        """Карточки по списку id; порядок сохраняется, неизвестные попадают в ``missing``."""
        with self._lock:
            self._reload_if_changed()
            by_id, cities, categories = self._by_id, self._cities, self._categories
        items: list[dict[str, Any]] = []
        missing: list[str] = []
        for listing_id in ids:
            card = by_id.get(listing_id)
            if card is None:
                missing.append(listing_id)
                continue
            enriched = dict(card)
            enriched["category_name"] = categories.get(card["category"], card["category"])
            enriched["city_name"] = cities.get(card["city_id"], card["city_name"] or card["city_id"])
            items.append(enriched)
        return items, missing


# --- вызов настоящего бэкенда -----------------------------------------------


def backend_env(offline: bool) -> dict[str, str]:
    """Окружение процесса бэкенда.

    ``--offline`` только ДОБАВЛЯЕТ ``ASSISTANT_OFFLINE=1``: если оператор сам
    выставил эту переменную, сервер обязан её уважать и не перетирать.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if offline:
        env["ASSISTANT_OFFLINE"] = "1"
    return env


def last_stderr_line(raw: bytes) -> str:
    """Последняя непустая строка логов бэкенда — причина сбоя для браузера."""
    text = raw.decode("utf-8", "replace")
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped[:300]
    return "бэкенд завершился с ошибкой и без сообщения"


#: Строка логов модели: ``модель: prompt=… completion=… total=…`` (app/llm/client.py).
TOKENS_PATTERN = re.compile(r"модель:\s*prompt=(\d+)\s+completion=(\d+)\s+total=(\d+)")
#: Итоговая строка ядра: ``готово: status=ok за 12 мс, всего 34 мс`` (app/main.py).
ELAPSED_PATTERN = re.compile(r"всего\s+(\d+)\s+мс")
NO_MODEL_MARKER = "модель: недоступна"


def backend_metrics(stderr: bytes) -> dict[str, Any]:
    """Метрики одного запуска бэкенда из его stderr.

    Токены берём из строки ``модель: prompt=… total=…``, которую печатает
    ``app/llm/client.py``. Если строки нет — модель не вызывалась: либо её нет,
    либо задача решена лексиконом. Это позволяет интерфейсу честно показать
    «модель не вызывалась», а не выдумать число.

    Отдельно отмечаем ``model_unavailable``: бэкенд сообщил, что модель
    недоступна. Это НЕ то же самое, что режим сервера ``--offline``: строка
    появляется и при отсутствующем ключе, поэтому режим ответа берётся из флага
    сервера (см. ``_metrics_headers``), а не отсюда.
    """
    text = stderr.decode("utf-8", "replace")
    usages = list(TOKENS_PATTERN.finditer(text))
    tokens = sum(int(match.group(3)) for match in usages)
    elapsed_match = ELAPSED_PATTERN.search(text)
    return {
        "model_calls": len(usages),
        "tokens": tokens,
        "elapsed_ms": int(elapsed_match.group(1)) if elapsed_match else 0,
        "model_unavailable": NO_MODEL_MARKER in text,
    }


def call_backend(
    payload: dict[str, Any], catalog_path: Path | str, env: dict[str, str], *, text_search: bool = False,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """CORE. Один запрос диалога → ответ бэкенда как есть.

    Вход:  payload      — ``{"messages": [...]}`` из тела POST-запроса;
           catalog_path — путь к каталогу (по умолчанию репозиторный);
           env          — окружение процесса бэкенда (см. :func:`backend_env`).
    Выход: (HTTP-код, документ, метрики). 200 — распарсенный stdout бэкенда
           БАЙТ-В-БАЙТ как объект; иначе ``{"error": ...}`` с последней строкой
           stderr. ``metrics`` уходит в заголовки ответа и в тело не попадает,
           поэтому контракт трёх статусов не меняется.
    Крайние случаи: таймаут 125 с → 504; ненулевой код → 502; stdout не JSON → 502.
    """
    command = [
        sys.executable,
        "-m",
        "app.main",
        "--catalog",
        str(catalog_path),
    ]
    if text_search:
        command.append("--text-search")
    try:
        with BACKEND_LOCK:
            completed = subprocess.run(
                command,
                input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                capture_output=True,
                cwd=str(ROOT),
                timeout=BACKEND_TIMEOUT_SECONDS,
                env=env,
            )
    except subprocess.TimeoutExpired:
        return 504, {"error": f"бэкенд не ответил за {BACKEND_TIMEOUT_SECONDS} с"}, {}
    except OSError as exc:
        return 500, {"error": f"не удалось запустить бэкенд: {exc}"}, {}

    metrics = backend_metrics(completed.stderr)
    if completed.returncode != 0:
        reason = last_stderr_line(completed.stderr)
        log(f"бэкенд вернул код {completed.returncode}: {reason}")
        return 502, {"error": reason}, metrics

    try:
        document = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        log(f"stdout бэкенда не разобран: {exc}")
        return 502, {"error": f"ответ бэкенда не является JSON: {exc}"}, metrics
    if not isinstance(document, dict):
        return 502, {"error": "ответ бэкенда не является JSON-объектом"}, metrics
    return 200, document, metrics


def validate_request_body(raw: bytes) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Проверяет тело POST-запроса ровно по правилам контракта.

    Это проверка ФОРМЫ запроса, а не логики подбора: JSON, набор полей верхнего
    уровня и у каждого сообщения, 1..20 реплик, роли ``user``/``assistant``,
    непустой ``text``, первая и последняя реплики — от ``user``, роли чередуются.
    Правила сознательно повторяют ``app/contract/request.py``: если бы шлюз
    пропускал то, что ядро отвергает, клиент получал бы 502 «сервис сломан»
    вместо честного 400 «неверный запрос». Поэтому 502 остаётся только для
    настоящего сбоя бэкенда.
    """
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return None, {"error": f"тело запроса не в UTF-8: {exc.reason}"}
    if not text.strip():
        return None, {"error": "пустое тело: ожидался JSON-объект с полем messages"}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, {"error": f"тело запроса не является JSON: {exc.msg} (строка {exc.lineno})"}

    if not isinstance(payload, dict):
        return None, {"error": "тело запроса должно быть JSON-объектом"}
    extra_keys = set(payload) - REQUEST_KEYS
    if extra_keys:
        return None, {"error": f"лишние поля запроса: {sorted(extra_keys)}"}
    if "messages" not in payload:
        return None, {"error": "отсутствует обязательное поле messages"}

    messages = payload["messages"]
    if not isinstance(messages, list):
        return None, {"error": "поле messages должно быть массивом"}
    if not 1 <= len(messages) <= MAX_MESSAGES:
        return None, {"error": f"messages: ожидалось 1..{MAX_MESSAGES} сообщений, получено {len(messages)}"}

    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            return None, {"error": f"сообщение #{index}: ожидался объект"}
        extra = set(message) - MESSAGE_KEYS
        if extra:
            return None, {"error": f"сообщение #{index}: лишние поля {sorted(extra)}"}
        missing = MESSAGE_KEYS - set(message)
        if missing:
            return None, {"error": f"сообщение #{index}: нет полей {sorted(missing)}"}
        role = message["role"]
        text_value = message["text"]
        if role not in ROLES:
            return None, {"error": f"сообщение #{index}: роль должна быть user или assistant"}
        if not isinstance(text_value, str) or not text_value.strip():
            return None, {"error": f"сообщение #{index}: text должен быть непустой строкой"}

    if messages[0]["role"] != "user":
        return None, {"error": "первое сообщение должно быть от user"}
    if messages[-1]["role"] != "user":
        return None, {"error": "последнее сообщение должно быть от user"}
    for index in range(1, len(messages)):
        if messages[index]["role"] == messages[index - 1]["role"]:
            role = messages[index]["role"]
            return None, {"error": f"сообщение #{index + 1}: роли должны чередоваться (повтор {role})"}
    return payload, None


# --- HTTP -------------------------------------------------------------------


class Api:
    """Состояние сервера: каталог, путь к нему и режим офлайна."""

    def __init__(self, catalog: Catalog, catalog_path: Path | str, offline: bool, *, text_search: bool = False) -> None:
        self.catalog = catalog
        self.catalog_path = catalog_path
        self.offline = offline
        self.text_search = text_search
        self.env = backend_env(offline)
        self.catalog_ok = True
        self.catalog_error: str | None = None

    def server_info(self) -> dict[str, Any]:
        """Состояние сервера для заголовков ответа: каталог и режим модели."""
        return {
            "catalog_ok": self.catalog_ok,
            "catalog_error": self.catalog_error,
            "offline": self.offline,
        }

    def listings(self, query: dict[str, list[str]]) -> tuple[int, dict[str, Any]]:
        """``GET /api/listings?ids=a,b,c`` — карточки товаров без описаний."""
        raw_ids = query.get("ids")
        if not raw_ids:
            return 400, {"error": "укажите параметр ids, например /api/listings?ids=demo-1-1-11,demo-1-2-05"}
        ids = [part.strip() for value in raw_ids for part in value.split(",")]
        ids = [listing_id for listing_id in ids if listing_id]
        if not ids:
            return 400, {"error": "параметр ids не содержит ни одного идентификатора"}
        if len(ids) > 50:
            return 400, {"error": "за один запрос можно запросить не больше 50 объявлений"}
        try:
            items, missing = self.catalog.find(ids)
        except RuntimeError as exc:
            return 500, {"error": str(exc)}
        return 200, {"total": len(items), "items": items, "missing": missing}

    def answer(self, raw_body: bytes) -> tuple[int, dict[str, Any], dict[str, Any]]:
        """``POST /api/answer`` — вызов бэкенда отдельным процессом."""
        payload, error = validate_request_body(raw_body)
        if error is not None:
            return 400, error, {}
        assert payload is not None
        return call_backend(payload, self.catalog_path, self.env, text_search=self.text_search)


def make_handler(api: Api) -> type[BaseHTTPRequestHandler]:
    """Обработчик, связанный с конкретным состоянием сервера."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "buyer-assistant-web/1.0"
        sys_version = ""
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802 — имя задано базовым классом
            self._handle_get(head=False)

        def do_HEAD(self) -> None:  # noqa: N802
            self._handle_get(head=True)

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path != "/api/answer":
                if path == "/api/listings" or static_name(path) in STATIC_FILES:
                    self._allowed("POST", "GET, HEAD")
                elif path.startswith("/api/"):
                    self._json(404, {"error": f"неизвестный эндпоинт {path}"})
                else:
                    self._text(404, "404: файл не найден\n", head=False)
                return
            try:
                body = self._read_body()
            except BodyError as exc:
                self._json(exc.status, {"error": exc.message})
                return
            status, payload, metrics = api.answer(body)
            self._json(status, payload, headers=self._metrics_headers(metrics))

        def do_PUT(self) -> None:  # noqa: N802
            self._allowed("PUT", "GET, HEAD, POST")

        def do_DELETE(self) -> None:  # noqa: N802
            self._allowed("DELETE", "GET, HEAD, POST")

        # --- GET -----------------------------------------------------------
        def _handle_get(self, *, head: bool) -> None:
            parts = urlsplit(self.path)
            path = parts.path
            try:
                if path == "/api/listings":
                    self._json(*api.listings(parse_qs(parts.query, keep_blank_values=True)), head=head)
                    return
                if path == "/api/answer":
                    self._allowed("GET", "POST", head=head)
                    return
                if path.startswith("/api/"):
                    self._json(404, {"error": f"неизвестный эндпоинт {path}"}, head=head)
                    return
                if path == "/favicon.ico":
                    # Файла иконки нет, и 404 здесь только шумит в логах. 204
                    # корректен: «есть, но содержимого нет» — браузер молчит.
                    self._no_content()
                    return
                if path == "/robots.txt":
                    # Явный запрет индексации: без файла краулер получает 404.
                    self._send(200, ROBOTS_TXT.encode("utf-8"), "text/plain; charset=utf-8",
                               head=head, cache="no-cache")
                    return
                self._static(path, head=head)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _read_body(self) -> bytes:
            """Читает тело POST-запроса с явными кодами ошибок.

            411 — нет ``Content-Length``; 413 — тело больше лимита (до 1 МБ
            дочитываем и выбрасываем, чтобы отдать внятный ответ; свыше —
            рвём соединение); 400 — заголовок есть, но бессмысленный.
            """
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise BodyError(411, "требуется заголовок Content-Length")
            try:
                length = int(raw_length)
            except ValueError:
                self.close_connection = True
                raise BodyError(400, "некорректный Content-Length") from None
            if length < 0:
                self.close_connection = True
                raise BodyError(400, "отрицательный Content-Length")
            if length > MAX_BODY_BYTES:
                if length > 1_048_576:
                    self.close_connection = True
                else:
                    self.rfile.read(length)  # дочитываем, чтобы клиент получил 413
                raise BodyError(413, f"тело больше {MAX_BODY_BYTES} байт")
            return self.rfile.read(length)

        # --- статика -------------------------------------------------------
        def _static(self, path: str, *, head: bool) -> None:
            name = static_name(path)
            content_type = STATIC_FILES.get(name)
            if content_type is None:
                self._text(404, "404: файл не найден\n", head=head)
                return
            try:
                body = (WEB_DIR / name).read_bytes()
            except OSError as exc:
                log(f"не удалось прочитать {name}: {exc}")
                self._text(500, "500: статика недоступна\n", head=head)
                return
            self._send(200, body, content_type, head=head, cache="no-cache")

        # --- ответы --------------------------------------------------------
        def _metrics_headers(self, metrics: dict[str, Any]) -> dict[str, str]:
            """Метрики вызова модели и время ответа — в заголовках, не в теле.

            Тело остаётся ровно документом контракта (три статуса,
            ``additionalProperties: false``), а браузер получает честные цифры:
            сколько вызвано моделей, сколько токенов и сколько занял ответ.
            Общие заголовки добавляет ``_send``.
            """
            if not metrics:
                return {}
            # Режим отвечает сервер, а не stderr бэкенда: строка «модель:
            # недоступна» появляется и при отсутствующем ключе, поэтому по ней
            # нельзя отличить --offline от обычного запуска. Иначе демо врало бы.
            mode = "offline" if api.offline else "on-demand"
            headers = {
                "X-Model-Calls": str(metrics.get("model_calls", 0)),
                "X-Model-Tokens": str(metrics.get("tokens", 0)),
                "X-Elapsed-Ms": str(metrics.get("elapsed_ms", 0)),
                "X-Model-Mode": mode,
            }
            if metrics.get("model_unavailable"):
                headers["X-Model-Unavailable"] = "1"
            return headers

        def _allowed(self, method: str, allow: str, *, head: bool = False) -> None:
            self._json(
                405,
                {"error": f"метод {method} не поддерживается"},
                head=head,
                headers={"Allow": allow},
            )

        def _json(
            self,
            status: int,
            payload: dict[str, Any],
            *,
            head: bool = False,
            headers: dict[str, str] | None = None,
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8", head=head, cache="no-store", headers=headers)

        def _text(self, status: int, text: str, *, head: bool) -> None:
            self._send(status, text.encode("utf-8"), "text/plain; charset=utf-8", head=head, cache="no-store")

        def _no_content(self) -> None:
            """204 без тела: корректный ответ на отсутствующий favicon.ico."""
            self.send_response(204)
            for name, value in server_headers(api.server_info()).items():
                self.send_header(name, value)
            self.end_headers()

        def _send(
            self,
            status: int,
            body: bytes,
            content_type: str,
            *,
            head: bool,
            cache: str,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            for name, value in server_headers(api.server_info()).items():
                self.send_header(name, value)
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            if not head and body:
                self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — сигнатура базового класса
            log(f"{self.address_string()} {format % args}")

    return Handler


class WebServer(ThreadingHTTPServer):
    """Потоковый сервер; обрыв соединения клиентом — не сбой сервера."""

    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        error = sys.exc_info()[1]
        if isinstance(error, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            log(f"клиент {client_address[0]} закрыл соединение")
            return
        super().handle_error(request, client_address)


def make_server(api: Api, host: str = "127.0.0.1", port: int = 8000) -> WebServer:
    """Создаёт сервер (не запускает): ``port=0`` выбирает свободный порт."""
    return WebServer((host, port), make_handler(api), api)


def build_api(catalog_path: Path | str, offline: bool, *, text_search: bool = False) -> Api:
    """Собирает состояние сервера.

    Недоступный или битый каталог НЕ мешает старту: сервер страны обязан
    подняться, чтобы отдать браузеру понятную ошибку, а карточки в таком случае
    не отдаются (``X-Catalog: bad``). Отсутствие файла видно и в ``/api/listings``
    (500), и при вызове бэкенда (502 от ``app.main``).
    """
    catalog = Catalog(catalog_path)
    api = Api(catalog, catalog_path, offline, text_search=text_search)
    try:
        catalog._reload_if_changed()
    except RuntimeError as exc:
        api.catalog_ok = False
        api.catalog_error = str(exc)
        log(str(exc))
    return api


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="web/serve.py", description="Демо-сайт «ИИ-помощник покупателя».")
    parser.add_argument("--host", default="127.0.0.1", help="адрес прослушивания (по умолчанию только loopback)")
    parser.add_argument("--port", type=int, default=8000, help="порт; 0 — любой свободный")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG), help="путь или HTTP(S)-URL каталога (по умолчанию data/catalog.json)")
    parser.add_argument("--offline", action="store_true", help="выставить ASSISTANT_OFFLINE=1 процессу бэкенда")
    parser.add_argument("--text-search", action="store_true", help="искать по названиям товаров без таксономии")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Запуск сервера: печатает адрес и работает до Ctrl+C."""
    for stream in (sys.stderr,):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    args = build_arg_parser().parse_args(argv)
    catalog_path = parse_catalog_source(args.catalog)
    if isinstance(catalog_path, Path):
        catalog_path = catalog_path.resolve()

    api = build_api(catalog_path, args.offline, text_search=args.text_search)

    try:
        server = make_server(api, args.host, args.port)
    except OSError as exc:
        log(f"не удалось занять {args.host}:{args.port}: {exc}")
        return 3

    host, port = server.server_address[:2]
    catalog_label = str(catalog_path) if isinstance(catalog_path, Path) else "по URL"
    log(f"каталог: {catalog_label} ({len(api.catalog._by_id)} объявлений)")
    log(f"режим: {'офлайн (модель не вызывается)' if api.offline else 'обычный (модель по необходимости)'}")
    log(f"сайт: http://{host}:{port}/ — Ctrl+C для остановки")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        log("остановка по Ctrl+C")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
