"""Один вызов шлюза модели: лимиты LLM_API.md, без циклов автоматических ретраев.

Почему клиент такой строгий:

* **Лимиты проверяются до отправки.** Вход ограничен в байтах UTF-8, полный
  запрос — 65 536 байт, выход — в токенах. Нарушение на шлюзе стоит квоты и
  времени, поэтому размер считается локально.
* **Никаких ретраев, кроме одного шанса на 429.** Шлюз не повторяет генерации,
  а каждая попытка расходует квоту 20 rpm. Для 429 выдерживается ``Retry-After``
  (если пауза не помещается в бюджет запуска — ошибка сразу).
* **Явный User-Agent.** ``workers.dev`` отвечает ``403 error code: 1010``
  обычным текстом на клиентов вроде ``Python-urllib/3.12``; такой ответ
  распознаётся и объясняется, а не превращается в «нет вариантов».
* **Редиректы не выполняются.** Ключ не должен уехать на чужой хост.
* **Ошибка модели — техническая.** Она поднимается наружу (:class:`LlmError`),
  логируется в stderr и завершает запуск ненулевым кодом; ``no_match``
  для этого не используется никогда (CONTRACT.md §1).

Модель в этом модуле — только извлечение условий. Подбор комплектов, цены и
объяснения считаются локально и токенов не тратят.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from app.config import (
    LLM_CALL_TIMEOUT_SECONDS,
    LLM_MAX_BODY_BYTES,
    LLM_MAX_INPUT_BYTES,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_RATE_LIMIT_DEFAULT_SLEEP_SECONDS,
    LLM_RATE_LIMIT_MAX_RETRIES,
    MAX_PRICE_RUB,
    LlmConfig,
)
from app.jsonio import JsonError, json_from, loads_strict
from app.model import CATEGORIES, Slots
from app.nlp.extract import Resolver
from app.prompt_safety import CATEGORY_SCHEMA, EXTRACTION_SCHEMA, EXTRACTION_SCHEMA_NAME, SYSTEM_PROMPT

#: Лимиты LLM_API.md, которые не заданы в конфигурации, но обязательны.
MAX_MESSAGES = 40
MAX_OUTPUT_TOKENS = 2_000
MAX_SCHEMA_BYTES = 16_384
MAX_RESPONSE_BYTES = 262_144

#: Роли сообщений, разрешённые шлюзом.
ALLOWED_ROLES = frozenset({"system", "user", "assistant"})

#: Минимальный остаток времени, при котором вызов ещё имеет смысл.
MIN_TIMEOUT_SECONDS = 0.5

#: Сколько символов тела ошибки показывать в stderr.
ERROR_EXCERPT_CHARS = 200

REDACTED = "[REDACTED]"
_LOG_PREFIX = "[buyer-assistant]"

#: Размер схемы генерации в компактной записи — считается один раз.
_SCHEMA_BYTES = len(
    json.dumps(EXTRACTION_SCHEMA, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
)


class LlmError(RuntimeError):
    """Технический сбой модели: stderr + ненулевой код, никогда не ``no_match``.

    ``status`` — HTTP-код, если ответ пришёл; ``kind`` — короткая категория
    причины для диагностики (``transport``, ``deadline``, ``request_too_large``,
    ``rate_limited``, ``access``, ``non_json``, ``model_output``, ``truncated``,
    ``config``). Текст сообщения уже очищен от ключа доступа.
    """

    def __init__(self, message: str, *, status: int | None = None, kind: str = "transport") -> None:
        super().__init__(message)
        self.status = status
        self.kind = kind


#: Opener без редиректов; создаётся при первом вызове модели (см. :func:`_build_opener`).
_OPENER: Any = None


def _build_opener() -> Any:
    """Opener urllib, запрещающий редиректы; импорт — при первом вызове модели.

    ``urllib.request`` стоит ≈53 мс старта процесса, а нужен только когда модель
    действительно вызывается (38 % запросов на стенде). Импорт внутри функции
    переносит эту цену на платный путь и не влияет на быстрый.
    """
    global _OPENER
    if _OPENER is None:
        import urllib.request

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            """Запрещает автоматические редиректы: ключ не должен уйти на чужой хост."""

            def redirect_request(self, request, fp, code, message, headers, newurl):  # noqa: ANN001, D401
                return None

        _OPENER = urllib.request.build_opener(_NoRedirect)
    return _OPENER


def build_payload(
    config: LlmConfig,
    messages: Sequence[Mapping[str, str]],
    max_tokens: int = LLM_MAX_OUTPUT_TOKENS,
    *,
    schema: Mapping[str, Any] = EXTRACTION_SCHEMA,
    schema_name: str = EXTRACTION_SCHEMA_NAME,
) -> bytes:
    """CORE. Собирает тело запроса и проверяет лимиты шлюза ДО отправки.

    Зачем: лимиты LLM_API.md (байты входа, байты полного JSON, токены выхода,
    число сообщений, размер схемы) считаются на стороне шлюза, но нарушение
    стоит квоты и времени запуска. Локальная проверка превращает «дорогую
    ошибку 400» в понятную техническую ошибку до сети.

    Вход:  config     — адрес, модель, ключ, User-Agent (ключ в тело не попадает);
           messages   — сообщения Chat Completions (``role`` + ``content``);
           max_tokens — потолок выхода, 1..2000 (для извлечения хватает 200).
    Выход: байты JSON-тела, готовые к ``POST {base_url}/chat/completions``.

    Алгоритм: валидация ролей и непустых текстов → сумма байтов всех ``content``
    (<= 32 768) → сборка тела с ``response_format: json_schema`` (``strict``) →
    проверка полного размера (<= 65 536) и размера схемы (<= 16 384).
    Сложность: время O(суммарная длина сообщений), память O(тело запроса).
    Почему именно он: тело собирается в одном месте, поэтому «в модель случайно
    уехал каталог» ловится не ревью, а тестом на состав сообщений.
    Крайние случаи: пустой список сообщений, роль ``tool``, ``content`` не строка,
    суммарный вход больше 32 КБ, тело больше 65 КБ — всё это LlmError до сети.
    Вызывается из: app/llm/client.py::complete, тесты.
    """
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= MAX_OUTPUT_TOKENS:
        raise LlmError(f"max_tokens вне диапазона 1..{MAX_OUTPUT_TOKENS}: {max_tokens!r}", kind="request_too_large")
    if not 1 <= len(messages) <= MAX_MESSAGES:
        raise LlmError(f"сообщений должно быть 1..{MAX_MESSAGES}, получено {len(messages)}", kind="request_too_large")
    schema_bytes = _SCHEMA_BYTES if schema is EXTRACTION_SCHEMA else len(
        json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    if schema_bytes > MAX_SCHEMA_BYTES:
        raise LlmError(f"схема генерации больше {MAX_SCHEMA_BYTES} байт", kind="request_too_large")

    normalized: list[dict[str, str]] = []
    input_bytes = 0
    for position, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            raise LlmError(f"сообщение #{position}: недопустимая роль {role!r}", kind="request_too_large")
        if not isinstance(content, str) or not content:
            raise LlmError(f"сообщение #{position}: content должен быть непустой строкой", kind="request_too_large")
        input_bytes += len(content.encode("utf-8"))
        normalized.append({"role": str(role), "content": content})

    if input_bytes > LLM_MAX_INPUT_BYTES:
        raise LlmError(
            f"тексты сообщений {input_bytes} байт > предела {LLM_MAX_INPUT_BYTES}",
            kind="request_too_large",
        )

    body = {
        "model": config.model,
        "messages": normalized,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
    }
    data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(data) > LLM_MAX_BODY_BYTES:
        raise LlmError(
            f"тело запроса {len(data)} байт > предела {LLM_MAX_BODY_BYTES}",
            kind="request_too_large",
        )
    return data


def complete(
    config: LlmConfig,
    messages: Sequence[Mapping[str, str]],
    deadline: float,
    *,
    max_tokens: int = LLM_MAX_OUTPUT_TOKENS,
    sleep: Callable[[float], None] = time.sleep,
    schema: Mapping[str, Any] = EXTRACTION_SCHEMA,
    schema_name: str = EXTRACTION_SCHEMA_NAME,
    sanitize: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """CORE. Один вызов модели: запрос, разбор оболочки, JSON модели, лимит времени.

    Зачем: весь сетевой риск собран в одной функции, чтобы остальная программа
    оставалась детерминированной. Здесь же решается, что считать технической
    ошибкой (всё, кроме успешного валидного ответа) и сколько раз пробовать.

    Вход:  config    — доступ к шлюзу (URL, модель, ключ, User-Agent);
           messages  — сообщения (см. :func:`build_payload`);
           deadline  — монотонное время, после которого запуск обязан кончиться;
           max_tokens— потолок выхода; ``sleep`` — пауза для 429 (подменяется в тестах).
    Выход: словарь условий, прошедший белый список полей (``city_id``,
           ``budget_rub``, ``categories``); пустой словарь — модель ничего не нашла.

    Алгоритм: проверка настроек → сборка тела с лимитами → ``POST`` с таймаутом
    ``min(90 с, остаток до дедлайна)`` → при 429 одна пауза по ``Retry-After``
    (если она помещается в бюджет) и одна повторная попытка → разбор оболочки
    Chat Completions → строгий разбор ``choices[0].message.content`` как JSON →
    фильтрация полей. Любой другой исход — :class:`LlmError`.
    Сложность: время O(сеть + длина ответа), память O(ответ); токенов — ровно
    один короткий обмен, независимо от размера каталога.
    Почему именно он: контракт требует ненулевой код при сбое сервиса и не
    разрешает превращать его в ``no_match``; единственный способ гарантировать
    это — не глушить ошибки и не ретраить бесконечно.
    Крайние случаи: ключ/адрес не заданы (``config``); пустой остаток бюджета
    (``deadline``); ответ без ``choices``; ``content`` не JSON или не объект
    (``model_output``); 429 дважды (``rate_limited``); 401/403 (``access``);
    не-JSON тело ошибки (``non_json``, включая ``error code: 1010``).
    Вызывается из: app/llm/client.py::make_resolver, тесты, bench/load.py.
    """
    if not config.configured:
        raise LlmError(
            "модель не настроена: нужны LLM_BASE_URL, LLM_MODEL и LLM_API_KEY",
            kind="config",
        )

    payload = build_payload(config, messages, max_tokens, schema=schema, schema_name=schema_name)
    url = f"{config.base_url}/chat/completions"
    headers = _headers(config)

    attempts = 0
    while True:
        attempts += 1
        timeout = _timeout_left(deadline)
        if attempts == 1:
            print(f"{_LOG_PREFIX} модель: запрос к шлюзу", file=sys.stderr, flush=True)
        status, response_headers, body = _http_post(url, payload, headers, timeout)

        if status == 200:
            envelope = _parse_envelope(body, config.api_key)
            _log_usage(envelope)
            return _extraction_content(envelope, sanitize or _sanitize_slots)

        error = _error_from_response(status, body, config.api_key, attempts)
        if status == 429 and attempts <= LLM_RATE_LIMIT_MAX_RETRIES:
            pause = _retry_after(response_headers)
            remaining = deadline - time.monotonic()
            if remaining <= pause + MIN_TIMEOUT_SECONDS:
                raise error
            print(
                f"{_LOG_PREFIX} модель занята (429), пауза {pause:.1f} с по Retry-After, "
                f"попытка {attempts + 1} из {LLM_RATE_LIMIT_MAX_RETRIES + 1}",
                file=sys.stderr,
                flush=True,
            )
            sleep(pause)
            continue
        raise error


def make_resolver(
    config: LlmConfig,
    deadline: float,
    *,
    complete_fn: Callable[..., dict[str, Any]] = complete,
) -> Resolver:
    """Заполняет недостающие категории по фразе покупателя."""
    system_message = {"role": "system", "content": SYSTEM_PROMPT}

    def resolve(text: str, slots: Slots) -> dict[str, Any]:  # noqa: ARG001 — подпись Resolver
        return complete_fn(
            config,
            [system_message, {"role": "user", "content": text}],
            deadline,
            schema=CATEGORY_SCHEMA,
            schema_name="buyer_categories",
            sanitize=_sanitize_categories,
        )

    resolve.__doc__ = "Один вызов модели для одной реплики покупателя (§5)."
    return resolve


def redact(text: str, api_key: str) -> str:
    """Убирает ключ доступа из любого текста, который может попасть в stderr."""
    if api_key:
        text = text.replace(api_key, REDACTED)
    return text


def _headers(config: LlmConfig) -> dict[str, str]:
    """Заголовки запроса: явный User-Agent обязателен для ``workers.dev``."""
    return {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "User-Agent": config.user_agent or "hackathon-client/1.0",
    }


def _timeout_left(deadline: float) -> float:
    """Таймаут HTTP: не больше 90 с и не больше остатка бюджета запуска."""
    remaining = deadline - time.monotonic()
    if remaining <= MIN_TIMEOUT_SECONDS:
        raise LlmError(
            f"не хватает времени на вызов модели (осталось {remaining:.1f} с)",
            kind="deadline",
        )
    return min(LLM_CALL_TIMEOUT_SECONDS, remaining)


def _http_post(
    url: str,
    payload: bytes,
    headers: Mapping[str, str],
    timeout: float,
) -> tuple[int, Mapping[str, str], bytes]:
    """Один POST без редиректов. Возвращает (код, заголовки, тело) для любого ответа.

    Сетевые сбои (DNS, TLS, таймаут, обрыв) — :class:`LlmError` с ``transport``:
    повтор запрещён, потому что судьба исходного вызова неизвестна.
    ``urllib`` импортируется здесь, а не на уровне модуля: это ≈53 мс старта,
    которые не нужны запросам без модели.
    """
    import socket
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, data=payload, headers=dict(headers), method="POST")
    try:
        with _build_opener().open(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        try:
            body = error.read(MAX_RESPONSE_BYTES + 1)
        except OSError:
            body = b""
        return error.code, dict(error.headers.items()) if error.headers else {}, body
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise LlmError(f"нет связи со шлюзом модели: {exc}", kind="transport") from exc
    except OSError as exc:
        raise LlmError(f"сбой ввода-вывода при вызове модели: {exc}", kind="transport") from exc


def _parse_envelope(body: bytes, api_key: str) -> dict[str, Any]:
    """Разбирает оболочку Chat Completions; не-JSON ответ объясняется."""
    if len(body) > MAX_RESPONSE_BYTES:
        raise LlmError(f"ответ модели больше {MAX_RESPONSE_BYTES} байт", kind="protocol")
    text = redact(body.decode("utf-8", errors="replace"), api_key)
    try:
        envelope = loads_strict(text, source="ответ модели")
    except JsonError as exc:
        hint = ""
        if "1010" in text:
            hint = " (похоже на блокировку клиента: проверьте User-Agent)"
        raise LlmError(f"не-JSON ответ шлюза: {exc}{hint}: {_excerpt(text)}", kind="non_json") from exc
    if not isinstance(envelope, dict):
        raise LlmError("ответ шлюза не является объектом", kind="non_json")
    return envelope


def _extraction_content(
    envelope: Mapping[str, Any],
    sanitize: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Достаёт JSON модели из ``choices[0].message.content`` и фильтрует поля.

    Разбор терпим к обрамлению (markdown-забор, проза до/после, служебные
    токены) — см. :func:`app.jsonio.json_from` — но не к самому JSON: NaN,
    дубли ключей и массив вместо объекта остаются ошибкой. Если объектов
    несколько, берётся первый, из которого после белого списка осталось хоть
    одно поле: ``{"error":"x"}{"city_id":…}`` не должен превращаться в пустой
    набор условий.

    Обрезка по лимиту выхода и «модель промолчала» различаются: ``finish_reason``
    ``length`` — это ``truncated`` (лечится увеличением ``max_tokens``), а
    остальное — ``model_output`` (модель не вернула объект).
    """
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlmError("в ответе модели нет choices", kind="protocol")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    finish_reason = first.get("finish_reason") if isinstance(first, dict) else None
    sanitize = sanitize or _sanitize_slots
    if not isinstance(content, str) or not content.strip():
        raise LlmError("в ответе модели нет текста message.content", kind="protocol")
    try:
        raw = json_from(
            content,
            source="choices[0].message.content",
            accept=lambda candidate: bool(sanitize(candidate)),
        )
    except JsonError as exc:
        kind = "truncated" if finish_reason == "length" else "model_output"
        hint = " (ответ модели обрезан по max_tokens)" if kind == "truncated" else ""
        raise LlmError(f"содержимое ответа не JSON{hint}: {exc}", kind=kind) from exc
    return sanitize(raw)


def _sanitize_slots(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Белый список полей ответа: типы проверяются, лишнее отбрасывается.

    Даже при ``strict: true`` проверка дублируется локально: правила подбора
    не должны зависеть от того, насколько аккуратно модель соблюла схему.
    """
    result: dict[str, Any] = {}
    city = raw.get("city_id")
    if isinstance(city, str) and city.strip():
        result["city_id"] = city.strip()[:32]
    budget = raw.get("budget_rub")
    if isinstance(budget, int) and not isinstance(budget, bool) and 0 < budget <= MAX_PRICE_RUB:
        result["budget_rub"] = budget
    categories = _sanitize_categories(raw).get("categories")
    if categories:
        result["categories"] = categories
    return result


def _sanitize_categories(raw: Mapping[str, Any]) -> dict[str, Any]:
    categories = raw.get("categories")
    if not isinstance(categories, list):
        return {}
    valid = [item for item in categories if isinstance(item, str) and item in CATEGORIES]
    return {"categories": valid[:len(CATEGORIES)]} if valid or not categories else {}


def _error_from_response(status: int, body: bytes, api_key: str, attempts: int) -> LlmError:
    """Превращает неуспешный HTTP-ответ в объяснённую техническую ошибку."""
    text = redact(body.decode("utf-8", errors="replace"), api_key)
    code, message, request_id = "", "", ""
    try:
        payload = loads_strict(text, source="ошибка шлюза")
    except JsonError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        error = payload["error"]
        code = str(error.get("code") or "")
        message = str(error.get("message") or "")
        request_id = str(error.get("request_id") or "")

    details = "; ".join(part for part in (f"code={code}" if code else "", message) if part)
    suffix = f" [{details}]" if details else ""
    if request_id:
        suffix += f" request_id={request_id}"
    if not details and text.strip():
        suffix += f" {_excerpt(text)}"

    if status in (401, 403):
        hint = " проверьте ключ и срок доступа у организатора"
        if "1010" in text:
            hint += "; похоже на блокировку клиента — нужен явный User-Agent"
        kind = "access"
    elif status == 429:
        hint = " превышена частота или параллельность вызовов"
        kind = "rate_limited"
    elif status in (502, 503):
        hint = " шлюз или модель временно недоступны"
        kind = "upstream"
    elif 300 <= status < 400:
        hint = " редиректы запрещены: ключ не отправляется на другой хост"
        kind = "redirect"
    else:
        hint = ""
        kind = "http"

    return LlmError(
        f"модель ответила HTTP {status} (попытка {attempts}){suffix}.{hint}",
        status=status,
        kind=kind,
    )


def _retry_after(headers: Mapping[str, str]) -> float:
    """Пауза из ``Retry-After`` (секунды); по умолчанию 2 с, но не больше 30 с."""
    for name, value in headers.items():
        if name.lower() != "retry-after":
            continue
        try:
            seconds = float(str(value).strip())
        except ValueError:
            break
        return min(max(seconds, 0.0), 30.0)
    return LLM_RATE_LIMIT_DEFAULT_SLEEP_SECONDS


def _excerpt(text: str) -> str:
    """Короткий однострочный фрагмент ответа для stderr (без переводов строк)."""
    flat = " ".join(text.split())
    if len(flat) > ERROR_EXCERPT_CHARS:
        return flat[: ERROR_EXCERPT_CHARS - 1] + "…"
    return flat


def _log_usage(envelope: Mapping[str, Any]) -> None:
    """Печатает расход токенов одного вызова: это и есть учёт бюджета §5."""
    usage = envelope.get("usage")
    if not isinstance(usage, dict):
        return
    counts = (
        value if type(value) is int and value >= 0 else None
        for value in (usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens"))
    )
    prompt, completion, total = counts
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    if total is None:
        return
    prompt_label = "-" if prompt is None else str(prompt)
    completion_label = "-" if completion is None else str(completion)
    print(
        f"{_LOG_PREFIX} модель: prompt={prompt_label} completion={completion_label} total={total}",
        file=sys.stderr,
        flush=True,
    )
