"""Входящий запрос: разбор и валидация по контракту (CONTRACT.md §2.3).

Контракт жёсткий: 1..20 сообщений, только роли ``user``/``assistant``,
ровно поля ``role`` и ``text``, первое и последнее сообщение — от покупателя,
роли чередуются. Любое нарушение — техническая ошибка входа (ненулевой exit),
а не ``clarify``: ``clarify`` отвечает на неполные *условия*, а не на битые данные.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.config import MAX_MESSAGES

Role = Literal["user", "assistant"]

ALLOWED_ROLES: frozenset[str] = frozenset({"user", "assistant"})
ALLOWED_MESSAGE_KEYS: frozenset[str] = frozenset({"role", "text"})
ALLOWED_REQUEST_KEYS: frozenset[str] = frozenset({"messages"})
USER_ROLE: Role = "user"


class RequestError(ValueError):
    """Запрос не соответствует контракту."""


@dataclass(frozen=True, slots=True)
class Message:
    """Одна реплика диалога."""

    role: Role
    text: str


@dataclass(frozen=True, slots=True)
class Request:
    """Провалидированный запрос: неизменяемая последовательность реплик."""

    messages: tuple[Message, ...]

    @property
    def user_texts(self) -> tuple[str, ...]:
        """Тексты только покупателя, в хронологическом порядке."""
        return tuple(message.text for message in self.messages if message.role == USER_ROLE)

    @property
    def last_user_text(self) -> str:
        """Последняя реплика покупателя (последнее слово в диалоге)."""
        return self.user_texts[-1]

    @property
    def dialogue(self) -> tuple[tuple[Role, str], ...]:
        """История целиком — для слияния слотов (merge) и трассировки."""
        return tuple((message.role, message.text) for message in self.messages)


def parse_request(payload: Any) -> Request:
    """CORE. Проверяет JSON-объект запроса и возвращает :class:`Request`.

    Зачем: ни один слой ниже не должен защищаться от структурного мусора
    (нет ключа ``messages``, лишние поля, роли вразнобой). Проверка одна,
    на границе процесса, — дальше данные уже типизированы.

    Вход:  payload — результат ``json.loads`` тела запроса (любой JSON-тип).
    Выход: Request с 1..20 репликами; при нарушении — RequestError с точной причиной.

    Алгоритм: один линейный проход по массиву сообщений с проверкой инвариантов
    (длина, тип элемента, набор ключей, значение роли, непустой текст, порядок ролей).
    Сложность: время O(m) по числу сообщений (m <= 20), память O(m).
    Почему именно он: проверок 8 штук и они независимы, поэтому один проход
    дешевле и читаемее, чем схема-машина состояний; JSON Schema применяется
    отдельно в тестах как эталон, но не в горячем пути.
    Крайние случаи: пустой массив, 21 сообщение, текст из пробелов,
    не-объект на входе, роль ``system``, повтор ``user``/``user``,
    последняя реплика не от покупателя — всё это RequestError.
    Вызывается из: app/main.py::main.
    """
    if not isinstance(payload, dict):
        raise RequestError(f"тело запроса должно быть JSON-объектом, получено {type(payload).__name__}")

    extra_keys = set(payload) - ALLOWED_REQUEST_KEYS
    if extra_keys:
        raise RequestError(f"лишние поля запроса: {sorted(extra_keys)}")
    if "messages" not in payload:
        raise RequestError("отсутствует обязательное поле messages")

    raw_messages = payload["messages"]
    if not isinstance(raw_messages, list):
        raise RequestError("messages должно быть массивом")
    if not 1 <= len(raw_messages) <= MAX_MESSAGES:
        raise RequestError(f"messages: ожидалось 1..{MAX_MESSAGES} сообщений, получено {len(raw_messages)}")

    messages: list[Message] = []
    for index, raw in enumerate(raw_messages):
        messages.append(_parse_message(raw, index))

    if messages[0].role != USER_ROLE:
        raise RequestError("первое сообщение должно быть от user")
    if messages[-1].role != USER_ROLE:
        raise RequestError("последнее сообщение должно быть от user")
    for index in range(1, len(messages)):
        if messages[index].role == messages[index - 1].role:
            raise RequestError(
                f"роли должны чередоваться: сообщения {index} и {index + 1} обе роли {messages[index].role!r}"
            )

    return Request(messages=tuple(messages))


def _parse_message(raw: Any, index: int) -> Message:
    """Разбирает одно сообщение; нумерация в тексте ошибки — с единицы."""
    position = index + 1
    if not isinstance(raw, dict):
        raise RequestError(f"сообщение #{position}: ожидался объект, получено {type(raw).__name__}")
    extra_keys = set(raw) - ALLOWED_MESSAGE_KEYS
    if extra_keys:
        raise RequestError(f"сообщение #{position}: лишние поля {sorted(extra_keys)}")
    missing_keys = ALLOWED_MESSAGE_KEYS - set(raw)
    if missing_keys:
        raise RequestError(f"сообщение #{position}: отсутствуют поля {sorted(missing_keys)}")

    role = raw["role"]
    if not isinstance(role, str) or role not in ALLOWED_ROLES:
        raise RequestError(f"сообщение #{position}: недопустимая роль {role!r}")
    text = raw["text"]
    if not isinstance(text, str):
        raise RequestError(f"сообщение #{position}: text должно быть строкой")
    if not text.strip():
        raise RequestError(f"сообщение #{position}: text не должен быть пустым")
    return Message(role=role, text=text)  # type: ignore[arg-type]
