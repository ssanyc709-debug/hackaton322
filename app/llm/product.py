"""Короткое извлечение названия по описанию покупателя."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from app.config import LlmConfig
from app.llm.client import complete

_SCHEMA = {
    "type": "object",
    "properties": {"product": {"type": "string"}},
    "required": ["product"],
    "additionalProperties": False,
}
_SYSTEM = (
    "Назови товар из описания покупателя в 1–6 словах. Сохрани названные свойства. "
    "Если возможны разные товары, верни пустую строку. Ответ: JSON с полем product."
)


def _sanitize(raw: Mapping[str, Any]) -> dict[str, str]:
    """Пропускает только строку в пределах короткого названия товара."""
    product = raw.get("product")
    if isinstance(product, str) and len(product) <= 120 and len(product.split()) <= 6:
        return {"product": product.strip()}
    return {}


def make_product_resolver(
    config: LlmConfig,
    deadline: float,
    *,
    complete_fn: Callable[..., dict[str, Any]] = complete,
) -> Callable[[str], str]:
    """Создаёт резолвер с лимитом 64 токена на ответ. Получает только описание."""

    def resolve(text: str) -> str:
        raw = complete_fn(
            config,
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": text}],
            deadline,
            max_tokens=64,
            schema=_SCHEMA,
            schema_name="product_phrase",
            sanitize=_sanitize,
        )
        return _sanitize(raw).get("product", "")

    return resolve
