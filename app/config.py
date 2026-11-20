"""Конфигурация запуска: окружение, .env-лоадер, лимиты контракта.

Ключ модели в этом файле отсутствует намеренно: единственные источники —
переменная окружения ``LLM_API_KEY`` либо локальный ``.env`` (он в .gitignore).
Настройки читаются один раз на процесс; секрет никогда не попадает в логи
(см. :meth:`LlmConfig.redacted`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SCHEMA_DIR = REPO_ROOT / "schemas"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_CACHE_DIR = DATA_DIR / "cache"

# --- Лимиты контракта (CONTRACT.md §2) -----------------------------------
MAX_MESSAGES = 20
MAX_CATALOG_LISTINGS = 250
MAX_OPTIONS = 3
MIN_PRICE_RUB = 1
MAX_PRICE_RUB = 10_000_000

# --- Бюджет времени на один запуск (контракт: до 120 с) -------------------
RUN_BUDGET_SECONDS = 120.0
# Оставляем запас на старт интерпретатора, запись stdout и служебные операции.
RUN_BUDGET_SAFETY_SECONDS = 5.0

# --- Бюджет одного вызова модели -----------------------------------------
LLM_MAX_OUTPUT_TOKENS = 200
LLM_CALL_TIMEOUT_SECONDS = 90.0
LLM_MAX_BODY_BYTES = 65_536
LLM_MAX_INPUT_BYTES = 32_768
# 429 означает «слишком часто / модель занята»: одна короткая пауза, не цикл ретраев.
LLM_RATE_LIMIT_MAX_RETRIES = 1
LLM_RATE_LIMIT_DEFAULT_SLEEP_SECONDS = 2.0

TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on", "да"})


class ConfigError(RuntimeError):
    """Некорректная конфигурация: сообщение уходит в stderr, процесс завершается с ошибкой."""


@dataclass(frozen=True, slots=True)
class LlmConfig:
    """Параметры доступа к шлюзу модели.

    ``api_key`` — единственный секрет в системе: он не попадает ни в ответ,
    ни в логи, а печатается только через :meth:`redacted`.
    """

    base_url: str = ""
    model: str = ""
    api_key: str = ""
    user_agent: str = "team-02/1.0"
    expires_at: str = ""

    @property
    def configured(self) -> bool:
        """True, если заданы все три обязательных поля (URL, модель, ключ)."""
        return bool(self.base_url and self.model and self.api_key)

    def redacted(self) -> dict[str, str]:
        """Безопасное представление настроек для stderr: ключ заменён на длину."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "user_agent": self.user_agent,
            "api_key": f"<set:{len(self.api_key)} chars>" if self.api_key else "<unset>",
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True, slots=True)
class Settings:
    """Полный набор настроек процесса."""

    llm: LlmConfig
    cache_dir: Path
    offline: bool = False
    debug: bool = False
    offline_cache_disabled: bool = False

    @property
    def llm_available(self) -> bool:
        """Модель можно вызывать: ключ есть и офлайн-режим выключен."""
        return self.llm.configured and not self.offline

    @classmethod
    def load(
        cls,
        env_file: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "Settings":
        """Собирает настройки: сначала файл .env, затем переменные окружения.

        Переменные окружения имеют приоритет над файлом — это позволяет
        переопределять ключ в CI, не редактируя .env.
        """
        env: dict[str, str] = {}
        path = env_file if env_file is not None else DEFAULT_ENV_FILE
        if path.is_file():
            env.update(read_env_file(path))
        source = os.environ if environ is None else environ
        env.update({key: value for key, value in source.items() if key.startswith(("LLM_", "TEAM_", "ASSISTANT_"))})

        llm = LlmConfig(
            base_url=(env.get("LLM_BASE_URL") or "").strip().rstrip("/"),
            model=(env.get("LLM_MODEL") or "").strip(),
            api_key=(env.get("LLM_API_KEY") or "").strip(),
            user_agent=(env.get("LLM_USER_AGENT") or "team-02/1.0").strip(),
            expires_at=(env.get("LLM_EXPIRES_AT") or "").strip(),
        )
        if llm.base_url:
            _validate_base_url(llm.base_url)
        if llm.api_key and not llm.model:
            raise ConfigError("LLM_API_KEY задан, но LLM_MODEL пуст")

        cache_dir = Path(env.get("ASSISTANT_CACHE_DIR") or DEFAULT_CACHE_DIR)
        return cls(
            llm=llm,
            cache_dir=cache_dir,
            offline=is_truthy(env.get("ASSISTANT_OFFLINE", "")),
            debug=is_truthy(env.get("ASSISTANT_DEBUG", "")),
            offline_cache_disabled=is_truthy(env.get("ASSISTANT_NO_CACHE", "")),
        )


def read_env_file(path: Path) -> dict[str, str]:
    """Читает простой KEY=VALUE файл.

    Поддерживаются комментарии ``#``, пустые строки, префикс ``export``,
    кавычки вокруг значения и значения, содержащие ``=``.
    """
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:  # файл есть, но недоступен — это ошибка конфигурации
        raise ConfigError(f"не удалось прочитать {path}: {exc}") from exc
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def is_truthy(value: str) -> bool:
    """Флаги окружения: «1», «true», «yes», «on», «да» (регистр не важен)."""
    return value.strip().lower() in TRUTHY_ENV_VALUES


def _validate_base_url(value: str) -> None:
    """Разрешаем только https, либо http на loopback (локальный мок для тестов)."""
    lowered = value.lower()
    if lowered.startswith("https://"):
        return
    if lowered.startswith("http://") and any(
        lowered.startswith(f"http://{host}") for host in ("localhost", "127.0.0.1", "[::1]")
    ):
        return
    raise ConfigError(f"LLM_BASE_URL должен быть https (или http на loopback): {value!r}")
