"""Кэш лексиконного разбора: ``data/cache/lexicon.json`` — 0 токенов на повторах.

Ключ — ``sha256`` нормализованного текста реплики (плюс версия лексикона):
одинаковая формулировка разбирается один раз, дальше слоты читаются с диска.
Значение кэша — только результат локального разбора, поэтому его нельзя
«отравить» описанием объявления: каталог в кэш не попадает вообще.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from app.model import Bound, NumericRange, Slots
from app.nlp.numbers import normalize_text

#: Версия формата файла кэша.
#: 1 → 2: в записи слота добавлено ``fields_removed`` (снятие ограничений);
#: записи прежнего формата не знают об этом поле, поэтому не переиспользуются.
CACHE_FORMAT_VERSION = 2
#: Версия лексикона: при изменении словарей старые разборы недействительны.
#: 4 — исправлено слияние пар (одиночная нижняя граница не теряется, Д1)
#: и потолок бюджета берётся только из верхних границ (Д2).
#: 5 — слово «ширина» делает число шириной без единицы «см»; появилось
#: снятие ограничений («расширим ширину», «убери регулировку»).
#: 7 — уточнены признаки городов, категорий и неоднозначных чисел.
#: 8 — распознавание отдельного запроса «нужно сиденье со спинкой».
#: 9 — прилагательные сами по себе не определяют товар.
#: 10 — сравнения, утверждённая поправка бюджета и альтернативные города.
LEXICON_VERSION = 10
#: Имя файла кэша разборов.
CACHE_FILE_NAME = "lexicon.json"
#: Предел числа записей: при переполнении половина старых отбрасывается.
MAX_CACHE_ENTRIES = 2000


def phrase_key(text: str) -> str:
    """CORE. Ключ кэша для реплики: ``sha256`` нормализованного текста.

    Зачем: повторная формулировка («До 15 000 ₽ за всё» и «до 15000 ₽ за все»)
    должна стоить 0 токенов, а ключ обязан быть стабильным и коротким.

    Вход:  text — сырая реплика покупателя.
    Выход: 64-символьная hex-строка.

    Алгоритм: :func:`app.nlp.numbers.normalize_text` (регистр, «ё», неразрывные
    пробелы, разряды чисел) → sha256 по UTF-8.
    Сложность: время O(len(text)), память O(1) сверх строки.
    Почему именно он: нормализация перед хешированием повышает hit-rate на
    естественных вариациях («15 000» / «15\u00a0000»), а sha256 даёт ключ
    фиксированной длины без хранения самого текста в индексе.
    Крайние случаи: пустая строка; текст из одних пробелов (нормализуется в "");
    очень длинная реплика (> 32 КБ) — ключ считается, вызов модели всё равно
    ограничен отдельно.
    Вызывается из: app/nlp/cache.py::LexiconCache.get, put.
    """
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class LexiconCache:
    """Файловый кэш разборов с ленивой загрузкой и атомарной записью."""

    def __init__(self, directory: Path, *, enabled: bool = True) -> None:
        self.path = directory / CACHE_FILE_NAME
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._entries: dict[str, dict[str, Any]] | None = None

    @property
    def hit_rate(self) -> float:
        """Доля попаданий среди обращений (для метрик в stderr)."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def get(self, text: str) -> Slots | None:
        """CORE. Возвращает разбор реплики из кэша — 0 исходящих токенов.

        Зачем: цель §5 — минимум токенов; повтор того же запроса (демонстрация,
        автопроверка, нагрузка) не должен ни вызывать модель, ни даже считать
        числа заново.

        Вход:  text — реплика покупателя.
        Выход: :class:`Slots` или ``None`` при промахе/отключённом кэше.

        Алгоритм: ключ :func:`phrase_key` → поиск в загруженном словаре записей →
        десериализация слотов (диапазоны восстанавливаются в :class:`NumericRange`).
        Сложность: попадание O(len(text)) на нормализацию + O(1) на словарь.
        Почему именно он: разбор реплики — самый «дорогой» локальный шаг после
        индекса; кэш по хешу делает его бесплатным, а версия лексикона защищает
        от устаревших разборов после правки словарей.
        Крайние случаи: файл отсутствует/битый → кэш считается пустым и
        перезаписывается; версия лексикона не совпала → полный промах;
        запись повреждена → промах и удаление записи.
        Вызывается из: app/nlp/extract.py::extract_message.
        """
        if not self.enabled:
            return None
        entries = self._load()
        key = phrase_key(text)
        payload = entries.get(key)
        if payload is None:
            self.misses += 1
            return None
        try:
            slots = _decode_slots(payload)
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[buyer-assistant] повреждённая запись кэша {key[:12]}: {exc}", file=sys.stderr, flush=True)
            entries.pop(key, None)
            self.misses += 1
            return None
        self.hits += 1
        return slots

    def put(self, text: str, slots: Slots) -> None:
        """Сохраняет разбор реплики; ошибки записи не ломают запрос."""
        if not self.enabled:
            return
        entries = self._load()
        entries[phrase_key(text)] = _encode_slots(slots)
        self._trim(entries)
        self._flush(entries)

    def _load(self) -> dict[str, dict[str, Any]]:
        """Ленивая загрузка файла кэша с проверкой версий."""
        if self._entries is not None:
            return self._entries
        self._entries = {}
        if not self.path.is_file():
            return self._entries
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("version") != CACHE_FORMAT_VERSION:
                raise ValueError(f"версия формата {payload.get('version')!r}")
            if payload.get("lexicon_version") != LEXICON_VERSION:
                raise ValueError(f"версия лексикона {payload.get('lexicon_version')!r}")
            entries = payload.get("entries")
            if not isinstance(entries, dict):
                raise ValueError("entries должен быть объектом")
        except (OSError, ValueError, AttributeError, TypeError) as exc:
            print(f"[buyer-assistant] кэш разборов игнорируется ({self.path.name}): {exc}",
                  file=sys.stderr, flush=True)
            return self._entries
        self._entries = {str(key): value for key, value in entries.items() if isinstance(value, dict)}
        return self._entries

    def _trim(self, entries: dict[str, dict[str, Any]]) -> None:
        """Ограничивает размер кэша: при переполнении выбрасывает половину старых."""
        if len(entries) <= MAX_CACHE_ENTRIES:
            return
        for key in list(entries)[: MAX_CACHE_ENTRIES // 2]:
            entries.pop(key, None)

    def _flush(self, entries: Mapping[str, dict[str, Any]]) -> None:
        """Атомарная запись кэша: временный файл + ``os.replace``."""
        payload = {
            "version": CACHE_FORMAT_VERSION,
            "lexicon_version": LEXICON_VERSION,
            "entries": entries,
        }
        temporary_path: str | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Уникальный временный файл на каждый запуск: параллельные процессы
            # не перезаписывают один и тот же `.tmp` (иначе кэш портился).
            descriptor, temporary_path = tempfile.mkstemp(
                dir=str(self.path.parent), prefix=self.path.name + ".", suffix=".tmp"
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            os.replace(temporary_path, self.path)
            temporary_path = None
        except OSError as exc:
            print(f"[buyer-assistant] не удалось записать кэш разборов ({self.path}): {exc}",
                  file=sys.stderr, flush=True)
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass


def _bound_to_json(bound: Bound | None) -> list[Any] | None:
    """Граница в JSON: ``[value, inclusive]`` или ``None``."""
    return None if bound is None else [bound.value, bound.inclusive]


def _bound_from_json(raw: Any) -> Bound | None:
    """Восстанавливает границу из JSON, проверяя типы (int не может быть bool)."""
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"граница должна быть [value, inclusive], получено {raw!r}")
    value, inclusive = raw
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"значение границы должно быть целым, получено {value!r}")
    if not isinstance(inclusive, bool):
        raise ValueError(f"признак включительности должен быть bool, получено {inclusive!r}")
    return Bound(value=value, inclusive=inclusive)


def _range_to_json(value: NumericRange | None) -> dict[str, Any] | None:
    """Диапазон в JSON или ``None``."""
    if value is None:
        return None
    return {"lower": _bound_to_json(value.lower), "upper": _bound_to_json(value.upper)}


def _range_from_json(raw: Any) -> NumericRange | None:
    """Диапазон из JSON или ``None``."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"диапазон должен быть объектом, получено {raw!r}")
    return NumericRange(lower=_bound_from_json(raw.get("lower")), upper=_bound_from_json(raw.get("upper")))


def _encode_slots(slots: Slots) -> dict[str, Any]:
    """Сериализует слоты в компактный JSON-объект."""
    return {
        "city_id": slots.city_id,
        "budget_rub": slots.budget_rub,
        "categories": list(slots.categories),
        "width": _range_to_json(slots.width),
        "color_temperature": _range_to_json(slots.color_temperature),
        "adjustable_height": slots.adjustable_height,
        "city_unknown": slots.city_unknown,
        "categories_unknown": list(slots.categories_unknown),
        "categories_removed": list(slots.categories_removed),
        "fields_removed": list(slots.fields_removed),
        "categories_explicit": slots.categories_explicit,
        "categories_additive": slots.categories_additive,
        "mentioned": sorted(slots.mentioned),
    }


def _decode_slots(payload: Mapping[str, Any]) -> Slots:
    """Восстанавливает слоты из JSON-объекта кэша."""
    categories = payload.get("categories") or []
    mentioned = payload.get("mentioned") or []
    if not isinstance(categories, list) or not isinstance(mentioned, list):
        raise ValueError("categories/mentioned должны быть массивами")
    budget = payload.get("budget_rub")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < 1):
        raise ValueError(f"budget_rub должен быть целым > 0, получено {budget!r}")
    adjustable = payload.get("adjustable_height")
    if adjustable is not None and not isinstance(adjustable, bool):
        raise ValueError(f"adjustable_height должен быть bool, получено {adjustable!r}")
    return Slots(
        city_id=payload.get("city_id"),
        budget_rub=budget,
        categories=tuple(str(category) for category in categories),
        width=_range_from_json(payload.get("width")),
        color_temperature=_range_from_json(payload.get("color_temperature")),
        adjustable_height=adjustable,
        city_unknown=payload.get("city_unknown"),
        categories_unknown=tuple(str(name) for name in (payload.get("categories_unknown") or [])),
        categories_removed=tuple(str(name) for name in (payload.get("categories_removed") or [])),
        fields_removed=tuple(str(field) for field in (payload.get("fields_removed") or [])),
        categories_explicit=bool(payload.get("categories_explicit")),
        categories_additive=bool(payload.get("categories_additive")),
        mentioned=frozenset(str(field) for field in mentioned),
        source="cache",
    )
