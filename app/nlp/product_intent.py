"""Распознавание запроса, в котором товар описан через назначение."""

from __future__ import annotations

import re

from app.nlp.numbers import normalize_text, tokenize

_DESCRIPTION = re.compile(
    r"\b(?:"
    r"(?:что|че)\s+(?:нибудь|то)\s+\w+|"
    r"то\s+что\s+\w+|что\s+(?:будет|может|умеет|посоветуешь|посоветуете)|"
    r"(?:на|в|за|с|под)\s+(?:чем|что|котор\w+)\s+\w+|"
    r"чем\s+можно|куда\s+можно|"
    r"(?:предмет|вещь|штук\w*|нечто)\s+(?:для|чтобы|с)\s+\w+"
    r")\b"
)
_USAGE = re.compile(
    r"\b(?:то\s+что|что\s+(?:будет|может|умеет)|"
    r"(?:на|в|за|с|под)\s+(?:чем|что|котор\w+)|чем\s+можно|куда\s+можно|для|чтобы)\b"
)


def describes_product(text: str) -> bool:
    """Проверяет, можно ли распознать назначение без города и бюджета."""
    return bool(_DESCRIPTION.search(" ".join(tokenize(normalize_text(text)))))


def terms_before_usage(text: str) -> frozenset[str]:
    """Выделяет слова перед описанием действия или назначения товара."""
    normalized = " ".join(tokenize(normalize_text(text)))
    return frozenset(_USAGE.split(normalized, maxsplit=1)[0].split())
