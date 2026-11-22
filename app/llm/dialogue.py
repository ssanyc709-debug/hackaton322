"""Передача распознанного товара между запросами одного веб-диалога."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Concatenate, ParamSpec, TypeVar, cast

from app.jsonio import JsonError, load_json_file

PREVIOUS_RESULT = "model-previous.json"
CURRENT_RESULT = "model-current.json"
_Params = ParamSpec("_Params")
_Result = TypeVar("_Result", str, dict[str, Any], dict[str, Any] | None)


class DialogueAnalysis:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        try:
            previous = load_json_file(directory / PREVIOUS_RESULT)
        except JsonError:
            previous = None
        self.previous = previous if isinstance(previous, dict) else {}

    def wrap(self, resolver: Callable[Concatenate[str, _Params], _Result]
             ) -> Callable[Concatenate[str, _Params], _Result]:
        def resolve(text: str, *args: _Params.args, **kwargs: _Params.kwargs) -> _Result:
            previous = self.previous.get("result")
            if self.previous.get("text") == text and isinstance(previous, (str, dict)):
                result = cast(_Result, previous)
            else:
                result = resolver(text, *args, **kwargs)

            recognized = result.get("categories") if isinstance(result, dict) else result
            if recognized:
                try:
                    (self.directory / CURRENT_RESULT).write_text(
                        json.dumps({"text": text, "result": result}, ensure_ascii=False), encoding="utf-8",
                    )
                except OSError:
                    # Потеря записи приводит только к повторному разбору.
                    pass
            return result

        return resolve
