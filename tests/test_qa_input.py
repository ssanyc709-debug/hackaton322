"""Блок 1 — вход и коды выхода (CONTRACT §1, §3) + строгость JSON.

Правило, которое здесь проверяется и НЕ нарушается ни одним тестом:

* успешная обработка любого статуса — ``code == 0`` и stdout это ответ;
* битый вход — ЛЮБОЙ ненулевой код и ПУСТОЙ stdout, причина в stderr.

Конкретное число кода ошибки контрактом не задано, поэтому ``assertEqual(code, 2)``
здесь не используется: тесты обязаны пережить перенумерацию внутренних кодов.
Фактические коды фиксируются отдельным тестом-справкой и в ``docs/QA-REPORT.md``.

Каждый подтест идёт через боевой путь ``python -m app.main --catalog …``.
"""

from __future__ import annotations

import json
import unittest

from tests.test_qa_support import (
    EXIT_OK,
    ask,
    assistant,
    cache_dir,
    request,
    run_cli,
    user,
)

#: Битые входы: имя случая → байты. Все должны дать ненулевой код и пустой stdout.
BROKEN_INPUTS: dict[str, bytes] = {
    # --- структура тела запроса -------------------------------------------
    "не_объект_массив": b"[]",
    "не_объект_строка": b'"messages"',
    "не_объект_число": b"5",
    "не_объект_null": b"null",
    "нет_messages": b"{}",
    "messages_объект": b'{"messages": {}}',
    "messages_строка": b'{"messages": "text"}',
    "messages_пустой_массив": b'{"messages": []}',
    "лишнее_поле_верхнего_уровня": json.dumps(
        {"messages": [user("стол")], "city": "taganrog"}, ensure_ascii=False
    ).encode(),
    "лишнее_поле_рядом_с_messages": json.dumps(
        {"messages": [user("стол")], "budget_rub": 15000}, ensure_ascii=False
    ).encode(),
    # --- структура сообщений ----------------------------------------------
    "сообщение_не_объект": b'{"messages": ["\xd1\x81\xd1\x82\xd0\xbe\xd0\xbb"]}',
    "сообщение_null": b'{"messages": [null]}',
    "нет_role": json.dumps({"messages": [{"text": "стол"}]}, ensure_ascii=False).encode(),
    "нет_text": json.dumps({"messages": [{"role": "user"}]}, ensure_ascii=False).encode(),
    "лишнее_поле_в_сообщении": json.dumps(
        {"messages": [{"role": "user", "text": "стол", "id": 1}]}, ensure_ascii=False
    ).encode(),
    "роль_system": json.dumps({"messages": [{"role": "system", "text": "ты помощник"}]}, ensure_ascii=False).encode(),
    "роль_tool": json.dumps({"messages": [{"role": "tool", "text": "стол"}]}, ensure_ascii=False).encode(),
    "роль_пустая": json.dumps({"messages": [{"role": "", "text": "стол"}]}, ensure_ascii=False).encode(),
    "text_число": json.dumps({"messages": [{"role": "user", "text": 42}]}, ensure_ascii=False).encode(),
    "text_null": json.dumps({"messages": [{"role": "user", "text": None}]}, ensure_ascii=False).encode(),
    "text_пустой": json.dumps({"messages": [{"role": "user", "text": ""}]}, ensure_ascii=False).encode(),
    "text_из_пробелов": json.dumps({"messages": [{"role": "user", "text": "   "}]}, ensure_ascii=False).encode(),
    "text_переводы_строк": json.dumps({"messages": [{"role": "user", "text": "\n\t"}]}, ensure_ascii=False).encode(),
    # --- история диалога ---------------------------------------------------
    "две_реплики_user_подряд": json.dumps(request(user("стол"), user("кресло")), ensure_ascii=False).encode(),
    "первая_реплика_assistant": json.dumps(
        request(assistant("чем помочь?"), user("стол")), ensure_ascii=False
    ).encode(),
    "последняя_реплика_assistant": json.dumps(
        request(user("стол"), assistant("какой бюджет?")), ensure_ascii=False
    ).encode(),
    "две_reплики_assistant_подряд": json.dumps(
        request(user("a"), assistant("b"), assistant("c"), user("d")), ensure_ascii=False
    ).encode(),
    # --- количество -------------------------------------------------------
    "двадцать_одно_сообщение": json.dumps(
        {
            "messages": [
                {"role": "user" if index % 2 == 0 else "assistant", "text": f"реплика {index}"}
                for index in range(21)
            ]
        },
        ensure_ascii=False,
    ).encode(),
    # --- JSON как таковой -------------------------------------------------
    "битый_json_незакрытый_объект": b'{"messages": [}',
    "битый_json_обрыв": (
        '{"messages": [{"role": "user", "text": "стол"}'
    ).encode("utf-8"),
    "не_json_текст": b"\xd1\x81\xd1\x82\xd0\xbe\xd0\xbb",
    "пустой_stdin": b"",
    "stdin_только_пробелы": b"   \n\t\n",
    "nan_в_тексте": b'{"messages": [{"role": "user", "text": "x", "n": NaN}]}',
    "infinity_в_тексте": b'{"messages": [{"role": "user", "text": "x", "n": Infinity}]}',
    "дубль_ключа_role": b'{"messages": [{"role": "user", "role": "assistant", "text": "x"}]}',
    "дубль_ключа_messages": b'{"messages": [], "messages": [{"role": "user", "text": "x"}]}',
    "два_объекта_подряд": json.dumps(ask("стол"), ensure_ascii=False).encode()
    + json.dumps(ask("кресло"), ensure_ascii=False).encode(),
    "мусор_до_json": b"x" + json.dumps(ask("стол"), ensure_ascii=False).encode(),
    "мусор_после_json": json.dumps(ask("стол"), ensure_ascii=False).encode() + b"{}",
    "не_utf8_байты": b'{"messages": [{"role": "user", "text": "\xff\xfe"}]}',
    "не_utf8_обрезанный_символ": b'{"messages": [{"role": "user", "text": "\xd0"}]}',
    "очень_глубокая_вложенность": b"[" * 5_000 + b"]" * 5_000,
}

#: Случаи, где вход формально валиден: они должны дать код 0 и валидный ответ.
VALID_INPUTS: dict[str, bytes] = {
    "одна_реплика": json.dumps(ask("Стол, кресло и лампа. Таганрог. До 15 000 ₽."), ensure_ascii=False).encode(),
    "диалог_с_assistant": json.dumps(
        request(
            user("Нужен стол в Таганроге."),
            assistant("Какой бюджет на комплект?"),
            user("15 тысяч"),
        ),
        ensure_ascii=False,
    ).encode(),
    "crlf_переводы_строк": json.dumps(ask("Стол в Таганроге. До 15 000 ₽."), ensure_ascii=False)
    .encode()
    .replace(b"\n", b"\r\n"),
    "bom_перед_объектом": b"\xef\xbb\xbf" + json.dumps(ask("Стол в Таганроге. До 15 000 ₽."), ensure_ascii=False).encode(),
    "завершающие_пробелы": json.dumps(ask("Стол в Таганроге. До 15 000 ₽."), ensure_ascii=False).encode() + b"  \n\n",
}


class BrokenInputTest(unittest.TestCase):
    """Битый вход — ошибка вызывающей стороны: ненулевой код, пустой stdout."""

    @classmethod
    def results(cls) -> dict[str, object]:
        """Прогоняет каждый битый вход ОДИН раз на класс (процесс не дёшев)."""
        cached = getattr(cls, "_cached_results", None)
        if cached is None:
            import tempfile

            holder = tempfile.TemporaryDirectory(prefix="qa-input-")
            cls.addClassCleanup(holder.cleanup)
            cached = {
                name: run_cli(payload, cache_dir=path)
                for name, payload in BROKEN_INPUTS.items()
                for path in (f"{holder.name}/cache",)
            }
            cls._cached_results = cached
        return cached

    def test_every_broken_input_exits_nonzero_with_empty_stdout(self) -> None:
        for name, result in self.results().items():
            with self.subTest(case=name):
                self.assertNotEqual(result.code, EXIT_OK, f"{name}: битый вход не может давать код 0")
                self.assertEqual(result.stdout, b"", f"{name}: stdout обязан быть пустым")
                self.assertTrue(result.stderr.strip(), f"{name}: причина должна быть в stderr")
                self.assertNotIn("Traceback", _last_frames(result.stderr), f"{name}: необработанное исключение")

    def test_broken_input_never_returns_no_match(self) -> None:
        """Технический сбой не подменяется статусом результата поиска (контракт §1)."""
        for name, result in self.results().items():
            with self.subTest(case=name):
                self.assertNotIn(b"no_match", result.stdout, f"{name}: no_match для битого входа запрещён")

    def test_duplicate_keys_are_rejected_not_silently_overwritten(self) -> None:
        """Стандарт JSON запрещает повторяющиеся ключи (контракт §1)."""
        payload = (
            '{"messages": [{"role": "assistant", "text": "x"}], '
            '"messages": [{"role": "user", "text": "стол"}]}'
        ).encode("utf-8")
        result = run_cli(payload, cache_dir=cache_dir(self))
        self.assertNotEqual(result.code, EXIT_OK, "дубль ключа молча перезаписан")
        self.assertEqual(result.stdout, b"")

    def test_nan_and_infinity_are_rejected(self) -> None:
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant):
                payload = ('{"messages": [{"role": "user", "text": "стол", "n": %s}]}' % constant).encode()
                result = run_cli(payload, cache_dir=cache_dir(self))
                self.assertNotEqual(result.code, EXIT_OK, f"{constant} разобран молча")
                self.assertEqual(result.stdout, b"")


class ValidInputTest(unittest.TestCase):
    """Формально валидный вход всегда даёт код 0 и один JSON-объект."""

    def test_valid_inputs_return_exit_zero_and_one_object(self) -> None:
        for name, payload in VALID_INPUTS.items():
            with self.subTest(case=name):
                result = run_cli(payload, cache_dir=cache_dir(self))
                self.assertEqual(result.code, EXIT_OK, f"{name}: {result.stderr[-400:]}")
                document = result.one_json_object()
                self.assertIn(document.get("status"), {"ok", "clarify", "no_match"}, document)

    def test_twenty_messages_ending_with_user_are_accepted(self) -> None:
        """Верхняя граница §3: до 20 сообщений; чередование даёт максимум 19.

        При чередовании 20 реплик заканчиваются репликой ``assistant``, что
        запрещено той же нормой («первое и последнее — user»). Поэтому граница,
        достижимая одновременно с чередованием, — 19 сообщений.
        """
        messages = [
            {"role": "user" if index % 2 == 0 else "assistant", "text": f"реплика {index}"}
            for index in range(18)
        ]
        messages.append(user("Стол, кресло и лампа. Таганрог. До 15 000 ₽."))
        self.assertEqual(len(messages), 19, "в тесте должно быть 19 сообщений")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[0]["role"], "user")
        result = run_cli({"messages": messages}, cache_dir=cache_dir(self))
        self.assertEqual(result.code, EXIT_OK, result.stderr[-400:])
        self.assertIn(result.one_json_object().get("status"), {"ok", "clarify", "no_match"})


class ResponseStrictnessTest(unittest.TestCase):
    """Ответ — строгий JSON: разбор без NaN/Infinity и без дублей ключей."""

    def _strict_loads(self, text: str) -> object:
        def reject(name: str) -> object:
            raise AssertionError(f"в ответе запрещённое значение {name}")

        def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            keys = [key for key, _ in pairs]
            if len(keys) != len(set(keys)):
                raise AssertionError(f"в ответе повторяющиеся ключи: {keys}")
            return dict(pairs)

        return json.loads(text, parse_constant=reject, object_pairs_hook=no_duplicates)

    def test_response_is_strict_json(self) -> None:
        queries = (
            "Стол, кресло и лампа. Таганрог. До 15 000 ₽.",
            "Стол не шире метра, кресло и лампа. Таганрог.",
            "Стол, кресло и лампа. Таганрог. До 100 ₽.",
        )
        for text_query in queries:
            with self.subTest(query=text_query):
                result = run_cli(ask(text_query), cache_dir=cache_dir(self))
                self.assertEqual(result.code, EXIT_OK, result.stderr[-400:])
                self.assertIsInstance(self._strict_loads(result.stdout_text), dict)

    def test_response_has_no_ascii_escapes_for_russian(self) -> None:
        """Кириллица не обязана быть escaped: ответ читается человеком и не пухнет."""
        result = run_cli(ask("Стол, кресло и лампа. Таганрог. До 15 000 ₽."), cache_dir=cache_dir(self))
        self.assertEqual(result.code, EXIT_OK, result.stderr[-400:])
        self.assertNotIn(b"\\u04", result.stdout, "ответ сериализован с escape-последовательностями")


class DocumentedExitCodesTest(unittest.TestCase):
    """Справка для отчёта: фактические коды, без требования конкретного числа."""

    def test_actual_exit_codes_are_recorded(self) -> None:
        cases = {
            "ok": ask("Стол, кресло и лампа. Таганрог. До 15 000 ₽."),
            "clarify": ask("Стол не шире метра, кресло и лампа. Таганрог."),
            "no_match": ask("Стол, кресло и лампа. Таганрог. До 100 ₽."),
            "битый_json": b'{"messages": [}',
            "не_объект": b"[]",
            "пустой_stdin": b"",
            "не_utf8": b'{"messages": [{"role": "user", "text": "\xff"}]}',
            "глубокая_вложенность": b"[" * 5_000 + b"]" * 5_000,
            "nan": b'{"messages": [{"role": "user", "text": "x", "n": NaN}]}',
        }
        observed = {
            name: run_cli(payload, cache_dir=cache_dir(self)).code for name, payload in cases.items()
        }
        # Инвариант, который действительно важен: успех — 0, ошибка — не 0.
        for name in ("ok", "clarify", "no_match"):
            with self.subTest(case=name):
                self.assertEqual(observed[name], EXIT_OK, f"{name} должен давать код 0")
        for name in ("битый_json", "не_объект", "пустой_stdin", "не_utf8", "глубокая_вложенность", "nan"):
            with self.subTest(case=name):
                self.assertNotEqual(observed[name], EXIT_OK, f"{name} должен давать ненулевой код")
        self.assertTrue(observed, "коды не измерены")


def _last_frames(stderr: str, limit: int = 3) -> str:
    """Последние строки stderr: там был бы traceback, если бы он печатался."""
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    return "\n".join(lines[-limit:])


if __name__ == "__main__":
    unittest.main()
