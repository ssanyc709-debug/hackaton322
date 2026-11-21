"""Диалог: слияние условий по истории, clarify, повторные уточнения (§2.3)."""

from __future__ import annotations

import unittest

from app.catalog.index import build_index
from app.config import Settings
from app.contract.request import parse_request
from app.model import FIELD_ADJUSTABLE, FIELD_BUDGET, FIELD_CATEGORIES, FIELD_CITY, FIELD_WIDTH
from app.nlp.extract import missing_fields
from app.nlp.merge import merge_history
from app.pipeline import answer
from tests.support import CATALOG_PATH

import json

DOCUMENT = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
INDEX = build_index(DOCUMENT)
SETTINGS = Settings.load(environ={"ASSISTANT_OFFLINE": "1"})


def run(texts: list[str]) -> dict:
    """Собирает запрос из реплик покупателя (с репликами помощника между ними)."""
    messages: list[dict[str, str]] = []
    for index, text in enumerate(texts):
        if index:
            messages.append({"role": "assistant", "text": "Уточняю."})
        messages.append({"role": "user", "text": text})
    request = parse_request({"messages": messages})
    return answer(request=request, index=INDEX, settings=SETTINGS, deadline=1e18)


class MergeBasicsTest(unittest.TestCase):
    """Правило «явное изменение заменяет, остальное сохраняется»."""

    def test_budget_from_second_message_is_kept(self) -> None:
        slots = merge_history(["Стол и кресло в Таганроге", "15 тысяч"])
        self.assertEqual(slots.city_id, "taganrog")
        self.assertEqual(slots.budget_rub, 15000)
        self.assertEqual(slots.categories, ("table", "chair"))
        self.assertEqual(missing_fields(slots), ())

    def test_city_change_replaces_previous(self) -> None:
        slots = merge_history(["Стол в Таганроге до 10000", "Нет, давай в Азове"])
        self.assertEqual(slots.city_id, "azov")
        self.assertEqual(slots.budget_rub, 10000)

    def test_budget_change_replaces_previous(self) -> None:
        slots = merge_history(["Стол в Таганроге до 10000", "Бюджет 15000"])
        self.assertEqual(slots.budget_rub, 15000)

    def test_width_change_replaces_previous(self) -> None:
        slots = merge_history(["Стол не шире 80 см, Таганрог, до 20000", "Ладно, до 100 см"])
        assert slots.width is not None and slots.width.upper is not None
        self.assertEqual(slots.width.upper.value, 100)

    def test_additive_categories_are_united(self) -> None:
        slots = merge_history(["Нужен стол в Таганроге до 20000", "Добавь ещё лампу"])
        self.assertEqual(slots.categories, ("table", "lamp"))

    def test_categories_are_replaced_without_additive_word(self) -> None:
        slots = merge_history(["Стол, кресло и лампа. Таганрог. До 20000", "Только стол"])
        self.assertEqual(slots.categories, ("table",))

    def test_removed_category_is_dropped(self) -> None:
        slots = merge_history(["Стол, кресло и лампа. Таганрог. До 20000", "Лампа не нужна"])
        self.assertEqual(slots.categories, ("table", "chair"))

    def test_removing_the_only_category_requires_clarification(self) -> None:
        slots = merge_history(["Стол в Таганроге до 20000", "Без стола"])
        self.assertEqual(slots.categories, ())
        self.assertIn(FIELD_CATEGORIES, missing_fields(slots))

    def test_unknown_city_overrides_known_one(self) -> None:
        slots = merge_history(["Стол в Таганроге до 20000", "Лучше в Москве"])
        self.assertIsNone(slots.city_id)
        self.assertEqual(slots.city_unknown, "москва")
        self.assertIn(FIELD_CITY, missing_fields(slots))

    def test_unrelated_message_changes_nothing(self) -> None:
        slots = merge_history(["Стол в Таганроге до 20000", "Спасибо!"])
        self.assertEqual(slots.city_id, "taganrog")
        self.assertEqual(slots.budget_rub, 20000)

    def test_mentioned_fields_are_accumulated(self) -> None:
        slots = merge_history(["Стол в Таганроге", "до 15000", "не шире 100 см"])
        self.assertTrue({FIELD_CITY, FIELD_BUDGET, FIELD_CATEGORIES, FIELD_WIDTH} <= set(slots.mentioned))

    def test_adjustable_from_later_message(self) -> None:
        slots = merge_history(["Кресло в Таганроге до 20000", "Нужно с регулировкой высоты"])
        self.assertIs(slots.adjustable_height, True)
        self.assertIn(FIELD_ADJUSTABLE, slots.mentioned)

    def test_removed_width_is_dropped_from_history(self) -> None:
        """«Расширим ширину» снимает прежнее ограничение, остальное сохраняется."""
        slots = merge_history(
            ["Стол не шире 80 см, Таганрог, до 20000", "Давай расширим допустимую ширину"]
        )
        self.assertIsNone(slots.width)
        self.assertEqual(slots.city_id, "taganrog")
        self.assertEqual(slots.budget_rub, 20000)

    def test_removal_with_new_value_keeps_the_value(self) -> None:
        slots = merge_history(
            ["Стол не шире 80 см, Таганрог, до 20000", "Расширим ширину до 250"]
        )
        assert slots.width is not None and slots.width.upper is not None
        self.assertEqual(slots.width.upper.value, 250)

    def test_removed_adjustable_is_dropped_from_history(self) -> None:
        slots = merge_history(
            ["Кресло с регулировкой высоты, Таганрог, до 20000", "Убери требование к регулировке"]
        )
        self.assertIsNone(slots.adjustable_height)
        self.assertEqual(slots.categories, ("chair",))


class ModelCallPolicyTest(unittest.TestCase):
    """§5: модель — один вызов на запрос и только если история не закрыла условия."""

    def setUp(self) -> None:
        self.calls: list[str] = []

    def resolver(self, text: str, slots) -> dict[str, object]:  # noqa: ARG002 — подпись Resolver
        self.calls.append(text)
        return {"budget_rub": 12000}

    def test_history_covering_everything_costs_zero_calls(self) -> None:
        """«Стол и кресло в Таганроге» + «12 тысяч»: по репликам модель звали бы дважды."""
        slots = merge_history(
            ["Стол и кресло в Таганроге", "12 тысяч"], resolver=self.resolver
        )
        self.assertEqual(self.calls, [])
        self.assertEqual(slots.budget_rub, 12000)
        self.assertEqual(slots.city_id, "taganrog")
        self.assertEqual(slots.categories, ("table", "chair"))

    def test_one_call_for_the_whole_dialogue(self) -> None:
        """Две реплики, условия закрывает только их сумма — ровно один вызов."""
        slots = merge_history(
            ["Хочу обставить кабинет в Таганроге", "12 тысяч"],
            resolver=self.resolver,
        )
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(slots.city_id, "taganrog")
        self.assertEqual(slots.budget_rub, 12000)
        self.assertIn("Хочу обставить кабинет в Таганроге", self.calls[0])
        self.assertIn("12 тысяч", self.calls[0])

    def test_no_call_in_offline_mode(self) -> None:
        merge_history(["Хочу мебель"], resolver=self.resolver, allow_model=False)
        self.assertEqual(self.calls, [])

    def test_no_call_when_history_has_no_material_for_the_model(self) -> None:
        """Ни города, ни бюджета: модель не выдумает их, вызов ничего не изменит."""
        merge_history(["Хочу мебель"], resolver=self.resolver)
        self.assertEqual(self.calls, [])

    def test_model_error_is_not_swallowed(self) -> None:
        def broken(text: str, slots) -> dict[str, object]:
            raise TimeoutError("шлюз не ответил")

        with self.assertRaises(TimeoutError):
            merge_history(["Нужна деревяшка", "Таганрог", "15 тысяч"], resolver=broken)

    def test_long_dialogue_falls_back_to_last_message(self) -> None:
        """История длиннее предела модели: в один вызов уходит последняя реплика."""
        from app.nlp.extract import MAX_MODEL_TEXT_BYTES
        from app.nlp.merge import dialogue_text

        long_text = "я" * (MAX_MODEL_TEXT_BYTES - 10)
        self.assertEqual(dialogue_text([long_text, "Таганрог"]), "Таганрог")
        self.assertEqual(dialogue_text([]), "")


class ClarifyFlowTest(unittest.TestCase):
    """Уточнения: спрашиваем только недостающее, ответ учитывается."""

    def fields(self, document: dict) -> list[str]:
        self.assertEqual(document["status"], "clarify", document)
        return [question["field"] for question in document["questions"]]

    def test_example_02_asks_only_budget(self) -> None:
        self.assertEqual(self.fields(run(["Стол не шире метра, кресло и лампа. Таганрог."])),
                         ["budget_rub"])

    def test_all_three_conditions_are_asked_at_once(self) -> None:
        self.assertEqual(self.fields(run(["Хочу купить мебель"])),
                         ["budget_rub", "city_id", "categories"])

    def test_answer_to_question_is_used(self) -> None:
        document = run(["Стол не шире метра, кресло и лампа. Таганрог.", "15 тысяч"])
        self.assertEqual(document["status"], "ok")

    def test_answered_field_is_not_asked_again(self) -> None:
        """Категория была известна с первой реплики — значит ответ на бюджет закрывает всё."""
        document = run(["Стол в Таганроге", "15 тысяч"])
        self.assertEqual(document["status"], "ok")

    def test_unknown_condition_is_still_asked(self) -> None:
        document = run(["Хочу стол и кресло", "Таганрог", "15 тысяч"])
        self.assertEqual(document["status"], "ok")

    def test_unknown_category_asks_categories(self) -> None:
        document = run(["Диван в Таганроге до 30000"])
        self.assertEqual(self.fields(document), ["categories"])

    def test_unknown_city_asks_city(self) -> None:
        document = run(["Стол в Москве до 30000"])
        self.assertEqual(self.fields(document), ["city_id"])

    def test_short_answer_with_city_only(self) -> None:
        document = run(["Стол и лампа до 15000", "Таганрог"])
        self.assertEqual(document["status"], "ok")

    def test_explicit_change_after_clarify(self) -> None:
        document = run(["Стол в Ростове-на-Дону до 15000", "Вообще-то нужен Азов"])
        self.assertEqual(document["status"], "ok")
        self.assertTrue(any(option["listing_ids"] for option in document["options"]))


class UnintelligibleMessageTest(unittest.TestCase):
    """Мусорная реплика не подтверждает прежний набор категорий (§3).

    Правило контракта: явное изменение условия заменяет прежнее значение,
    остальные условия сохраняются. Реплика, в которой не распознано НИ ОДНОГО
    условия и нет ни одного числа, не является подтверждением прежнего подбора:
    иначе «ыаллфдылааввы» молча означает «продолжай, я согласен», и покупатель
    получает прежний комплект вместо уточнения.

    Ожидаемое поведение: категории считаются неизвестными → ``clarify`` по
    ``categories``, а уже названные город и бюджет повторно не спрашиваются.
    """

    def fields(self, document: dict) -> list[str]:
        self.assertEqual(document.get("status"), "clarify", document)
        self.assertNotIn("options", document, "в статусе clarify комплектов быть не может")
        return [question["field"] for question in document["questions"]]

    def test_gibberish_after_complete_conditions_asks_categories(self) -> None:
        document = run(["Стол до 5000. Таганрог.", "ыаллфдылааввы"])
        self.assertEqual(self.fields(document), ["categories"])

    def test_repeated_letters_after_complete_conditions_asks_categories(self) -> None:
        document = run(["Стол до 5000. Таганрог.", "фффффф"])
        self.assertEqual(self.fields(document), ["categories"])

    def test_latin_gibberish_after_complete_conditions_asks_categories(self) -> None:
        document = run(["Стол до 5000. Таганрог.", "asdfgh"])
        self.assertEqual(self.fields(document), ["categories"])

    def test_question_in_clarify_is_not_empty(self) -> None:
        document = run(["Стол до 5000. Таганрог.", "ыаллфдылааввы"])
        self.assertEqual(self.fields(document), ["categories"])
        for question in document["questions"]:
            self.assertTrue(question["text"].strip(), "вопрос без текста")

    def test_gibberish_from_the_bug_report_with_assistant_reply(self) -> None:
        """Точная последовательность из отчёта: реплика помощника между запросами."""
        request = parse_request(
            {
                "messages": [
                    {"role": "user", "text": "Стол до 5000. Таганрог."},
                    {"role": "assistant", "text": "Готово."},
                    {"role": "user", "text": "ыаллфдылааввы"},
                ]
            }
        )
        document = answer(request=request, index=INDEX, settings=SETTINGS, deadline=1e18)
        self.assertEqual(self.fields(document), ["categories"])

    def test_gibberish_without_conditions_asks_everything(self) -> None:
        """Прежних условий нет — спрашиваем все основные, как и раньше."""
        document = run(["Хочу мебель", "ыаллфдылааввы"])
        self.assertEqual(
            self.fields(document),
            ["budget_rub", "city_id", "categories"],
        )


class UnintelligibleRegressionTest(unittest.TestCase):
    """Короткие осмысленные ответы обязаны остаться рабочими (§3).

    Детектор «непонятной» реплики не должен ломать легитимные подтверждения:
    число, «спасибо», «ок» — это ответы покупателя, а не мусор.
    """

    def test_numeric_answer_keeps_the_selection(self) -> None:
        document = run(["Стол и кресло в Таганроге", "15 тысяч"])
        self.assertEqual(document.get("status"), "ok", document)
        self.assertTrue(document["options"], "ответ ok без вариантов")

    def test_numeric_answer_after_width_and_categories(self) -> None:
        document = run(["Стол не шире метра, кресло и лампа. Таганрог.", "15 тысяч"])
        self.assertEqual(document.get("status"), "ok", document)

    def test_thanks_keeps_the_selection(self) -> None:
        document = run(["Стол до 5000. Таганрог.", "Спасибо!"])
        self.assertEqual(document.get("status"), "ok", document)

    def test_short_confirmation_keeps_the_selection(self) -> None:
        document = run(["Стол до 5000. Таганрог.", "ок"])
        self.assertEqual(document.get("status"), "ok", document)

    def test_unsupported_category_in_one_message_still_asks_categories(self) -> None:
        document = run(["Диван в Таганроге до 30000"])
        self.assertEqual(document.get("status"), "clarify", document)
        fields = [question["field"] for question in document["questions"]]
        self.assertEqual(fields, ["categories"])


class DialogueSelectionTest(unittest.TestCase):
    """Подбор по истории: все ограничения сохранены."""

    def test_budget_and_width_survive_short_answer(self) -> None:
        document = run(["Стол не шире метра, кресло и лампа. Таганрог. До 15 000 ₽ за всё."])
        self.assertEqual(document["status"], "ok")
        by_id = {listing["id"]: listing for listing in DOCUMENT}
        for option in document["options"]:
            self.assertLessEqual(option["total_rub"], 15000)
            for listing_id in option["listing_ids"]:
                self.assertEqual(by_id[listing_id]["city_id"], "taganrog")
                listing = by_id[listing_id]
                if listing["category"] == "table":
                    width = listing["attributes"].get("width_cm")
                    self.assertIsNotNone(width, "стол без ширины не подтверждает условие")
                    self.assertLessEqual(width, 100)

    def test_lamp_temperature_survives(self) -> None:
        document = run(["Лампа 3000 к в Таганроге, до 5000"])
        self.assertEqual(document["status"], "ok")
        by_id = {listing["id"]: listing for listing in DOCUMENT}
        for option in document["options"]:
            for listing_id in option["listing_ids"]:
                self.assertEqual(by_id[listing_id]["attributes"].get("color_temperature_k"), 3000)

    def test_adjustable_true_filter(self) -> None:
        document = run(["Кресло с регулировкой высоты, Таганрог, до 6000"])
        self.assertEqual(document["status"], "ok")
        by_id = {listing["id"]: listing for listing in DOCUMENT}
        for option in document["options"]:
            for listing_id in option["listing_ids"]:
                self.assertIs(by_id[listing_id]["attributes"].get("adjustable_height"), True)

    def test_adjustable_false_filter_is_not_dropped(self) -> None:
        document = run(["Кресло без регулировки, Таганрог, до 6000"])
        self.assertEqual(document["status"], "ok")
        by_id = {listing["id"]: listing for listing in DOCUMENT}
        for option in document["options"]:
            for listing_id in option["listing_ids"]:
                self.assertIs(by_id[listing_id]["attributes"].get("adjustable_height"), False)


if __name__ == "__main__":
    unittest.main()
