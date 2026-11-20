"""Тесты лексикона: Snowball, хеш-словарь, триграммы + Левенштейн, сленг."""

from __future__ import annotations

import unittest

from app.nlp.lexicon import (
    Lexicon,
    candidate_forms,
    get_lexicon,
    levenshtein,
    order_categories,
    stem,
    trigrams,
)
from app.nlp.numbers import normalize_text, tokenize


def scan(text: str) -> tuple[str, ...]:
    """Нормализует текст и возвращает токены."""
    return tokenize(normalize_text(text))


class StemmerTest(unittest.TestCase):
    """Snowball: словоформы сводятся к одной основе."""

    def test_noun_forms_collapse(self) -> None:
        self.assertEqual(stem("кресла"), stem("кресло"))
        self.assertEqual(stem("лампы"), stem("лампа"))
        self.assertEqual(stem("таганроге"), stem("таганрог"))

    def test_porter_quirk_is_compensated_by_dictionary(self) -> None:
        """Porter может переусердствовать («стол» и «стола» → «сто»), но словарь
        хранит и точные формы, поэтому распознавание от этого не страдает."""
        self.assertEqual(stem("стол"), "сто")  # артефакт Porter, не баг
        lexicon = get_lexicon()
        for text in ("стол", "стола", "столов", "столы", "столик"):
            with self.subTest(text=text):
                hits = lexicon.scan_categories(scan(text))
                self.assertEqual([hit.value for hit in hits], ["table"])

    def test_adjective_and_reflexive_forms(self) -> None:
        self.assertEqual(stem("письменного"), stem("письменный"))
        self.assertTrue(stem("моющийся").startswith("моющ"))

    def test_short_words_are_untouched(self) -> None:
        for word in ("стол", "азов", "и"):
            with self.subTest(word=word):
                self.assertTrue(stem(word))

    def test_stem_is_idempotent_for_most_forms(self) -> None:
        """Второй прогон не должен «съедать» основу дальше одного раза."""
        for word in ("таганроге", "креслами", "лампами", "ростове"):
            once = stem(word)
            self.assertLessEqual(len(stem(once)), len(once))

    def test_candidate_forms_include_raw_and_stems(self) -> None:
        forms = candidate_forms("ростове")
        self.assertEqual(forms[0], "ростове")
        self.assertIn("рост", forms)


class TrigramAndLevenshteinTest(unittest.TestCase):
    """Триграммы и расстояние редактирования."""

    def test_levenshtein_basic(self) -> None:
        self.assertEqual(levenshtein("кресло", "кресло", 2), 0)
        self.assertEqual(levenshtein("кресло", "креслр", 2), 1)
        self.assertEqual(levenshtein("лампа", "лампаа", 2), 1)
        self.assertEqual(levenshtein("стол", "стул", 2), 1)
        self.assertEqual(levenshtein("", "стол", 5), 4)

    def test_levenshtein_early_exit(self) -> None:
        self.assertGreater(levenshtein("стол", "холодильник", 2), 2)

    def test_trigrams_are_padded(self) -> None:
        grams = trigrams("стол")
        self.assertIn("  с", grams)
        self.assertIn("тол", grams)


class LexiconLookupTest(unittest.TestCase):
    """Города, категории, сленг — без модели, за 0 токенов."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.lexicon: Lexicon = get_lexicon()

    def cities(self, text: str) -> list[str]:
        return [hit.value for hit in self.lexicon.scan_cities(scan(text))]

    def categories(self, text: str) -> list[str]:
        return order_categories(hit.value for hit in self.lexicon.scan_categories(scan(text)))

    def test_city_in_any_case_form(self) -> None:
        for text in ("Таганрог", "в Таганроге", "из таганрога", "ТАГАНРОГ"):
            with self.subTest(text=text):
                self.assertEqual(self.cities(text), ["taganrog"])

    def test_rostov_with_hyphen_and_spaces(self) -> None:
        for text in ("Ростов-на-Дону", "Ростов на Дону", "в Ростове", "Ростове-на-Дону"):
            with self.subTest(text=text):
                self.assertEqual(self.cities(text), ["rostov-na-donu"])

    def test_azov(self) -> None:
        self.assertEqual(self.cities("в Азове"), ["azov"])

    def test_categories_in_any_form(self) -> None:
        self.assertEqual(self.categories("стол, кресло и лампа"), ("table", "chair", "lamp"))
        self.assertEqual(self.categories("столы, кресла, лампы"), ("table", "chair", "lamp"))

    def test_slang_and_common_synonyms(self) -> None:
        cases = {
            "стул и торшер": ("chair", "lamp"),
            "парта и светильник": ("table", "lamp"),
            "табуретка и ночник": ("chair", "lamp"),
            "люстра": ("lamp",),
            "пуфик": ("chair",),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.categories(text), expected)

    def test_unknown_furniture_is_reported_not_mapped(self) -> None:
        hits = self.lexicon.scan_unsupported_categories(scan("диван и шкаф"))
        self.assertEqual({hit.value for hit in hits}, {"диван", "шкаф"})
        self.assertEqual(self.categories("диван и шкаф"), ())

    def test_unknown_city_is_reported(self) -> None:
        hits = self.lexicon.scan_unsupported_cities(scan("нужен стол в Москве"))
        self.assertEqual([hit.value for hit in hits], ["москва"])
        self.assertEqual(self.cities("нужен стол в Москве"), [])

    def test_mat_does_not_become_a_product(self) -> None:
        text = "бля, срочно нужен сраный стол и кресло"
        self.assertEqual(self.categories(text), ("table", "chair"))

    def test_typo_is_recovered_by_trigram_and_levenshtein(self) -> None:
        self.assertEqual(self.categories("креслр и лампп"), ("chair", "lamp"))
        self.assertEqual(self.cities("в Таганрг"), ["taganrog"])

    def test_short_word_is_not_fuzzily_matched(self) -> None:
        """«лом» не должен превращаться в «стол» — слишком короткое слово."""
        self.assertEqual(self.categories("лом"), ())

    def test_light_stopword_guard(self) -> None:
        """«свет» — не категория: иначе «выключи свет» подбирал бы лампу."""
        self.assertEqual(self.categories("нужен свет"), ())

    def test_order_is_canonical(self) -> None:
        self.assertEqual(order_categories(["lamp", "table", "chair", "table"]),
                         ("table", "chair", "lamp"))


if __name__ == "__main__":
    unittest.main()
