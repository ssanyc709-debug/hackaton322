"""Локальный подбор по названиям товаров из каталога без таксономии."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from app.catalog import load_json_data
from app.catalog.explain import money
from app.config import MAX_OPTIONS
from app.contract import response
from app.model import CatalogError
from app.nlp.extract import is_unintelligible
from app.nlp.lexicon import CITY_WORDS, UNSUPPORTED_CITY_WORDS, stem
from app.nlp.numbers import (
    CAPPING_KINDS,
    FILLER_WORDS,
    MONEY_HINTS,
    UPPER_STRICT,
    UNIT_RUB,
    find_amounts,
    normalize_text,
    tokenize,
)
from app.nlp.product_intent import GENERIC_PRODUCT_WORDS, describes_product, terms_before_usage

_OPTION_WORDS = frozenset({"вариант", "варианта", "варианты", "вариантов", "варианте"})
_PURPOSE_WORDS = frozenset({"для", "под", "совместим", "совместима", "совместимый"})
_OWN_ATTRIBUTES = frozenset({"color", "colour", "material", "brand", "model", "цвет", "материал", "бренд", "модель"})
_NON_MONEY_UNITS = frozenset({"мм", "mm", "шт", "штука", "штуки", "штук", "кг", "kg", "г", "л"})
_PRICE_WORDS = frozenset({"цена", "ценой", "стоимость", "стоимостью"})
_BUDGET_WORDS = MONEY_HINTS | {
    "от", "до", "за", "дешевле", "дороже", "меньше", "менее", "максимум", "примерно",
    "около", "более", "больше", "пределах", "не", "в",
} | _PRICE_WORDS
_QUERY_FILLER = (FILLER_WORDS - {"не"}) | {
    "дай", "дайте", "покажи", "покажите", "предложи", "предложите", "куплю", "купить",
    "товар", "товары", "варианты", "вариант", "варианта", "вариантов", "подбор",
    "покажешь", "показывай", "посмотреть", "поищи", "ищем", "интересует", "лучше", "давай",
    "самовывоз", "город", "теперь", "поменяем", "ок", "спасибо", "подходит",
    "хорошо", "ага", "угу", "да", "ладно", "еще", "привет", "здравствуйте",
    "благодарю", "понятно", "понял", "ясно", "все", "доступные", "г", "либо",
}
_FLEETING_VOWEL = re.compile(r"([бвгджзклмнпрстфхцчшщ])[ое]к$")
_PREFERENCE_CLAUSE = re.compile(r"\b(?:чтобы|чтоб|желательно|по возможности)\b")
_STRICT_CLAUSE_WORDS = frozenset({"не", "без", "из", "только", "кроме", "обязательно", "должен", "должна", "должно"})
_ADJECTIVE_ENDING = re.compile(r"[а-я]{3,}(?:ый|ий|ой|ая|яя|ое|ее|ые|ие|ую|юю|ого|его|ых|их|ому|ему|ыми|ими)$")
_REPEATED_LAST_LETTER = re.compile(r"([а-я])\1+$")

ProductResolver = Callable[[str], str]
PreferenceChecker = Callable[[str], bool]
_PRODUCT_FIELDS = frozenset({"id", "title", "price_rub", "city_id"})


def _terms(text: str) -> tuple[str, ...]:
    """Разбивает название или реплику на поисковые слова и номера моделей."""
    return tuple(token for token in tokenize(normalize_text(text)) if token != "₽")


def _forms(word: str) -> frozenset[str]:
    """Точное слово, основа и форма с беглой гласной («балок», «решёток»)."""
    base = stem(word)
    forms = {word, base} if len(base) >= 3 else {word}
    if len(word) >= 5 and _FLEETING_VOWEL.search(word):
        forms.add(_FLEETING_VOWEL.sub(r"\1к", word))
    return frozenset(forms)


def _looks_like_adjective(word: str) -> bool:
    return bool(_ADJECTIVE_ENDING.fullmatch(_REPEATED_LAST_LETTER.sub(r"\1", word)))


def _city_key(words: Sequence[str]) -> tuple[str, ...]:
    return tuple(stem(stem(word)) for word in words)


def extract_all_strings(data: object) -> Iterator[str]:
    """Обходит вложенные текстовые значения поисковых полей."""
    if isinstance(data, dict):
        for value in data.values():
            yield from extract_all_strings(value)
    elif isinstance(data, list):
        for value in data:
            yield from extract_all_strings(value)
    elif isinstance(data, str) and data.strip():
        yield data.strip()


@dataclass(frozen=True, slots=True)
class Product:
    """Факты о товаре, необходимые для поиска и выдачи."""

    id: str
    title: str
    price_rub: int
    city_id: str
    city_name: str
    search_texts: tuple[str, ...] = ()
    own_attributes: tuple[str, ...] = ()
    city_aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QueryStep:
    """Условия, названные в одной реплике."""

    terms: tuple[str, ...]
    phrase: str
    city_id: str | None
    city_mentioned: bool
    budget: int | None
    limit: int | None


def _parse_product(raw: object, position: int) -> Product:
    """Проверяет поля товара. Допускает распространённые имена полей."""
    if not isinstance(raw, dict):
        raise CatalogError(f"товар #{position}: ожидался JSON-объект")
    listing_id = raw.get("id")
    title = raw.get("title", raw.get("name"))
    price = raw.get("price_rub", raw.get("price"))
    city_id = raw.get("city_id", raw.get("city"))
    city_name = raw.get("city_name", raw.get("city", city_id))
    if any(not isinstance(value, str) or not value.strip() for value in (listing_id, title, city_id, city_name)):
        raise CatalogError(f"товар #{position}: нужны непустые id, title/name и city_id/city")
    if type(price) is not int or price < 1:
        raise CatalogError(f"товар #{position}: price_rub/price должен быть целым числом > 0")
    search_fields = [raw.get(key) for key in ("aliases", "keywords")]
    search_texts = tuple(extract_all_strings(search_fields))
    attributes = raw.get("attributes")
    own = tuple(
        value for key, value in attributes.items() if normalize_text(key) in _OWN_ATTRIBUTES and isinstance(value, str)
    ) if isinstance(attributes, dict) else ()
    city_aliases = tuple(extract_all_strings(raw.get("city_aliases")))
    return Product(listing_id, title.strip(), price, city_id.strip(), city_name.strip(), search_texts, own, city_aliases)


def is_text_catalog(document: Any) -> bool:
    """Распознаёт форматы, отличимые от массива объявлений по ТЗ."""
    if isinstance(document, dict):
        document = document.get("listings", document.get("products"))
    return isinstance(document, list) and any(
        isinstance(row, dict) and (
            any(short in row and standard not in row for short, standard in (
                ("name", "title"), ("price", "price_rub"), ("city", "city_id"),
            ))
            or (_PRODUCT_FIELDS <= row.keys() and "category" not in row and "description" not in row)
        )
        for row in document
    )


class TextCatalog:
    """Индекс заголовков, поисковых псевдонимов и городов текущего JSON."""

    def __init__(self, products: Sequence[Product]) -> None:
        self.products = {product.id: product for product in products}
        if len(self.products) != len(products):
            raise CatalogError("в каталоге повторяются id товаров")
        self.head_postings: dict[str, set[str]] = defaultdict(set)
        self.purpose_postings: dict[str, set[str]] = defaultdict(set)
        self.name_scopes: dict[str, list[tuple[set[str], set[str]]]] = defaultdict(list)
        aliases: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for city_id, words in (CITY_WORDS | UNSUPPORTED_CITY_WORDS).items():
            for word in (*words, city_id):
                aliases[_terms(word)].add(city_id)
        for form in ("ростов", "ростове", "ростова", "ростову", "ростовом"):
            aliases[(form, "на", "дону")].add("rostov-na-donu")
        catalog_aliases: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for product in products:
            facts = {form for value in product.own_attributes for word in _terms(value) for form in _forms(word)}
            for phrase in (product.title, *product.search_texts):
                head = set(facts)
                purpose: set[str] = set()
                scope = head
                for position, word in enumerate(_terms(phrase)):
                    if position and word in _PURPOSE_WORDS:
                        scope = purpose
                    scope.update(_forms(word))
                self.name_scopes[product.id].append((head, purpose))
                for form in head:
                    self.head_postings[form].add(product.id)
                for form in purpose:
                    self.purpose_postings[form].add(product.id)
            for name in (product.city_id, product.city_name, *product.city_aliases):
                if terms := _terms(name):
                    catalog_aliases[terms].add(product.city_id)
        catalog_stems: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for alias, ids in catalog_aliases.items():
            catalog_stems[_city_key(alias)].update(ids)
        for alias in aliases:
            ids = catalog_stems.get(_city_key(alias))
            if ids:
                aliases[alias] = ids
        aliases.update(catalog_aliases)
        self.city_aliases = aliases
        self.city_forms: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for alias, ids in aliases.items():
            self.city_forms[_city_key(alias)].update(ids)
        self.city_forms.update(catalog_stems)
        self.city_lengths = sorted({len(alias) for alias in aliases if alias}, reverse=True)

    @classmethod
    def load(cls, source: Path | str) -> TextCatalog:
        """Читает текущий документ из заданного источника и строит индекс."""
        return cls.from_document(load_json_data(source))

    @classmethod
    def from_document(cls, raw: Any) -> TextCatalog:
        """Строит индекс по одному снимку каталога."""
        if isinstance(raw, dict):
            raw = raw.get("listings", raw.get("products"))
        if not isinstance(raw, list):
            raise CatalogError("для текстового поиска нужен массив товаров или объект с listings/products")
        return cls([_parse_product(row, position) for position, row in enumerate(raw, 1)])

    def _postings(self, word: str, index: dict[str, set[str]]) -> set[str]:
        """Объединяет объявления для допустимых словоформ одного слова."""
        return set().union(*(index.get(form, ()) for form in _forms(word)))

    def matching_ids(self, terms: Sequence[str]) -> set[str]:
        """Возвращает только объявления, в которых совпали все поисковые слова."""
        if not terms:
            return set()
        separator = next((i for i, word in enumerate(terms) if word in _PURPOSE_WORDS), len(terms))
        postings = [
            self._postings(word, self.head_postings if i < separator else self.purpose_postings)
            for i, word in enumerate(terms)
        ]
        smallest, *others = sorted(postings, key=len)
        candidates = smallest.intersection(*others)
        head_terms = tuple(map(_forms, terms[:separator]))
        purpose_terms = tuple(map(_forms, terms[separator:]))
        return {
            item_id for item_id in candidates
            if any(all(forms & head for forms in head_terms) and all(forms & purpose for forms in purpose_terms)
                   for head, purpose in self.name_scopes[item_id])
        }

    def _city(self, tokens: tuple[str, ...]) -> tuple[str | None, set[int]]:
        """Находит последнее упоминание города и позиции кусочка с названием."""
        selected: set[str] = set()
        positions: set[int] = set()
        bases = _city_key(tokens)
        previous_end = 0
        for start in range(len(tokens)):
            if start in positions:
                continue
            for length in self.city_lengths:
                end = start + length
                if end > len(tokens):
                    continue
                cities = self.city_aliases.get(tokens[start:end])
                if cities is None and (start == 0 or tokens[start - 1] in {"в", "во", "из"}):
                    cities = self.city_forms.get(bases[start:end])
                if cities:
                    positions.update(range(start, end))
                    qualifier = start - 1 if start and tokens[start - 1] in {"в", "во", "из"} else start
                    if qualifier and tokens[qualifier - 1] == "не":
                        positions.update(range(qualifier - 1, start))
                        selected.difference_update(cities)
                    elif {"или", "либо"}.intersection(tokens[previous_end:start]):
                        selected.update(cities)
                    else:
                        selected = set(cities)
                    previous_end = end
                    break
        return (next(iter(selected)) if len(selected) == 1 else None), positions

    def parse_query(self, text: str) -> QueryStep:
        """Отделяет город, бюджет и число альтернатив от слов товара."""
        normalized = normalize_text(text)
        tokens = tokenize(normalized)
        city, ignored = self._city(tokens)
        city_mentioned = bool(ignored)
        options = None
        budget_amounts = []
        for amount in find_amounts(normalized):
            end = amount.position + len(amount.raw.split())
            if amount.following_position is not None and tokens[amount.following_position] in _OPTION_WORDS:
                options = min(MAX_OPTIONS, max(1, amount.value))
                ignored.update(range(amount.position, end + 1))
                continue
            previous_word = tokens[amount.position - 1] if amount.position else ""
            has_money_context = amount.trigger in _BUDGET_WORDS or previous_word in _PRICE_WORDS | {"за"}
            short_amount = amount.position == 0 and end == len(tokens)
            currency_suffix = amount.raw.endswith((" тысяч", " тысячи", " тыс", " к", " k"))
            if amount.following_position is not None and tokens[amount.following_position] in _NON_MONEY_UNITS:
                continue
            if amount.unit == UNIT_RUB or (
                amount.unit is None and (
                    has_money_context
                    or currency_suffix
                    or (short_amount and amount.value >= 1000 and not self._postings(tokens[0], self.head_postings))
                )
            ):
                budget_amounts.append(amount)
                ignored.update(range(amount.position, end))
                for previous in range(amount.position - 1, -1, -1):
                    if tokens[previous] not in _BUDGET_WORDS:
                        break
                    ignored.add(previous)

        # Верхняя граница из денежной группы. Строгое «дешевле N» в рублях — N−1.
        budgets = [
            amount.value - (amount.kind == UPPER_STRICT)
            for amount in budget_amounts
            if amount.kind in CAPPING_KINDS
        ]
        budget = budgets[-1] if budgets else None
        if tokens[:1] == ("тогда",) and (city_mentioned or budget is not None):
            ignored.add(0)
        selected = {
            i for i, word in enumerate(tokens)
            if i not in ignored and word != "₽" and word not in _QUERY_FILLER and word not in _PURPOSE_WORDS
        }
        first, last = (min(selected), max(selected)) if selected else (0, 0)
        # «Для» значимо, когда стоит между словами названия товара.
        terms = tuple(
            word for i, word in enumerate(tokens)
            if i in selected or (first < i < last and i not in ignored and word in _PURPOSE_WORDS)
        )
        phrase = " ".join(word for i, word in enumerate(tokens) if i not in ignored and word != "₽")
        return QueryStep(terms, phrase, city, city_mentioned, budget, options)

    def answer(
        self, texts: Sequence[str], *, resolver: ProductResolver | None = None,
        preference_checker: PreferenceChecker | None = None,
    ) -> dict[str, object]:
        """Подбирает альтернативы одной товарной позиции по истории диалога."""
        city = None
        budget = None
        limit = MAX_OPTIONS
        terms: tuple[str, ...] = ()
        phrase = ""
        for text in texts:
            step = self.parse_query(text)
            if step.city_mentioned:
                city = step.city_id
            if step.budget is not None:
                budget = step.budget
            if step.limit is not None:
                limit = step.limit
            if step.terms:
                terms = step.terms
                phrase = step.phrase
        descriptive = describes_product(phrase)
        if descriptive:
            before_usage = terms_before_usage(phrase)
            required_terms = tuple(word for word in terms if word in before_usage)
        else:
            required_terms = terms

        missing = []
        if budget is None:
            missing.append(response.QUESTION_BUDGET)
        if city is None:
            missing.append(response.QUESTION_CITY)
        ids = set()
        if not terms or "не" in required_terms or "без" in required_terms:
            missing.append(response.QUESTION_CATEGORIES)
        else:
            ids = self.matching_ids(terms)
            if not ids:
                preference = not descriptive and self._has_catalog_subject(phrase)
                ids, wanted_description = self._resolve_product(
                    terms, required_terms, phrase, descriptive or preference, resolver,
                    check_preference=preference, preference_checker=preference_checker,
                )
                if wanted_description and not ids:
                    missing.append(response.QUESTION_CATEGORIES)
        if missing:
            titles = []
            if (response.QUESTION_CATEGORIES in missing and descriptive
                    and not {"не", "без"}.intersection(required_terms)):
                titles = self._related_titles(terms, city, budget)
            cities = sorted({product.city_name for product in self.products.values()})
            return _clarify(missing, titles, cities)

        if not ids:
            return response.no_match("В каталоге нет совпадений по названию. Проверьте название товара.")
        matches = sorted(
            (self.products[item_id] for item_id in ids if self.products[item_id].city_id == city),
            key=lambda product: (product.price_rub, product.id),
        )
        if not matches:
            return response.no_match(
                "В выбранном городе нет товара с таким названием. Уточните название или выберите другой город."
            )
        if matches[0].price_rub > budget:
            return response.no_match(
                f"В выбранном городе подходящие товары стоят от {money(matches[0].price_rub)}. "
                "Увеличьте бюджет, чтобы увидеть варианты."
            )
        return response.ok([
            response.option(
                [product.id],
                product.price_rub,
                f"{product.title} — {money(product.price_rub)}. Город: {product.city_name}.",
            )
            for product in matches[:limit]
            if product.price_rub <= budget
        ])

    def _related_titles(self, terms: Sequence[str], city: str | None, budget: int | None) -> list[str]:
        """Предлагает товары с совпадающими словами из каталога."""
        scores: dict[str, int] = defaultdict(int)
        for term in set(terms) - _PURPOSE_WORDS - GENERIC_PRODUCT_WORDS:
            for item_id in self._postings(term, self.head_postings) | self._postings(term, self.purpose_postings):
                scores[item_id] += 1
        available = [
            self.products[item_id] for item_id in scores
            if (city is None or self.products[item_id].city_id == city)
            and (budget is None or self.products[item_id].price_rub <= budget)
        ]
        available.sort(key=lambda product: (-scores[product.id], product.price_rub, product.id))
        return list(dict.fromkeys(product.title for product in available))[:MAX_OPTIONS]

    def _has_catalog_subject(self, phrase: str) -> bool:
        """Распознаёт известный предмет с отдельной фразой о пожеланиях."""
        head, *tail = _PREFERENCE_CLAUSE.split(phrase, maxsplit=1)
        if not tail:
            return False
        name = tuple(term for term in _terms(head) if term not in _QUERY_FILLER)
        return bool(self.matching_ids(name))

    def _resolve_product(
        self, terms: Sequence[str], required_terms: Sequence[str], phrase: str,
        descriptive: bool, resolver: ProductResolver | None,
        *, check_preference: bool, preference_checker: PreferenceChecker | None,
    ) -> tuple[set[str], bool]:
        """Разбирает осмысленную фразу, которую индекс не сопоставил с названием."""
        postings = {word: self._postings(word, self.head_postings)
                    for word in required_terms if word not in _PURPOSE_WORDS}
        anchors = [hits for hits in postings.values() if hits]
        unknown_words = {
            word for word in terms if word.isalpha() and word not in (GENERIC_PRODUCT_WORDS | _PURPOSE_WORDS)
        }
        unmatched_modifier = (
            not descriptive and not _PURPOSE_WORDS.intersection(terms)
            and any(hits and not _looks_like_adjective(word) for word, hits in postings.items())
            and any(not hits and _looks_like_adjective(word) for word, hits in postings.items())
        )
        uncertain = descriptive or unmatched_modifier or (len(unknown_words) > 1 and not anchors)
        if not uncertain or is_unintelligible(phrase):
            return set(), False
        if check_preference or unmatched_modifier:
            detail = _PREFERENCE_CLAUSE.split(phrase, maxsplit=1)[1] if check_preference else phrase
            if _STRICT_CLAUSE_WORDS.intersection(_terms(detail)) or any(char.isdigit() for char in detail):
                return set(), True
        if unmatched_modifier:
            known = tuple(word for word, hits in postings.items() if hits)
            ids = self.matching_ids(known)
            if not ids or preference_checker is None:
                return set(), True
            remaining = " ".join(word for word, hits in postings.items() if not hits)
            context = f"Запрос: {phrase}\nСовпало: {' '.join(known)}\nОстальное: {remaining}"
            if len(context.encode("utf-8")) > 4000:
                return set(), True
            return (ids if preference_checker(context) else set()), True
        if resolver is None or len(phrase.encode("utf-8")) > 4000:
            return set(), True

        # Сохраняем распознанные свойства товара при перефразировании.
        rewritten = _terms(resolver(phrase))
        return self.matching_ids(rewritten).intersection(*anchors), True


def _clarify(fields: Sequence[str], titles: Sequence[str], cities: Sequence[str]) -> dict[str, object]:
    """Задаёт вопросы в форме ответа контракта без списка фиксированных классов."""
    questions = response.clarify(fields)
    texts = {
        response.QUESTION_CATEGORIES: "Какой товар ищете? Укажите название или опишите, как он выглядит.",
        response.QUESTION_CITY: "В каком городе нужен товар?",
        response.QUESTION_BUDGET: "Сколько рублей готовы потратить на один вариант?",
    }
    if titles:
        choices = ", ".join(f"«{title}»" for title in titles)
        texts[response.QUESTION_CATEGORIES] = f"Какой товар ищете? Подойдёт один из этих вариантов: {choices}?"
    if cities:
        prefix = "Например" if len(cities) > 4 else "Города в каталоге"
        texts[response.QUESTION_CITY] += f" {prefix}: {', '.join(cities[:4])}."
    for question in questions["questions"]:
        question["text"] = texts[question["field"]]
    return questions
