"""Локальный подбор по названиям товаров из каталога без таксономии."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

from app.catalog import load_json_data
from app.catalog.explain import money
from app.config import MAX_OPTIONS
from app.contract import response
from app.model import CatalogError
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
from app.nlp.product_intent import describes_product, terms_before_usage

_OPTION_WORDS = frozenset({"вариант", "варианта", "варианты", "вариантов", "варианте"})
_PURPOSE_WORDS = frozenset({"для", "под", "совместим", "совместима", "совместимый"})
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
    "благодарю", "понятно", "понял", "ясно", "все", "доступные", "г",
}
_FLEETING_VOWEL = re.compile(r"([бвгджзклмнпрстфхцчшщ])[ое]к$")

ProductResolver = Callable[[str], str]


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
    search_fields = [raw.get(key) for key in ("aliases", "keywords", "attributes")]
    search_texts = tuple(extract_all_strings(search_fields))
    return Product(listing_id, title.strip(), price, city_id.strip(), city_name.strip(), search_texts)


class TextCatalog:
    """Индекс заголовков, поисковых псевдонимов и городов текущего JSON."""

    def __init__(self, products: Sequence[Product]) -> None:
        self.products = {product.id: product for product in products}
        if len(self.products) != len(products):
            raise CatalogError("в каталоге повторяются id товаров")
        self.head_postings: dict[str, set[str]] = defaultdict(set)
        self.purpose_postings: dict[str, set[str]] = defaultdict(set)
        aliases: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for city_id, words in (CITY_WORDS | UNSUPPORTED_CITY_WORDS).items():
            for word in (*words, city_id):
                aliases[_terms(word)].add(city_id)
        for form in ("ростов", "ростове", "ростова", "ростову", "ростовом"):
            aliases[(form, "на", "дону")].add("rostov-na-donu")
        catalog_aliases: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for product in products:
            for phrase in (product.title, *product.search_texts):
                postings = self.head_postings
                for position, word in enumerate(_terms(phrase)):
                    if position and word in _PURPOSE_WORDS:
                        postings = self.purpose_postings
                    for form in _forms(word):
                        postings[form].add(product.id)
            catalog_aliases[_terms(product.city_id)].add(product.city_id)
            catalog_aliases[_terms(product.city_name)].add(product.city_id)
        catalog_stems: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for alias, ids in catalog_aliases.items():
            catalog_stems[tuple(stem(word) for word in alias)].update(ids)
        for alias in aliases:
            ids = catalog_stems.get(tuple(stem(word) for word in alias))
            if ids:
                aliases[alias] = ids
        aliases.update(catalog_aliases)
        self.city_aliases = aliases
        self.city_forms: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for alias, ids in aliases.items():
            self.city_forms[tuple(stem(word) for word in alias)].update(ids)
        self.city_forms.update(catalog_stems)
        self.city_lengths = sorted({len(alias) for alias in aliases if alias}, reverse=True)

    @classmethod
    def load(cls, source: Path | str) -> TextCatalog:
        """Читает текущий документ из заданного источника и строит индекс."""
        raw = load_json_data(source)
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
        return smallest.intersection(*others)

    def _city(self, tokens: tuple[str, ...]) -> tuple[str | None, set[int]]:
        """Находит последнее упоминание города и позиции кусочка с названием."""
        city: str | None = None
        positions: set[int] = set()
        bases = tuple(stem(token) for token in tokens)
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
                    city = next(iter(cities)) if len(cities) == 1 else None
                    positions.update(range(start, end))
                    break
        return city, positions

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
            if descriptive and not ids:
                if resolver is not None and len(phrase.encode("utf-8")) <= 4000:
                    # Совпавшие свойства товара обязательны. Слова о его применении
                    # могут называть другой предмет, например бумагу для письма.
                    anchors = [
                        hits for word in required_terms
                        if word not in _PURPOSE_WORDS and (hits := self._postings(word, self.head_postings))
                    ]
                    rewritten = _terms(resolver(phrase))
                    ids = self.matching_ids(rewritten).intersection(*anchors)
                if not ids:
                    missing.append(response.QUESTION_CATEGORIES)
        if missing:
            return _clarify(missing)

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
                f"{product.title} — {money(product.price_rub)}. Самовывоз: {product.city_name}.",
            )
            for product in matches[:limit]
            if product.price_rub <= budget
        ])


def _clarify(fields: Sequence[str]) -> dict[str, object]:
    """Задаёт вопросы в форме ответа контракта без списка фиксированных классов."""
    questions = response.clarify(fields)
    texts = {
        response.QUESTION_CATEGORIES: "Какой товар ищете? Укажите название и важные слова, например «железная решётка».",
        response.QUESTION_CITY: "В каком городе нужен товар?",
        response.QUESTION_BUDGET: "Сколько рублей готовы потратить на один вариант?",
    }
    for question in questions["questions"]:
        question["text"] = texts[question["field"]]
    return questions
