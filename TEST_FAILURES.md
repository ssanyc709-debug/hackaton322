# Отчет об ошибках (21.11.2026 06:59)

Прогон: `python3 -m unittest discover -s tests -t .` (Python 3.14.4)
Итог: **628 тестов, 623 успешно, 5 упавших, 6 пропущенных.**

Все 5 падений — новый баг «мусорная реплика подтверждает прежний подбор».
Новые тесты добавлены в `tests/test_dialogue.py` (`UnintelligibleMessageTest`,
`UnintelligibleRegressionTest`). Бэкенд не изменялся.

---

## Упавшие тесты (FAILED)

### 1. **test_gibberish_after_complete_conditions_asks_categories** (файл `tests/test_dialogue.py`)

- **Что проверяли:** после полного набора условий («Стол до 5000. Таганрог.»)
  реплика без распознанных условий и без чисел («ыаллфдылааввы») не должна
  подтверждать прежний подбор — категории считаются неизвестными.
- **Ожидалось:** `status == "clarify"`, `fields == ["categories"]`, ключа
  `options` в ответе нет.
- **Получено:** `status == "ok"`, повторён прежний подбор из трёх столов
  (`demo-1-1-11`, `demo-1-1-10`, `demo-1-1-04`, суммы 2500 / 2700 / 3200 ₽).
- **Трейсбек:**
  ```
  AssertionError: 'ok' != 'clarify'
  - ok
  + clarify :
  {'status': 'ok', 'options': [{'listing_ids': ['demo-1-1-11'], 'total_rub': 2500, …},
                               {'listing_ids': ['demo-1-1-10'], 'total_rub': 2700, …},
                               {'listing_ids': ['demo-1-1-04'], 'total_rub': 3200, …}]}
  ```

### 2. **test_repeated_letters_after_complete_conditions_asks_categories** (файл `tests/test_dialogue.py`)

- **Что проверяли:** то же для реплики «фффффф» (шесть одинаковых букв).
- **Ожидалось:** `clarify`, `fields == ["categories"]`.
- **Получено:** `ok` с прежними тремя столами (2500 / 2700 / 3200 ₽).
- **Трейсбек:**
  ```
  AssertionError: 'ok' != 'clarify'
  - ok
  + clarify
  ```

### 3. **test_latin_gibberish_after_complete_conditions_asks_categories** (файл `tests/test_dialogue.py`)

- **Что проверяли:** то же для латиницы «asdfgh» (набор с клавиатуры).
- **Ожидалось:** `clarify`, `fields == ["categories"]`.
- **Получено:** `ok` с прежними тремя столами.
- **Трейсбек:**
  ```
  AssertionError: 'ok' != 'clarify'
  - ok
  + clarify
  ```

### 4. **test_question_in_clarify_is_not_empty** (файл `tests/test_dialogue.py`)

- **Что проверяли:** в `clarify` у каждого вопроса есть непустой `text`
  (§4: «непустой массив `questions`», у вопроса два поля `field` и `text`).
- **Ожидалось:** `clarify`, `fields == ["categories"]`, `text` не пустой.
- **Получено:** `ok` — до проверки текста вопроса дело не дошло, потому что
  статус не `clarify`.
- **Трейсбек:**
  ```
  AssertionError: 'ok' != 'clarify'
  - ok
  + clarify
  ```

### 5. **test_gibberish_from_the_bug_report_with_assistant_reply** (файл `tests/test_dialogue.py`)

- **Что проверяли:** точная последовательность из отчёта о баге, включая реплику
  помощника между запросами покупателя:
  `user: «Стол до 5000. Таганрог.»` → `assistant: «Готово.»` →
  `user: «ыаллфдылааввы»`.
- **Ожидалось:** `clarify` с полями `["categories"]`.
- **Получено:** `ok` с прежним подбором (воспроизводит баг на уровне контракта,
  не только через внутренний API).
- **Трейсбек:**
  ```
  AssertionError: 'ok' != 'clarify'
  - ok
  + clarify
  ```

---

## Причина (для бэкендера)

Данные из разбора, файлы и строки указаны по текущему состоянию репозитория.

1. `app/nlp/merge.py::merge_history` (~76–84) — левая свёртка `extract_message`
   по репликам покупателя.
2. Для реплики «ыаллфдылааввы» `step.mentioned` пуст (нет города, бюджета,
   категорий, ширины, температуры, регулировки).
3. `app/nlp/merge.py::merge_step` (~140–205) применяет только поля из
   `step.mentioned`, поэтому все прежние слоты (`city= taganrog`, `budget=5000`,
   `categories=('table',)`) остаются как есть — реплика неотличима от «спасибо».
4. `app/nlp/extract.py::missing_fields` (~317–331) возвращает пустой кортеж.
5. `app/pipeline.py::answer` (~57–68) не видит недостающих условий и уходит
   в `build_plan` → снова `ok` с тем же комплектом.

Требуемое поведение (по постановке):

* локальный детектор `is_unintelligible(text)` в `app/nlp/extract.py` (0 токенов
  модели); реплика «непонятна», если одновременно:
  `slots.mentioned` пуст, `find_amounts` пуст, и есть хотя бы один алфавитный
  токен длиной ≥ 4, который не является городом/категорией (включая
  неподдерживаемые), `NOISE_WORDS`/`STOPWORD_GUARDS`, `FILLER_WORDS` или
  подтверждением из белого списка (спасибо, пожалуйста, да, нет, ок, окей,
  хорошо, ладно, понятно, ясно, ага, угу, привет, здравствуйте, благодарю,
  беру, подходит);
* в `app/pipeline.py::answer` ПОСЛЕ `merge_history` (то есть после возможного
  вызова модели `resolve_missing`) обнулить категории
  (`categories=()`, `categories_unknown=()`, `categories_explicit=False`,
  `categories_additive=False`, `categories_removed=()`) — тогда
  `missing_fields` сам вернёт `categories` → `response.clarify(...)`.

Контракт не меняется: статусы только `ok` / `clarify` / `no_match`,
`clarify` → exit 0, логи только в stderr, `description` не читается.

---

## Пропущенные тесты (SKIPPED из-за багов)

Пропусков, вызванных этим багом, нет. Все 6 `skip` — известные и не связаны
с текущей задачей:

- **test_request_document_matches_schema** (`tests/test_contract.py`):
  Причина — не установлен пакет `jsonschema` (проверка схемы запроса
  пропускается; используется встроенная структурная проверка).
- **test_documents_pass_json_schema** (`tests/test_contract.py`):
  Причина — не установлен пакет `jsonschema` (проверка документов ответа
  по `schemas/response.schema.json`).
- **test_both_cities_are_presented** (`tests/test_qa_history.py::MultipleCitiesTest`):
  Причина — вне ТЗ §3: «извлекается один город», варианты в обоих не требуются.
- **test_three_cities_are_all_presented** (`tests/test_qa_history.py::MultipleCitiesTest`):
  Причина — вне ТЗ §3: «извлекается один город», три города не требуются.
- **test_no_match_when_neither_city_fits** (`tests/test_qa_history.py::MultipleCitiesTest`):
  Причина — вне ТЗ §3: `no_match` объясняет выбранный город, а не все
  упомянутые.
- **test_or_in_a_follow_up_adds_the_city** (`tests/test_qa_history.py::MultipleCitiesTest`):
  Причина — вне ТЗ §3: «или в Азове» заменяет город — извлекается один город.

---

## Регрессии, которые обязаны остаться зелёными (сейчас зелёные)

Проверены в том же прогоне, чтобы исполнитель не «починил» баг слишком широко:

| Тест | Диалог | Сейчас |
|---|---|---|
| `test_numeric_answer_keeps_the_selection` | «Стол и кресло в Таганроге» → «15 тысяч» | `ok` ✅ |
| `test_numeric_answer_after_width_and_categories` | «Стол не шире метра, кресло и лампа. Таганрог.» → «15 тысяч» | `ok` ✅ |
| `test_thanks_keeps_the_selection` | «Стол до 5000. Таганрог.» → «Спасибо!» | `ok` ✅ |
| `test_short_confirmation_keeps_the_selection` | «Стол до 5000. Таганрог.» → «ок» | `ok` ✅ |
| `test_unsupported_category_in_one_message_still_asks_categories` | «Диван в Таганроге до 30000» | `clarify` `["categories"]` ✅ |
| `test_gibberish_without_conditions_asks_everything` | «Хочу мебель» → «ыаллфдылааввы» | `clarify` `["budget_rub","city_id","categories"]` ✅ |
| `test_unrelated_message_changes_nothing` (существующий) | «Стол в Таганроге до 20000» → «Спасибо!» | слоты не изменились ✅ |

---

## Воспроизведение через `run` (после исправления должно быть так)

```bash
# 1) мусорная реплика → clarify categories, exit 0
printf '%s' '{"messages":[{"role":"user","text":"Стол до 5000. Таганрог."},{"role":"assistant","text":"Готово."},{"role":"user","text":"ыаллфдылааввы"}]}' \
  | ASSISTANT_OFFLINE=1 ./run --catalog data/catalog.json

# Сейчас:   {"status":"ok", …}          ← баг
# Ожидается: {"status":"clarify","questions":[{"field":"categories","text":"…"}]}

# 2) без мусорной реплики → ok
printf '%s' '{"messages":[{"role":"user","text":"Стол до 5000. Таганрог."}]}' \
  | ASSISTANT_OFFLINE=1 ./run --catalog data/catalog.json
# Сейчас и ожидается: {"status":"ok", …}
```

Примечание: `./run` в текущем чекауте имеет права `100644`, поэтому запуск
через него требует `bash run` или `python3 -m app.main` (это отдельное
наблюдение, к данному багу не относится).
