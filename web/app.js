// Логика демо-сайта «ИИ-помощник покупателя».
//
// Жёсткие правила, заложенные в код:
//   * любой текст с сервера выводится ТОЛЬКО через textContent — ни innerHTML,
//     ни insertAdjacentHTML, ни eval/new Function здесь нет сознательно: названия
//     и объяснения приходят из данных, которые мы не контролируем;
//   * история диалога живёт в памяти страницы и каждый раз уходит ЦЕЛИКОМ;
//   * роли обязаны чередоваться, первая и последняя — user. Поэтому, ответив на
//     уточнение, перед репликой покупателя добавляем assistant-сообщение с текстом
//     всех вопросов — иначе бэкенд отвергнет вход;
//   * сбой сервера/сети показывается отдельным блоком и НИКОГДА не превращается
//     в «нет вариантов».
"use strict";

var REQUEST_TIMEOUT_MS = 120000; // 120 секунд: дольше fetch просто не ждём
var MAX_MESSAGES = 20; // столько сообщений принимает контракт

// Русские подписи для поля field из ответа clarify.
var FIELD_LABELS = {
  budget_rub: "бюджет",
  city_id: "город",
  categories: "товары",
  width_cm: "ширина",
  adjustable_height: "регулировка высоты",
  color_temperature_k: "температура",
};

// Панель примеров: три реплики разбираются лексиконом, четвёртая — моделью.
var EXAMPLES = [
  { text: "Стол, кресло и лампа. Таганрог. До 15000.", hint: "полный подбор" },
  { text: "Нужен стол в Таганроге", hint: "спросит бюджет" },
  { text: "Нужен стол в Таганроге до 100 ₽", hint: "невыполнимо" },
  { text: "Нужна деревяшка с четырьмя ножками. Таганрог, до 15000.", hint: "через модель" },
];

var state = {
  history: [], // [{role, text}] — отправляется целиком
  cards: new Map(), // id → карточка из того же ответа, что и подбор
  dialogueToken: "",
  busy: false,
};

var els = {
  status: document.getElementById("status"),
  chat: document.getElementById("chat"),
  answer: document.getElementById("answer"),
  metrics: document.getElementById("metrics"),
  form: document.getElementById("composer"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  newDialog: document.getElementById("new-dialog"),
  examples: document.getElementById("examples"),
  examplePanel: document.getElementById("example-panel"),
};

var PLACEHOLDER = "Введите запрос, чтобы увидеть подходящие товары.";

// --- сеть -------------------------------------------------------------------

function setStatus(text, broken) {
  els.status.textContent = text;
  els.status.dataset.state = broken ? "error" : "ok";
}

/** fetch с таймаутом через AbortController. Никакого «вечного спиннера». */
async function fetchWithTimeout(url, options) {
  var controller = new AbortController();
  var timer = setTimeout(function () {
    controller.abort();
  }, REQUEST_TIMEOUT_MS);
  try {
    var merged = Object.assign({}, options, { signal: controller.signal });
    return await fetch(url, merged);
  } finally {
    clearTimeout(timer);
  }
}

function serviceError(message) {
  var error = new Error(message || "Сервис недоступен, попробуйте ещё раз");
  error.serviceDown = true;
  return error;
}

/** Возвращает подбор, карточки из того же снимка каталога и метрики. */
async function askBackend(messages, dialogueToken) {
  var started = performance.now();
  var response;
  var headers = { "Content-Type": "application/json", Accept: "application/json" };
  if (dialogueToken) headers["X-Dialogue-Token"] = dialogueToken;
  try {
    response = await fetchWithTimeout("/api/answer?include=cards", {
      method: "POST",
      headers: headers,
      body: JSON.stringify({ messages: messages }),
    });
  } catch (error) {
    if (error && error.name === "AbortError") {
      throw serviceError("запрос длился дольше 120 секунд и был прерван");
    }
    throw serviceError();
  }
  var payload = await response.json().catch(function () {
    return null;
  });
  if (!response.ok) {
    var reason = payload && payload.error ? payload.error : "HTTP " + response.status;
    var error = serviceError(reason);
    error.metrics = readMetrics(response, started);
    throw error;
  }
  return {
    document: payload.answer,
    items: payload.items,
    dialogueToken: response.headers.get("X-Dialogue-Token") || "",
    metrics: readMetrics(response, started),
  };
}

function readMetrics(response, started) {
  var tokenHeader = response.headers.get("X-Model-Tokens");
  var modelElapsedHeader = response.headers.get("X-Model-Elapsed-Ms");
  return {
    modelCalls: Number(response.headers.get("X-Model-Calls") || 0),
    tokens: tokenHeader === null ? null : Number(tokenHeader),
    elapsedMs: Math.round(performance.now() - started),
    modelElapsedMs: modelElapsedHeader === null ? null : Number(modelElapsedHeader),
    modelUnavailable: response.headers.get("X-Model-Unavailable") === "1",
    modelReused: response.headers.get("X-Model-Reused") === "1",
  };
}

// --- мелкие помощники -------------------------------------------------------

function el(tag, className, text) {
  var node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text; // только текст
  return node;
}

function money(value) {
  return new Intl.NumberFormat("ru-RU").format(Number(value) || 0) + " ₽";
}

function fieldLabel(field) {
  return FIELD_LABELS[field] || field;
}

function attributesText(card) {
  var attributes = card.attributes || {};
  var parts = [];
  if (typeof attributes.width_cm === "number") parts.push("ширина " + attributes.width_cm + " см");
  if (typeof attributes.adjustable_height === "boolean") {
    parts.push(attributes.adjustable_height ? "регулировка высоты: есть" : "регулировка высоты: нет");
  }
  if (typeof attributes.color_temperature_k === "number") {
    parts.push("температура " + attributes.color_temperature_k + " K");
  }
  Object.keys(attributes).forEach(function (key) {
    if (["width_cm", "adjustable_height", "color_temperature_k"].indexOf(key) !== -1) return;
    parts.push(fieldLabel(key) + ": " + String(attributes[key]));
  });
  return parts.join(", ");
}

// --- чат --------------------------------------------------------------------

function addBubble(role, text) {
  var item = el("li", "bubble bubble--" + role);
  item.append(el("p", null, text));
  els.chat.append(item);
  els.chat.scrollTop = els.chat.scrollHeight;
  return item;
}

// --- панель ответа ----------------------------------------------------------

function renderOption(option, index) {
  var listingIds = option.listing_ids || [];
  var single = listingIds.length === 1 ? state.cards.get(listingIds[0]) : null;
  if (single) {
    var listing = el("article", "option option--single");
    var listingBody = el("div", "option__body");
    listingBody.append(el("h3", "option__title", single.title || listingIds[0]));
    if (single.city_name) listingBody.append(el("p", "option__explanation", single.city_name));
    var attributes = attributesText(single);
    if (attributes) listingBody.append(el("p", "option__explanation", attributes));
    listing.append(listingBody, el("p", "option__total", money(option.total_rub)));
    return listing;
  }
  var card = el("article", "option");
  var head = el("div", "option__head");
  head.append(el("h3", "option__title", "Вариант " + (index + 1)));
  head.append(el("p", "option__total", money(option.total_rub)));
  card.append(head);
  card.append(el("p", "option__explanation", option.explanation));

  var list = el("ul", "items");
  listingIds.forEach(function (id) {
    var item = state.cards.get(id);
    var row = el("li", "item");
    if (!item) {
      row.append(el("span", "item__title", id));
      list.append(row);
      return;
    }
    row.append(el("span", "item__category", item.category_name || item.category || ""));
    row.append(el("span", "item__title", item.title || id));
    var details = attributesText(item);
    if (details) row.append(el("span", "item__details", details));
    var place = item.city_name ? item.city_name + " · " : "";
    row.append(el("span", "item__price", place + money(item.price_rub)));
    list.append(row);
  });
  card.append(list);
  return card;
}

function renderOk(document_) {
  var wrap = el("div", "answer answer--ok");
  document_.options.forEach(function (option, index) {
    wrap.append(renderOption(option, index));
  });
  return wrap;
}

function renderClarify(document_) {
  var wrap = el("div", "answer answer--clarify");
  wrap.append(el("p", "answer__lead", "Чтобы подобрать, нужно уточнить:"));
  var list = el("ul", "questions");
  document_.questions.forEach(function (question) {
    var row = el("li", "question");
    row.append(el("span", "question__field", fieldLabel(question.field)));
    row.append(el("span", "question__text", question.text));
    list.append(row);
  });
  wrap.append(list);
  wrap.append(el("p", "answer__hint", "Ответьте в поле слева — помощник учтёт уточнение."));
  return wrap;
}

function renderNoMatch(document_) {
  var wrap = el("div", "answer answer--no-match");
  wrap.append(el("p", "answer__lead", "Подобрать не удалось."));
  wrap.append(el("p", "answer__message", document_.message));
  return wrap;
}

function renderServiceError(error) {
  var wrap = el("div", "answer answer--error");
  wrap.append(el("p", "answer__lead", "Сервис недоступен, попробуйте ещё раз."));
  wrap.append(el("p", "answer__message", error.message));
  return wrap;
}

function renderDocument(document_) {
  if (!document_ || typeof document_.status !== "string") {
    return renderServiceError({ message: "ответ сервера без поля status" });
  }
  if (document_.status === "ok") return renderOk(document_);
  if (document_.status === "clarify") return renderClarify(document_);
  if (document_.status === "no_match") return renderNoMatch(document_);
  return renderServiceError({ message: "неизвестный статус: " + document_.status });
}

function showInPanel(node) {
  els.answer.replaceChildren(node);
}

/** Короткая реплика помощника в чате; подробности — в панели справа. */
function echoInChat(document_) {
  if (document_.status === "ok") {
    var summaries = ["Нашёл подходящий вариант.", "Нашёл два варианта.", "Нашёл три варианта."];
    addBubble("assistant", summaries[document_.options.length - 1] || "Подобрал варианты.");
    return;
  }
  if (document_.status === "clarify") {
    document_.questions.forEach(function (question) {
      addBubble("assistant", fieldLabel(question.field) + ": " + question.text);
    });
    return;
  }
  if (document_.status === "no_match") {
    addBubble("assistant", document_.message);
    return;
  }
  addBubble("assistant", "Не удалось разобрать ответ сервера.");
}

/** Текст ассистента для истории: он нужен, чтобы роли чередовались. */
function assistantHistoryText(document_) {
  var text = "";
  if (document_.status === "clarify") {
    text = document_.questions
      .map(function (question) {
        return question.text;
      })
      .join(" ");
  } else if (document_.status === "no_match") {
    text = document_.message;
  } else {
    text = (document_.options || [])
      .map(function (option) {
        return option.explanation;
      })
      .join(" ");
  }
  // Пустая реплика нарушила бы контракт (text должен быть непустым).
  return text && text.trim() ? text : "Готово.";
}

/**
 * Готовит историю к отправке: контракт принимает не больше 20 сообщений.
 * Старые пары user/assistant отбрасываем, чтобы диалог не упирался в лимит
 * мёртвой кнопкой. Инварианты (первая реплика — user, роли чередуются)
 * сохраняются, иначе бэкенд вернёт 400.
 */
function fitHistory(history) {
  var trimmed = history.slice();
  while (trimmed.length + 1 > MAX_MESSAGES) {
    trimmed.splice(0, 2); // самая старая пара user + assistant
  }
  while (trimmed.length && trimmed[0].role !== "user") {
    trimmed.shift();
  }
  return trimmed;
}

function showMetrics(metrics) {
  var model;
  if (metrics.modelCalls) {
    var cost = metrics.tokens === null ? "расход токенов неизвестен" : metrics.tokens + " токенов";
    var modelTime = metrics.modelElapsedMs === null ? "" : ", ожидание шлюза " + metrics.modelElapsedMs + " мс";
    model = "вызов модели: да (" + cost + modelTime + ")";
  } else if (metrics.modelReused) {
    model = "использован разбор модели из диалога (0 токенов)";
  } else if (metrics.modelUnavailable) {
    model = "вызов модели: нет (модель недоступна, разбор лексиконом)";
  } else {
    model = "0 токенов";
  }
  els.metrics.hidden = false;
  els.metrics.textContent = metrics.elapsedMs ? "Ответ за " + metrics.elapsedMs + " мс · " + model : model;
}

// --- отправка ---------------------------------------------------------------

function setBusy(busy) {
  state.busy = busy;
  els.send.disabled = busy;
  els.input.disabled = busy;
  els.send.textContent = busy ? "Подбираю…" : "Подобрать";
}

function normalize(text) {
  return text.replace(/\s+/g, " ").trim();
}

async function submit() {
  var text = normalize(els.input.value);
  if (!text || state.busy) return;

  var before = state.history.length;
  state.history = fitHistory(state.history);
  if (state.history.length < before) {
    addBubble(
      "assistant",
      "Ранние реплики больше не передаются: контракт принимает не больше " + MAX_MESSAGES + " сообщений."
    );
  }

  els.input.value = "";
  state.history.push({ role: "user", text: text });
  addBubble("user", text);
  setBusy(true);
  setStatus("обрабатываю запрос…", false);

  try {
    var result = await askBackend(state.history, state.dialogueToken);
    var document_ = result.document;
    state.dialogueToken = result.dialogueToken;
    state.cards = new Map(result.items.map(function (item) { return [item.id, item]; }));
    state.history.push({ role: "assistant", text: assistantHistoryText(document_) });
    showInPanel(renderDocument(document_));
    echoInChat(document_);
    showMetrics(result.metrics);

    setStatus("ответ получен", false);
  } catch (error) {
    // Неудачную реплику в историю не пишем: иначе следующая отправка сломает
    // чередование ролей. Показываем отдельный технический блок.
    state.history.pop();
    showInPanel(renderServiceError(error));
    addBubble("assistant", "Не удалось получить ответ: " + error.message);
    if (error.metrics && error.metrics.modelCalls) {
      showMetrics(error.metrics);
    } else {
      els.metrics.hidden = true;
    }
    setStatus("сервис недоступен", true);
  } finally {
    setBusy(false);
    els.input.focus();
  }
}

// --- примеры и сброс --------------------------------------------------------

async function loadExamples() {
  try {
    var response = await fetch("/api/examples", { cache: "no-store" });
    if (!response.ok) return;
    var data = await response.json();
    renderExamples(data.kind === "furniture" ? EXAMPLES : data.examples);
  } catch (_) {
    // Форма поиска доступна и при сбое загрузки примеров.
  }
}

function renderExamples(examples) {
  els.examples.replaceChildren();
  els.examplePanel.hidden = !examples.length;
  examples.forEach(function (example) {
    var button = el("button", "button button--example", example.text);
    if (example.hint) button.title = example.hint;
    button.type = "button";
    button.addEventListener("click", function () {
      if (state.busy) return;
      resetDialog();
      els.input.value = example.text;
      submit();
    });
    els.examples.append(button);
  });
}

function resetDialog() {
  state.history = [];
  state.dialogueToken = "";
  state.cards.clear();
  els.chat.replaceChildren();
  els.answer.replaceChildren(el("p", "placeholder", PLACEHOLDER));
  els.metrics.hidden = true;
  setStatus("готов к работе", false);
}

// --- запуск -----------------------------------------------------------------

function boot() {
  els.form.addEventListener("submit", function (event) {
    event.preventDefault();
    submit();
  });
  els.input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  });
  els.newDialog.addEventListener("click", function () {
    resetDialog();
    loadExamples();
    els.input.focus();
  });
  loadExamples();
  addBubble("assistant", "Какой товар ищете? Можно сразу указать город и бюджет.");
}

boot();
