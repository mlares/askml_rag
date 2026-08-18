const form = document.querySelector("#ask-form");
const question = document.querySelector("#question");
const submitButton = document.querySelector("#submit-button");
const result = document.querySelector("#result");
const resultStatus = document.querySelector("#result-status");
const answerState = document.querySelector("#answer-state");
const abstentionState = document.querySelector("#abstention-state");
const errorState = document.querySelector("#error-state");
const answerText = document.querySelector("#answer-text");
const citations = document.querySelector("#citations");
const abstentionMessage = document.querySelector("#abstention-message");
const errorMessage = document.querySelector("#error-message");
const languageInputs = document.querySelectorAll('input[name="language"]');

const copy = {
  es: {
    eyebrow: "Asistente de investigación del portfolio",
    intro:
      "Pregunta sobre el trabajo profesional y la investigación públicos de Marcelo Lares. Las respuestas se fundamentan en fuentes recuperadas.",
    disclaimerTitle: "Asistente con datos públicos.",
    disclaimerText:
      "Esta herramienta experimental usa un corpus público curado y puede estar incompleto. Verifica las fuentes citadas; no envíes información privada, sensible o confidencial.",
    askHeading: "Haz una pregunta",
    languageLegend: "Idioma de la pregunta y la respuesta",
    spanish: "Español",
    english: "English",
    questionLabel: "Pregunta",
    questionPlaceholder:
      "Por ejemplo: ¿Qué investigación hizo Marcelo sobre vacíos cósmicos?",
    formHint: "Solo se usan fuentes públicas del corpus.",
    askButton: "Preguntar",
    loading: "Buscando fuentes públicas y verificando la evidencia…",
    loadingButton: "Pensando…",
    aboutHeading: "Acerca de este asistente",
    aboutText:
      "Busca un corpus público curado sobre Marcelo Lares y solo responde cuando las fuentes recuperadas aportan evidencia.",
    aboutItemOne: "Abre cada enlace y verifica la evidencia citada.",
    aboutItemTwo:
      "“No lo sé” significa que las fuentes disponibles no fueron suficientes.",
    aboutItemThree:
      "Las preguntas no se conservan en los registros de solicitudes.",
    aboutHint:
      "Para proteger el servicio, las preguntas tienen límites de frecuencia y tiempo. Este es un proyecto experimental de portfolio, no un buscador web general.",
    answerHeading: "Respuesta",
    sourcesHeading: "Fuentes",
    abstentionHeading: "No lo sé a partir de las fuentes disponibles",
    abstentionFallback:
      "Las fuentes públicas recuperadas no aportan evidencia suficiente para responder.",
    abstentionHint:
      "Prueba con una pregunta más específica o consulta sobre el trabajo público, proyectos, docencia o publicaciones de Marcelo.",
    errorHeading: "El asistente no está disponible temporalmente",
    errorHint: "Por favor, inténtalo de nuevo en unos instantes.",
    requestFailed: "La solicitud falló.",
    serverReturned: "El servidor respondió con el estado {status}.",
  },
  en: {
    eyebrow: "Portfolio research assistant",
    intro:
      "Ask about Marcelo Lares's public professional work and research. Answers are grounded in retrieved sources.",
    disclaimerTitle: "Public-data assistant.",
    disclaimerText:
      "This experimental tool uses a curated public corpus and may be incomplete. Verify cited sources; do not submit private, sensitive, or confidential information.",
    askHeading: "Ask a question",
    languageLegend: "Question and answer language",
    spanish: "Español",
    english: "English",
    questionLabel: "Question",
    questionPlaceholder:
      "For example: What research has Marcelo done on cosmic voids?",
    formHint: "Only public corpus sources are used.",
    askButton: "Ask",
    loading: "Searching public sources and checking evidence…",
    loadingButton: "Thinking…",
    aboutHeading: "About this assistant",
    aboutText:
      "It searches a curated public corpus about Marcelo Lares, then returns an answer only when the retrieved sources provide support.",
    aboutItemOne: "Open each source link and verify the quoted evidence.",
    aboutItemTwo:
      "“I don't know” means the available sources were insufficient.",
    aboutItemThree: "Questions are not retained in application request logs.",
    aboutHint:
      "To protect the service, questions are rate limited and time bounded. This is an experimental portfolio project, not a general web search.",
    answerHeading: "Answer",
    sourcesHeading: "Sources",
    abstentionHeading: "I don't know from the available sources",
    abstentionFallback:
      "The retrieved public sources do not provide enough evidence to answer that.",
    abstentionHint:
      "Try a more specific question, or ask about Marcelo's public work, projects, teaching, or publications.",
    errorHeading: "The assistant is temporarily unavailable",
    errorHint: "Please try again shortly.",
    requestFailed: "The request failed.",
    serverReturned: "The server returned {status}.",
  },
};

function selectedLanguage() {
  return document.querySelector('input[name="language"]:checked').value;
}

function textFor(key) {
  return copy[selectedLanguage()][key];
}

function applyInterfaceLanguage() {
  const language = selectedLanguage();
  document.documentElement.lang = language;
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = copy[language][element.dataset.i18n];
  });
  question.placeholder = copy[language].questionPlaceholder;
  if (!submitButton.disabled) {
    submitButton.textContent = copy[language].askButton;
  }
}

function resetResult() {
  result.hidden = false;
  resultStatus.hidden = true;
  answerState.hidden = true;
  abstentionState.hidden = true;
  errorState.hidden = true;
  citations.replaceChildren();
}

function showLoading() {
  resetResult();
  resultStatus.textContent = textFor("loading");
  resultStatus.hidden = false;
  submitButton.disabled = true;
  submitButton.textContent = textFor("loadingButton");
}

function citationElement(citation) {
  const article = document.createElement("article");
  article.className = "citation";

  const source = citation.source_url
    ? document.createElement("a")
    : document.createElement("span");
  source.className = "citation-title";
  source.textContent = citation.title;
  if (citation.source_url) {
    source.href = citation.source_url;
    source.target = "_blank";
    source.rel = "noopener noreferrer";
  }

  const quote = document.createElement("blockquote");
  quote.textContent = `“${citation.quote}”`;
  article.append(source, quote);
  return article;
}

function showAnswer(payload) {
  answerText.textContent = payload.answer;
  payload.citations.forEach((citation) => {
    citations.append(citationElement(citation));
  });
  answerState.hidden = false;
}

function showAbstention(payload) {
  abstentionMessage.textContent =
    payload.limitations?.[0] ||
    textFor("abstentionFallback");
  abstentionState.hidden = false;
}

function showError(message) {
  errorMessage.textContent = message;
  errorState.hidden = false;
}

languageInputs.forEach((input) => {
  input.addEventListener("change", applyInterfaceLanguage);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const questionText = question.value.trim();
  if (!questionText) {
    question.focus();
    return;
  }

  showLoading();
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: questionText, language: selectedLanguage() }),
    });
    if (!response.ok) {
      throw new Error(textFor("serverReturned").replace("{status}", response.status));
    }

    const payload = await response.json();
    resultStatus.hidden = true;
    if (payload.answerable) {
      showAnswer(payload);
    } else {
      showAbstention(payload);
    }
  } catch (error) {
    resultStatus.hidden = true;
    showError(error instanceof Error ? error.message : textFor("requestFailed"));
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = textFor("askButton");
  }
});

applyInterfaceLanguage();
