const form = document.querySelector("#ask-form");
const question = document.querySelector("#question");
const submitButton = document.querySelector("#submit-button");
const promptList = document.querySelector("#example-prompts");
const result = document.querySelector("#result");
const resultStatus = document.querySelector("#result-status");
const emptyState = document.querySelector("#empty-state");
const answerState = document.querySelector("#answer-state");
const abstentionState = document.querySelector("#abstention-state");
const errorState = document.querySelector("#error-state");
const answerText = document.querySelector("#answer-text");
const abstentionMessage = document.querySelector("#abstention-message");
const errorMessage = document.querySelector("#error-message");
const languageInputs = document.querySelectorAll('input[name="language"]');
let promptIndex = 0;
let promptOrder = [];
let carouselTimer;
let typewriterTimer;

const copy = {
  es: {
    heroHeading: "Hablemos de mi trabajo.", heroCopy: "Liderazgo en IA aplicada, entrega de ML, profundidad de investigación, docencia y proyectos seleccionados.", bookNav: "Agendar una conversación", bookCta: "Agendar una conversación", privacyLink: "Privacidad y uso de datos", askInstruction: "Preguntá naturalmente sobre mi trabajo; responderé a partir de las fuentes públicas disponibles.",
    languageLegend: "Idioma de la pregunta y la respuesta", spanish: "Español", english: "English", questionLabel: "Tu pregunta", questionPlaceholder: "Por ejemplo: ¿Tenés experiencia con sistemas de recomendación?",
    askButton: "Preguntar", examplesHeading: "Probá una pregunta",
    prompts: ["¿Tenés experiencia con sistemas de recomendación?", "Contame sobre vos.", "¿Investigaste sobre vacíos cósmicos?", "¿Quién dirigió tu tesis?", "¿Qué experiencia tenés trabajando en retail?", "¿Qué experiencia tenés trabajando en segmentación de clientes?", "¿Qué equipos de IA lideraste?", "¿Cómo aplicaste machine learning en producción?", "¿Qué materias enseñás?"],
    howItWorksSummary: "Cómo funciona este asistente", howItWorksText: "Busca material público revisado y responde solo cuando puede mostrar fuentes que lo respaldan. No es un buscador web general.", privacyText: "Las preguntas no se conservan en los registros de solicitudes. No compartas información privada, sensible o confidencial.",
    loading: "Buscando fuentes públicas y verificando la evidencia…", loadingButton: "Buscando…", answerKicker: "Respuesta con evidencia", sourcesHeading: "Evidencia utilizada",
    abstentionKicker: "Un próximo paso útil", abstentionHeading: "No puedo verificarlo con las fuentes disponibles.", abstentionFallback: "Las fuentes públicas recuperadas no aportan evidencia suficiente para responder.",
    abstentionHint: "Probá una de las preguntas sugeridas o explorá directamente el portfolio y el CV.", errorHeading: "El asistente no está disponible temporalmente", errorHint: "Por favor, intentá de nuevo en unos instantes.",
    requestFailed: "La solicitud falló.", serverReturned: "El servidor respondió con el estado {status}.", footerLink: "Volver al portfolio de Marcelo", footerNote: "Este asistente usa fuentes públicas revisadas y muestra evidencia de respaldo.", emptyState: "Tu respuesta y la evidencia que la respalda aparecerán aquí.",
  },
  en: {
    heroHeading: "Let’s talk about my work.", heroCopy: "Applied AI leadership, ML delivery, research depth, teaching, and selected projects.", bookNav: "Book a conversation", bookCta: "Book a conversation", privacyLink: "Privacy and data use", askInstruction: "Ask naturally about my work; I’ll answer from the available public sources.",
    languageLegend: "Question and answer language", spanish: "Español", english: "English", questionLabel: "Your question", questionPlaceholder: "For example: Do you have experience with recommendation systems?",
    askButton: "Ask", examplesHeading: "Try a question",
    prompts: ["Do you have experience with recommendation systems?", "Tell me about yourself.", "Did you research cosmic voids?", "Who was your thesis director?", "What is your experience working in retail?", "What is your experience working in customer segmentation?", "What AI teams have you led?", "How have you used machine learning in production?", "What do you teach?"],
    howItWorksSummary: "How this assistant works", howItWorksText: "It searches reviewed public material, then provides an answer only when it can show supporting sources. It is not a general web search.", privacyText: "Questions are not retained in application request logs. Do not submit private, sensitive, or confidential information.",
    loading: "Searching public sources and checking evidence…", loadingButton: "Searching…", answerKicker: "Evidence-backed answer", sourcesHeading: "Evidence used",
    abstentionKicker: "A useful next step", abstentionHeading: "I can’t verify that from the available sources.", abstentionFallback: "The retrieved public sources do not provide enough evidence to answer that.",
    abstentionHint: "Try a suggested question, or explore the portfolio and CV directly.", errorHeading: "The assistant is temporarily unavailable", errorHint: "Please try again shortly.",
    requestFailed: "The request failed.", serverReturned: "The server returned {status}.", footerLink: "Back to Marcelo’s portfolio", footerNote: "This assistant uses reviewed public sources and shows supporting evidence.", emptyState: "Your answer and supporting evidence will appear here.",
  },
};

function selectedLanguage() { return document.querySelector('input[name="language"]:checked').value; }
function textFor(key) { return copy[selectedLanguage()][key]; }
function setLanguage(language) {
  const input = document.querySelector(`input[name="language"][value="${language}"]`);
  if (input && !input.checked) { input.checked = true; applyInterfaceLanguage(); }
}
function detectedLanguage(value) {
  const text = value.toLocaleLowerCase();
  const spanish = /[áéíóúñ¿¡]|\b(qué|cómo|cuál|cuáles|tenés|tienes|trabajaste|has trabajado|investigación|docencia|materia|curso|podés|puedes|sobre|con|para)\b/g;
  const english = /\b(what|which|how|have|has|you|your|worked|research|teaching|course|about|with|for|does|do)\b/g;
  const spanishScore = (text.match(spanish) || []).length;
  const englishScore = (text.match(english) || []).length;
  if (spanishScore === englishScore) return null;
  return spanishScore > englishScore ? "es" : "en";
}
function shuffledPrompts(prompts, previousPrompt) {
  const order = [...prompts];
  for (let index = order.length - 1; index > 0; index -= 1) {
    const replacement = Math.floor(Math.random() * (index + 1));
    [order[index], order[replacement]] = [order[replacement], order[index]];
  }
  if (order.length > 1 && order[0] === previousPrompt) {
    [order[0], order[1]] = [order[1], order[0]];
  }
  return order;
}
function renderPrompt() {
  const prompts = copy[selectedLanguage()].prompts;
  if (promptIndex >= promptOrder.length) {
    promptOrder = shuffledPrompts(prompts, promptList.dataset.prompt);
    promptIndex = 0;
  }
  const prompt = promptOrder[promptIndex];
  clearTimeout(carouselTimer);
  clearTimeout(typewriterTimer);
  promptList.dataset.prompt = prompt;
  promptList.setAttribute("aria-label", prompt);
  promptList.textContent = "";

  if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
    promptList.textContent = prompt;
    scheduleNextPrompt();
    return;
  }

  promptList.classList.add("is-typing");
  let characterIndex = 0;
  const typeNextCharacter = () => {
    promptList.textContent = prompt.slice(0, characterIndex + 1);
    characterIndex += 1;
    if (characterIndex < prompt.length) {
      typewriterTimer = window.setTimeout(typeNextCharacter, 35);
      return;
    }
    promptList.classList.remove("is-typing");
    scheduleNextPrompt();
  };
  typeNextCharacter();
}
function scheduleNextPrompt() {
  carouselTimer = window.setTimeout(() => {
    promptIndex += 1;
    renderPrompt();
  }, 3600);
}
function renderPrompts() {
  const previousPrompt = promptList.dataset.prompt;
  promptIndex = 0;
  promptOrder = shuffledPrompts(copy[selectedLanguage()].prompts, previousPrompt);
  clearTimeout(carouselTimer);
  clearTimeout(typewriterTimer);
  renderPrompt();
}
function applyInterfaceLanguage() {
  const language = selectedLanguage();
  document.documentElement.lang = language;
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    const translated = copy[language][element.dataset.i18n];
    if (translated) element.textContent = translated;
  });
  question.placeholder = copy[language].questionPlaceholder;
  renderPrompts();
  if (!submitButton.disabled) submitButton.textContent = copy[language].askButton;
}
function resetResult() { resultStatus.hidden = true; emptyState.hidden = true; answerState.hidden = true; abstentionState.hidden = true; errorState.hidden = true; }
function showLoading() { resetResult(); resultStatus.textContent = textFor("loading"); resultStatus.hidden = false; submitButton.disabled = true; submitButton.textContent = textFor("loadingButton"); }
function showAnswer(payload) { emptyState.hidden = true; answerText.textContent = payload.answer; answerState.hidden = false; }
function showAbstention(payload) { abstentionMessage.textContent = payload.limitations?.[0] || textFor("abstentionFallback"); abstentionState.hidden = false; }
function showError(message) { errorMessage.textContent = message; errorState.hidden = false; }

languageInputs.forEach((input) => input.addEventListener("change", applyInterfaceLanguage));
question.addEventListener("input", () => {
  const language = detectedLanguage(question.value);
  if (language) setLanguage(language);
});
promptList.addEventListener("click", () => { question.value = promptList.dataset.prompt; question.focus(); });
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const questionText = question.value.trim();
  if (!questionText) { question.focus(); return; }
  showLoading();
  try {
    const response = await fetch("/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: questionText, language: selectedLanguage() }) });
    if (!response.ok) throw new Error(textFor("serverReturned").replace("{status}", response.status));
    const payload = await response.json(); resultStatus.hidden = true;
    if (payload.answerable) showAnswer(payload); else showAbstention(payload);
  } catch (error) {
    resultStatus.hidden = true; showError(error instanceof Error ? error.message : textFor("requestFailed"));
  } finally {
    submitButton.disabled = false; submitButton.textContent = textFor("askButton");
  }
});
applyInterfaceLanguage();
