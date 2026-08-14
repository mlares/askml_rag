# Web interface

AskML RAG now has a small browser interface at `GET /`. It is intentionally a
thin client over the existing `POST /ask` boundary: all retrieval, OpenAI
access, citation validation, and abstention decisions remain on the server.

## Run it

Follow the setup in [api.md](api.md), then start the local server:

```bash
uv run python scripts/serve_api.py
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in a browser. Submitting
a question makes a real OpenAI-backed request, so it requires the local corpus
and `OPENAI_API_KEY` already described in [openai_integration.md](openai_integration.md).

## Architecture

```text
browser
  -> GET /                         FastAPI returns index.html
  -> GET /static/styles.css        browser loads presentation rules
  -> GET /static/app.js            browser loads interaction code
  -> POST /ask { question }        same FastAPI process runs the RAG pipeline
  <- GroundedAnswer JSON           browser renders answer or abstention
```

The relevant files are:

- `src/askml_rag/api/app.py`: mounts `/static` and serves the HTML at `/`.
- `src/askml_rag/api/static/index.html`: semantic page structure.
- `src/askml_rag/api/static/styles.css`: responsive visual design.
- `src/askml_rag/api/static/app.js`: form submission and response rendering.
- `tests/test_api.py`: verifies that the page and JavaScript asset are served,
  as well as the existing JSON endpoint.

## Why this design

### One same-origin application

The HTML, JavaScript, and API are served from the same host and port. Therefore
the browser can call relative URL `/ask` without CORS configuration. More
importantly, the browser receives no OpenAI credential: only the Python server
reads `OPENAI_API_KEY` and makes the model request.

This is the smallest deployable portfolio architecture. It is a good fit while
the interface has one page and no client-side routing, login, or build-time
asset optimization. A separate React/Vue frontend would add a second build and
deployment pipeline without improving the core RAG behavior yet.

### Plain HTML, CSS, and JavaScript

No frontend framework is used. `index.html` contains the durable document
structure; `styles.css` contains the visual rules; `app.js` provides the small
amount of interaction. Splitting the three makes the code easy to inspect and
change while avoiding a bundler, Node dependency tree, and generated assets.

This is a conscious scope decision, not a claim that frameworks are bad. Move
to a component framework only when repeated interactive components, complex
client state, or a larger product makes that complexity worthwhile.

### Server-owned RAG configuration

The UI sends only the question. It cannot choose retriever, `k`, model, source
paths, or prompt version. Those choices remain inside `AskService`, where they
are evaluated and controlled. This avoids a user-visible interface accidentally
turning into an unevaluated experiment-control panel.

### Explicit answer states

After form submission, `app.js` renders exactly one of these states:

| State | Meaning | What the user sees |
| --- | --- | --- |
| Loading | The server is retrieving and generating. | A progress message and disabled button. |
| Supported answer | `answerable` is `true`. | Answer plus source title/link and literal quote for every citation. |
| Abstention | HTTP succeeded but `answerable` is `false`. | “I don't know from the available sources” and the recorded limitation. |
| Request failure | The HTTP request failed or returned a non-success status. | A temporary-unavailability message. |

The last two are intentionally distinct. An abstention is a correct product
outcome: the system refuses to manufacture evidence. A request failure means
the infrastructure needs attention.

### Safe rendering of model-controlled text

The answer, title, quote, and limitation may ultimately originate in a model or
document. The browser inserts them with `textContent`, not `innerHTML`, so they
are displayed as text rather than interpreted as browser markup. Source links
are only created from the server-provided `source_url`; external links use
`rel="noopener noreferrer"` to isolate the new tab from this page.

## Interface behavior

The form enforces `required` and a 1,000-character limit for immediate browser
feedback. The server is still the authority: `AskRequest` validates the same
constraints and rejects malformed requests with HTTP 422.

For each citation, the interface shows:

- source title;
- a link when the corpus has a public source URL;
- a literal supporting quote.

It does not expose raw prompt text, API keys, or internal model reasoning.
`citation_warnings` and `validation_errors` remain available in the JSON API
for debugging, but are intentionally not presented as normal reader-facing
content.

The page also has a persistent public-data disclaimer: it states that the
corpus is curated public material, answers may be incomplete, citations should
be checked, and visitors must not submit private, sensitive, or confidential
information.

Below the question box, an “About this assistant” panel explains the evidence
boundary in visitor-facing terms: it uses a curated public corpus rather than
the open web, supports abstention, does not retain questions in application
request logs, and has rate/time limits. It deliberately links visitors to the
source quotes rather than asking them to trust generated prose alone.

## Test the UI boundary

```bash
uv run pytest -q tests/test_api.py
```

These tests verify that `/` returns the interface, `/static/app.js` is served,
and `/ask` works with a fake LLM. They do not visually test the CSS and they do
not make a paid API call. For a manual check, open the local page and try one
supported question and one deliberately unsupported question.

## What remains before public deployment

The local UI is ready for exploration and a portfolio demo. Its current
operational safeguards are described in [operations.md](operations.md). Before
embedding it in a public personal site, add appropriate access policy and a
deployment plan. If the final personal webpage lives on another domain, either
serve this UI from the API domain or add a narrowly scoped CORS policy; do not
loosen it globally by default.
