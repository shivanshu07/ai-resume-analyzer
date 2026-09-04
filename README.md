# AI Resume Analyzer

An AI-powered resume analysis system that evaluates how well a resume matches a specific job description — combining semantic embeddings, FAISS vector search, structured rule-based matching, and an LLM-generated improvement summary into a single hybrid score with a requirement-by-requirement breakdown.

**[Try the live demo →](#https://ai-resume-analyzer-saawhwwh8kdi9w4vswsxpw.streamlit.app/)** &nbsp;|&nbsp; **[API docs →](#https://ai-resume-analyzer-e5lg.onrender.com)** &nbsp;|&nbsp; ![Tests](https://github.com/shivanshu07/ai-resume-analyzer/actions/workflows/tests.yml/badge.svg)

> Replace the two links above with your actual deployed URLs before publishing this README — see [Live Deployments](#live-deployments) below for where to find them.

---

## What it does

Give it a resume (PDF) and a job description, and it returns:

- An overall **ATS-style alignment score** (0–100) with a plain-language interpretation
- A **requirement-by-requirement breakdown** — which JD requirements are strongly met, partially met, weak, or missing entirely
- **Matched vs. missing** skills, concepts, and education fields
- A ranked list of **priority gaps** — the highest-impact things to fix first
- An optional **LLM-generated summary** (via Groq) explaining, in plain English, what to actually change on the resume

Unlike a keyword-matching ATS screen, the scoring is hybrid: semantic similarity (does this resume chunk *mean* something related to this requirement?) combined with structured evidence (does the resume literally contain this skill, this degree, this many years of experience?). Neither signal alone is reliable — see [Why Hybrid Matching](#why-hybrid-matching) for why.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Three ways to run it](#three-ways-to-run-it)
3. [Local setup (CLI)](#local-setup-cli)
4. [Running the API locally](#running-the-api-locally)
5. [Running the Streamlit UI locally](#running-the-streamlit-ui-locally)
6. [Running with Docker](#running-with-docker)
7. [Live Deployments](#live-deployments)
8. [Why there are four requirements files](#why-there-are-four-requirements-files)
9. [Environment variables](#environment-variables)
10. [Testing](#testing)
11. [LLM Output Evaluation](#llm-output-evaluation)
12. [CI/CD](#cicd)
13. [Project structure](#project-structure)
14. [Module responsibilities](#module-responsibilities)
15. [Understanding the output](#understanding-the-output)
16. [Why Hybrid Matching](#why-hybrid-matching)
17. [Known limitations](#known-limitations)
18. [Possible future improvements](#possible-future-improvements)

---

## Architecture

The project has two independently deployable pieces talking over HTTP — not one monolithic script:

```text
┌─────────────────────┐         HTTP          ┌──────────────────────────┐
│   Streamlit UI       │ ─────────────────────▶ │   FastAPI backend         │
│   (Streamlit Cloud)  │ ◀───────────────────── │   (Docker, on Render)     │
└─────────────────────┘        JSON            └──────────────────────────┘
                                                            │
                                                            ▼
                                          ┌──────────────────────────────────┐
                                          │  ResumeAnalysisPipeline            │
                                          │                                    │
                                          │  PDF/JD parsing → cleaning →       │
                                          │  chunking → embedding (Sentence-   │
                                          │  Transformers) → FAISS vector      │
                                          │  search → hybrid scoring →         │
                                          │  ATS/gap analysis                  │
                                          └──────────────────────────────────┘
                                                            │
                                                            ▼
                                          ┌──────────────────────────────────┐
                                          │  LLMGapExplainer (optional)        │
                                          │  → Groq API → plain-English        │
                                          │    improvement summary             │
                                          └──────────────────────────────────┘
```

The Streamlit app is a **thin client** — it doesn't import the pipeline directly, it calls the deployed API's `/analyze` endpoint over HTTP, the same way any other consumer could. This is a deliberate separation: the API is the actual product; the UI is one interface to it.

---

## Three ways to run it

Pick whichever matches what you're trying to do:

| Goal | How |
|---|---|
| Just see it work, no setup | Use the [live demo](#live-deployments) |
| Hack on the pipeline itself | [Local CLI](#local-setup-cli) |
| Hack on the API | [Local API](#running-the-api-locally) |
| Hack on the UI | [Local Streamlit](#running-the-streamlit-ui-locally) |
| Reproduce the exact deployed environment | [Docker](#running-with-docker) |

---

## Local setup (CLI)

### Step 1 — Clone and create a virtual environment

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI-Resume-Analyzer

python -m venv venv
source venv/Scripts/activate   # Git Bash on Windows
# venv\Scripts\activate        # Command Prompt on Windows
# source venv/bin/activate     # macOS/Linux
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Add your input files

```text
data/raw/resume/resume.pdf
data/raw/job_description/job_description.txt
```

### Step 4 — Run the pipeline

```bash
python -m app.main
```

This runs the full pipeline end to end and writes intermediate and final outputs to `data/processed/`, including `ats_analysis.json` — the main result.

---

## Running the API locally

```bash
pip install fastapi "uvicorn[standard]" python-multipart openai
uvicorn app.api:app --reload
```

Then open `http://127.0.0.1:8000/docs` for interactive Swagger UI, or `http://127.0.0.1:8000/health` to confirm it's up.

`POST /analyze` accepts a resume PDF and a job description (as pasted text or an uploaded `.txt` file), and returns the same `ats_analysis` JSON the CLI produces — plus an optional `llm_summary` field if `include_llm_summary=true` is passed.

---

## Running the Streamlit UI locally

```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

By default it points at the live deployed API (see [Live Deployments](#live-deployments)). To test against your own local API instead, change the "API base URL" field in the sidebar to `http://localhost:8000`.

---

## Running with Docker

```bash
docker build -t resume-analyzer .
docker run --env-file .env -p 8000:8000 resume-analyzer
```

Then hit `http://localhost:8000/docs` exactly as with the local API. The embedding model is baked into the image at build time (not downloaded on every container start), and the image installs CPU-only PyTorch explicitly rather than the default CUDA build — both deliberate choices to keep the container's memory footprint small enough for free-tier hosting (see [Why there are four requirements files](#why-there-are-four-requirements-files)).

---

## Live Deployments

| Component | Platform | Notes |
|---|---|---|
| FastAPI backend | [Render](https://render.com) (free tier) | `<YOUR_RENDER_URL>` |
| Streamlit UI | [Streamlit Community Cloud](https://streamlit.io/cloud) (free) | `<YOUR_STREAMLIT_URL>` |

**Fill in the two URLs above** — find them in your Render dashboard (top of the service page) and Streamlit Cloud dashboard respectively, then replace the placeholder links at the top of this README too.

**Cold starts:** both platforms' free tiers sleep after a period of inactivity. The first request after a while can take 30–60 seconds to wake the service back up — this is expected behavior, not a bug. If a demo link seems unresponsive at first, give it a moment and try again.

---

## Why there are four requirements files

This tripped up the Docker and CI setup more than once, so it's worth explaining rather than leaving implicit:

| File | Used by | Why it's separate |
|---|---|---|
| `requirements.txt` | Local dev environment | The full, real dev environment — includes test/eval tooling like `deepeval` |
| `requirements-docker.txt` | `Dockerfile` | Runtime-only deps. The full `requirements.txt` has a genuine version conflict (`click` is pinned to a version `deepeval` and `huggingface-hub` can't simultaneously satisfy) that has nothing to do with what the API needs to run |
| `requirements-test.txt` | GitHub Actions CI | Same conflict, same fix, needed again because CI runs on a clean Linux environment, not your local one |
| `app/requirements.txt` | Streamlit Community Cloud | Streamlit Cloud auto-detects a file named exactly `requirements.txt` in the same directory as the entrypoint script. Without this, it silently falls back to the repo-root `requirements.txt` and tries (and fails) to install the same conflicting dev dependencies |

The short version: `requirements.txt` reflects everything installed in your local dev environment via `pip freeze`, which mixes real runtime dependencies with unrelated dev/test tooling. Each deployment target gets its own minimal, conflict-free list of only what it actually needs.

---

## Environment variables

Create a `.env` file at the project root (never commit it):

```env
GROQ_API_KEY=your_groq_key_here
```

Get a free key at [console.groq.com/keys](https://console.groq.com/keys) — no credit card required. Used for the LLM gap-explanation summary and the LLM-judge evaluation tests. If unset, both features degrade gracefully (the summary field comes back `null`, evaluation tests skip) rather than breaking the rest of the pipeline.

```env
MODEL_NAME=openai/gpt-oss-120b
```

Optional override. Defaults to `openai/gpt-oss-120b` — Groq's current recommended free-tier model. (An earlier default, `llama-3.3-70b-versatile`, was deprecated by Groq on June 17, 2026 and no longer works; if you see a `model_not_found` error, this is why.)

---

## Testing

```bash
pytest -v
```

or, equivalently and more robustly regardless of how it's invoked:

```bash
python -m pytest -v
```

A root-level `conftest.py` ensures the repo root is always on `sys.path`, so `from src...` imports resolve correctly whether you run bare `pytest`, `python -m pytest`, or trigger tests from an IDE.

Tests use a small, generic, safe-to-commit sample resume fixture (`tests/fixtures/sample_resume.pdf`) rather than any real private resume file.

---

## LLM Output Evaluation

Deterministic code (PDF parsing, section detection, scoring math) is covered by regular `pytest` assertions. The LLM-generated gap summary is different — there's no single "correct" wording to assert against, so `tests/test_llm_eval.py` uses [`deepeval`](https://github.com/confident-ai/deepeval)'s `GEval` metric instead: an LLM-as-judge that scores the summary against a written rubric.

Two things are checked:

- **Groundedness** — does the summary avoid asserting false facts about the candidate's resume or the job's requirements? (Clearly-hedged illustrative examples, like "e.g., coursework in X" don't count as false claims — that's expected, useful advice, not hallucination.)
- **Actionability** — does it give specific, resume-focused suggestions rather than generic career advice?

The judge model is Groq (`src/llm/groq_eval_model.py`, a custom `DeepEvalBaseLLM` subclass), not OpenAI — evaluation runs on the same free tier as the rest of the project.

```bash
pytest tests/test_llm_eval.py -v -s
```

The `-s` flag is required to actually see the score — by default pytest only shows output for failing tests. Alternatively:

```bash
deepeval test run tests/test_llm_eval.py
```

uses deepeval's own CLI runner, which prints a results table without needing `-s`.

---

## CI/CD

GitHub Actions (`.github/workflows/tests.yml`) runs the full `pytest` suite automatically on every push and pull request to `main`, using `requirements-test.txt`. `GROQ_API_KEY` is available in CI as a repository secret, so the LLM evaluation tests run for real in CI, not just locally.

---

## Project structure

```text
AI-Resume-Analyzer/
│
├── app/
│   ├── main.py                  # CLI entrypoint
│   ├── api.py                   # FastAPI app
│   └── requirements.txt         # Minimal deps, for Streamlit Cloud's auto-detection
│
├── streamlit_app.py             # Streamlit UI (thin HTTP client)
├── .streamlit/config.toml       # UI theme
│
├── config/
│   └── settings.py              # Env var loading, Groq config
│
├── src/
│   ├── pipeline.py              # ResumeAnalysisPipeline -- shared by CLI and API
│   │
│   ├── extraction/
│   │   ├── pdf_parser.py
│   │   ├── jd_parser.py
│   │   └── requirement_extractor.py
│   │
│   ├── preprocessing/
│   │   ├── cleaner.py
│   │   ├── chunker.py
│   │   └── requirement_normalizer.py
│   │
│   ├── llm/
│   │   ├── embedder.py
│   │   ├── matcher.py           # FAISS-backed semantic matching
│   │   ├── gap_explainer.py     # Groq LLM summary generation
│   │   └── groq_eval_model.py   # Groq judge for deepeval
│   │
│   ├── evaluation/
│   │   ├── hybrid_scorer.py
│   │   ├── ats_scorer.py
│   │   ├── gap_analyzer.py
│   │   └── analysis.py
│   │
│   └── utils/
│       ├── file_handler.py
│       ├── helper.py
│       └── logger.py
│
├── tests/
│   ├── fixtures/sample_resume.pdf
│   ├── test_parser.py
│   ├── test_cleaner.py
│   ├── test_llm_eval.py
│   └── ...
│
├── .github/workflows/tests.yml  # CI
├── conftest.py                  # sys.path fix + warning filters
├── Dockerfile
├── .dockerignore
├── requirements.txt             # Full local dev environment
├── requirements-docker.txt      # Minimal, for Docker
├── requirements-test.txt        # Minimal, for CI
└── requirements-streamlit.txt   # Minimal, for local Streamlit dev
```

---

## Module responsibilities

**`src/pipeline.py`** — `ResumeAnalysisPipeline`, the reusable class both `app/main.py` (CLI) and `app/api.py` (FastAPI) call into, so both entrypoints run the exact same logic rather than two versions that can drift apart. Loads the embedding model once at instantiation, not per request.

**`src/extraction/pdf_parser.py`** — extracts raw text from the resume PDF via PyMuPDF.

**`src/extraction/jd_parser.py`** — reads and lightly cleans the raw job description text.

**`src/extraction/requirement_extractor.py`** — splits the JD into structured requirements (required / preferred / responsibility / education / other), detecting section headings across a broad range of real-world phrasings ("Qualifications", "What You'll Do", "Nice to Have", etc.) rather than a narrow fixed list. Falls back to an "other" bucket instead of dropping content when no known heading matches.

**`src/preprocessing/cleaner.py`** — normalizes extracted text and detects resume section boundaries (`SUMMARY`, `SKILLS`, `WORK EXPERIENCE`, `PROJECTS`, `EDUCATION`, etc.).

**`src/preprocessing/chunker.py`** — splits cleaned resume sections into semantically coherent chunks (keeping each job/project together where possible) for embedding.

**`src/preprocessing/requirement_normalizer.py`** — extracts structured fields (skills, education fields, experience, concepts) from each JD requirement's raw text via pattern matching.

**`src/llm/embedder.py`** — generates normalized embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions).

**`src/llm/matcher.py`** — semantic matching between JD requirements and resume chunks. Uses a FAISS `IndexFlatIP` vector index (exact search, not approximate — since embeddings are pre-normalized, inner product is mathematically identical to cosine similarity, so this produces the same scores as brute-force search, just via a real vector index instead of a hand-rolled loop). The index is built once per resume and reused across every requirement lookup.

**`src/llm/gap_explainer.py`** — `LLMGapExplainer`, calls Groq to generate a short natural-language summary of the top priority gaps. Degrades gracefully (returns `None`) if no API key is configured or the request fails, rather than breaking the rest of the response.

**`src/llm/groq_eval_model.py`** — `GroqEvalModel`, a custom `DeepEvalBaseLLM` subclass so the evaluation test suite judges LLM output using Groq instead of defaulting to OpenAI.

**`src/evaluation/hybrid_scorer.py`** — combines semantic similarity with structured evidence (skills, education, experience, concepts) into a single per-requirement score and assessment (`STRONG_ALIGNMENT` / `PARTIAL_ALIGNMENT` / `WEAK_ALIGNMENT` / `NO_ALIGNMENT`).

**`src/evaluation/ats_scorer.py`** — aggregates requirement-level scores into an overall weighted ATS-style score and interpretation.

**`src/evaluation/gap_analyzer.py`** — produces matched/missing skill, concept, and education lists, plus a priority-ranked list of the gaps most worth addressing.

**`src/evaluation/analysis.py`** — `ResumeAnalysisEngine`, combines ATS scoring and gap analysis into the final `ats_analysis` output.

---

## Understanding the output

**Overall score:**

```text
Overall ATS Score: 45.6/100
Interpretation: Weak Match
```

An analytical score produced by this project's own scoring logic — **not** a real employer ATS system's score, and not a prediction of whether an application will be rejected or accepted. Use it to identify what to improve, not as a pass/fail signal.

**Per-requirement:**

```text
Semantic similarity: 0.5633
Hybrid score: 0.5527
Assessment: PARTIAL_ALIGNMENT
```

Hybrid score is the primary signal — it factors in more than raw embedding similarity alone (see below for why that matters).

**Missing ≠ doesn't have.** If a skill or concept is reported missing, it means the resume's current *wording* didn't provide clear evidence for it — not necessarily that the candidate actually lacks it. This is a resume-wording problem to fix, not necessarily a skills gap.

---

## Why Hybrid Matching

Semantic similarity alone can mislead in both directions:

- A resume mentioning `Python, SQL, Machine Learning` will look semantically related to almost any data-science requirement — even one that also needs `Marketing Analytics` or `Client Engagement`, which the resume never mentions and semantic similarity alone won't reliably flag as absent.
- Conversely, an exact keyword match doesn't guarantee the requirement is actually satisfied in context.

The hybrid layer combines semantic similarity with structured, explicit evidence (skills, education, experience, concepts) so the final score reflects more than "these two texts use similar words."

---

## Known limitations

- **Single resume / single JD at a time** — not currently designed for batch comparison across many resumes or many jobs.
- **Pretrained models only** — no custom-trained ranking model; `all-MiniLM-L6-v2` is general-purpose and may not capture highly specialized domain relationships.
- **Heuristic scoring** — the ATS score is an engineered metric, not learned or calibrated against real hiring outcomes.
- **Evidence quality depends on resume wording** — a strong candidate can still score lower than expected if relevant experience isn't phrased in terms that match the JD's terminology.
- **LLM-judge evaluation has inherent noise** — GEval scores can vary slightly between runs of identical input, the same way the embedding model's own floating-point output has minor run-to-run variance. This is expected behavior for LLM-as-judge evaluation generally, not a defect.

---

## Possible future improvements

- Cross-encoder reranking of top matches for higher precision
- Batch comparison of one resume against multiple JDs (the FAISS index already makes this architecturally straightforward — it's currently rebuilt per-request rather than persisted, since that's all a single resume/JD comparison needs)
- More sophisticated experience/date extraction
- Domain-specific fine-tuned embeddings
- Persisted evaluation score history / trend tracking (e.g., via Confident AI)
- Resume rewriting suggestions with inline diffs, not just a text summary

---

## Author

**Shivanshu Kumar**

[LinkedIn](https://linkedin.com/in/shivanshu19476) · [GitHub](https://github.com/shivanshu07)
