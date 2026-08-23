# AI Resume Analyzer

An AI-powered resume analysis system that evaluates how well a resume matches a specific job description.

The project combines **PDF parsing, text preprocessing, structured job-requirement extraction, semantic embeddings, semantic matching, hybrid scoring, ATS-style analysis, gap analysis, and a Streamlit interface** into a reproducible end-to-end pipeline.

> **Current project scope:** The system is designed around evaluating a resume against a job description using pretrained models. It does not require training a custom ML model.

---

## 1. Project Overview

The AI Resume Analyzer takes two primary inputs:

1. A **resume PDF**
2. A **job description text file**

It processes both inputs and produces:

- Extracted resume text
- Cleaned resume text
- Resume sections and chunks
- Structured job requirements
- Resume and job-description embeddings
- Semantic similarity results
- Hybrid matching scores
- ATS-style overall score
- Skill matches and gaps
- Concept matches and gaps
- Education alignment
- Experience analysis
- Priority gaps that require attention

The main processing pipeline is executed through:

```bash
python -m app.main
```

---

## 2. Current Architecture

The project is organized into application code, source modules, utilities, data, tests, configuration, and supporting files.

```text
AI-Resume-Analyzer/
│
├── .deepeval/
├── .github/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── streamlit_app.py
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── data/
│   ├── processed/
│   │   ├── ats_analysis.json
│   │   ├── hybrid_results.json
│   │   ├── jd_embeddings.json
│   │   ├── jd_requirements.json
│   │   ├── match_results.json
│   │   ├── resume_chunks.json
│   │   ├── resume_clean.txt
│   │   ├── resume_embeddings.json
│   │   └── resume_raw.txt
│   │
│   └── raw/
│       ├── job_description/
│       │   └── job_description.txt
│       │
│       └── resume/
│           └── resume.pdf
│
├── src/
│   ├── __init__.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── analysis.py
│   │   ├── ats_scorer.py
│   │   ├── gap_analyzer.py
│   │   └── hybrid_scorer.py
│   │
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── jd_parser.py
│   │   ├── pdf_parser.py
│   │   └── requirement_extractor.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── embedder.py
│   │   └── matcher.py
│   │
│   └── preprocessing/
│       ├── __init__.py
│       ├── chunker.py
│       ├── cleaner.py
│       └── requirement_normalizer.py
│
├── utils/
│   ├── __init__.py
│   ├── file_handler.py
│   ├── helper.py
│   └── logger.py
│
├── tests/
│   ├── test_ats_analysis.py
│   ├── test_cleaner.py
│   ├── test_normalizer.py
│   └── test_parser.py
│
├── .env
├── .gitignore
├── README.md
├── requirements.txt
└── venv/
```

> `venv/`, Python cache directories, and other generated/local files should normally not be committed to GitHub.

---

# 3. Module Responsibilities

## `app/`

### `app/main.py`

This is the main command-line orchestration layer.

It coordinates the complete pipeline:

```text
Resume PDF
    ↓
PDF parsing
    ↓
Text cleaning
    ↓
Section detection / chunking
    ↓
Resume embeddings
    ↓
Job description parsing
    ↓
Requirement extraction
    ↓
Requirement normalization
    ↓
JD embeddings
    ↓
Semantic matching
    ↓
Hybrid matching
    ↓
ATS scoring
    ↓
Gap analysis
    ↓
JSON outputs
```

Run it with:

```bash
python -m app.main
```

---

### `app/streamlit_app.py`

Provides the Streamlit-based interface for interacting with the resume analysis system.

Run with:

```bash
streamlit run app/streamlit_app.py
```

---

# 4. Configuration

## `config/settings.py`

Contains project configuration and settings used by the application.

This keeps configuration separate from the processing logic.

---

# 5. Source Modules

## 5.1 Extraction

The extraction layer converts the raw resume and job description into usable text and structured requirements.

### `src/extraction/pdf_parser.py`

Responsible for extracting text from the resume PDF.

Input:

```text
data/raw/resume/resume.pdf
```

Output:

```text
data/processed/resume_raw.txt
```

---

### `src/extraction/jd_parser.py`

Responsible for reading and processing the job description.

Input:

```text
data/raw/job_description/job_description.txt
```

The extracted job description is passed to the requirement extraction stage.

---

### `src/extraction/requirement_extractor.py`

Converts the job description into structured requirements.

Requirements are categorized into areas such as:

- Required skills
- Preferred skills
- Responsibilities
- Education
- Other requirements

The resulting structured requirements are stored in:

```text
data/processed/jd_requirements.json
```

---

# 6. Preprocessing

The preprocessing layer prepares extracted text for matching.

## `src/preprocessing/cleaner.py`

Cleans extracted resume/JD text.

Typical processing includes normalization of whitespace and formatting artifacts produced during PDF/text extraction.

Output:

```text
data/processed/resume_clean.txt
```

---

## `src/preprocessing/chunker.py`

Divides the cleaned resume into meaningful sections/chunks.

The current pipeline detects sections such as:

```text
SUMMARY
SKILLS
WORK EXPERIENCE
PROJECTS
EDUCATION
CERTIFICATIONS & LANGUAGES
```

Output:

```text
data/processed/resume_chunks.json
```

Each chunk retains information about its resume section so that matching can identify the most relevant evidence.

---

## `src/preprocessing/requirement_normalizer.py`

Normalizes extracted job requirements into structured components.

For example, a requirement can be represented using fields such as:

```text
category
importance
skills
education
concepts
experience
original requirement
```

This makes it possible to compare the JD requirements against different aspects of the resume.

---

# 7. Embedding and Matching

## `src/llm/embedder.py`

Generates semantic embeddings using a pretrained sentence-transformer model.

The current pipeline uses:

```text
all-MiniLM-L6-v2
```

The embedding dimension reported by the current pipeline is:

```text
384
```

Resume embeddings are saved to:

```text
data/processed/resume_embeddings.json
```

Job-description requirement embeddings are saved to:

```text
data/processed/jd_embeddings.json
```

The model is pretrained; this project does not train the embedding model.

---

## `src/llm/matcher.py`

Performs semantic matching between:

- normalized job requirements
- resume chunks

The system identifies the resume chunk with the strongest semantic relationship to each requirement.

The semantic matching output contains:

- Requirement
- Category
- Best evidence section
- Resume chunk
- Similarity score
- Match level

Output:

```text
data/processed/match_results.json
```

---

# 8. Evaluation and Scoring

## `src/evaluation/hybrid_scorer.py`

Combines semantic similarity with structured resume/JD evidence.

The purpose of hybrid matching is to avoid relying entirely on embedding similarity.

The system considers multiple types of evidence, including:

- semantic similarity
- skills
- education
- concepts
- experience
- requirement category

The output is:

```text
data/processed/hybrid_results.json
```

Each requirement receives a hybrid score and an assessment such as:

```text
STRONG_ALIGNMENT
PARTIAL_ALIGNMENT
WEAK_ALIGNMENT
NO_ALIGNMENT
```

---

## `src/evaluation/ats_scorer.py`

Calculates the overall ATS-style score from the requirement-level analysis.

The output includes:

- Overall ATS score
- Interpretation
- Requirement summary
- Category scores

---

## `src/evaluation/gap_analyzer.py`

Identifies gaps between the resume and job description.

The analysis covers areas such as:

### Skills

Matched skills and missing skills.

Example:

```text
Matched:
- Python
- SQL
- Statistics
- Machine Learning

Missing:
- Marketing Analytics
- Problem Scoping
```

### Concepts

Matched and missing concepts such as:

```text
Business Insights
Stakeholder Management
Client Engagement
Decision Making
Innovation
```

### Education

Checks whether the resume's education aligns with education-related requirements.

### Experience

Extracts experience requirements and estimates professional experience from the resume.

### Priority Gaps

Ranks the requirements that need the most attention.

---

## `src/evaluation/analysis.py`

Provides the higher-level analysis functionality used to bring together the scoring and gap-analysis results.

---

# 9. Utilities

## `utils/file_handler.py`

Provides reusable methods for:

- saving text
- loading text
- saving JSON
- loading JSON

It also creates parent directories automatically when saving files.

This keeps file I/O logic separate from the application pipeline.

---

## `utils/helper.py`

Contains general helper functionality, including directory creation utilities.

---

## `utils/logger.py`

Contains project logging functionality.

---

# 10. Input Files

The current pipeline expects the following structure:

```text
data/raw/
│
├── job_description/
│   └── job_description.txt
│
└── resume/
    └── resume.pdf
```

## Resume

Place the resume PDF at:

```text
data/raw/resume/resume.pdf
```

## Job Description

Place the job description text at:

```text
data/raw/job_description/job_description.txt
```

The job description should contain the complete text of the target job posting.

---

# 11. Installation

## Step 1 — Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI-Resume-Analyzer
```

---

## Step 2 — Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it in Git Bash:

```bash
source venv/Scripts/activate
```

Or in Command Prompt:

```cmd
venv\Scripts\activate
```

---

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

# 12. Environment Variables

If the project configuration requires environment variables, create a `.env` file in the project root.

Example:

```text
HF_TOKEN=your_huggingface_token
```

The current embedding pipeline can download the pretrained model without authentication, but Hugging Face may display a warning about unauthenticated requests and rate limits.

If a Hugging Face token is used, keep it private.

Do **not** commit `.env` to GitHub.

---

# 13. Reproducing the Analysis

After installation, make sure these two files exist:

```text
data/raw/resume/resume.pdf
data/raw/job_description/job_description.txt
```

Then activate the virtual environment and run:

```bash
python -m app.main
```

The program will execute the complete pipeline.

---

# 14. Expected Processing Stages

When the pipeline runs, the console displays stages similar to:

```text
RESUME PROCESSING

RESUME EMBEDDING GENERATION

JOB DESCRIPTION PROCESSING

JOB DESCRIPTION EMBEDDINGS

SEMANTIC MATCHING

HYBRID MATCHING

DAY 6 - ATS ANALYSIS
```

The exact console output can vary depending on the input files and scoring implementation.

---

# 15. Generated Output Files

After a successful run, the following files are generated in:

```text
data/processed/
```

### `resume_raw.txt`

Raw text extracted from the resume PDF.

### `resume_clean.txt`

Cleaned resume text.

### `resume_chunks.json`

Structured resume sections/chunks.

### `jd_requirements.json`

Structured job-description requirements.

### `resume_embeddings.json`

Embedding vectors generated from resume chunks.

### `jd_embeddings.json`

Embedding vectors generated from JD requirements.

### `match_results.json`

Semantic matching results.

### `hybrid_results.json`

Hybrid requirement-level matching results.

### `ats_analysis.json`

Final ATS-style analysis and gap analysis.

---

# 16. Understanding the Results

## Semantic Matching

Semantic similarity measures how closely a resume chunk is related to a job requirement.

Example:

```text
Similarity: 0.5633
Match level: moderate
```

This is based on embedding similarity and should not be interpreted as a literal percentage probability of getting the job.

---

## Hybrid Matching

Hybrid matching combines semantic evidence with structured evidence.

Example:

```text
Semantic score: 0.5633
Score: 0.5527
Assessment: PARTIAL_ALIGNMENT
```

The hybrid score is the project's primary requirement-level matching signal because it incorporates more information than semantic similarity alone.

---

## ATS Score

The ATS score summarizes the requirement-level analysis.

Example interpretation:

```text
Overall ATS Score: 45.60/100
Interpretation: Weak Match
```

The score is an **ATS-style project metric**, not an actual score produced by an employer's Applicant Tracking System.

---

# 17. Example Analysis

For a sample resume/job-description pair, the system may produce output similar to:

```text
Overall ATS Score: 45.60/100
Interpretation: Weak Match
```

It can also report:

```text
Strong alignment
Partial alignment
Weak alignment
No alignment
```

along with:

- required-category score
- preferred-category score
- responsibility score
- matched skills
- missing skills
- matched concepts
- missing concepts
- education alignment
- experience analysis
- priority gaps

The exact numbers depend on the resume and job description supplied as input.

---

# 18. Running the Streamlit Interface

The project also contains a Streamlit interface.

Start it using:

```bash
streamlit run app/streamlit_app.py
```

This provides a graphical interface for interacting with the analysis pipeline.

The command-line pipeline remains the reproducible backend workflow:

```bash
python -m app.main
```

---

# 19. Running Tests

The project contains tests for important processing components.

Run the test suite using:

```bash
pytest
```

Or:

```bash
python -m pytest
```

The current test files include:

```text
tests/test_parser.py
tests/test_cleaner.py
tests/test_normalizer.py
tests/test_ats_analysis.py
```

---

# 20. Development Workflow

A typical development workflow is:

```text
1. Update input resume/JD
        ↓
2. Run tests
        ↓
3. Run the main pipeline
        ↓
4. Inspect generated JSON files
        ↓
5. Review ATS analysis
        ↓
6. Modify processing/scoring logic if required
        ↓
7. Run tests again
```

Recommended commands:

```bash
python -m pytest
python -m app.main
```

---

# 21. Reproducibility

To reproduce the project's results as closely as possible:

1. Use the same resume PDF.
2. Use the same job description text.
3. Use the same Python environment.
4. Install the versions specified in `requirements.txt`.
5. Use the same embedding model:
   ```text
   all-MiniLM-L6-v2
   ```
6. Run:
   ```bash
   python -m app.main
   ```

Because semantic embeddings and dependency versions can affect results, changing model or package versions may produce slightly different scores.

---

# 22. Important Interpretation Notes

### This is not a real employer ATS

The project's ATS score is an analytical score created by this application.

It should be used to understand:

- resume/JD alignment
- missing skills
- missing concepts
- education alignment
- experience alignment
- priority areas for resume improvement

It should not be treated as a prediction of whether an employer will reject or select an application.

### Semantic similarity is not keyword matching

A requirement can receive semantic similarity even when the exact phrase does not appear in the resume.

Conversely, an exact keyword match does not automatically mean the candidate satisfies the requirement.

This is why the project uses a hybrid approach.

### Missing evidence is different from actual lack of experience

If the analyzer reports a missing skill or concept, it means that the current resume content did not provide sufficient evidence for the analyzer.

It does not necessarily prove that the candidate lacks that skill.

---

# 23. Project Technology Stack

The project currently uses the following technologies/components:

- **Python**
- **FastAPI-related project architecture**
- **Streamlit**
- **Sentence Transformers**
- **FAISS-related semantic-search architecture**
- **Large Language Model / pretrained NLP components**
- **PyMuPDF/PDF text extraction components**
- **JSON-based intermediate outputs**
- **pytest**
- **Hugging Face pretrained embedding models**

The exact installed package versions should be taken from:

```text
requirements.txt
```

rather than inferred from this README.

---

# 24. Project Design

The project follows a modular architecture:

```text
Input
  │
  ├── Resume PDF
  │
  └── Job Description
          │
          ▼
     Extraction Layer
          │
          ▼
   Preprocessing Layer
          │
          ▼
   Requirement Normalization
          │
          ▼
     Embedding Layer
          │
          ▼
    Semantic Matching
          │
          ▼
     Hybrid Scoring
          │
          ▼
      ATS Scoring
          │
          ▼
      Gap Analysis
          │
          ▼
   JSON Results / UI
```

This separation allows individual components to be improved without rewriting the entire application.

---

# 25. Why the Project Uses Hybrid Matching

Pure semantic similarity can produce misleading results.

For example, a resume may contain:

```text
Python
SQL
Machine Learning
```

and therefore appear semantically related to a requirement involving data science.

However, the job may additionally require:

```text
Marketing Analytics
Client Engagement
Problem Scoping
Product/Engineering Collaboration
```

Semantic similarity alone may not distinguish these missing requirements reliably.

The hybrid layer therefore incorporates structured evidence from:

- skills
- concepts
- education
- experience
- requirement category
- semantic similarity

This makes the final ATS-style analysis more interpretable.

---

# 26. Current Project Limitations

The current version is intentionally focused and has several limitations.

### Single resume / JD evaluation

The project is currently designed around evaluating one resume against one job description at a time.

### Pretrained models

The system uses pretrained models rather than training a custom resume-ranking model.

### ATS score is heuristic

The final score is an engineered analytical metric rather than a learned score calibrated against real hiring outcomes.

### Semantic model limitations

`all-MiniLM-L6-v2` is a general-purpose sentence embedding model and may not understand every domain-specific relationship, especially highly specialized job requirements.

### Evidence quality

The quality of the final analysis depends heavily on the quality of:

- PDF text extraction
- section detection
- requirement extraction
- requirement normalization
- resume wording

### Job-specific terminology

A resume can be strong for a role while still receiving a lower score if important domain-specific terminology is not explicitly represented in the resume.

---

# 27. Future Improvements

Possible future improvements include:

- Better resume section detection
- More robust PDF parsing
- Improved job-requirement extraction
- Better handling of compound requirements
- Domain-specific embedding models
- Cross-encoder reranking
- More sophisticated experience/date extraction
- Improved education matching
- LLM-based evidence verification
- Explainable requirement-level recommendations
- Resume rewriting recommendations
- Multiple resume/JD comparisons
- Historical evaluation against real job outcomes
- More comprehensive automated tests
- Production API deployment
- Improved Streamlit dashboard
- FAISS-based large-scale candidate/repository search

---

# 28. GitHub Usage

Before pushing the project to GitHub, verify that local/generated files are excluded where appropriate.

In particular, avoid committing:

```text
venv/
.env
__pycache__/
.pytest_cache/
*.pyc
```

Generated analysis files under `data/processed/` may be committed if they are intentionally included as reproducible example outputs. Otherwise, they can be excluded and regenerated using:

```bash
python -m app.main
```

---

# 29. Quick Start

For someone who wants to reproduce the project with minimal steps:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI-Resume-Analyzer

python -m venv venv
source venv/Scripts/activate

pip install -r requirements.txt
```

Place:

```text
resume.pdf
```

at:

```text
data/raw/resume/resume.pdf
```

and:

```text
job_description.txt
```

at:

```text
data/raw/job_description/job_description.txt
```

Then run:

```bash
python -m app.main
```

Inspect:

```text
data/processed/ats_analysis.json
```

for the final ATS-style analysis.

To run the UI:

```bash
streamlit run app/streamlit_app.py
```

To run tests:

```bash
pytest
```

---

# 30. Final Output

The most important generated file is:

```text
data/processed/ats_analysis.json
```

It contains the consolidated ATS-style evaluation, including:

- overall score
- requirement-level alignment
- category scores
- matched skills
- missing skills
- matched concepts
- missing concepts
- education analysis
- experience analysis
- priority gaps

The project therefore provides a complete pipeline from:

**Resume + Job Description → Structured Requirements → Semantic Matching → Hybrid Matching → ATS Analysis → Actionable Gaps**

---

## Author

**Shivanshu Kumar**

AI / Data Engineering project focused on practical application of NLP, semantic similarity, information extraction, and resume-to-job matching.
