# AI Resume Analyzer

An AI-powered resume analysis system that evaluates how well a resume matches a given job description using PDF parsing, text cleaning, section detection, semantic embeddings, structured requirement extraction, hybrid matching, ATS-style scoring, and gap analysis.

The project is designed as a focused, reproducible prototype using one resume and one job description. It uses pretrained NLP models rather than training an LLM from scratch.

---

## 1. Project Overview

The goal of the project is to answer:

> How well does a candidate's resume match a particular job description, and what are the most important gaps?

The system processes both the resume and job description and produces:

- Resume text extraction
- Resume section detection
- Resume chunk generation
- Job-description requirement extraction
- Requirement categorization
- Semantic similarity scores
- Hybrid requirement matching
- Skill analysis
- Concept analysis
- Education analysis
- Experience analysis
- Priority gap identification
- Overall ATS-style match score

The current implementation is intended as a proof-of-concept AI Resume Analyzer rather than a production ATS.

---

## 2. Key Features

### Resume Processing

- PDF resume parsing
- Text cleaning and normalization
- Resume section detection
- Section-based chunking

Recognized sections include:

```text
SUMMARY
SKILLS
WORK EXPERIENCE
PROJECTS
EDUCATION
CERTIFICATIONS & LANGUAGES
```

### Job Description Processing

The job description is converted into structured requirements such as:

- Required skills
- Preferred skills
- Responsibilities
- Education requirements
- Experience requirements
- Other requirements

Example:

```text
Requirement:
Experience in data science, statistics, or a related field.

Category:
required

Importance:
high

Skills:
Statistics
Data Science
```

---

## 3. AI / NLP Components

The project uses the pretrained Sentence Transformer:

```text
all-MiniLM-L6-v2
```

The model converts resume chunks and job-description requirements into numerical embeddings.

The current embedding dimension is:

```text
384
```

Semantic similarity is calculated between requirements and resume chunks.

---

## 4. Hybrid Matching

Semantic similarity alone is not sufficient for resume evaluation.

Different requirement types use different evidence:

| Requirement type | Most relevant resume evidence |
|---|---|
| Education | EDUCATION |
| Skills | SKILLS |
| Years of experience | WORK EXPERIENCE |
| Responsibilities | WORK EXPERIENCE |
| Business concepts | WORK EXPERIENCE / SUMMARY |
| Technical concepts | SKILLS / PROJECTS / EXPERIENCE |

The hybrid layer combines semantic evidence with structured evidence and produces assessments such as:

```text
STRONG_ALIGNMENT
PARTIAL_ALIGNMENT
WEAK_ALIGNMENT
NO_ALIGNMENT
```

---

## 5. ATS-Style Analysis

The system generates an ATS-style score and category scores.

Example:

```text
Overall ATS Score: 45.60/100
Interpretation: Weak Match
```

Category scores include:

```text
required
preferred
responsibility
```

The score is an internal project metric and is not the proprietary score of any recruiting platform.

---

## 6. Skill Analysis

The system extracts skills from the job description and compares them against the resume.

Example:

```text
Matched skills:

+ Statistics
+ Python
+ R
+ SQL
+ Data Analytics
+ Machine Learning
+ Artificial Intelligence
```

It also identifies missing skills:

```text
Missing skills:

- Data Science
- Statistical Analysis
- Database
- MATLAB
- Statistical Methods
- Marketing Analytics
- Modeling
- Problem Scoping
- Marketing Effectiveness
```

---

## 7. Concept Analysis

The analyzer also evaluates concepts that may not be simple technical skills.

Example matched concepts:

```text
+ Business Insights
+ Stakeholder Management
+ Business Processes
+ Product Engineering Collaboration
```

Potential gaps include:

```text
- Client Engagement
- Customer Collaboration
- Proof of Concept
- Decision Making
- Strategic Insights
- Innovation
- Metrics
```

This is useful for roles emphasizing business impact, client interaction, stakeholder management, or product collaboration.

---

## 8. Education and Experience Analysis

Education requirements are analyzed separately from general semantic similarity.

Example:

```text
Matched:
Statistics
Data Science
Engineering
```

Experience requirements are extracted and compared against estimated professional experience.

Example:

```text
Required experience:
2 years

Estimated experience:
2.0 years
```

---

## 9. Priority Gaps

The analyzer ranks important requirements that are weakly supported by the resume.

Example:

```text
PRIORITY GAPS

1. Requirement 1
   Importance: high
   Assessment: WEAK_ALIGNMENT

2. Requirement 11
   Importance: medium
   Assessment: NO_ALIGNMENT

3. Requirement 3
   Importance: medium
   Assessment: NO_ALIGNMENT
```

This makes the output actionable for resume improvement.

---

## 10. Project Architecture

A simplified project structure is:

```text
AI-Resume-Analyzer/
│
├── app/
│   └── main.py
│
├── data/
│   ├── raw/
│   │   ├── resume/
│   │   └── job_description/
│   │
│   └── processed/
│
├── src/
│   ├── preprocessing/
│   │   ├── pdf_parser.py
│   │   ├── cleaner.py
│   │   └── chunker.py
│   │
│   ├── embeddings/
│   │   └── embedding_generator.py
│   │
│   ├── matching/
│   │   ├── semantic_matcher.py
│   │   └── hybrid_matcher.py
│   │
│   ├── analysis/
│   │   ├── ats_scorer.py
│   │   └── gap_analyzer.py
│   │
│   └── jd_parser.py
│
├── utils/
│   ├── file_handler.py
│   └── helper.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

The exact directory names may differ slightly depending on the current repository version.

---

## 11. Utility Files

### `utils/file_handler.py`

Provides reusable functions for:

- saving text
- loading text
- saving JSON
- loading JSON
- creating parent directories when required

### `utils/helper.py`

Contains common utility functionality such as creating directories when they do not already exist.

---

## 12. End-to-End Data Flow

```text
Resume PDF
    │
    ▼
PDF Parser
    │
    ▼
Text Cleaning
    │
    ▼
Section Detection
    │
    ▼
Resume Chunking
    │
    ▼
Resume Embeddings
    │
    │
    │
Job Description
    │
    ▼
JD Parser
    │
    ▼
Requirement Extraction
    │
    ▼
Requirement Normalization
    │
    ▼
JD Embeddings
    │
    └──────────────┐
                   ▼
             Semantic Matching
                   │
                   ▼
             Hybrid Matching
                   │
                   ▼
             ATS Scoring
                   │
                   ▼
             Gap Analysis
                   │
                   ▼
             Final Results
```

---

## 13. Input Data

The current project intentionally uses a small evaluation setup.

### Resume

Place the resume PDF in:

```text
data/raw/resume/
```

Example:

```text
data/raw/resume/resume.pdf
```

### Job Description

Place the job description in:

```text
data/raw/job_description/
```

Example:

```text
data/raw/job_description/job_description.txt
```

The exact filenames and paths must match those expected by the current `app/main.py`.

---

## 14. Installation and Reproduction

### Step 1 — Clone the repository

```bash
git clone <your-github-repository-url>
cd AI-Resume-Analyzer
```

### Step 2 — Create a virtual environment

Windows:

```bash
python -m venv venv
```

Git Bash:

```bash
source venv/Scripts/activate
```

Command Prompt:

```cmd
venv\Scripts\activate
```

PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Add the evaluation files

Put the resume and job description in their expected `data/raw/` locations.

### Step 5 — Run the application

From the project root:

```bash
python -m app.main
```

Running it as a module is recommended so that package imports work correctly.

---

## 15. First Model Run

On the first execution, Sentence Transformers may download:

```text
all-MiniLM-L6-v2
```

A Hugging Face warning about an unauthenticated request may appear. This is not an application error. For the current prototype, a Hugging Face token is not required for normal model use.

---

## 16. Expected Pipeline Output

A successful execution displays stages similar to:

```text
============================================================
RESUME PROCESSING
============================================================

============================================================
RESUME EMBEDDING GENERATION
============================================================

============================================================
JOB DESCRIPTION PROCESSING
============================================================

============================================================
JOB DESCRIPTION EMBEDDINGS
============================================================

============================================================
SEMANTIC MATCHING
============================================================

============================================================
HYBRID MATCHING
============================================================

============================================================
DAY 6 - ATS ANALYSIS
============================================================
```

Resume processing may report:

```text
Characters extracted: 3183
Characters after cleaning: 2978
Resume sections detected: 6
Resume chunks created: 7
```

Exact values depend on the input documents.

---

## 17. Generated Files

Processed artifacts are generated under:

```text
data/processed/
```

Important files include:

### `resume_embeddings.json`

Vector representations of resume chunks.

### `jd_embeddings.json`

Vector representations of job requirements.

### `match_results.json`

Semantic matching results.

### `hybrid_results.json`

Requirement-level hybrid scores and assessments.

### `ats_analysis.json`

Final analysis containing:

- ATS score
- category scores
- matched skills
- missing skills
- matched concepts
- missing concepts
- education analysis
- experience analysis
- priority gaps

---

## 18. Reproducing the Current Results

To reproduce the current evaluation as closely as possible:

1. Use the same resume.
2. Use the same job description.
3. Use the repository's `requirements.txt`.
4. Use the same Python environment/version where possible.
5. Use `all-MiniLM-L6-v2`.
6. Run from the repository root.
7. Execute:

```bash
python -m app.main
```

Then inspect:

```text
data/processed/resume_embeddings.json
data/processed/jd_embeddings.json
data/processed/match_results.json
data/processed/hybrid_results.json
data/processed/ats_analysis.json
```

For reproducibility, keep the input documents, model, dependencies, and scoring configuration fixed.

---

## 19. Example Evaluation

A representative semantic matching run produced scores such as:

```text
Requirement 1: 0.4162
Requirement 2: 0.5633
Requirement 4: 0.6309
Requirement 5: 0.5849
Requirement 6: 0.4831
```

A recent calibrated run produced an overall ATS-style score in the mid-40s out of 100.

The exact score can change when the implementation, scoring configuration, resume, or job description changes.

These results are prototype evaluation output, not an authoritative hiring decision.

---

## 20. Semantic vs Hybrid Scores

Semantic similarity measures how similar the language of a requirement is to resume text:

```text
Requirement
    ↓
Sentence Transformer
    ↓
384-dimensional embedding
    ↓
Cosine similarity
```

The hybrid score incorporates additional structured evidence.

Therefore:

```text
Semantic Score ≠ Hybrid Score
```

A requirement can have moderate semantic similarity while having weak structured evidence, or vice versa.

---

## 21. Important Design Considerations

Different requirement types should use different evidence.

### Education

```text
EDUCATION
```

### Skills

```text
SKILLS
PROJECTS
WORK EXPERIENCE
```

### Experience

```text
WORK EXPERIENCE
```

### Responsibilities

```text
WORK EXPERIENCE
PROJECTS
SUMMARY
```

This requirement-aware evidence strategy is important for avoiding misleading hybrid scores.

---

## 22. Limitations

### Single-resume evaluation

The current scope uses one resume and one job description. It is not a statistically validated ATS benchmark.

### Pretrained embedding model

The project uses `all-MiniLM-L6-v2` rather than a domain-specific resume/job matching model.

### Semantic similarity is not qualification

High semantic similarity does not necessarily mean that a candidate satisfies a requirement.

For example, a Bachelor's degree may be semantically similar to a Master's-degree requirement even when the candidate does not possess the required Master's degree.

### Ambiguous requirements

Job descriptions can combine multiple conditions in one sentence. More advanced logical interpretation may therefore be necessary.

### ATS score is an internal metric

The generated score is an ATS-style score for this project. It is not equivalent to the proprietary scoring mechanism of Google, IBM, LinkedIn, Workday, Greenhouse, or another recruiting platform.

---

## 23. Development Roadmap

### Phase 1 — Matching Improvements

- Better requirement-aware weighting
- Improved education matching
- Better experience extraction
- Better synonym handling
- Improved skill normalization

### Phase 2 — Advanced NLP

- Cross-encoder reranking
- Domain-specific embeddings
- LLM-based requirement verification
- Better logical requirement parsing

### Phase 3 — Application

- Streamlit dashboard
- Resume upload interface
- Job-description upload/paste interface
- Visual ATS score
- Skill-gap visualization
- Requirement-by-requirement explanations

### Phase 4 — Evaluation

- Multiple resumes
- Multiple job descriptions
- Ground-truth labels
- Precision/recall evaluation
- Ranking evaluation
- Model comparison

---

## 24. Future LLM Integration

The architecture can later incorporate an LLM for deeper reasoning:

```text
Semantic Matching
        +
Structured Matching
        +
LLM Verification
        ↓
Final Assessment
```

An LLM could verify whether the candidate actually demonstrates the required experience rather than simply checking whether similar words occur in the resume.

Potential applications include:

- transferable skills
- equivalent experience
- business impact
- project relevance
- leadership
- client-facing experience

---

## 25. Git Workflow

After making changes:

```bash
git status
git diff
git add .
git commit -m "Improve resume matching and ATS analysis"
git push origin main
```

Avoid committing:

```text
venv/
__pycache__/
large model files
temporary output files
personal resume files
API tokens
.env
```

---

## 26. Recommended `.gitignore`

```gitignore
venv/
.venv/
__pycache__/
*.pyc
.env
.vscode/
.idea/

data/raw/
```

Whether `data/processed/` should be ignored depends on whether generated evaluation artifacts should be committed to GitHub.

---

## 27. Project Learning Outcomes

This project demonstrates practical experience with:

- Python
- PDF processing
- Natural Language Processing
- Sentence Transformers
- Embeddings
- Cosine similarity
- Semantic search
- Structured information extraction
- Requirement normalization
- Hybrid matching
- ATS-style scoring
- Gap analysis
- JSON-based data pipelines
- Modular Python architecture
- Reproducible ML workflows

The project therefore goes beyond a simple keyword-based resume checker.

---

## 28. Summary

The Resume Analyzer follows this pipeline:

```text
PDF Resume
    ↓
Parsing
    ↓
Cleaning
    ↓
Section Detection
    ↓
Chunking
    ↓
Embeddings
    ↓
Job Description Parsing
    ↓
Requirement Normalization
    ↓
Semantic Matching
    ↓
Hybrid Matching
    ↓
ATS Scoring
    ↓
Skill Analysis
    ↓
Concept Analysis
    ↓
Education Analysis
    ↓
Experience Analysis
    ↓
Priority Gap Analysis
```

The project is currently a working AI/NLP prototype for resume-to-job-description analysis, with an architecture that can later support more advanced matching models, LLM reasoning, larger evaluation datasets, and a Streamlit interface.
