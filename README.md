# AI Resume Analyzer

An AI-powered Resume Analyzer built with Python that analyzes how well a resume matches a specific Job Description (JD).

The current implementation uses NLP, semantic embeddings, structured requirement extraction, and hybrid matching to move beyond simple keyword-based resume screening.

## Current Scope

The project currently processes:

- One resume PDF
- One job description

and produces:

- Structured resume sections and chunks
- Structured and normalized JD requirements
- Requirement-level resume–JD matching results
- Semantic, skill, concept, education, and experience matching evidence

The system is being developed incrementally, with the core processing and matching pipeline completed before adding the final UI and LLM feedback layers.

---

## 1. Project Roadmap

```text
DAY 1
Project Setup
     |
     v
DAY 2
Resume Processing
     |
     v
DAY 3
Job Description Processing
     |
     v
DAY 4
Semantic Matching
     |
     v
DAY 5
Hybrid Matching
     |
     v
NEXT
ATS Score + Resume Gap Analysis
     |
     v
FINAL
LLM Feedback + FastAPI + Streamlit + Evaluation
```

---

# 2. Project Structure

```text
AI-Resume-Analyzer/
|
├── app/
│   └── main.py
|
├── data/
│   ├── raw/
│   │   ├── resume/
│   │   │   └── resume.pdf
│   │   |
│   │   └── jd/
│   │       └── job_description.txt
│   |
│   └── processed/
│       ├── resume_raw.txt
│       ├── resume_clean.txt
│       ├── resume_chunks.json
│       ├── jd_requirements.json
│       └── hybrid_results.json
|
├── src/
│   ├── extraction/
│   │   ├── pdf_parser.py
│   │   ├── jd_parser.py
│   │   └── requirement_extractor.py
│   |
│   ├── preprocessing/
│   │   ├── cleaner.py
│   │   ├── chunker.py
│   │   └── requirement_normalizer.py
│   |
│   ├── evaluation/
│   │   ├── embedder.py
│   │   ├── matcher.py
│   │   └── hybrid_scorer.py
│   |
│   └── utils/
│       └── file_handler.py
|
├── requirements.txt
├── README.md
└── .gitignore
```

> The exact structure may evolve as additional application and evaluation components are added.

---

# 3. Day 1 — Project Setup

The first stage established the Python environment and project structure.

## Create the project

```bash
mkdir AI-Resume-Analyzer
cd AI-Resume-Analyzer
```

## Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies are maintained in `requirements.txt`.

---

# 4. Day 2 — Resume Processing

The resume-processing pipeline converts the PDF into structured information.

```text
resume.pdf
    |
    v
PDF Parser
    |
    v
Raw Text
    |
    v
Text Cleaner
    |
    v
Section Detection
    |
    v
Semantic Chunker
    |
    v
resume_chunks.json
```

## 4.1 PDF extraction

Input:

```text
data/raw/resume/resume.pdf
```

Raw extracted text:

```text
data/processed/resume_raw.txt
```

The raw file preserves the text extracted from the PDF before further processing.

## 4.2 Text cleaning

The cleaning stage handles issues such as:

- Extra whitespace
- Empty lines
- Formatting artifacts
- Inconsistent spacing
- Unnecessary characters

Output:

```text
data/processed/resume_clean.txt
```

## 4.3 Section detection

The resume is divided into meaningful sections such as:

```text
SUMMARY
SKILLS
WORK EXPERIENCE
PROJECTS
EDUCATION
CERTIFICATIONS & LANGUAGES
```

A section-aware representation allows the matching stage to distinguish between different types of evidence.

## 4.4 Semantic chunking

Large sections are divided into manageable chunks.

Example:

```json
{
    "section": "SKILLS",
    "text": "Programming: Python, SQL...",
    "chunk_id": "resume_002"
}
```

Output:

```text
data/processed/resume_chunks.json
```

---

# 5. Day 3 — Job Description Processing

The JD pipeline converts unstructured job-description text into structured requirements.

```text
Job Description
       |
       v
JD Parser
       |
       v
JD Lines
       |
       v
Requirement Extractor
       |
       v
Requirement Normalizer
       |
       v
jd_requirements.json
```

## 5.1 JD parsing

The JD parser converts the description into logical lines and identifies sections such as:

```text
Minimum qualifications
Preferred qualifications
About the job
Responsibilities
```

## 5.2 Requirement extraction

Requirements are categorized into:

```text
required
preferred
responsibility
education
other
```

This prevents unrelated job-description text from being treated as a candidate requirement.

## 5.3 Requirement normalization

A single JD sentence may contain multiple pieces of information.

For example:

```text
2 years of experience using analytics to solve product
or business problems, coding (e.g., Python, R, SQL)...
```

is converted into structured information such as:

```json
{
    "original_text": "...",
    "category": "preferred",
    "importance": "medium",
    "skills": [
        "Python",
        "R",
        "SQL",
        "Statistical Analysis",
        "Data Analytics",
        "Database"
    ],
    "education_fields": [],
    "experience": [
        "2 years"
    ],
    "concepts": [
        "Business Problems"
    ]
}
```

The normalized requirements are saved to:

```text
data/processed/jd_requirements.json
```

Each requirement can contain:

- Original text
- Category
- Importance
- Skills
- Education fields
- Experience
- Concepts

This structured representation is the input to the matching stage.

---

# 6. Day 4 — Semantic Matching

Once both the resume and JD are structured, semantic matching compares their meaning.

Inputs:

```text
resume_chunks.json
+
jd_requirements.json
```

Conceptually:

```text
Resume Chunk
     |
     v
Embedding Model
     |
     v
Resume Vector
     |
     | similarity
     v
JD Requirement Vector
     ^
     |
Embedding Model
     ^
     |
JD Requirement
```

Semantic similarity is useful when the resume and JD use different wording but describe related concepts.

For example:

```text
JD:
develop statistical models

Resume:
built predictive machine learning models
```

These expressions may be semantically related even without an exact keyword match.

---

# 7. Day 5 — Hybrid Matching

Semantic similarity alone is not sufficient.

A resume may be semantically related to a requirement but still miss explicit skills or evidence.

The current hybrid approach combines:

```text
Semantic Similarity
        +
Skill Matching
        +
Concept Matching
        +
Education Matching
        +
Experience Matching
        +
Resume Evidence
```

This makes the final result more explainable and reduces the limitations of purely keyword-based or purely semantic matching.

---

# 8. Skill Matching

For example, suppose the JD requires:

```text
Python
R
SQL
Machine Learning
Marketing Analytics
```

and the resume contains:

```text
Python
SQL
Machine Learning
```

The analyzer can distinguish:

```text
Matched:
Python
SQL
Machine Learning

Missing:
R
Marketing Analytics
```

This is important because semantic similarity should not incorrectly treat a related skill as an explicit skill match.

---

# 9. Concept Matching

Some JD requirements describe broader capabilities rather than individual technologies.

Examples include:

```text
Customer Collaboration
Stakeholder Management
Proof of Concept
Business Insights
Decision Making
Strategic Insights
```

The system therefore evaluates concepts in addition to individual skills.

---

# 10. Education Matching

Education requirements are evaluated separately.

For example:

```text
JD:

Bachelor's degree in Statistics, Data Science,
Mathematics, Physics, Economics, Operations Research,
Engineering, or a related quantitative field.
```

The resume may contain:

```text
Bachelor of Technology in Computer Science
and Artificial Intelligence
```

The education information is evaluated as education evidence rather than being treated as an ordinary skill.

---

# 11. Experience Matching

Experience requirements are also evaluated separately.

For example:

```text
2 years of experience using analytics
to solve product or business problems
```

The system can use resume evidence such as:

```text
Technical Lead
Sep 2024 - Present

Designed and delivered enterprise reporting
and analytics solutions...
```

This allows the analyzer to assess whether the resume provides supporting experience evidence.

---

# 12. Evidence Identification

The matcher does not only produce a score.

It also identifies supporting resume evidence where available.

For example:

```text
Requirement:
Experience with Python, SQL and Machine Learning

Resume evidence:
Programming: Python, SQL

Machine Learning:
Scikit-learn, Regression, Classification,
Clustering, Model Evaluation
```

This makes the result easier to inspect and explain.

---

# 13. Requirement-Level Assessment

Each JD requirement receives an assessment.

The current pipeline uses classifications such as:

```text
STRONG_ALIGNMENT
PARTIAL_ALIGNMENT
WEAK_ALIGNMENT
NO_ALIGNMENT
```

A result can contain information such as:

```text
Requirement
    |
    +-- Score
    +-- Assessment
    +-- Matched Skills
    +-- Missing Skills
    +-- Matched Concepts
    +-- Missing Concepts
    +-- Best Evidence
```

The purpose is to explain why a requirement was or was not satisfied instead of returning only a single overall number.

---

# 14. Generated Files

The current pipeline produces the following files under:

```text
data/processed/
```

## `resume_raw.txt`

Raw text extracted from the resume PDF.

```text
resume.pdf
    |
    v
resume_raw.txt
```

## `resume_clean.txt`

Cleaned and section-aware resume text.

```text
resume_raw.txt
    |
    v
resume_clean.txt
```

## `resume_chunks.json`

Structured resume chunks.

Example:

```json
{
    "section": "PROJECTS",
    "text": "Book Recommender System...",
    "chunk_id": "resume_004"
}
```

## `jd_requirements.json`

Normalized job-description requirements.

Example:

```json
{
    "original_text": "...",
    "category": "preferred",
    "importance": "medium",
    "skills": [
        "Python",
        "R",
        "SQL"
    ],
    "education_fields": [],
    "experience": [
        "2 years"
    ],
    "concepts": [
        "Business Problems"
    ]
}
```

## `hybrid_results.json`

Requirement-level resume–JD matching results.

This is the primary output of the current matching stage.

---

# 15. Running the Current Project

Place the input files at:

```text
data/raw/resume/resume.pdf
data/raw/jd/job_description.txt
```

Then activate the virtual environment and run:

```bash
python -m app.main
```

The pipeline processes the resume and JD and generates the structured and matching outputs under:

```text
data/processed/
```

---

# 16. Expected Processing Flow

A successful run should approximately follow:

```text
============================================================
RESUME PROCESSING COMPLETED
============================================================

Resume sections detected
Resume chunks created


============================================================
JOB DESCRIPTION PROCESSING COMPLETED
============================================================

JD lines extracted
Structured requirements


============================================================
NORMALIZED REQUIREMENTS
============================================================

Requirement 1
Requirement 2
...
Requirement N


============================================================
MATCHING
============================================================

Resume <-> Job Description semantic matching
Hybrid scoring
Requirement assessment


============================================================
PROCESSING COMPLETED
============================================================
```

The exact number of sections, chunks, and requirements depends on the supplied resume and JD.

---

# 17. Important Design Decision

The initial implementation intentionally uses:

```text
One Resume
+
One Job Description
```

instead of immediately building a large dataset.

This allows the core pipeline to be developed and validated first.

Future versions can expand to:

- Multiple resumes
- Multiple job descriptions
- Dataset-based evaluation
- Model benchmarking
- Batch processing

---

# 18. Why Hybrid Matching?

A purely keyword-based system has limitations.

For example:

```text
JD:
statistical modeling

Resume:
predictive modeling using Scikit-learn
```

Keyword matching may fail to recognize the relationship.

However, semantic matching alone can also be misleading.

For example:

```text
JD:
marketing analytics using MATLAB and SQL

Resume:
machine learning using Python and SQL
```

The concepts may be related, but the resume still does not explicitly demonstrate:

```text
MATLAB
Marketing Analytics
```

Hybrid matching addresses this by combining:

```text
Semantic similarity
+
Explicit evidence
```

---

# 19. Current Architecture

```text
                +---------------------+
                |     Resume PDF      |
                +----------+----------+
                           |
                           v
                  +-----------------+
                  |   PDF Parser    |
                  +--------+--------+
                           |
                           v
                  +-----------------+
                  | Text Cleaning   |
                  +--------+--------+
                           |
                           v
                  +-----------------+
                  | Section Detect  |
                  +--------+--------+
                           |
                           v
                  +-----------------+
                  | Semantic Chunker|
                  +--------+--------+
                           |
                           v
                 resume_chunks.json
                           |
                           |
                           v
                  +-----------------+
                  | Semantic Matcher|
                  +--------+--------+
                           |
                           |
                           ^
                           |
                 jd_requirements.json
                           ^
                           |
                  +-----------------+
                  |   Normalizer    |
                  +--------+--------+
                           ^
                           |
                  +-----------------+
                  | Requirement     |
                  | Extractor       |
                  +--------+--------+
                           ^
                           |
                  +-----------------+
                  |    JD Parser    |
                  +--------+--------+
                           ^
                           |
                  +-----------------+
                  | Job Description |
                  +-----------------+

                           |
                           v
                  +-----------------+
                  | Hybrid Scoring  |
                  +--------+--------+
                           |
                           v
                  hybrid_results.json
```

---

# 20. Current Development Status

| Component | Status |
|---|:---:|
| Project setup | Complete |
| Virtual environment | Complete |
| Dependency management | Complete |
| Resume PDF extraction | Complete |
| Resume text cleaning | Complete |
| Resume section detection | Complete |
| Resume semantic chunking | Complete |
| JD parsing | Complete |
| JD requirement extraction | Complete |
| Requirement normalization | Complete |
| Skill extraction | Complete |
| Education extraction | Complete |
| Experience extraction | Complete |
| Concept extraction | Complete |
| Semantic embeddings | Complete |
| Resume–JD semantic matching | Complete |
| Explicit skill matching | Complete |
| Concept matching | Complete |
| Education matching | Complete |
| Experience matching | Complete |
| Hybrid scoring | Complete |
| Requirement-level assessment | Complete |
| Evidence identification | Complete |
| ATS overall score | Next |
| Resume gap analysis | Next |
| AI resume recommendations | Planned |
| LLM-powered feedback | Planned |
| Resume tailoring | Planned |
| FastAPI backend | Planned |
| Streamlit UI | Planned |
| DeepEval evaluation | Planned |

---

# 21. Next Development Stage

The next stage will build on the completed requirement-level matching system.

Planned flow:

```text
hybrid_results.json
        |
        v
Overall ATS Score
        |
        v
Resume Strengths
        |
        v
Resume Gaps
        |
        v
Improvement Recommendations
        |
        v
LLM-powered Explanation
```

After that, the project can be exposed through:

```text
FastAPI
+
Streamlit
```

and evaluated using:

```text
DeepEval
```

---

# 22. Future End-to-End Architecture

```text
                    USER
                     |
                     v
             +---------------+
             |   Streamlit   |
             |      UI       |
             +-------+-------+
                     |
                     v
             +---------------+
             |    FastAPI    |
             |    Backend    |
             +-------+-------+
                     |
          +----------+----------+
          |                     |
          v                     v
    Resume Pipeline        JD Pipeline
          |                     |
          +----------+----------+
                     |
                     v
              Hybrid Matching
                     |
                     v
                ATS Scoring
                     |
                     v
               LLM Analysis
                     |
                     v
             Recommendations
                     |
                     v
                   USER
```

---

# 23. Conclusion

The current version establishes the core intelligence required for a Resume Analyzer.

The system can now:

```text
Read a resume
     |
     v
Understand its sections
     |
     v
Read a job description
     |
     v
Extract structured requirements
     |
     v
Understand skills and concepts
     |
     v
Compare resume and JD semantically
     |
     v
Check explicit evidence
     |
     v
Evaluate individual requirements
     |
     v
Identify matches and gaps
     |
     v
Generate structured matching results
```

The next goal is to transform these requirement-level results into a complete candidate-facing analysis containing:

- Overall ATS score
- Resume strengths
- Resume weaknesses
- Missing skills
- Requirement coverage
- Actionable recommendations
- AI-generated feedback

---

## License

This project is intended for educational and portfolio purposes.
