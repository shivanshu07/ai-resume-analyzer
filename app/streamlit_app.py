import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import tempfile

import numpy as np
import streamlit as st


from src.extraction.pdf_parser import PDFParser
from src.extraction.jd_parser import JobDescriptionParser
from src.extraction.requirement_extractor import RequirementExtractor

from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.chunker import SemanticChunker
from src.preprocessing.requirement_normalizer import RequirementNormalizer

from src.llm.embedder import TextEmbedder
from src.llm.matcher import ResumeJDMatcher

from src.evaluation.hybrid_scorer import HybridMatcher
from src.evaluation.analysis import ResumeAnalysisEngine
# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# INITIALIZE COMPONENTS
# ============================================================

@st.cache_resource
def initialize_components():

    return {
        "pdf_parser": PDFParser(),

        "cleaner": TextCleaner(),

        "chunker": SemanticChunker(
            max_characters=1800
        ),

        "requirement_extractor":
            RequirementExtractor(),

        "requirement_normalizer":
            RequirementNormalizer(),

        "embedder":
            TextEmbedder(),

        "matcher":
            ResumeJDMatcher(
                similarity_threshold=0.35
            ),

        "hybrid_scorer":
            HybridMatcher(),

        "analysis_engine":
            ResumeAnalysisEngine()
    }


components = initialize_components()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_semantic_scores(
    requirement_embedding,
    resume_embeddings
):
    """
    Calculate cosine similarity between one JD
    requirement embedding and every resume chunk.
    """

    query = np.asarray(
        requirement_embedding,
        dtype=float
    )

    matrix = np.asarray(
        resume_embeddings,
        dtype=float
    )

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    query = query.reshape(-1)

    if matrix.shape[1] != query.shape[0]:

        raise ValueError(
            "Embedding dimension mismatch."
        )

    query_norm = np.linalg.norm(
        query
    )

    matrix_norms = np.linalg.norm(
        matrix,
        axis=1
    )

    if query_norm == 0:

        return [
            0.0
            for _ in range(len(matrix))
        ]

    matrix_norms = np.where(
        matrix_norms == 0,
        1e-12,
        matrix_norms
    )

    scores = (
        matrix @ query
    ) / (
        matrix_norms * query_norm
    )

    return [
        round(
            float(score),
            4
        )
        for score in scores
    ]


def save_uploaded_file(
    uploaded_file,
    suffix
):
    """
    Save a Streamlit uploaded file temporarily.
    """

    temporary_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temporary_file.write(
        uploaded_file.getvalue()
    )

    temporary_file.close()

    return temporary_file.name


# ============================================================
# COMPLETE ANALYSIS PIPELINE
# ============================================================

def run_analysis(
    resume_file,
    jd_file
):

    pdf_parser = components["pdf_parser"]
    cleaner = components["cleaner"]
    chunker = components["chunker"]

    requirement_extractor = (
        components["requirement_extractor"]
    )

    requirement_normalizer = (
        components["requirement_normalizer"]
    )

    embedder = components["embedder"]
    matcher = components["matcher"]
    hybrid_scorer = components["hybrid_scorer"]

    analysis_engine = (
        components["analysis_engine"]
    )

    # --------------------------------------------------------
    # Temporary files
    # --------------------------------------------------------

    resume_path = save_uploaded_file(
        resume_file,
        ".pdf"
    )

    jd_path = save_uploaded_file(
        jd_file,
        ".txt"
    )

    try:

        # ====================================================
        # STEP 1 — RESUME PDF EXTRACTION
        # ====================================================

        with st.status(
            "Processing resume...",
            expanded=True
        ) as status:

            st.write(
                "Extracting text from PDF..."
            )

            raw_resume_text = (
                pdf_parser.extract_text(
                    resume_path
                )
            )

            if not raw_resume_text.strip():

                raise ValueError(
                    "No text could be extracted "
                    "from the uploaded resume."
                )

            # =================================================
            # STEP 2 — RESUME CLEANING
            # =================================================

            st.write(
                "Cleaning resume text..."
            )

            sections = cleaner.clean(
                raw_resume_text
            )

            if not sections:

                raise ValueError(
                    "No resume sections were detected."
                )

            # =================================================
            # STEP 3 — RESUME CHUNKING
            # =================================================

            st.write(
                "Creating semantic resume chunks..."
            )

            chunks = chunker.create_chunks(
                sections
            )

            if not chunks:

                raise ValueError(
                    "No resume chunks were created."
                )

            # =================================================
            # STEP 4 — RESUME EMBEDDINGS
            # =================================================

            st.write(
                "Generating resume embeddings..."
            )

            resume_texts = [
                chunk["text"]
                for chunk in chunks
            ]

            resume_embeddings = (
                embedder.generate_embeddings(
                    resume_texts
                )
            )

            # =================================================
            # STEP 5 — JD PARSING
            # =================================================

            st.write(
                "Parsing job description..."
            )

            jd_parser = JobDescriptionParser(
                jd_path
            )

            parsed_jd = jd_parser.parse()

            if not parsed_jd["lines"]:

                raise ValueError(
                    "The uploaded job description "
                    "contains no readable text."
                )

            # =================================================
            # STEP 6 — REQUIREMENT EXTRACTION
            # =================================================

            st.write(
                "Extracting job requirements..."
            )

            jd_sections = (
                requirement_extractor.extract(
                    parsed_jd["lines"]
                )
            )

            requirements = (
                requirement_extractor
                .build_requirement_objects(
                    jd_sections
                )
            )

            if not requirements:

                raise ValueError(
                    "No job requirements could "
                    "be extracted from the JD."
                )

            # =================================================
            # STEP 7 — REQUIREMENT NORMALIZATION
            # =================================================

            st.write(
                "Normalizing requirements..."
            )

            normalized_requirements = (
                requirement_normalizer
                .normalize_all(
                    requirements
                )
            )

            # =================================================
            # STEP 8 — JD EMBEDDINGS
            # =================================================

            st.write(
                "Generating job-description embeddings..."
            )

            jd_texts = [
                requirement["original_text"]
                for requirement
                in normalized_requirements
            ]

            jd_embeddings = (
                embedder.generate_embeddings(
                    jd_texts
                )
            )

            # =================================================
            # STEP 9 — SEMANTIC MATCHING
            # =================================================

            st.write(
                "Performing semantic matching..."
            )

            match_results = (
                matcher.match_all(
                    normalized_requirements,
                    jd_embeddings,
                    chunks,
                    resume_embeddings
                )
            )

            # =================================================
            # STEP 10 — HYBRID MATCHING
            # =================================================

            st.write(
                "Performing hybrid matching..."
            )

            hybrid_results = []

            for index, requirement in enumerate(
                normalized_requirements
            ):

                semantic_scores = (
                    calculate_semantic_scores(
                        jd_embeddings[index],
                        resume_embeddings
                    )
                )

                result = (
                    hybrid_scorer.match_requirement(
                        requirement=requirement,
                        resume_chunks=chunks,
                        resume_sections=chunks,
                        semantic_scores=semantic_scores
                    )
                )

                result["requirement_id"] = (
                    index + 1
                )

                result["requirement"] = (
                    requirement["original_text"]
                )

                result["category"] = (
                    requirement.get(
                        "category",
                        "preferred"
                    )
                )

                result["importance"] = (
                    requirement.get(
                        "importance",
                        "medium"
                    )
                )

                result["requirement_type"] = (
                    requirement.get(
                        "category",
                        "preferred"
                    )
                )

                hybrid_results.append(
                    result
                )

            # =================================================
            # STEP 11 — ATS + GAP ANALYSIS
            # =================================================

            st.write(
                "Calculating ATS score and resume gaps..."
            )

            final_analysis = (
                analysis_engine.analyze(
                    hybrid_results
                )
            )

            status.update(
                label="Analysis completed successfully.",
                state="complete",
                expanded=False
            )

        return {
            "sections": sections,
            "chunks": chunks,
            "requirements": normalized_requirements,
            "match_results": match_results,
            "hybrid_results": hybrid_results,
            "analysis": final_analysis
        }

    finally:

        # ----------------------------------------------------
        # Remove temporary files
        # ----------------------------------------------------

        Path(resume_path).unlink(
            missing_ok=True
        )

        Path(jd_path).unlink(
            missing_ok=True
        )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def display_list(
    values,
    empty_message="None"
):

    if not values:

        st.caption(
            empty_message
        )

        return

    for value in values:

        st.write(
            f"• {value}"
        )


def assessment_label(
    assessment
):

    mapping = {
        "STRONG_ALIGNMENT":
            "Strong Alignment",

        "PARTIAL_ALIGNMENT":
            "Partial Alignment",

        "WEAK_ALIGNMENT":
            "Weak Alignment",

        "NO_ALIGNMENT":
            "No Alignment"
    }

    return mapping.get(
        str(assessment).upper(),
        str(assessment)
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "📄 AI Resume Analyzer"
)

st.markdown(
    """
### Resume ↔ Job Description Analysis

Upload a resume and a job description to run the complete
NLP and semantic matching pipeline.

The analyzer evaluates:

- Resume structure
- JD requirements
- Semantic similarity
- Explicit skills
- Concepts
- Education
- Experience
- ATS-style score
- Resume strengths
- Resume gaps
- Priority gaps
"""
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header(
    "1. Upload Documents"
)

col1, col2 = st.columns(2)

with col1:

    resume_file = st.file_uploader(
        "Upload Resume",
        type=["pdf"],
        help="Upload the candidate's resume as a PDF."
    )

with col2:

    jd_file = st.file_uploader(
        "Upload Job Description",
        type=["txt"],
        help="Upload the job description as a .txt file."
    )


# ============================================================
# INPUT PREVIEW
# ============================================================

if jd_file:

    st.subheader(
        "Job Description Preview"
    )

    jd_preview = jd_file.getvalue().decode(
        "utf-8",
        errors="replace"
    )

    st.text_area(
        "JD content",
        jd_preview,
        height=200,
        disabled=True
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🚀 Analyze Resume",
    type="primary",
    use_container_width=True
)


if analyze_button:

    if resume_file is None:

        st.error(
            "Please upload a resume PDF."
        )

        st.stop()

    if jd_file is None:

        st.error(
            "Please upload a job description TXT file."
        )

        st.stop()

    try:

        with st.spinner(
            "Running complete resume analysis..."
        ):

            results = run_analysis(
                resume_file,
                jd_file
            )

        st.session_state["results"] = results

    except Exception as error:

        st.error(
            "Analysis failed."
        )

        st.exception(error)

        st.stop()


# ============================================================
# RESULTS
# ============================================================

if "results" in st.session_state:

    results = st.session_state["results"]

    analysis = results["analysis"]

    hybrid_results = results["hybrid_results"]

    # ========================================================
    # ATS SCORE
    # ========================================================

    st.divider()

    st.header(
        "2. ATS Match Score"
    )

    score = analysis[
        "overall_ats_score"
    ]

    interpretation = analysis[
        "score_interpretation"
    ]

    score_col, interpretation_col = (
        st.columns(2)
    )

    with score_col:

        st.metric(
            "Overall ATS-style Score",
            f"{score:.2f}/100"
        )

    with interpretation_col:

        st.metric(
            "Interpretation",
            interpretation
        )

    st.caption(
        "This is a project-specific ATS-style score, "
        "not an official ATS score from a hiring platform."
    )

    # ========================================================
    # REQUIREMENT SUMMARY
    # ========================================================

    st.header(
        "3. Requirement Coverage"
    )

    summary = analysis[
        "requirement_summary"
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Strong",
            summary.get("strong", 0)
        )

    with c2:

        st.metric(
            "Partial",
            summary.get("partial", 0)
        )

    with c3:

        st.metric(
            "Weak",
            summary.get("weak", 0)
        )

    with c4:

        st.metric(
            "No Alignment",
            summary.get(
                "no_alignment",
                0
            )
        )

    # ========================================================
    # CATEGORY SCORES
    # ========================================================

    st.header(
        "4. Match by Requirement Category"
    )

    category_summary = analysis[
        "category_summary"
    ]

    if category_summary:

        category_columns = st.columns(
            len(category_summary)
        )

        for column, (
            category,
            category_data
        ) in zip(
            category_columns,
            category_summary.items()
        ):

            with column:

                st.metric(
                    category.title(),
                    f"{category_data['score']:.1f}"
                )

                category_counts = (
                    category_data[
                        "summary"
                    ]
                )

                st.caption(
                    f"Strong: "
                    f"{category_counts.get('strong', 0)} | "
                    f"Partial: "
                    f"{category_counts.get('partial', 0)} | "
                    f"Weak: "
                    f"{category_counts.get('weak', 0)}"
                )

    # ========================================================
    # SKILL ANALYSIS
    # ========================================================

    st.header(
        "5. Skills Analysis"
    )

    skills = analysis[
        "skills"
    ]

    skill_col1, skill_col2 = (
        st.columns(2)
    )

    with skill_col1:

        st.subheader(
            "Matched Skills"
        )

        display_list(
            skills.get("matched", [])
        )

    with skill_col2:

        st.subheader(
            "Missing Skills"
        )

        display_list(
            skills.get("missing", [])
        )

    # ========================================================
    # CONCEPT ANALYSIS
    # ========================================================

    st.header(
        "6. Concept Analysis"
    )

    concepts = analysis[
        "concepts"
    ]

    concept_col1, concept_col2 = (
        st.columns(2)
    )

    with concept_col1:

        st.subheader(
            "Matched Concepts"
        )

        display_list(
            concepts.get("matched", [])
        )

    with concept_col2:

        st.subheader(
            "Missing Concepts"
        )

        display_list(
            concepts.get("missing", [])
        )

    # ========================================================
    # EDUCATION
    # ========================================================

    st.header(
        "7. Education Analysis"
    )

    education = analysis[
        "education"
    ]

    education_col1, education_col2 = (
        st.columns(2)
    )

    with education_col1:

        st.subheader(
            "Matched Education"
        )

        display_list(
            education.get("matched", [])
        )

    with education_col2:

        st.subheader(
            "Education Gaps"
        )

        display_list(
            education.get("missing", [])
        )

    # ========================================================
    # EXPERIENCE
    # ========================================================

    st.header(
        "8. Experience Analysis"
    )

    experience = analysis[
        "experience"
    ]

    exp_col1, exp_col2 = (
        st.columns(2)
    )

    with exp_col1:

        st.metric(
            "Estimated Experience",
            f"{experience.get('estimated_years', 0):.1f} years"
        )

    with exp_col2:

        required_experience = experience.get(
            "required",
            []
        )

        st.write(
            "**Required:**"
        )

        display_list(
            required_experience
        )

    if experience.get("evidence"):

        st.subheader(
            "Experience Evidence"
        )

        display_list(
            experience["evidence"]
        )

    # ========================================================
    # PRIORITY GAPS
    # ========================================================

    st.header(
        "9. Priority Gaps"
    )

    priority_gaps = analysis[
        "priority_gaps"
    ]

    if priority_gaps:

        for index, gap in enumerate(
            priority_gaps,
            start=1
        ):

            if isinstance(
                gap,
                dict
            ):

                requirement = gap.get(
                    "requirement",
                    "Unknown requirement"
                )

                st.warning(
                    f"**{index}.** {requirement}"
                )

            else:

                st.warning(
                    f"**{index}.** {gap}"
                )

    else:

        st.success(
            "No high-priority gaps were identified."
        )

    # ========================================================
    # REQUIREMENT DETAILS
    # ========================================================

    st.header(
        "10. Requirement-by-Requirement Analysis"
    )

    for result in hybrid_results:

        requirement_id = result.get(
            "requirement_id",
            ""
        )

        requirement = result.get(
            "requirement",
            ""
        )

        score = float(
            result.get(
                "hybrid_score",
                0.0
            )
        )

        assessment = result.get(
            "assessment",
            ""
        )

        with st.expander(
            f"Requirement {requirement_id} — "
            f"{assessment_label(assessment)}"
        ):

            st.write(
                requirement
            )

            detail_col1, detail_col2, detail_col3 = (
                st.columns(3)
            )

            with detail_col1:

                st.metric(
                    "Hybrid Score",
                    f"{score:.3f}"
                )

            with detail_col2:

                st.write(
                    "**Category**"
                )

                st.write(
                    result.get(
                        "category",
                        "N/A"
                    ).title()
                )

            with detail_col3:

                st.write(
                    "**Importance**"
                )

                st.write(
                    result.get(
                        "importance",
                        "N/A"
                    ).title()
                )

            # ------------------------------------------------
            # Skills
            # ------------------------------------------------

            skill_match = result.get(
                "skill_match",
                {}
            )

            if skill_match:

                st.subheader(
                    "Skill Evidence"
                )

                st.write(
                    f"Score: "
                    f"{skill_match.get('score', 0):.2f}"
                )

                matched = skill_match.get(
                    "matched",
                    []
                )

                missing = skill_match.get(
                    "missing",
                    []
                )

                if matched:

                    st.write(
                        "**Matched:** "
                        + ", ".join(matched)
                    )

                if missing:

                    st.write(
                        "**Missing:** "
                        + ", ".join(missing)
                    )

            # ------------------------------------------------
            # Concepts
            # ------------------------------------------------

            concept_match = result.get(
                "concept_match",
                {}
            )

            if concept_match:

                st.subheader(
                    "Concept Evidence"
                )

                matched = concept_match.get(
                    "matched",
                    []
                )

                missing = concept_match.get(
                    "missing",
                    []
                )

                if matched:

                    st.write(
                        "**Matched:** "
                        + ", ".join(matched)
                    )

                if missing:

                    st.write(
                        "**Missing:** "
                        + ", ".join(missing)
                    )

            # ------------------------------------------------
            # Best Evidence
            # ------------------------------------------------

            evidence = result.get(
                "best_evidence"
            )

            if evidence:

                st.subheader(
                    "Best Resume Evidence"
                )

                st.info(
                    f"**Section:** "
                    f"{evidence.get('section', 'N/A')}\n\n"
                    f"{evidence.get('text', '')}"
                )

    # ========================================================
    # PIPELINE INFORMATION
    # ========================================================

    with st.expander(
        "🔍 Pipeline Information"
    ):

        st.write(
            f"Resume sections detected: "
            f"{len(results['sections'])}"
        )

        st.write(
            f"Resume chunks created: "
            f"{len(results['chunks'])}"
        )

        st.write(
            f"JD requirements extracted: "
            f"{len(results['requirements'])}"
        )

        st.write(
            "Processing pipeline:"
        )

        st.code(
            """
Resume PDF
    ↓
PDF Parser
    ↓
Text Cleaner
    ↓
Semantic Chunker
    ↓
Resume Embeddings
    ↓
JD Parser
    ↓
Requirement Extractor
    ↓
Requirement Normalizer
    ↓
JD Embeddings
    ↓
Semantic Matching
    ↓
Hybrid Matching
    ↓
ATS Scoring
    ↓
Gap Analysis
    ↓
Final Analysis
            """,
            language="text"
        )

    # ========================================================
    # RESET
    # ========================================================

    st.divider()

    if st.button(
        "🔄 Analyze Another Resume"
    ):

        del st.session_state["results"]

        st.rerun()