import numpy as np

from src.extraction.pdf_parser import PDFParser
from src.extraction.jd_parser import JobDescriptionParser
from src.extraction.requirement_extractor import RequirementExtractor

from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.chunker import SemanticChunker
from src.preprocessing.requirement_normalizer import (
    RequirementNormalizer
)

from src.llm.embedder import TextEmbedder
from src.llm.matcher import ResumeJDMatcher

from src.evaluation.hybrid_scorer import HybridMatcher
from src.evaluation.analysis import ResumeAnalysisEngine

from src.utils.file_handler import FileHandler


# ============================================================
# FILE PATHS
# ============================================================

PDF_PATH = (
    "data/raw/resume/resume.pdf"
)

RAW_TEXT_PATH = (
    "data/processed/resume_raw.txt"
)

CLEAN_TEXT_PATH = (
    "data/processed/resume_clean.txt"
)

CHUNKS_PATH = (
    "data/processed/resume_chunks.json"
)

JD_PATH = (
    "data/raw/job_description/job_description.txt"
)

JD_REQUIREMENTS_PATH = (
    "data/processed/jd_requirements.json"
)

RESUME_EMBEDDINGS_PATH = (
    "data/processed/resume_embeddings.json"
)

JD_EMBEDDINGS_PATH = (
    "data/processed/jd_embeddings.json"
)

MATCH_RESULTS_PATH = (
    "data/processed/match_results.json"
)

HYBRID_RESULTS_PATH = (
    "data/processed/hybrid_results.json"
)

ATS_ANALYSIS_PATH = (
    "data/processed/ats_analysis.json"
)


# ============================================================
# INITIALIZE COMPONENTS
# ============================================================

parser = PDFParser()

cleaner = TextCleaner()

chunker = SemanticChunker(
    max_characters=1800
)

handler = FileHandler()

jd_parser = JobDescriptionParser(
    JD_PATH
)

requirement_extractor = (
    RequirementExtractor()
)

requirement_normalizer = (
    RequirementNormalizer()
)

embedder = TextEmbedder()

matcher = ResumeJDMatcher(
    similarity_threshold=0.35
)

hybrid_scorer = HybridMatcher()

analysis_engine = (
    ResumeAnalysisEngine()
)


# ============================================================
# HELPER: COSINE SIMILARITY
# ============================================================

def calculate_semantic_scores(
    requirement_embedding,
    resume_embeddings
):

    query = np.asarray(
        requirement_embedding,
        dtype=np.float32
    ).reshape(-1)

    matrix = np.asarray(
        resume_embeddings,
        dtype=np.float32
    )

    if matrix.ndim == 1:
        matrix = matrix.reshape(
            1,
            -1
        )

    if matrix.ndim != 2:
        raise ValueError(
            "Resume embeddings must be a 2D matrix."
        )

    if matrix.shape[1] != query.shape[0]:

        raise ValueError(
            "Embedding dimension mismatch: "
            f"resume embeddings have dimension "
            f"{matrix.shape[1]}, while JD embedding "
            f"has dimension {query.shape[0]}."
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
            for _ in range(
                len(matrix)
            )
        ]

    matrix_norms = np.where(
        matrix_norms == 0,
        1e-12,
        matrix_norms
    )

    scores = (
        matrix @ query
    ) / (
        matrix_norms
        *
        query_norm
    )

    scores = np.clip(
        scores,
        -1.0,
        1.0
    )

    return [
        round(
            float(score),
            4
        )
        for score in scores
    ]


# ============================================================
# HELPER: VALIDATE INPUTS
# ============================================================

def validate_pipeline_inputs(
    requirements,
    jd_embeddings,
    chunks,
    resume_embeddings
):

    if not requirements:
        raise ValueError(
            "No normalized JD requirements were generated."
        )

    if not chunks:
        raise ValueError(
            "No resume chunks were generated."
        )

    if jd_embeddings is None:
        raise ValueError(
            "JD embeddings could not be loaded."
        )

    if resume_embeddings is None:
        raise ValueError(
            "Resume embeddings could not be loaded."
        )

    if len(requirements) != len(
        jd_embeddings
    ):

        raise ValueError(
            "Requirement/embedding mismatch: "
            f"{len(requirements)} requirements but "
            f"{len(jd_embeddings)} JD embeddings."
        )

    if len(chunks) != len(
        resume_embeddings
    ):

        raise ValueError(
            "Resume chunk/embedding mismatch: "
            f"{len(chunks)} chunks but "
            f"{len(resume_embeddings)} resume embeddings."
        )


# ============================================================
# HELPER: STANDARDIZE CHUNKS
# ============================================================

def build_resume_evidence_chunks(
    chunks
):

    output = []

    for index, chunk in enumerate(
        chunks
    ):

        if not isinstance(
            chunk,
            dict
        ):

            raise TypeError(
                f"Resume chunk {index} "
                "is not a dictionary."
            )

        output.append(
            {
                **chunk,

                "section": str(
                    chunk.get(
                        "section",
                        "UNKNOWN"
                    )
                ),

                "text": str(
                    chunk.get(
                        "text",
                        ""
                    )
                ),

                "chunk_id": str(
                    chunk.get(
                        "chunk_id",
                        f"resume_{index + 1:03d}"
                    )
                )
            }
        )

    return output


# ============================================================
# DAY 2
# RESUME PROCESSING
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "RESUME PROCESSING"
)

print(
    "=" * 60
)


raw_text = parser.extract_text(
    PDF_PATH
)

if not raw_text.strip():

    raise ValueError(
        "No text could be extracted from the resume PDF."
    )

handler.save_text(
    raw_text,
    RAW_TEXT_PATH
)


sections = cleaner.clean(
    raw_text
)

if not sections:

    raise ValueError(
        "No resume sections were detected."
    )


cleaned_lines = []

for section in sections:

    cleaned_lines.append(
        str(
            section.get(
                "section",
                ""
            )
        )
    )

    cleaned_lines.extend(
        [
            str(line)
            for line in section.get(
                "content",
                []
            )
        ]
    )

    cleaned_lines.append("")


clean_text = "\n".join(
    cleaned_lines
).strip()

handler.save_text(
    clean_text,
    CLEAN_TEXT_PATH
)


chunks = chunker.create_chunks(
    sections
)

chunks = build_resume_evidence_chunks(
    chunks
)

if not chunks:

    raise ValueError(
        "No resume chunks were created."
    )

handler.save_json(
    chunks,
    CHUNKS_PATH
)


# ============================================================
# RESUME EMBEDDINGS
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "RESUME EMBEDDING GENERATION"
)

print(
    "=" * 60
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

embedder.save_embeddings(
    resume_embeddings,
    RESUME_EMBEDDINGS_PATH
)


# ============================================================
# DAY 3
# JOB DESCRIPTION PROCESSING
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "JOB DESCRIPTION PROCESSING"
)

print(
    "=" * 60
)


parsed_jd = jd_parser.parse()

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

normalized_requirements = (
    requirement_normalizer
    .normalize_all(
        requirements
    )
)

if not normalized_requirements:

    raise ValueError(
        "No normalized JD requirements were generated."
    )

handler.save_json(
    normalized_requirements,
    JD_REQUIREMENTS_PATH
)


# ============================================================
# JD EMBEDDINGS
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "JOB DESCRIPTION EMBEDDINGS"
)

print(
    "=" * 60
)


jd_texts = [
    requirement.get(
        "original_text",
        ""
    )
    for requirement
    in normalized_requirements
]

jd_embeddings = (
    embedder.generate_embeddings(
        jd_texts
    )
)

embedder.save_embeddings(
    jd_embeddings,
    JD_EMBEDDINGS_PATH
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

resume_embeddings_loaded = (
    embedder.load_embeddings(
        RESUME_EMBEDDINGS_PATH
    )
)

jd_embeddings_loaded = (
    embedder.load_embeddings(
        JD_EMBEDDINGS_PATH
    )
)


validate_pipeline_inputs(
    normalized_requirements,
    jd_embeddings_loaded,
    chunks,
    resume_embeddings_loaded
)


# ============================================================
# DAY 4
# SEMANTIC MATCHING
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "SEMANTIC MATCHING"
)

print(
    "=" * 60
)


match_results = matcher.match_all(

    normalized_requirements,

    jd_embeddings_loaded,

    chunks,

    resume_embeddings_loaded
)

handler.save_json(
    match_results,
    MATCH_RESULTS_PATH
)


# ============================================================
# DAY 5
# HYBRID MATCHING
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "HYBRID MATCHING"
)

print(
    "=" * 60
)


hybrid_results = []


for index, requirement in enumerate(
    normalized_requirements
):

    print(
        f"\nProcessing requirement "
        f"{index + 1}/"
        f"{len(normalized_requirements)}"
    )

    requirement_embedding = (
        jd_embeddings_loaded[index]
    )

    semantic_scores = (
        calculate_semantic_scores(
            requirement_embedding,
            resume_embeddings_loaded
        )
    )

    result = (
        hybrid_scorer
        .match_requirement(

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
        requirement.get(
            "original_text",
            ""
        )
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


handler.save_json(
    hybrid_results,
    HYBRID_RESULTS_PATH
)


# ============================================================
# DAY 6
# ATS + GAP ANALYSIS
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "DAY 6 - ATS ANALYSIS"
)

print(
    "=" * 60
)


ats_analysis = (
    analysis_engine.analyze(
        hybrid_results
    )
)

handler.save_json(
    ats_analysis,
    ATS_ANALYSIS_PATH
)


# ============================================================
# SEMANTIC RESULTS
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "SEMANTIC MATCHING RESULTS"
)

print(
    "=" * 60
)


for result in match_results:

    best = result.get(
        "best_match",
        {}
    )

    print(
        f"\nRequirement "
        f"{result.get('requirement_id')}"
    )

    print(
        f"Category: "
        f"{result.get('category')}"
    )

    print(
        f"Requirement: "
        f"{result.get('requirement')}"
    )

    print(
        f"Best evidence section: "
        f"{best.get('section', 'N/A')}"
    )

    print(
        f"Resume chunk: "
        f"{best.get('chunk_id', 'N/A')}"
    )

    print(
        f"Similarity: "
        f"{best.get('similarity', 0.0)}"
    )

    print(
        f"Match level: "
        f"{result.get('match_level', 'N/A')}"
    )


# ============================================================
# HYBRID RESULTS
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "HYBRID MATCHING COMPLETED"
)

print(
    "=" * 60
)


print(
    f"Requirements processed: "
    f"{len(hybrid_results)}"
)

print(
    "\nOutput saved to:"
)

print(
    f"  {HYBRID_RESULTS_PATH}"
)


print(
    "\nRequirement summary:"
)


for result in hybrid_results:

    print(
        f"\nRequirement "
        f"{result.get('requirement_id')}"
    )

    print(
        f"Category: "
        f"{result.get('category')}"
    )

    print(
        f"Semantic score: "
        f"{result.get('semantic_score', 0.0)}"
    )

    print(
        f"Score: "
        f"{result.get('hybrid_score', 0.0)}"
    )

    print(
        f"Assessment: "
        f"{result.get('assessment', 'N/A')}"
    )

    best = result.get(
        "best_evidence",
        {}
    )

    if best:

        print(
            f"Best evidence: "
            f"{best.get('chunk_id', 'N/A')} "
            f"({best.get('section', 'N/A')})"
        )


# ============================================================
# ATS SCORE
# ============================================================

print(
    "\n"
    + "-" * 60
)

print(
    "ATS SCORE"
)

print(
    "-" * 60
)

print(
    f"Overall ATS Score: "
    f"{ats_analysis['overall_ats_score']}/100"
)

print(
    f"Interpretation: "
    f"{ats_analysis['score_interpretation']}"
)


# ============================================================
# REQUIREMENT SUMMARY
# ============================================================

summary = (
    ats_analysis[
        "requirement_summary"
    ]
)

print(
    "\n"
    + "-" * 60
)

print(
    "REQUIREMENT SUMMARY"
)

print(
    "-" * 60
)

print(
    f"Total requirements: "
    f"{summary['total']}"
)

print(
    f"Strong alignment: "
    f"{summary['strong']}"
)

print(
    f"Partial alignment: "
    f"{summary['partial']}"
)

print(
    f"Weak alignment: "
    f"{summary['weak']}"
)

print(
    f"No alignment: "
    f"{summary['no_alignment']}"
)


# ============================================================
# CATEGORY SCORES
# ============================================================

print(
    "\n"
    + "-" * 60
)

print(
    "CATEGORY SCORES"
)

print(
    "-" * 60
)

for category, data in (
    ats_analysis[
        "category_summary"
    ].items()
):

    print(
        f"{category}: "
        f"{data['score']}/100"
    )


# ============================================================
# SKILLS
# ============================================================

skills = ats_analysis[
    "skills"
]

print(
    "\n"
    + "-" * 60
)

print(
    "SKILL ANALYSIS"
)

print(
    "-" * 60
)

print(
    "\nMatched skills:"
)

for skill in skills["matched"]:

    print(
        f"  + {skill}"
    )

print(
    "\nMissing skills:"
)

for skill in skills["missing"]:

    print(
        f"  - {skill}"
    )


# ============================================================
# CONCEPTS
# ============================================================

concepts = ats_analysis[
    "concepts"
]

print(
    "\n"
    + "-" * 60
)

print(
    "CONCEPT ANALYSIS"
)

print(
    "-" * 60
)

print(
    "\nMatched concepts:"
)

for concept in concepts["matched"]:

    print(
        f"  + {concept}"
    )

print(
    "\nMissing concepts:"
)

for concept in concepts["missing"]:

    print(
        f"  - {concept}"
    )


# ============================================================
# EDUCATION
# ============================================================

education = ats_analysis[
    "education"
]

print(
    "\n"
    + "-" * 60
)

print(
    "EDUCATION ANALYSIS"
)

print(
    "-" * 60
)

print(
    f"Matched: "
    f"{education['matched']}"
)

print(
    f"Related: "
    f"{education['related']}"
)

print(
    f"Missing: "
    f"{education['missing']}"
)


# ============================================================
# EXPERIENCE
# ============================================================

experience = ats_analysis[
    "experience"
]

print(
    "\n"
    + "-" * 60
)

print(
    "EXPERIENCE ANALYSIS"
)

print(
    "-" * 60
)

print(
    f"Required experience: "
    f"{experience['required']}"
)

print(
    f"Estimated years: "
    f"{experience['estimated_years']}"
)

print(
    f"Evidence: "
    f"{experience['evidence']}"
)


# ============================================================
# PRIORITY GAPS
# ============================================================

priority_gaps = ats_analysis[
    "priority_gaps"
]

print(
    "\n"
    + "-" * 60
)

print(
    "PRIORITY GAPS"
)

print(
    "-" * 60
)


for index, gap in enumerate(
    priority_gaps[:5],
    start=1
):

    print(
        f"\n{index}. Requirement "
        f"{gap.get('requirement_id')}"
    )

    print(
        f"   Importance: "
        f"{gap.get('importance')}"
    )

    print(
        f"   Assessment: "
        f"{gap.get('assessment')}"
    )

    print(
        f"   Score: "
        f"{gap.get('hybrid_score', 0.0)}"
    )

    print(
        f"   Requirement: "
        f"{gap.get('requirement', '')}"
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "DAY 6 COMPLETED SUCCESSFULLY"
)

print(
    "=" * 60
)

print(
    "\nATS analysis saved to:"
)

print(
    f"  {ATS_ANALYSIS_PATH}"
)