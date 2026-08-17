from src.extraction.pdf_parser import PDFParser

from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.chunker import SemanticChunker

from src.utils.file_handler import FileHandler

from src.extraction.jd_parser import JobDescriptionParser
from src.extraction.requirement_extractor import RequirementExtractor

from src.preprocessing.requirement_normalizer import RequirementNormalizer

from src.llm.embedder import TextEmbedder
from src.llm.matcher import ResumeJDMatcher

from src.evaluation.hybrid_scorer import HybridMatcher

import numpy as np


# ============================================================
# FILE PATHS
# ============================================================

PDF_PATH = "data/raw/resume/resume.pdf"

RAW_TEXT_PATH = "data/processed/resume_raw.txt"

CLEAN_TEXT_PATH = "data/processed/resume_clean.txt"

CHUNKS_PATH = "data/processed/resume_chunks.json"

JD_PATH = "data/raw/job_description/job_description.txt"

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

requirement_extractor = RequirementExtractor()

requirement_normalizer = RequirementNormalizer()

embedder = TextEmbedder()

matcher = ResumeJDMatcher(
    similarity_threshold=0.35
)

hybrid_scorer = HybridMatcher()


# ============================================================
# HELPER
# ============================================================

def calculate_semantic_scores(
    requirement_embedding,
    resume_embeddings
):
    """
    Calculate cosine similarity between one JD requirement
    embedding and every resume chunk embedding.

    Returns:
        list[float]
    """

    query = np.asarray(
        requirement_embedding,
        dtype=float
    )

    matrix = np.asarray(
        resume_embeddings,
        dtype=float
    )

    # --------------------------------------------------------
    # Ensure correct dimensions
    # --------------------------------------------------------

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    query = query.reshape(-1)

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if matrix.shape[1] != query.shape[0]:

        raise ValueError(
            "Embedding dimension mismatch: "
            f"resume embeddings have dimension "
            f"{matrix.shape[1]}, while JD embedding has "
            f"dimension {query.shape[0]}."
        )

    # --------------------------------------------------------
    # Calculate norms
    # --------------------------------------------------------

    query_norm = np.linalg.norm(
        query
    )

    matrix_norms = np.linalg.norm(
        matrix,
        axis=1
    )

    # --------------------------------------------------------
    # Prevent division by zero
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

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


# ============================================================
# DAY 2
# RESUME PROCESSING PIPELINE
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


# ------------------------------------------------------------
# 1. Extract text from resume PDF
# ------------------------------------------------------------

raw_text = parser.extract_text(
    PDF_PATH
)


# ------------------------------------------------------------
# 2. Save raw extracted text
# ------------------------------------------------------------

handler.save_text(
    raw_text,
    RAW_TEXT_PATH
)


# ------------------------------------------------------------
# 3. Clean text and detect sections
# ------------------------------------------------------------

sections = cleaner.clean(
    raw_text
)


# ------------------------------------------------------------
# 4. Reconstruct cleaned text
# ------------------------------------------------------------

cleaned_lines = []

for section in sections:

    cleaned_lines.append(
        section["section"]
    )

    cleaned_lines.extend(
        section["content"]
    )

    cleaned_lines.append("")


clean_text = "\n".join(
    cleaned_lines
).strip()


# ------------------------------------------------------------
# 5. Save cleaned text
# ------------------------------------------------------------

handler.save_text(
    clean_text,
    CLEAN_TEXT_PATH
)


# ------------------------------------------------------------
# 6. Create semantic chunks
# ------------------------------------------------------------

chunks = chunker.create_chunks(
    sections
)


# ------------------------------------------------------------
# 7. Save resume chunks
# ------------------------------------------------------------

handler.save_json(
    chunks,
    CHUNKS_PATH
)


# ============================================================
# DAY 3
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


resume_texts = []

for chunk in chunks:

    resume_texts.append(
        chunk["text"]
    )


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


# ------------------------------------------------------------
# 8. Load and parse job description
# ------------------------------------------------------------

parsed_jd = jd_parser.parse()


# ------------------------------------------------------------
# 9. Extract JD sections
# ------------------------------------------------------------

jd_sections = requirement_extractor.extract(
    parsed_jd["lines"]
)


# ------------------------------------------------------------
# 10. Build structured requirements
# ------------------------------------------------------------

requirements = (
    requirement_extractor
    .build_requirement_objects(
        jd_sections
    )
)


# ------------------------------------------------------------
# 11. Normalize requirements
# ------------------------------------------------------------

normalized_requirements = (
    requirement_normalizer
    .normalize_all(
        requirements
    )
)


# ------------------------------------------------------------
# 12. Save normalized JD requirements
# ------------------------------------------------------------

handler.save_json(
    normalized_requirements,
    JD_REQUIREMENTS_PATH
)


# ============================================================
# DAY 3
# JOB DESCRIPTION EMBEDDINGS
# ============================================================

jd_texts = []

for requirement in normalized_requirements:

    jd_texts.append(
        requirement["original_text"]
    )


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
# DAY 4
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


matcher.save_results(
    match_results,
    MATCH_RESULTS_PATH
)


# ============================================================
# DAY 5
# HYBRID SCORING
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


# ------------------------------------------------------------
# IMPORTANT:
#
# normalized_requirements is a LIST.
#
# match_requirement() expects ONE requirement DICTIONARY.
#
# Therefore we process every requirement separately.
# ------------------------------------------------------------

for index, requirement in enumerate(
    normalized_requirements
):

    requirement_number = index + 1

    print(
        f"\nProcessing requirement "
        f"{requirement_number}/"
        f"{len(normalized_requirements)}"
    )

    # --------------------------------------------------------
    # Get the embedding belonging to THIS requirement
    # --------------------------------------------------------

    requirement_embedding = (
        jd_embeddings_loaded[index]
    )

    # --------------------------------------------------------
    # Calculate semantic similarity between this JD
    # requirement and EVERY resume chunk.
    # --------------------------------------------------------

    semantic_scores = (
        calculate_semantic_scores(
            requirement_embedding,
            resume_embeddings_loaded
        )
    )

    # --------------------------------------------------------
    # Hybrid matching
    # --------------------------------------------------------

    result = hybrid_scorer.match_requirement(

        requirement=requirement,

        resume_chunks=chunks,

        resume_sections=chunks,

        semantic_scores=semantic_scores
    )

    # --------------------------------------------------------
    # Add requirement metadata to result
    # --------------------------------------------------------

    result["requirement_id"] = (
        requirement_number
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


# ============================================================
# SAVE HYBRID RESULTS
# ============================================================

handler.save_json(
    hybrid_results,
    HYBRID_RESULTS_PATH
)


# ============================================================
# OUTPUT
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "RESUME PROCESSING COMPLETED"
)

print(
    "=" * 60
)

print(
    f"Characters extracted: "
    f"{len(raw_text)}"
)

print(
    f"Characters after cleaning: "
    f"{len(clean_text)}"
)

print(
    f"Resume sections detected: "
    f"{len(sections)}"
)

print(
    f"Resume chunks created: "
    f"{len(chunks)}"
)


print(
    "\nDetected resume sections:"
)

for section in sections:

    print(
        f"  - {section['section']}"
    )


# ============================================================
# JD INFORMATION
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "JOB DESCRIPTION PROCESSING COMPLETED"
)

print(
    "=" * 60
)

print(
    f"JD lines extracted: "
    f"{len(parsed_jd['lines'])}"
)

print(
    f"Structured requirements: "
    f"{len(requirements)}"
)


print(
    "\nExtracted JD sections:"
)

for section, items in jd_sections.items():

    print(
        f"\n{section.upper()}"
    )

    if not items:

        print(
            "  - None"
        )

        continue

    for item in items:

        print(
            f"  - {item}"
        )


# ============================================================
# NORMALIZED REQUIREMENTS
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "NORMALIZED REQUIREMENTS"
)

print(
    "=" * 60
)


for index, requirement in enumerate(
    normalized_requirements,
    start=1
):

    print(
        f"\nRequirement {index}"
    )

    print(
        f"  Original: "
        f"{requirement['original_text']}"
    )

    print(
        f"  Category: "
        f"{requirement['category']}"
    )

    print(
        f"  Importance: "
        f"{requirement['importance']}"
    )

    print(
        f"  Skills: "
        f"{requirement['skills']}"
    )

    print(
        f"  Education: "
        f"{requirement.get('education_fields', [])}"
    )

    print(
        f"  Concepts: "
        f"{requirement.get('concepts', [])}"
    )

    print(
        f"  Experience: "
        f"{requirement['experience']}"
    )


# ============================================================
# EMBEDDING INFORMATION
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "EMBEDDING GENERATION COMPLETED"
)

print(
    "=" * 60
)

print(
    f"Resume embeddings: "
    f"{len(resume_embeddings)}"
)

print(
    f"JD embeddings: "
    f"{len(jd_embeddings)}"
)

print(
    f"Embedding dimensions: "
    f"{resume_embeddings.shape[1]}"
)

print(
    "\nResume embeddings saved to:"
)

print(
    f"  {RESUME_EMBEDDINGS_PATH}"
)

print(
    "\nJD embeddings saved to:"
)

print(
    f"  {JD_EMBEDDINGS_PATH}"
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

    print(
        f"\nRequirement "
        f"{result['requirement_id']}"
    )

    print(
        f"Category: "
        f"{result['category']}"
    )

    print(
        f"Requirement: "
        f"{result['requirement']}"
    )

    print(
        f"Best evidence section: "
        f"{result['best_match']['section']}"
    )

    print(
        f"Resume chunk: "
        f"{result['best_match']['chunk_id']}"
    )

    print(
        f"Similarity: "
        f"{result['best_match']['similarity']}"
    )

    print(
        f"Match level: "
        f"{result['match_level']}"
    )


print(
    "\nMatch results saved to:"
)

print(
    f"  {MATCH_RESULTS_PATH}"
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
        f"{result['requirement_id']}"
    )

    print(
        f"Category: "
        f"{result['category']}"
    )

    print(
        f"Score: "
        f"{result['hybrid_score']}"
    )

    print(
        f"Assessment: "
        f"{result['assessment']}"
    )

    best = result.get(
        "best_evidence"
    )

    if best:

        print(
            f"Best evidence: "
            f"{best['chunk_id']} "
            f"({best['section']})"
        )


print(
    "\n"
    + "=" * 60
)

print(
    "DAY 5 COMPLETED SUCCESSFULLY"
)

print(
    "=" * 60
)