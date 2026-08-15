from src.extraction.pdf_parser import PDFParser
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.chunker import SemanticChunker
from src.utils.file_handler import FileHandler

from src.extraction.jd_parser import JobDescriptionParser
from src.extraction.requirement_extractor import RequirementExtractor
from src.preprocessing.requirement_normalizer import RequirementNormalizer

from src.llm.embedder import TextEmbedder

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

# ============================================================
# DAY 2
# RESUME PROCESSING PIPELINE
# ============================================================

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
# OUTPUT / INFORMATION
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
        f"  Experience: "
        f"{requirement['experience']}"
    )


print(
    "\n"
    + "=" * 60
)

print(
    "DAY 3 PROCESSING COMPLETED"
)

print(
    "=" * 60
)

print(
    f"\nResume chunks saved to:"
)

print(
    f"  {CHUNKS_PATH}"
)

print(
    f"\nNormalized JD requirements saved to:"
)

print(
    f"  {JD_REQUIREMENTS_PATH}"
)

print(
    "\nNext stage:"
)

print(
    "  Resume <-> Job Description semantic matching"
)

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