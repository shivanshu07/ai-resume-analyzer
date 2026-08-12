from src.extraction.pdf_parser import PDFParser
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.chunker import SemanticChunker
from src.utils.file_handler import FileHandler


PDF_PATH = "data/raw/resume/resume.pdf"

RAW_TEXT_PATH = "data/processed/resume_raw.txt"

CLEAN_TEXT_PATH = "data/processed/resume_clean.txt"

CHUNKS_PATH = "data/processed/resume_chunks.json"


# ===================================
# Initialize components
# ===================================

parser = PDFParser()

cleaner = TextCleaner()

chunker = SemanticChunker(
    max_characters=1800
)

handler = FileHandler()


# ===================================
# 1. Extract text from PDF
# ===================================

raw_text = parser.extract_text(
    PDF_PATH
)


# ===================================
# 2. Save raw extracted text
# ===================================

handler.save_text(
    raw_text,
    RAW_TEXT_PATH
)


# ===================================
# 3. Clean text and detect sections
# ===================================

sections = cleaner.clean(
    raw_text
)


# ===================================
# 4. Reconstruct cleaned text
# ===================================

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


# ===================================
# 5. Save cleaned text
# ===================================

handler.save_text(
    clean_text,
    CLEAN_TEXT_PATH
)


# ===================================
# 6. Create semantic chunks
# ===================================

chunks = chunker.create_chunks(
    sections
)


# ===================================
# 7. Save chunks
# ===================================

handler.save_json(
    chunks,
    CHUNKS_PATH
)


# ===================================
# 8. Print processing information
# ===================================

print(
    "\nResume processing completed."
)

print(
    f"Characters extracted: {len(raw_text)}"
)

print(
    f"Characters after cleaning: {len(clean_text)}"
)

print(
    f"Sections detected: {len(sections)}"
)

print(
    f"Chunks created: {len(chunks)}"
)

print(
    "\nDetected sections:"
)

for section in sections:

    print(
        f"  - {section['section']}"
    )