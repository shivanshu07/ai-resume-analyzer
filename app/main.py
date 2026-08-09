from src.extraction.pdf_parser import PDFParser
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.chunker import SemanticChunker
from src.utils.file_handler import FileHandler


PDF_PATH = "data/raw/resume/resume.pdf"

CLEAN_TEXT_PATH = "data/processed/resume_clean.txt"

CHUNKS_PATH = "data/processed/resume_chunks.json"


parser = PDFParser()

cleaner = TextCleaner()

chunker = SemanticChunker(
    max_characters=1500
)

handler = FileHandler()


# -----------------------------------
# 1. Extract PDF text
# -----------------------------------

raw_text = parser.extract_text(
    PDF_PATH
)


# -----------------------------------
# 2. Clean text
# -----------------------------------

clean_text = cleaner.clean(
    raw_text
)


# -----------------------------------
# 3. Save cleaned text
# -----------------------------------

handler.save_text(
    clean_text,
    CLEAN_TEXT_PATH
)


# -----------------------------------
# 4. Detect resume sections
# -----------------------------------

sections = cleaner.detect_sections(
    clean_text
)


# -----------------------------------
# 5. Create semantic chunks
# -----------------------------------

chunks = chunker.create_chunks(
    sections
)


# -----------------------------------
# 6. Save chunks
# -----------------------------------

handler.save_json(
    chunks,
    CHUNKS_PATH
)


print("Resume processing completed.")

print(f"Sections detected: {len(sections)}")

print(f"Chunks created: {len(chunks)}")