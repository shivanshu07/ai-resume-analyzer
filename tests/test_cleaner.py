"""
Tests that TextCleaner correctly extracts and structures
resume sections from parsed PDF text.

Uses the same committed sample fixture as test_parser.py --
see that file's docstring for why a fixture is used instead of
a private local resume path.
"""

from pathlib import Path

from src.extraction.pdf_parser import PDFParser
from src.preprocessing.cleaner import TextCleaner

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "resume.pdf"
)


def _get_cleaned_sections():

    parser = PDFParser()
    cleaner = TextCleaner()

    text = parser.extract_text(str(FIXTURE_PATH))

    return cleaner.clean(text)


def test_clean_returns_list_of_sections():

    sections = _get_cleaned_sections()

    assert isinstance(sections, list)
    assert len(sections) > 0


def test_clean_detects_known_section_headings():

    sections = _get_cleaned_sections()

    section_names = {s["section"] for s in sections}

    # The fixture PDF has these exact headings -- confirms
    # TextCleaner.RESUME_SECTIONS recognizes them correctly.
    assert "SUMMARY" in section_names
    assert "WORK EXPERIENCE" in section_names
    assert "EDUCATION" in section_names


def test_clean_preserves_content_under_sections():

    sections = _get_cleaned_sections()

    work_experience = next(
        s for s in sections if s["section"] == "WORK EXPERIENCE"
    )

    joined_content = " ".join(work_experience["content"])

    assert "Software Engineer" in joined_content