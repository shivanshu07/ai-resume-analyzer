"""
Tests text extraction from a resume PDF.

Uses a small, generic sample PDF committed at
tests/fixtures/sample_resume.pdf -- NOT your actual resume.
Your real resume shouldn't be committed to a public repo, and
CI has no access to files that only exist on your local
machine, so this fixture exists specifically so the test is
runnable anywhere (locally or in CI) without depending on
private files or OS-specific paths.
"""

from pathlib import Path

from src.extraction.pdf_parser import PDFParser

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "resume.pdf"
)


def test_extract_text_returns_nonempty_string():

    parser = PDFParser()

    text = parser.extract_text(str(FIXTURE_PATH))

    assert isinstance(text, str)
    assert text.strip() != ""


def test_extract_text_contains_known_content():

    parser = PDFParser()

    text = parser.extract_text(str(FIXTURE_PATH))

    # The fixture PDF deliberately contains these section
    # headings -- confirms extraction preserves real content,
    # not just that it returns *some* non-empty string.
    assert "SUMMARY" in text
    assert "WORK EXPERIENCE" in text


def test_extract_text_raises_on_missing_file():

    parser = PDFParser()

    missing_path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "does_not_exist.pdf"
    )

    try:
        parser.extract_text(str(missing_path))
        assert False, "Expected FileNotFoundError was not raised"

    except FileNotFoundError:
        pass