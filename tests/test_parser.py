"""
    File for testing the text extraction from Resume PDF
"""

from src.extraction.pdf_parser import PDFParser

parser = PDFParser()

text = parser.extract_text(r"data\raw\resume\resume.pdf")

print(text)