from src.extraction.pdf_parser import PDFParser
from src.preprocessing.cleaner import TextCleaner

parser = PDFParser()

cleaner = TextCleaner()

text = parser.extract_text(
    r"data\raw\resume\resume.pdf"
)

clean_text = cleaner.clean(text)

print(clean_text)