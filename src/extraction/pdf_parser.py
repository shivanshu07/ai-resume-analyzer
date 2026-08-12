import fitz
from pathlib import Path


class PDFParser:
    """
    Extract text from PDF documents using PyMuPDF.
    """

    def extract_text(self, pdf_path):
        """
        Extract text from all pages of a PDF.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file.

        Returns
        -------
        str
            Raw extracted text.
        """

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected a PDF file, got: {pdf_path.suffix}"
            )

        try:
            document = fitz.open(pdf_path)

        except Exception as e:
            raise RuntimeError(
                f"Unable to open PDF: {e}"
            )

        extracted_pages = []

        try:

            for page in document:

                page_text = page.get_text("text")

                if page_text:
                    extracted_pages.append(page_text)

        finally:

            document.close()

        return "\n".join(extracted_pages).strip()