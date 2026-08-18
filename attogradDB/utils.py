from pypdf import PdfReader


def read_pdf(file_path: str) -> str:
    """Extract text from every page of a PDF."""
    with open(file_path, "rb") as pdf_file:
        reader = PdfReader(pdf_file)
        return "".join(page.extract_text() for page in reader.pages)
