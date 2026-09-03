from pathlib import Path

from pypdf import PdfReader
from docx import Document
from pptx import Presentation
from openpyxl import load_workbook


def extract_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)

    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n\n".join(pages)


def extract_docx(file_path: str) -> str:
    document = Document(file_path)

    paragraphs = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n".join(paragraphs)


def extract_pptx(file_path: str) -> str:
    presentation = Presentation(file_path)

    slides = []

    for index, slide in enumerate(presentation.slides, start=1):
        slide_text = []

        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_text.append(shape.text)

        if slide_text:
            slides.append(
                f"Slide {index}:\n" + "\n".join(slide_text)
            )

    return "\n\n".join(slides)


def extract_xlsx(file_path: str) -> str:
    workbook = load_workbook(
        file_path,
        read_only=True,
        data_only=True
    )

    sheets = []

    for worksheet in workbook.worksheets:
        rows = []

        for row in worksheet.iter_rows(values_only=True):
            values = [
                str(value) if value is not None else ""
                for value in row
            ]

            if any(values):
                rows.append(" | ".join(values))

        if rows:
            sheets.append(
                f"Sheet: {worksheet.title}\n" +
                "\n".join(rows)
            )

    return "\n\n".join(sheets)


def extract_text(file_path: str) -> str:
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_path)

    if extension == ".docx":
        return extract_docx(file_path)

    if extension == ".pptx":
        return extract_pptx(file_path)

    if extension in [".xlsx", ".xlsm"]:
        return extract_xlsx(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )