import uuid
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Read a PDF and return list of pages with text + page number.
    Each item: { "text": "...", "page_number": 1 }
    """
    reader = PdfReader(file_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "text": text.strip(),
                "page_number": i + 1
            })

    return pages


def extract_text_from_txt(file_path: str) -> list[dict]:
    """
    Read a plain text file and return as single page.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return [{"text": text.strip(), "page_number": 1}]


def chunk_pages(pages: list[dict], document_name: str) -> list[dict]:
    """
    Split pages into smaller overlapping chunks.
    Each chunk keeps track of which page and document it came from.

    Why overlapping? So context at page boundaries is never lost.
    chunk_size=500  → each chunk is ~500 characters
    chunk_overlap=50 → 50 chars shared between consecutive chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk_text in page_chunks:
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "page_number": page["page_number"],
                "document_name": document_name,
            })

    return chunks