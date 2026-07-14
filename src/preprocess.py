"""PDF text extraction and normalization."""

import re
from pathlib import Path

from pypdf import PdfReader

from . import config


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract raw text from every page of a PDF."""
    try:
        reader = PdfReader(pdf_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as exc:  # corrupt/encrypted PDFs
        print(f"[pdf] Failed to read {pdf_path.name}: {exc}")
        return ""


def normalize_text(text: str) -> str:
    """Normalize extracted contract text.

    - unify unicode quotes/dashes and strip control characters
    - collapse runs of spaces/tabs, limit blank lines
    - drop page-number-only lines and repeated 'Source:' header artifacts
    """
    replacements = {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", " ": " ", "": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if re.fullmatch(r"(page\s*)?-?\s*\d{1,3}\s*-?", line, flags=re.I):
            continue  # bare page numbers / "Page 12"
        if line.lower().startswith("source:") and ".pdf" in line.lower():
            continue  # CUAD provenance headers repeated on every page
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_usable(text: str) -> bool:
    """True when the PDF produced enough text to analyse (not a scan)."""
    return len(text) >= config.MIN_TEXT_CHARS


def chunk_text(text: str) -> list[str]:
    """Split very long contracts into overlapping chunks for the LLM."""
    if len(text) <= config.CHUNK_CHARS:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + config.CHUNK_CHARS, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - config.CHUNK_OVERLAP
    return chunks
