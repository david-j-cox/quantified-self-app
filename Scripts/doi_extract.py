"""DOI extraction from PDFs.

Tries three layers, in order, returning as soon as one succeeds:
  1. PDF metadata fields (subject, doi, identifier).
  2. Regex over text extracted from the first 3 pages.
  3. OCR of the first 2 pages (fallback for scanned PDFs).

Returns the DOI as a lowercased string with no surrounding punctuation, or
None if nothing matches.

The DOI regex follows Crossref's recommended pattern:
    10.\\d{4,9}/[-._;()/:A-Z0-9]+
We additionally trim trailing punctuation that often gets pulled in from
running text (".", ",", ")") since the DOI grammar excludes them at end-of-token.
"""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber
import pypdf

DOI_REGEX = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)
TRAILING_PUNCT = re.compile(r"[.,;)\]>]+$")


def _clean(doi: str) -> str:
    return TRAILING_PUNCT.sub("", doi).lower().strip()


def _from_metadata(path: Path) -> str | None:
    try:
        reader = pypdf.PdfReader(str(path))
        meta = reader.metadata or {}
    except Exception:
        return None
    for key in ("/doi", "/DOI", "/Subject", "/Title", "/Identifier", "/prism:doi"):
        val = meta.get(key)
        if not val:
            continue
        m = DOI_REGEX.search(str(val))
        if m:
            return _clean(m.group(1))
    return None


def _from_text(path: Path, max_pages: int = 3) -> str | None:
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages[:max_pages]:
                txt = page.extract_text() or ""
                m = DOI_REGEX.search(txt)
                if m:
                    return _clean(m.group(1))
    except Exception:
        return None
    return None


def _from_ocr(path: Path, max_pages: int = 2) -> str | None:
    """OCR the first few pages — slow, only invoke when the cheaper layers
    have already failed."""
    try:
        from PIL import Image
        import pytesseract
        import pypdfium2 as pdfium
    except ImportError:
        return None
    try:
        doc = pdfium.PdfDocument(str(path))
    except Exception:
        return None
    try:
        for i in range(min(max_pages, len(doc))):
            page = doc[i]
            pil_image = page.render(scale=2.0).to_pil()
            txt = pytesseract.image_to_string(pil_image)
            m = DOI_REGEX.search(txt)
            if m:
                return _clean(m.group(1))
    finally:
        doc.close()
    return None


def extract_doi(path: Path, allow_ocr: bool = True) -> tuple[str | None, str]:
    """Return (doi, source) where source is 'metadata' / 'text' / 'ocr' / 'none'."""
    doi = _from_metadata(path)
    if doi:
        return doi, "metadata"
    doi = _from_text(path)
    if doi:
        return doi, "text"
    if allow_ocr:
        doi = _from_ocr(path)
        if doi:
            return doi, "ocr"
    return None, "none"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python doi_extract.py <pdf-path> [<pdf-path> ...]")
        sys.exit(1)
    for arg in sys.argv[1:]:
        p = Path(arg)
        doi, src = extract_doi(p, allow_ocr=True)
        print(f"{p.name}: {doi or '<not found>'} ({src})")
