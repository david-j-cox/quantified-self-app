#!/usr/bin/env python
"""One-time backfill of publication_stats with DOIs and Crossref metadata.

For each PDF in My Articles/:
  1. Extract DOI (metadata -> text -> OCR fallback).
  2. If DOI already attached to a publication_stats row -> silent skip.
  3. If DOI looks up in Crossref:
        a. If Crossref title (normalized) matches an existing DB row -> ATTACH:
              prompt 'attach DOI to row X? [Y/n]'.
        b. Else -> INSERT NEW: prompt with full Crossref metadata for Y/n.
  4. If DOI extraction fails or Crossref returns nothing:
        Manual prompt with three options:
          [1] match to existing DB row (show top 3 fuzzy candidates),
          [2] insert as new row (you type fields),
          [3] skip this PDF (will not be re-prompted in future runs because
              the filename gets added to .academic_ignore.txt).

This script is idempotent: re-running it skips PDFs whose DOIs are already
in the DB.  Cancelling halfway and restarting is fine.

Usage:
    python Scripts/backfill_publications.py
    python Scripts/backfill_publications.py --no-ocr     # skip the slow path
    python Scripts/backfill_publications.py --limit 5    # first N PDFs only
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).parent))
from doi_extract import extract_doi  # noqa: E402
from crossref_lookup import lookup_doi_with_retry  # noqa: E402

# ---------------------------------------------------------------------------
# Config / DB
# ---------------------------------------------------------------------------
load_dotenv()
engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

ARTICLES_DIR = Path("/Users/davidjcox/Dropbox/Articles/My Articles")
IGNORE_FILE = Path(__file__).parent / ".academic_ignore.txt"


# ---------------------------------------------------------------------------
# Title normalization (matches update_academic_data.py exactly)
# ---------------------------------------------------------------------------
def normalize_title(raw: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", raw or "")
    return re.sub(r"\s+", " ", cleaned).strip().lower()


_ARTICLE_PATTERN = re.compile(r"^\s*(?:\((\d{4})\)\s*)?(.+?)\.pdf$", re.IGNORECASE)


def parse_article_filename(name: str) -> tuple[int | None, str]:
    m = _ARTICLE_PATTERN.match(name)
    if not m:
        return None, name
    year = int(m.group(1)) if m.group(1) else None
    title = m.group(2).strip()
    if ", " in title:
        head, _, tail = title.rpartition(", ")
        tail_low = tail.lower()
        if "et al" in tail_low or "&" in tail or re.search(r"\b[A-Z][a-zA-Z\-]+$", tail):
            title = head
    return year, title.strip()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    return (input(f"{msg}{suffix}: ").strip() or default)


def prompt_yes_no(msg: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{msg} [{d}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False


def prompt_int(msg: str, default: int | None = None) -> int:
    while True:
        raw = prompt(msg, str(default) if default is not None else "")
        try:
            return int(raw)
        except ValueError:
            print("  must be an integer")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def load_db_state() -> tuple[dict[str, dict], set[str]]:
    """Return (rows-by-normalized-title, set-of-DOIs-already-attached)."""
    by_title: dict[str, dict] = {}
    dois: set[str] = set()
    with engine.connect() as conn:
        for r in conn.execute(text("SELECT * FROM publication_stats")).fetchall():
            d = dict(r._mapping)
            if d.get("title"):
                by_title[normalize_title(d["title"])] = d
            if d.get("doi"):
                dois.add(d["doi"].lower())
    return by_title, dois


def attach_doi(title: str, doi: str, crossref: dict) -> None:
    """Attach DOI + Crossref metadata to an existing row, identified by exact
    title match.  Overwrites journal/year/authors with Crossref's values
    (per the design: 'overwrite with canonical title' implies trusting the
    rest too)."""
    updates = {"doi": doi}
    if crossref.get("title"):
        updates["title_new"] = crossref["title"]  # rename in place
    if crossref.get("journal"):
        updates["journal"] = crossref["journal"]
    if crossref.get("year"):
        updates["year"] = crossref["year"]
    if crossref.get("authors"):
        updates["authors"] = crossref["authors"]

    set_parts = []
    params = {"title_match": title}
    for k, v in updates.items():
        if k == "title_new":
            set_parts.append("title = :title_new")
            params["title_new"] = v
        else:
            set_parts.append(f"{k} = :{k}")
            params[k] = v

    sql = f"UPDATE publication_stats SET {', '.join(set_parts)} WHERE title = :title_match"
    with engine.begin() as conn:
        r = conn.execute(text(sql), params)
        if r.rowcount != 1:
            raise RuntimeError(f"attach_doi: expected 1 update, got {r.rowcount} for title {title!r}")


def insert_new(crossref: dict, doi: str | None, role: str,
               theoretical: int = 0, empirical: int = 0,
               commentary: int = 0, book_content: int = 0) -> None:
    row = {
        "year": crossref.get("year"),
        "title": crossref.get("title"),
        "journal": crossref.get("journal"),
        "authors": crossref.get("authors"),
        "doi": doi,
        "isbn": None,
        "role": role,
        "new_journal": "No",
        "number_words": 0,
        "number_pages": 0,
        "number_journals": 0,
        "theoretical_articles": theoretical,
        "empirical_articles": empirical,
        "commentary_replies": commentary,
        "book_content": book_content,
    }
    cols = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row.keys())
    with engine.begin() as conn:
        conn.execute(text(f"INSERT INTO publication_stats ({cols}) VALUES ({placeholders})"), row)


# ---------------------------------------------------------------------------
# Per-PDF flows
# ---------------------------------------------------------------------------
ARTICLE_TYPE_CHOICES = """
  article type:
    1) theoretical
    2) empirical
    3) commentary / reply
    4) book content (for book chapters)
""".strip()


def ask_article_type() -> dict[str, int]:
    print(ARTICLE_TYPE_CHOICES)
    while True:
        c = prompt("  choice (1-4)", "2")
        if c in {"1", "2", "3", "4"}:
            break
    return {
        "theoretical": 1 if c == "1" else 0,
        "empirical":   1 if c == "2" else 0,
        "commentary":  1 if c == "3" else 0,
        "book_content": 1 if c == "4" else 0,
    }


def show_crossref(crossref: dict) -> None:
    print(f"    title:    {crossref.get('title')}")
    print(f"    journal:  {crossref.get('journal')}")
    print(f"    year:     {crossref.get('year')}")
    print(f"    authors:  {crossref.get('authors')}")
    print(f"    type:     {crossref.get('type')}")


def role_from_crossref_type(t: str | None) -> str:
    return {
        "journal-article": "article",
        "book-chapter": "chapter",
        "book": "author",
        "monograph": "author",
        "edited-book": "editor",
        "posted-content": "preprint",
        "report": "report",
    }.get(t or "", "article")


def article_type_from_crossref(t: str | None) -> dict[str, int]:
    if t == "book-chapter":
        return {"theoretical": 0, "empirical": 0, "commentary": 0, "book_content": 1}
    return {"theoretical": 0, "empirical": 1, "commentary": 0, "book_content": 0}


def fuzzy_title_candidates(file_title: str, by_title: dict[str, dict],
                           top: int = 3) -> list[dict]:
    """Cheap candidate ranking: count shared word-runs of length 3+ between
    the file title and each DB title."""
    fn = normalize_title(file_title).split()
    scored = []
    for db_norm, row in by_title.items():
        db_words = db_norm.split()
        # count word overlap (Jaccard-ish)
        shared = len(set(fn) & set(db_words))
        if shared < 3:
            continue
        scored.append((shared, row))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:top]]


def add_to_ignore(filename: str) -> None:
    line = filename.replace(".pdf", "").lstrip("()0123456789 ")
    with IGNORE_FILE.open("a") as f:
        f.write(f"\n# Skipped during backfill {Path(__file__).name}\n{line}\n")


def process_pdf(pdf: Path, by_title: dict[str, dict], dois: set[str],
                allow_ocr: bool) -> str:
    """Returns one of: 'attached', 'inserted', 'skipped', 'already-known'."""
    file_year, file_title = parse_article_filename(pdf.name)
    print(f"\n[{pdf.name}]")

    # Step 1: extract DOI
    doi, src = extract_doi(pdf, allow_ocr=allow_ocr)
    if doi:
        print(f"  DOI: {doi} (via {src})")
        if doi in dois:
            print("  already in DB -> skipping")
            return "already-known"
    else:
        print(f"  DOI: <not found>")

    # Step 2: Crossref lookup if we have a DOI
    crossref = lookup_doi_with_retry(doi) if doi else None
    if crossref:
        print("  Crossref:")
        show_crossref(crossref)
        # Try to match an existing DB row using EITHER the file-derived title
        # OR the Crossref canonical title.  This catches the common case
        # where the DB's title is a renamed-from-filename shorthand and the
        # Crossref title is the formal version (or vice versa).
        file_norm = normalize_title(file_title)
        canon_norm = normalize_title(crossref.get("title") or "")
        match = by_title.get(file_norm) or by_title.get(canon_norm)
        old_norm = file_norm if file_norm in by_title else canon_norm
        if match:
            print(f"  -> matches existing row: {match['title']!r}")
            if prompt_yes_no("  attach DOI and update with Crossref values?", True):
                attach_doi(match["title"], doi, crossref)
                # Refresh in-memory state under the new canonical title
                new_norm = normalize_title(crossref["title"])
                by_title.pop(old_norm, None)
                fresh = dict(match)
                fresh["title"] = crossref["title"]
                fresh["doi"] = doi
                by_title[new_norm] = fresh
                dois.add(doi)
                return "attached"
            return "skipped"
        # No exact match — insert new
        if prompt_yes_no("  insert as new publication_stats row?", True):
            role = role_from_crossref_type(crossref.get("type"))
            type_flags = article_type_from_crossref(crossref.get("type"))
            insert_new(crossref, doi, role, **type_flags)
            by_title[canon_norm] = {
                "title": crossref["title"], "doi": doi, "year": crossref.get("year"),
                "journal": crossref.get("journal"), "authors": crossref.get("authors"),
            }
            dois.add(doi)
            return "inserted"
        return "skipped"

    # Step 3: manual fallback (no DOI or Crossref miss)
    candidates = fuzzy_title_candidates(file_title, by_title)
    print(f"  filename title: {file_title!r}")
    if candidates:
        print("  closest existing rows:")
        for i, c in enumerate(candidates, 1):
            print(f"    [{i}] ({c.get('year')}) {c.get('title')}  | {c.get('journal')}")
    print("  options:")
    print("    [1-3] match to candidate above")
    print("    [n]   insert as new (manual entry)")
    print("    [s]   skip this PDF and add filename to ignore list")
    while True:
        choice = prompt("  choice", "s").lower()
        if choice == "s":
            add_to_ignore(pdf.name)
            return "skipped"
        if choice == "n":
            year = file_year or prompt_int("  year", file_year or 2026)
            title = prompt("  title", file_title)
            journal = prompt("  journal")
            authors = prompt("  authors (optional)")
            type_flags = ask_article_type()
            crossref_like = {
                "year": year, "title": title, "journal": journal, "authors": authors,
                "type": None,
            }
            insert_new(crossref_like, doi, "article", **type_flags)
            by_title[normalize_title(title)] = {
                "title": title, "doi": doi, "year": year, "journal": journal,
            }
            if doi:
                dois.add(doi)
            return "inserted"
        if choice in ("1", "2", "3"):
            idx = int(choice) - 1
            if idx >= len(candidates):
                print("  no such candidate")
                continue
            match = candidates[idx]
            # Attach DOI (no Crossref values to overwrite with)
            if doi:
                with engine.begin() as conn:
                    conn.execute(
                        text("UPDATE publication_stats SET doi = :d WHERE title = :t"),
                        {"d": doi, "t": match["title"]},
                    )
                dois.add(doi)
                print(f"  attached DOI to existing row")
                return "attached"
            else:
                print(f"  no DOI to attach; nothing to do")
                return "already-known"
        print("  unrecognized choice")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ocr", action="store_true",
                    help="Skip OCR fallback (faster but misses some DOIs)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N PDFs (for testing)")
    args = ap.parse_args()

    if not ARTICLES_DIR.exists():
        print(f"ERROR: {ARTICLES_DIR} not found")
        return 1

    by_title, dois = load_db_state()
    print(f"DB starting state: {len(by_title)} rows, {len(dois)} with DOIs")

    pdfs: Iterable[Path] = sorted(ARTICLES_DIR.glob("*.pdf"))
    if args.limit:
        pdfs = list(pdfs)[: args.limit]
    pdfs = list(pdfs)
    print(f"Scanning {len(pdfs)} PDFs (OCR {'OFF' if args.no_ocr else 'ON'})\n")

    counts = {"attached": 0, "inserted": 0, "skipped": 0, "already-known": 0}
    try:
        for pdf in pdfs:
            result = process_pdf(pdf, by_title, dois, allow_ocr=not args.no_ocr)
            counts[result] += 1
    except KeyboardInterrupt:
        print("\n\n[interrupted by user]")
    finally:
        print("\n=== summary ===")
        for k, v in counts.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
