#!/usr/bin/env python
"""Weekly academic data updater.

Scans the active CV and the Manuscripts Under Review / With Decisions folders
to keep cv_additions and publication_stats current.  Prompts the user only
when an extraction rule produces an ambiguous result.
"""

import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

CV_PATH = Path("/Users/davidjcox/Dropbox/Curriculum Vitae/2026/David J. Cox CV.docx")
ARTICLES_DIR = Path("/Users/davidjcox/Dropbox/Articles/My Articles")
BOOKS_DIR = Path("/Users/davidjcox/Dropbox/Articles/My Books")

# Sidecar for once-per-season prompts (peer_reviewer is once-per-run).
STATE_FILE = Path(__file__).parent / ".academic_last_prompted.json"

CURRENT_YEAR = date.today().year
CURRENT_MONTH = date.today().month

# CV_SECTIONS: dict of all-caps header substrings the parser treats as the
# start of a top-level section.  Order mirrors the CV roughly.
CV_SECTIONS = [
    "BOARD CERTIFICATIONS",
    "EDUCATION",
    "ACADEMIC APPOINTMENTS",
    "MAJOR RESEARCH INTERESTS",
    "PUBLICATIONS",
    "BOOKS",
    "INDUSTRY WHITE PAPERS",
    "RESEARCH SUPPORT",
    "HONORS AND AWARDS",
    "PUBLISHED OPEN-SOURCE CODE",
    "INVITED TALKS",
    "INTERNATIONAL & NATIONAL PRESENTATIONS",
    "REGIONAL AND STATE PRESENTATIONS",
    "LOCAL PRESENTATIONS",
    "POPULAR MEDIA",
    "CHIEF EDITOR",
    "ASSOCIATE EDITOR",
    "INVITED ASSOCIATE EDITOR",
    "AD HOC / GUEST ASSOCIATE EDITOR",
    "BOARD OF EDITORS",
    "EDITORIAL REVIEWER",
    "BOARD OF REVIEWERS",
    "AD HOC JOURNAL REVIEWER",
    "AD HOC BOOK REVIEWER",
    "TEACHING EXPERIENCE",
    "MENTORSHIP",
    "NON-ACADEMIC WORK EXPERIENCE",
    "ADVISORY BOARDS",
    "SERVICE",
    "ACTIVE MEMBERSHIPS",
]


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    resp = input(f"{msg}{suffix}: ").strip()
    return resp or default


def prompt_int(msg: str, default: int = 0) -> int:
    while True:
        raw = prompt(msg, str(default))
        try:
            return int(raw)
        except ValueError:
            print("  must be an integer")


def prompt_yes_no(msg: str, default: bool = False) -> bool:
    d = "y" if default else "n"
    while True:
        raw = prompt(msg + " (y/n)", d).lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False


# ---------------------------------------------------------------------------
# CV parsing
# ---------------------------------------------------------------------------
def load_cv_sections() -> dict[str, list[str]]:
    """Return {section_header: [line, ...]} by walking the body in document
    order, mixing paragraphs and table-row text.  Each body-order table row
    becomes a single line in the current section."""
    from docx.oxml.ns import qn

    doc = Document(str(CV_PATH))
    sections: dict[str, list[str]] = {}
    current = "_preamble"
    sections[current] = []

    body = doc.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            text_ = "".join(t.text or "" for t in child.iter(qn("w:t"))).strip()
            if not text_:
                continue
            matched_header = None
            for header in CV_SECTIONS:
                if text_.upper().startswith(header):
                    matched_header = header
                    break
            if matched_header:
                current = matched_header
                sections[current] = []
            else:
                sections[current].append(text_)
        elif tag == "tbl":
            for row in child.findall(qn("w:tr")):
                row_text = " ".join(
                    "".join(t.text or "" for t in cell.iter(qn("w:t"))).strip()
                    for cell in row.findall(qn("w:tc"))
                ).strip()
                if row_text:
                    sections[current].append(row_text)
    return sections


def count_year_in_presentations(lines: list[str], year: int) -> int:
    """Count presentation entries whose year tag matches.  Presentation
    lines may start with 'YYYY<header>' (invited-talks paragraphs, where
    the tab between year and title gets stripped) or 'NN. Author (Month,
    YYYY). Title...' (table-sourced rows), so we accept YYYY with a
    non-digit char to its left only."""
    pattern = re.compile(rf"(?<!\d){year}")
    return sum(1 for line in lines if pattern.search(line))


def count_year_in_citations(lines: list[str], year: int) -> int:
    """Count publication-style citations tagged with (YYYY)."""
    pattern = re.compile(rf"\({year}\)")
    return sum(1 for line in lines if pattern.search(line))


def count_year_any(lines: list[str], year: int) -> int:
    """Count lines that mention the year anywhere (for honors, grants,
    service that uses date ranges like '2025 – Present').  Use a digit-
    boundary look-around because tabs between year and adjacent text are
    stripped during paragraph extraction, making \\b unreliable."""
    pattern = re.compile(rf"(?<!\d){year}")
    return sum(1 for line in lines if pattern.search(line))


def count_publications_accepted(sections: dict, year: int) -> int:
    """Count 'Accepted/In Press/Published' entries tagged with current year,
    (in press), or (accepted).  Bounded by the next known sub-header inside
    PUBLICATIONS."""
    pub_lines = sections.get("PUBLICATIONS", [])
    subheaders = [
        "Preprints in Review",
        "Accepted/In Press/Published",
        "In Review",
        "In-Preparation",
    ]
    start_idx = None
    end_idx = len(pub_lines)
    for i, line in enumerate(pub_lines):
        if "Accepted/In Press/Published" in line:
            start_idx = i + 1
            continue
        if start_idx is not None:
            if any(sub in line for sub in subheaders if "Accepted" not in sub):
                end_idx = i
                break
    if start_idx is None:
        return 0
    chunk = pub_lines[start_idx:end_idx]
    count = 0
    year_pattern = re.compile(rf"\({year}\)|\(in press\)|\(accepted\)", re.IGNORECASE)
    for line in chunk:
        if year_pattern.search(line):
            count += 1
    return count


def parse_cv_counts(year: int) -> dict:
    """Derive the cv_additions counts we can compute from the CV alone."""
    sections = load_cv_sections()

    counts = {
        "publications": count_publications_accepted(sections, year),
        "research": count_year_any(sections.get("RESEARCH SUPPORT", []), year),
        "awards": count_year_any(sections.get("HONORS AND AWARDS", []), year),
        "invited_presentations": count_year_in_presentations(
            sections.get("INVITED TALKS", []), year
        ),
        "international_national_presentations": count_year_in_presentations(
            sections.get("INTERNATIONAL & NATIONAL PRESENTATIONS", []), year
        ),
        "regional_state_presentations": count_year_in_presentations(
            sections.get("REGIONAL AND STATE PRESENTATIONS", []), year
        ),
        "local_presentations": count_year_in_presentations(
            sections.get("LOCAL PRESENTATIONS", []), year
        ),
        "volunteer": count_year_any(sections.get("SERVICE", []), year),
    }
    return counts


def popular_media_totals(sections: dict) -> dict[str, int]:
    """Parse 'Newsletters to Date: N' / 'Episodes to Date: N' lines from
    the POPULAR MEDIA section.  Returns {label: count}.  Used to compute
    the delta across weekly runs."""
    totals: dict[str, int] = {}
    lines = sections.get("POPULAR MEDIA", [])
    current_label: str | None = None
    for line in lines:
        low = line.lower()
        if low.startswith("newsletter:"):
            current_label = line.split(":", 1)[1].strip()
        elif low.startswith("podcast:"):
            current_label = line.split(":", 1)[1].strip()
        to_date = re.search(r"(?:newsletters|episodes)\s+to\s+date\s*:\s*(\d+)",
                            line, re.IGNORECASE)
        if to_date and current_label:
            totals[current_label] = int(to_date.group(1))
    return totals


# ---------------------------------------------------------------------------
# Article / book scanning
# ---------------------------------------------------------------------------
def normalize_title(raw: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for title matching."""
    cleaned = re.sub(r"[^\w\s]", " ", raw)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


# Filename pattern: '(YYYY) Title, Authors.pdf'.  Year is optional so we can
# also process files that lack the prefix.
_ARTICLE_PATTERN = re.compile(r"^\s*(?:\((\d{4})\)\s*)?(.+?)\.pdf$", re.IGNORECASE)


def parse_article_filename(name: str) -> tuple[int | None, str]:
    """Pull (year, title-without-author-tail) from a 'My Articles' filename.

    Filenames look like:
      '(2017) Application of the matching law to pitch selection..., Cox et al..pdf'
    Title is everything after the year, with the trailing ', Author...'
    chunk stripped so the title alone is used for dedup and CV lookup.
    """
    m = _ARTICLE_PATTERN.match(name)
    if not m:
        return None, name
    year = int(m.group(1)) if m.group(1) else None
    title_with_authors = m.group(2).strip()

    # Strip trailing author chunk: split on the last ', ' and drop it if the
    # tail looks like an author list (contains a known marker).  Otherwise
    # keep the whole title.
    if ", " in title_with_authors:
        head, _, tail = title_with_authors.rpartition(", ")
        tail_low = tail.lower()
        if (
            "et al" in tail_low
            or "&" in tail
            or re.search(r"\b[A-Z][a-zA-Z\-]+$", tail)
        ):
            title_with_authors = head
    return year, title_with_authors.strip()


def parse_book_folder_name(name: str) -> tuple[int | None, str]:
    """Pull (year, title) from a 'My Books' folder name.  Year may be absent."""
    m = re.match(r"^\s*(?:\((\d{4})\)\s*)?(.+?)\s*$", name)
    if not m:
        return None, name
    year = int(m.group(1)) if m.group(1) else None
    title = m.group(2).strip()
    if ", " in title:
        head, _, tail = title.rpartition(", ")
        if "et al" in tail.lower() or "&" in tail:
            title = head
    return year, title


_journals_cache: set[str] | None = None


def known_journals() -> set[str]:
    """Cached set of normalized journal names already in publication_stats."""
    global _journals_cache
    if _journals_cache is None:
        df = pd.read_sql("SELECT DISTINCT journal FROM publication_stats", engine)
        _journals_cache = {normalize_title(j) for j in df["journal"].dropna().tolist()}
    return _journals_cache


def remember_journal(journal: str) -> None:
    """Add a freshly-inserted journal so subsequent rows in this run know it."""
    if journal:
        known_journals().add(normalize_title(journal))


# ---------------------------------------------------------------------------
# publication_stats read/write
# ---------------------------------------------------------------------------
def insert_publication_row(row: dict) -> None:
    cols = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row.keys())
    with engine.begin() as conn:
        conn.execute(text(f"INSERT INTO publication_stats ({cols}) VALUES ({placeholders})"), row)


# ---------------------------------------------------------------------------
# Article / book processing — DOI-keyed since 2026-05
#
# Articles are matched to existing publication_stats rows by DOI.  When a
# DOI is found in the PDF and already attached to a row, the file is
# silently skipped.  When the DOI is new, Crossref provides title / journal
# / year / authors and a single Y/n prompt confirms insertion.  When DOI
# extraction fails entirely, the file is reported and skipped (not auto-
# inserted) — drop a manual row in publication_stats or run the backfill
# script for those.
# ---------------------------------------------------------------------------
from doi_extract import extract_doi  # noqa: E402
from crossref_lookup import lookup_doi_with_retry  # noqa: E402


def load_db_doi_set() -> set[str]:
    """All DOIs already attached to publication_stats rows."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT doi FROM publication_stats WHERE doi IS NOT NULL")
        ).fetchall()
    return {r[0].lower() for r in rows}


def load_db_titles() -> set[str]:
    """Normalized titles in publication_stats — used for book-folder dedup
    as a fallback when no ISBN match is found."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT title FROM publication_stats WHERE title IS NOT NULL")
        ).fetchall()
    return {normalize_title(r[0]) for r in rows}


def load_db_isbns() -> set[str]:
    """All ISBNs already attached to publication_stats rows (digits only)."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT isbn FROM publication_stats WHERE isbn IS NOT NULL")
        ).fetchall()
    out: set[str] = set()
    for r in rows:
        digits = "".join(c for c in r[0] if c.isdigit())
        if digits:
            out.add(digits)
    return out


def isbn_in_folder_name(name: str) -> str | None:
    """If an ISBN-13 (978...) appears in the folder name, return digits-only.
    Most My Books/ folder names won't have this, but it's free to check."""
    digits = "".join(c if c.isdigit() else " " for c in name).split()
    for run in digits:
        if len(run) == 13 and run.startswith("978"):
            return run
    return None


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


def article_type_from_crossref(t: str | None) -> dict:
    """Default category flags inferred from Crossref's work type."""
    if t == "book-chapter":
        return {"theoretical_articles": 0, "empirical_articles": 0,
                "commentary_replies": 0, "book_content": 1}
    return {"theoretical_articles": 0, "empirical_articles": 1,
            "commentary_replies": 0, "book_content": 0}


def insert_from_crossref(crossref: dict, doi: str) -> None:
    """Build and INSERT a publication_stats row from Crossref metadata."""
    journals = known_journals()
    is_new_journal = (
        bool(crossref.get("journal"))
        and normalize_title(crossref["journal"]) not in journals
    )
    row = {
        "year": crossref.get("year"),
        "number_words": 0,
        "number_pages": 0,
        "number_journals": 1 if is_new_journal else 0,
        "title": crossref.get("title"),
        "journal": crossref.get("journal"),
        "new_journal": "Yes" if is_new_journal else "No",
        "authors": crossref.get("authors"),
        "doi": doi,
        "role": role_from_crossref_type(crossref.get("type")),
    }
    row.update(article_type_from_crossref(crossref.get("type")))
    insert_publication_row(row)
    if crossref.get("journal"):
        remember_journal(crossref["journal"])


def process_article_pdf(pdf_path: Path, doi_set: set[str]) -> str:
    """Returns one of: 'skipped' (already known) / 'inserted' / 'no-doi'
    / 'declined' / 'crossref-miss'."""
    doi, src = extract_doi(pdf_path, allow_ocr=True)
    if not doi:
        # No DOI — flag once at end-of-run; don't insert anything.
        return "no-doi"
    if doi in doi_set:
        return "skipped"

    crossref = lookup_doi_with_retry(doi)
    if not crossref:
        print(f"\n[NEW article] {pdf_path.name}")
        print(f"  DOI: {doi} ({src})")
        print(f"  Crossref returned nothing — skipping (try the backfill script)")
        return "crossref-miss"

    print(f"\n[NEW article] {pdf_path.name}")
    print(f"  DOI: {doi} ({src})")
    print(f"    title:    {crossref.get('title')}")
    print(f"    journal:  {crossref.get('journal')}")
    print(f"    year:     {crossref.get('year')}")
    print(f"    authors:  {crossref.get('authors')}")
    print(f"    type:     {crossref.get('type')}")
    if not prompt_yes_no("  insert as new publication_stats row?", True):
        return "declined"

    insert_from_crossref(crossref, doi)
    doi_set.add(doi)
    print("  + inserted publication_stats row")
    return "inserted"


def process_book_folder(folder: Path, db_titles: set[str],
                        db_isbns: set[str]) -> str:
    """Match book folder to existing rows by ISBN (preferred) or title.
    Folders that match neither are reported but NOT inserted to avoid
    duplicates — ISBN entry needs a human anyway."""
    _, title = parse_book_folder_name(folder.name)
    if not title:
        return "skipped"
    folder_isbn = isbn_in_folder_name(folder.name)
    if folder_isbn and folder_isbn in db_isbns:
        return "skipped"
    if normalize_title(title) in db_titles:
        return "skipped"
    return "no-match"


def scan_articles(doi_set: set[str]) -> dict[str, int]:
    counts = {"skipped": 0, "inserted": 0, "no-doi": 0,
              "declined": 0, "crossref-miss": 0}
    if not ARTICLES_DIR.exists():
        print(f"warning: {ARTICLES_DIR} not found")
        return counts
    pdfs = sorted(p for p in ARTICLES_DIR.glob("*.pdf"))
    no_doi_files: list[str] = []
    for pdf in pdfs:
        result = process_article_pdf(pdf, doi_set)
        counts[result] += 1
        if result == "no-doi":
            no_doi_files.append(pdf.name)
    print(f"  scanned {len(pdfs)} article PDFs: "
          f"{counts['skipped']} known, {counts['inserted']} inserted, "
          f"{counts['no-doi']} no-DOI, {counts['declined']} declined, "
          f"{counts['crossref-miss']} crossref-miss")
    if no_doi_files:
        print("  PDFs without an extractable DOI (manual entry needed):")
        for f in no_doi_files:
            print(f"    {f}")
    return counts


def scan_books(db_titles: set[str], db_isbns: set[str]) -> dict[str, int]:
    counts = {"skipped": 0, "no-match": 0}
    if not BOOKS_DIR.exists():
        print(f"warning: {BOOKS_DIR} not found")
        return counts
    folders = sorted(
        p for p in BOOKS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name.lower() != "icon"
    )
    unmatched: list[str] = []
    for f in folders:
        result = process_book_folder(f, db_titles, db_isbns)
        counts[result] += 1
        if result == "no-match":
            unmatched.append(f.name)
    print(f"  scanned {len(folders)} book folders: "
          f"{counts['skipped']} known, {counts['no-match']} not yet in DB")
    if unmatched:
        print("  Book folders not in DB (add via backfill or manual entry):")
        for n in unmatched:
            print(f"    {n}")
    return counts


# ---------------------------------------------------------------------------
# cv_additions update
# ---------------------------------------------------------------------------
def seasonal_prompts(state: dict) -> dict:
    """Peer review + editorial every run; teaching Jan/Jun/Aug; mentorship Aug."""
    results = {
        "peer_reviewer_delta": 0,
        "editorial_decisions_delta": 0,
        "teaching_delta": 0,
        "mentorship_delta": 0,
    }

    print("\n--- weekly peer review ---")
    results["peer_reviewer_delta"] = prompt_int(
        "peer reviews completed since last run", 0
    )

    print("\n--- weekly editorial decisions ---")
    results["editorial_decisions_delta"] = prompt_int(
        "editorial decisions rendered since last run", 0
    )

    last_teaching = state.get("teaching_last_prompted_season")
    this_season = None
    if CURRENT_MONTH in (1, 6, 8):
        this_season = f"{CURRENT_YEAR}-{CURRENT_MONTH}"
    if this_season and last_teaching != this_season:
        print("\n--- seasonal: teaching ---")
        if prompt_yes_no("teaching any new courses this semester?", False):
            results["teaching_delta"] = prompt_int("  how many new courses", 1)
        state["teaching_last_prompted_season"] = this_season

    last_mentorship = state.get("mentorship_last_prompted_year")
    if CURRENT_MONTH == 8 and last_mentorship != CURRENT_YEAR:
        print("\n--- annual: mentorship ---")
        if prompt_yes_no("new mentees this academic year?", False):
            results["mentorship_delta"] = prompt_int("  how many new mentees", 1)
        state["mentorship_last_prompted_year"] = CURRENT_YEAR

    return results


def current_cv_row() -> dict:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM cv_additions WHERE year = :y"), {"y": CURRENT_YEAR}
        ).fetchone()
    if row:
        return dict(row._mapping)
    cols = [
        "year", "total_additions", "certifications", "education", "clinical",
        "research", "publications", "editorial_decisions", "peer_reviewer",
        "invited_presentations", "international_national_presentations",
        "regional_state_presentations", "local_presentations",
        "popular_media_podcasts", "teaching", "mentorship", "volunteer", "awards",
    ]
    new_row = {c: 0 for c in cols}
    new_row["year"] = CURRENT_YEAR
    with engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO cv_additions ({', '.join(cols)})
            VALUES ({', '.join(':' + c for c in cols)})
        """), new_row)
    return new_row


def update_cv_additions(cv_counts: dict, deltas: dict, media_delta: int) -> None:
    row = current_cv_row()
    updates = dict(cv_counts)
    updates["peer_reviewer"] = (row.get("peer_reviewer") or 0) + deltas["peer_reviewer_delta"]
    updates["editorial_decisions"] = (row.get("editorial_decisions") or 0) + deltas["editorial_decisions_delta"]
    updates["popular_media_podcasts"] = (row.get("popular_media_podcasts") or 0) + media_delta
    if deltas["teaching_delta"]:
        updates["teaching"] = (row.get("teaching") or 0) + deltas["teaching_delta"]
    if deltas["mentorship_delta"]:
        updates["mentorship"] = (row.get("mentorship") or 0) + deltas["mentorship_delta"]

    merged = {**{k: row.get(k) or 0 for k in row if k != "total_additions"}, **updates}
    total_cols = [
        "certifications", "education", "clinical", "research", "publications",
        "editorial_decisions", "peer_reviewer", "invited_presentations",
        "international_national_presentations", "regional_state_presentations",
        "local_presentations", "popular_media_podcasts", "teaching",
        "mentorship", "volunteer", "awards",
    ]
    updates["total_additions"] = sum(int(merged.get(c) or 0) for c in total_cols)

    set_clause = ", ".join(f"{k} = :{k}" for k in updates.keys())
    params = dict(updates)
    params["year"] = CURRENT_YEAR
    with engine.begin() as conn:
        conn.execute(
            text(f"UPDATE cv_additions SET {set_clause} WHERE year = :year"),
            params,
        )
    print(f"\ncv_additions[{CURRENT_YEAR}] updated — total_additions={updates['total_additions']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    if not CV_PATH.exists():
        print(f"ERROR: CV not found at {CV_PATH}")
        return 1

    print(f"=== academic data update — {date.today()} ===")

    state = load_state()

    print("\nparsing CV...")
    sections = load_cv_sections()
    cv_counts = parse_cv_counts(CURRENT_YEAR)
    for k, v in cv_counts.items():
        print(f"  {k}: {v}")

    # popular-media delta: sum of increases in 'Newsletters/Episodes to Date'
    # across all tracked shows since last run.
    media_totals = popular_media_totals(sections)
    prior_totals = state.get("popular_media_totals", {})
    media_delta = 0
    for label, total in media_totals.items():
        prior = prior_totals.get(label, 0)
        if total > prior:
            media_delta += total - prior
    print(f"\npopular media delta since last run: {media_delta}")
    state["popular_media_totals"] = media_totals

    doi_set = load_db_doi_set()
    db_titles = load_db_titles()
    db_isbns = load_db_isbns()
    print(f"\npublication_stats: {len(db_titles)} rows, "
          f"{len(doi_set)} with DOIs, {len(db_isbns)} with ISBNs")

    print("\nscanning Articles/My Articles...")
    scan_articles(doi_set)
    print("\nscanning Articles/My Books...")
    scan_books(db_titles, db_isbns)

    deltas = seasonal_prompts(state)
    update_cv_additions(cv_counts, deltas, media_delta)
    save_state(state)

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
