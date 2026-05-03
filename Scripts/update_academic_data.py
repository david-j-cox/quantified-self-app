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
PROJECTS_ROOT = Path("/Users/davidjcox/Dropbox/Projects")
UNDER_REVIEW_DIR = PROJECTS_ROOT / "Manuscripts Under Review"
PUBLISHED_DIR = PROJECTS_ROOT / "Manuscripts With Decisions" / "Published"
REJECTED_DIR = PROJECTS_ROOT / "Manuscripts With Decisions" / "Rejected"

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
# Manuscript scanning
# ---------------------------------------------------------------------------
def normalize_title(raw: str) -> str:
    """Lowercase, strip punctuation/whitespace for title matching."""
    return re.sub(r"[^\w\s]", "", raw).strip().lower()


def find_main_docx(folder: Path) -> Path | None:
    """Pick the likely manuscript .docx inside a manuscript folder.

    Preference order:
      1. 'Manuscript.docx'
      2. 'Main Manuscript*.docx' (prefers 'With Title' over 'No Title')
      3. file name matching folder name (ignoring parenthetical prefixes)
      4. any .docx that is NOT cover letter / title page / figures / tables
         / appendix / preprint / response
      5. one level of recursion into subfolders, same rules
    """
    def search(dir_: Path) -> Path | None:
        docxs = [
            p for p in dir_.glob("*.docx")
            if not p.name.startswith("~$") and p.stat().st_size > 0
        ]
        if not docxs:
            return None

        for p in docxs:
            if p.stem.lower() == "manuscript":
                return p

        main_manuscripts = [p for p in docxs if p.stem.lower().startswith("main manuscript")]
        if main_manuscripts:
            with_title = [p for p in main_manuscripts if "with title" in p.stem.lower()]
            if with_title:
                return with_title[0]
            return main_manuscripts[0]

        folder_norm = normalize_title(dir_.name)
        for p in docxs:
            if normalize_title(p.stem) == folder_norm:
                return p

        excluded = (
            "cover letter", "title page", "figures", "tables", "appendix",
            "preprint", "response", "revision notes", "reply",
        )
        candidates = [p for p in docxs if not any(x in p.stem.lower() for x in excluded)]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]
        return None

    hit = search(folder)
    if hit:
        return hit
    for sub in folder.iterdir():
        if sub.is_dir() and not sub.name.startswith("."):
            hit = search(sub)
            if hit:
                return hit
    return None


def docx_word_count(path: Path) -> int:
    doc = Document(str(path))
    total = 0
    for p in doc.paragraphs:
        total += len(p.text.split())
    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    total += len(p.text.split())
    return total


def docx_page_count(path: Path) -> int | None:
    """Pull pages from core/app properties.  Often absent until Word saves."""
    doc = Document(str(path))
    try:
        props = doc.core_properties
        if hasattr(props, "pages") and props.pages:
            return int(props.pages)
    except Exception:
        pass
    try:
        app_props = doc.part.package.part_related_by.get("app")
    except Exception:
        app_props = None
    return None


def infer_journal_from_cv_in_review(title: str) -> str | None:
    """If the CV's 'In Review' subsection lists this manuscript, try to
    pull the journal name from the citation line."""
    sections = load_cv_sections()
    pub_lines = sections.get("PUBLICATIONS", [])
    in_review_chunk = []
    capture = False
    for line in pub_lines:
        if "In Review" in line and "In-Preparation" not in line:
            capture = True
            continue
        if capture and "In-Preparation" in line:
            break
        if capture:
            in_review_chunk.append(line)

    title_norm = normalize_title(title)
    for line in in_review_chunk:
        if title_norm in normalize_title(line):
            parts = re.split(r"\.\s+", line)
            if parts:
                last = parts[-1].strip().rstrip(".")
                if 5 <= len(last) <= 80:
                    return last
    return None


def known_journals() -> set[str]:
    df = pd.read_sql("SELECT DISTINCT journal FROM publication_stats", engine)
    return {normalize_title(j) for j in df["journal"].dropna().tolist()}


# ---------------------------------------------------------------------------
# Tracking table
# ---------------------------------------------------------------------------
def ensure_tracking_table() -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS academic_manuscript_tracking (
                title_normalized TEXT PRIMARY KEY,
                title TEXT,
                status TEXT,
                folder_path TEXT,
                first_seen DATE,
                last_updated DATE
            )
        """))


def tracking_get(title_norm: str) -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM academic_manuscript_tracking WHERE title_normalized = :t"),
            {"t": title_norm},
        ).fetchone()
    return dict(row._mapping) if row else None


def tracking_upsert(title_norm: str, title: str, status: str, folder_path: str) -> None:
    today = date.today()
    existing = tracking_get(title_norm)
    with engine.begin() as conn:
        if existing:
            conn.execute(text("""
                UPDATE academic_manuscript_tracking
                SET status = :s, folder_path = :f, last_updated = :d, title = :t
                WHERE title_normalized = :n
            """), {"s": status, "f": folder_path, "d": today, "t": title, "n": title_norm})
        else:
            conn.execute(text("""
                INSERT INTO academic_manuscript_tracking
                    (title_normalized, title, status, folder_path, first_seen, last_updated)
                VALUES (:n, :t, :s, :f, :d, :d)
            """), {"n": title_norm, "t": title, "s": status, "f": folder_path, "d": today})


# ---------------------------------------------------------------------------
# publication_stats read/write
# ---------------------------------------------------------------------------
def publication_row_by_title(title: str) -> dict | None:
    title_norm = normalize_title(title)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT ctid, * FROM publication_stats WHERE title IS NOT NULL")
        ).fetchall()
    for r in rows:
        d = dict(r._mapping)
        if normalize_title(d.get("title") or "") == title_norm:
            return d
    return None


def ask_article_type() -> dict:
    print("  article type?")
    print("    1) theoretical")
    print("    2) empirical")
    print("    3) commentary / reply")
    print("    4) book content")
    while True:
        c = prompt("  choice (1-4)", "2")
        if c in {"1", "2", "3", "4"}:
            break
    return {
        "theoretical_articles": 1 if c == "1" else 0,
        "empirical_articles": 1 if c == "2" else 0,
        "commentary_replies": 1 if c == "3" else 0,
        "book_content": 1 if c == "4" else 0,
    }


def insert_publication_row(row: dict) -> None:
    cols = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row.keys())
    with engine.begin() as conn:
        conn.execute(text(f"INSERT INTO publication_stats ({cols}) VALUES ({placeholders})"), row)


def update_publication_row(title: str, updates: dict) -> None:
    set_clause = ", ".join(f"{k} = :{k}" for k in updates.keys())
    params = dict(updates)
    params["title"] = title
    with engine.begin() as conn:
        conn.execute(
            text(f"UPDATE publication_stats SET {set_clause} WHERE title = :title"),
            params,
        )


# ---------------------------------------------------------------------------
# Manuscript processing
# ---------------------------------------------------------------------------
def process_manuscript(folder: Path, status: str, journal_hint: str | None) -> None:
    title = folder.name
    title_norm = normalize_title(title)
    tracked = tracking_get(title_norm)
    existing_row = publication_row_by_title(title)

    if tracked and tracked["status"] == status and existing_row:
        return

    main_doc = find_main_docx(folder)
    if main_doc is None:
        print(f"\n[{status}] {title}")
        print("  ⚠ could not locate main .docx — skipping")
        return

    print(f"\n[{status}] {title}")
    print(f"  main doc: {main_doc.name}")

    try:
        words = docx_word_count(main_doc)
        pages = docx_page_count(main_doc)
    except Exception as e:
        print(f"  ⚠ could not read {main_doc.name} ({e}) — skipping")
        return
    if pages is None:
        print(f"  words: {words}, pages not in metadata")
        if prompt_yes_no(f"  use estimate ({words // 250} pages from words/250)?", True):
            pages = max(1, words // 250)
        else:
            pages = prompt_int("  enter actual page count")
    else:
        print(f"  words: {words}, pages: {pages}")

    journal = journal_hint
    if not journal:
        inferred = infer_journal_from_cv_in_review(title)
        if inferred:
            if prompt_yes_no(f"  journal '{inferred}' — correct?", True):
                journal = inferred
    if not journal:
        journal = prompt("  journal name")

    journals = known_journals()
    new_journal_auto = normalize_title(journal) not in journals
    print(f"  '{journal}' {'NEW to catalog' if new_journal_auto else 'already in catalog'}")
    if prompt_yes_no(f"  mark as new journal?", new_journal_auto):
        new_journal_val = "Yes"
        number_journals = 1
    else:
        new_journal_val = "No"
        number_journals = 0

    if existing_row:
        updates = {
            "number_words": words,
            "number_pages": pages,
            "journal": journal,
            "new_journal": new_journal_val,
            "number_journals": number_journals,
        }
        if status == "published":
            if prompt_yes_no(
                f"  update year from {existing_row.get('year')} to {CURRENT_YEAR}?",
                False,
            ):
                updates["year"] = CURRENT_YEAR
        update_publication_row(title, updates)
        print(f"  ↻ updated publication_stats row")
    else:
        row = {
            "year": CURRENT_YEAR,
            "number_words": words,
            "number_pages": pages,
            "number_journals": number_journals,
            "title": title,
            "journal": journal,
            "new_journal": new_journal_val,
        }
        row.update(ask_article_type())
        insert_publication_row(row)
        print(f"  + inserted publication_stats row")

    tracking_upsert(title_norm, title, status, str(folder))


def scan_under_review() -> None:
    if not UNDER_REVIEW_DIR.exists():
        print(f"warning: {UNDER_REVIEW_DIR} not found")
        return
    for folder in sorted(p for p in UNDER_REVIEW_DIR.iterdir() if p.is_dir()):
        process_manuscript(folder, "under_review", None)


def scan_published() -> None:
    if not PUBLISHED_DIR.exists():
        print(f"warning: {PUBLISHED_DIR} not found")
        return
    for journal_folder in sorted(p for p in PUBLISHED_DIR.iterdir() if p.is_dir()):
        journal_name = journal_folder.name
        for manuscript_folder in sorted(p for p in journal_folder.iterdir() if p.is_dir()):
            process_manuscript(manuscript_folder, "published", journal_name)


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
    ensure_tracking_table()

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

    print("\nscanning Manuscripts Under Review...")
    scan_under_review()
    print("\nscanning Manuscripts With Decisions / Published...")
    scan_published()

    deltas = seasonal_prompts(state)
    update_cv_additions(cv_counts, deltas, media_delta)
    save_state(state)

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
