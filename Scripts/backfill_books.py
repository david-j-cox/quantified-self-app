#!/usr/bin/env python
"""One-time backfill of role + isbn on existing book_content=1 rows.

For each row in publication_stats with book_content=1:
  1. If row already has both role and isbn -> skip silently.
  2. Try to derive ISBN automatically:
       a. From chapter DOI (Springer/Elsevier/Routledge embed parent ISBN).
       b. From OpenLibrary search by title + author.
  3. Show row + candidate ISBNs (if any), prompt for role and ISBN.

Roles:
  author  - David authored this book
  editor  - David edited this volume
  chapter - David authored a chapter inside an edited volume

Run from project root:
    venv/bin/python Scripts/backfill_books.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).parent))
from openlibrary_lookup import lookup_book, isbn_from_chapter_doi  # noqa: E402

load_dotenv()
engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    return (input(f"{msg}{suffix}: ").strip() or default)


def prompt_choice(msg: str, options: list[str], default_idx: int = 0) -> int:
    """Prompt user to pick one of `options` by 1-based index.  Returns 0-based idx."""
    for i, opt in enumerate(options, 1):
        print(f"    [{i}] {opt}")
    while True:
        raw = prompt(msg, str(default_idx + 1))
        try:
            n = int(raw) - 1
            if 0 <= n < len(options):
                return n
        except ValueError:
            pass
        print("    invalid choice")


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


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def fetch_book_rows() -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT title, year, journal, doi, role, isbn, authors
            FROM publication_stats
            WHERE book_content = 1
            ORDER BY year, title
        """)).fetchall()
    return [dict(r._mapping) for r in rows]


def update_book_row(title: str, role: str, isbn: str | None) -> None:
    params = {"role": role, "title": title}
    set_parts = ["role = :role"]
    if isbn:
        params["isbn"] = isbn
        set_parts.append("isbn = :isbn")
    sql = f"UPDATE publication_stats SET {', '.join(set_parts)} WHERE title = :title"
    with engine.begin() as conn:
        r = conn.execute(text(sql), params)
        if r.rowcount != 1:
            raise RuntimeError(
                f"update_book_row: expected 1 update, got {r.rowcount} for {title!r}"
            )


# ---------------------------------------------------------------------------
# ISBN suggestion logic
# ---------------------------------------------------------------------------
def suggest_isbns(row: dict) -> list[tuple[str, str]]:
    """Return [(isbn, source-description), ...] of candidates for this row."""
    out: list[tuple[str, str]] = []

    # Source 1: chapter DOI
    if row.get("doi"):
        from_doi = isbn_from_chapter_doi(row["doi"])
        if from_doi:
            out.append((from_doi, f"chapter DOI {row['doi']}"))

    # Source 2: OpenLibrary search (use 'Cox' as author hint since David is
    # always either author/editor/chapter-author of these rows)
    try:
        results = lookup_book(row["title"], "Cox", limit=3)
    except Exception:
        results = []
    for r in results:
        for isbn in r.get("isbns", []):
            if isbn not in (i for i, _ in out):
                year_match = "" if r.get("year") == row.get("year") else f" (yr={r.get('year')}, mismatch)"
                out.append((isbn, f"OpenLibrary: {r.get('title')!r}{year_match}"))
        if len(out) >= 6:
            break
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
ROLES = ["author", "editor", "chapter", "report"]


def process_row(row: dict) -> str:
    """Returns 'updated' or 'skipped'."""
    print(f"\n[{row['year']}] {row['title']}")
    print(f"  journal/publisher: {row.get('journal')}")
    if row.get("doi"):
        print(f"  doi:               {row['doi']}")
    if row.get("authors"):
        print(f"  authors:           {row['authors']}")
    print(f"  current role={row.get('role') or '-'}  isbn={row.get('isbn') or '-'}")

    if row.get("role") and row.get("isbn"):
        print("  already complete -> skipping")
        return "skipped"

    # Role
    print("  role:")
    role_idx = prompt_choice("  pick role", ROLES, default_idx=2)  # default 'chapter'
    role = ROLES[role_idx]

    # ISBN
    isbn = row.get("isbn")
    if not isbn:
        print("  finding ISBN candidates...")
        candidates = suggest_isbns(row)
        if candidates:
            options = [f"{i}  ({src})" for i, src in candidates] + ["enter manually", "skip ISBN"]
            print("  candidates:")
            idx = prompt_choice("  pick ISBN", options, default_idx=0)
            if idx < len(candidates):
                isbn = candidates[idx][0]
            elif idx == len(candidates):  # manual
                manual = prompt("  ISBN (paste; blank to skip)")
                isbn = manual or None
            else:  # skip
                isbn = None
        else:
            print("  no candidates from auto-search")
            manual = prompt("  ISBN (paste; blank to skip)")
            isbn = manual or None

    update_book_row(row["title"], role, isbn)
    print(f"  + role={role}  isbn={isbn or '-'}")
    return "updated"


def main() -> int:
    rows = fetch_book_rows()
    print(f"book_content rows in publication_stats: {len(rows)}")
    counts = {"updated": 0, "skipped": 0}
    try:
        for row in rows:
            counts[process_row(row)] += 1
    except KeyboardInterrupt:
        print("\n\n[interrupted]")
    finally:
        print("\n=== summary ===")
        for k, v in counts.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
