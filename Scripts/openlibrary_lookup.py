"""OpenLibrary book search by title + author.

OpenLibrary's REST API (openlibrary.org/search.json) is free, no auth.
Returns all known ISBNs for a work, which the caller is expected to
disambiguate (different editions/formats share a work but have separate
ISBNs).

Usage:
    candidates = lookup_book('Research Ethics in Behavior Analysis', 'Cox')
    # -> [{'title': '...', 'year': 2022, 'isbns': [...], 'key': '...'}]
"""

from __future__ import annotations

import requests

USER_AGENT = "quantified-self-app/1.0 (mailto:cox.david.j@gmail.com)"
BASE_URL = "https://openlibrary.org/search.json"
TIMEOUT = 15


def lookup_book(title: str, author: str | None = None,
                limit: int = 5) -> list[dict]:
    """Search OpenLibrary by title (and optional author).  Returns up to
    `limit` work records, each with all known ISBNs."""
    params = {
        "title": title,
        "fields": "title,first_publish_year,isbn,key,author_name",
        "limit": limit,
    }
    if author:
        params["author"] = author
    try:
        r = requests.get(BASE_URL, params=params,
                         headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    except requests.RequestException:
        return []
    if r.status_code != 200:
        return []
    try:
        docs = r.json().get("docs") or []
    except ValueError:
        return []

    out = []
    for d in docs:
        isbns = d.get("isbn") or []
        # Keep ISBN-13s only — ISBN-10s are the same book, just legacy format.
        isbn13s = [i for i in isbns if len(i) == 13]
        out.append({
            "title": d.get("title"),
            "year": d.get("first_publish_year"),
            "isbns": isbn13s or isbns,  # fall back to ISBN-10 if no 13
            "key": d.get("key"),
            "authors": d.get("author_name") or [],
        })
    return out


def isbn_from_chapter_doi(doi: str) -> str | None:
    """Several publishers embed the parent book's ISBN in chapter DOIs:
        Springer:    10.1007/978-3-030-96478-8_4    -> 9783030964788
        Elsevier:    10.1016/b978-0-323-99594-8.00009-x -> 9780323995948
        Routledge:   10.4324/9781003380603-7        -> 9781003380603
    Returns ISBN-13 (digits only) or None if no pattern matches.
    """
    import re
    if not doi:
        return None
    # ISBN-13 starts with 978, has 13 digits total.  Publishers embed it
    # in DOIs in different ways: hyphenated (Springer 978-3-030-96478-8),
    # half-hyphenated (Elsevier b978-0-323-99594-8), or unhyphenated
    # (Routledge 9781003380603).  Strip non-digits, then look for any
    # 13-digit run starting with 978.
    digits_only = re.sub(r"[^\d]", "", doi)
    m = re.search(r"(978\d{10})", digits_only)
    return m.group(1) if m else None


if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) < 2:
        print("usage: python openlibrary_lookup.py 'title' [author]")
        sys.exit(1)
    title = sys.argv[1]
    author = sys.argv[2] if len(sys.argv) > 2 else None
    results = lookup_book(title, author)
    print(json.dumps(results, indent=2))
