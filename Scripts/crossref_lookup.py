"""Crossref metadata lookup by DOI.

Crossref's REST API at api.crossref.org is free, no auth required.  Their
docs ask for a User-Agent string with contact info so they can reach you if
your traffic looks abusive.

Usage:
    info = lookup_doi('10.1002/jaba.381')
    # -> {'title': 'Application of the matching law...',
    #     'journal': 'Journal of Applied Behavior Analysis',
    #     'year': 2017,
    #     'authors': 'Cox, Sosine, Dallery'}
"""

from __future__ import annotations

import time

import requests

USER_AGENT = "quantified-self-app/1.0 (mailto:cox.david.j@gmail.com)"
BASE_URL = "https://api.crossref.org/works/"
TIMEOUT = 15


def _author_string(authors: list[dict]) -> str:
    """Join authors into 'Cox, Sosine, Dallery' style.  Crossref returns one
    dict per author with 'given' + 'family' (sometimes only 'name')."""
    out = []
    for a in authors:
        family = a.get("family") or a.get("name") or ""
        if family:
            out.append(family)
    return ", ".join(out)


def _year_from(message: dict) -> int | None:
    """Crossref publication year hides in several places.  Try in order."""
    for key in ("published-print", "published-online", "issued", "created"):
        d = message.get(key, {})
        parts = d.get("date-parts") or []
        if parts and parts[0]:
            return int(parts[0][0])
    return None


def lookup_doi(doi: str) -> dict | None:
    """Return canonical metadata for a DOI, or None on miss/error."""
    url = BASE_URL + doi
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        msg = r.json().get("message") or {}
    except ValueError:
        return None

    # Title is a list of strings; subtitle is a separate list.  Concatenate
    # cleanly so 'Foo: Bar' style citations come out right.
    titles = msg.get("title") or []
    subtitles = msg.get("subtitle") or []
    title = (titles[0] if titles else "").strip()
    if subtitles and subtitles[0].strip():
        sep = ": " if not title.endswith((":", "?", "!")) else " "
        title = f"{title}{sep}{subtitles[0].strip()}"

    # Container ('container-title') is journal/book name.
    containers = msg.get("container-title") or []
    journal = (containers[0] if containers else "").strip()

    return {
        "title": title or None,
        "journal": journal or None,
        "year": _year_from(msg),
        "authors": _author_string(msg.get("author") or []) or None,
        "type": msg.get("type"),  # 'journal-article', 'book-chapter', 'book', etc.
        "publisher": (msg.get("publisher") or "").strip() or None,
    }


def lookup_doi_with_retry(doi: str, retries: int = 2, backoff: float = 1.5) -> dict | None:
    """Crossref occasionally rate-limits or has transient 5xx responses.
    Retry once with exponential backoff before giving up."""
    for attempt in range(retries + 1):
        result = lookup_doi(doi)
        if result is not None:
            return result
        if attempt < retries:
            time.sleep(backoff * (2 ** attempt))
    return None


if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) < 2:
        print("usage: python crossref_lookup.py <doi> [<doi> ...]")
        sys.exit(1)
    for d in sys.argv[1:]:
        info = lookup_doi_with_retry(d)
        print(f"{d}: {json.dumps(info, indent=2) if info else '<not found>'}")
