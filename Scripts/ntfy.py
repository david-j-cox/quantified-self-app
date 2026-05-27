"""Minimal Ntfy push helper.  Reads NTFY_TOPIC from .env / env."""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

load_dotenv()

NTFY_URL = "https://ntfy.sh"
TOPIC = os.getenv("NTFY_TOPIC")


def push(message: str, title: str | None = None,
         priority: str | None = None, tags: list[str] | None = None) -> bool:
    """Send a push.  Returns True on HTTP 2xx, False otherwise (silent fail
    is fine — we don't want a missed notification to crash run_all)."""
    if not TOPIC:
        return False
    headers = {}
    if title:
        headers["Title"] = title
    if priority:
        headers["Priority"] = priority
    if tags:
        headers["Tags"] = ",".join(tags)
    try:
        r = requests.post(
            f"{NTFY_URL}/{TOPIC}",
            data=message.encode("utf-8"),
            headers=headers,
            timeout=10,
        )
        return r.status_code // 100 == 2
    except requests.RequestException:
        return False


if __name__ == "__main__":
    import sys
    msg = " ".join(sys.argv[1:]) or "test from quantified-self-app"
    ok = push(msg, title="agent_health test")
    print(f"sent: {ok}")
