#!/usr/bin/env python
"""Daily Ntfy push summarizing Launch Agent health.

Run by ~/Library/LaunchAgents/com.davidjcox.laptop-health-summary.plist
once per day.  Sends a short summary of each davidjcox.* agent's recent
activity.  Stale agents trigger a louder alert (priority=high + tag).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent_health import check_all, format_summary  # noqa: E402
from ntfy import push  # noqa: E402


def main() -> int:
    checks = check_all()
    summary = format_summary(checks)
    stale_agents = [c for c in checks if c["stale"]]
    if stale_agents:
        title = f"Laptop health: {len(stale_agents)} agent(s) STALE"
        priority = "high"
    else:
        title = "Laptop health OK"
        priority = "low"
    push(summary, title=title, priority=priority)
    print(f"[{title}]\n{summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
