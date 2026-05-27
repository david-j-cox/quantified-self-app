"""Launch Agent health checks for the macOS LaunchAgents under ~/Library/LaunchAgents/com.davidjcox.*.plist.

Three things this module does:

1. `check_all()` — walks every davidjcox.* plist, reports load status and
   recency of activity (latest mtime among the plist's StandardOut/Error
   log files).  Returns a list of dicts.

2. `format_summary()` — turns that list into a short human-readable string
   suitable for a Ntfy push or terminal print.

3. `rebootstrap(label)` — bootout + bootstrap a single agent by its plist
   path.  Used by the auto-heal logic in run_all.py.

Expected cadences are baked in per agent (we know what we configured).  An
agent whose latest log mtime exceeds 2x its expected cadence is "stale".
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import time
from pathlib import Path
from typing import Iterable

LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
AGENT_PREFIX = "com.davidjcox."

# Expected interval (seconds) between successful runs, per agent label.
# Agents missing from this dict use a default of 86400s (1 day).
# A cadence of None means "run-at-load only, never check staleness".
EXPECTED_CADENCE: dict[str, int | None] = {
    "com.davidjcox.spotify-fetch": 3600,            # hourly
    "com.davidjcox.baseball-entry": 86400,          # daily morning batch
    "com.davidjcox.daily-food-update": 86400,       # daily food sync
    "com.davidjcox.laptop-health-summary": 86400,   # daily push
    "com.davidjcox.project-tracker": 86400,
    "com.davidjcox.golf-entry": 604800,             # weekly (Mondays 7 AM)
    "com.davidjcox.ensure-agents": None,            # RunAtLoad-only
}

STALE_MULTIPLIER = 3  # an agent is "stale" if last log > N × cadence


def _gui_uid() -> int:
    return os.getuid()


def _list_loaded_agents() -> set[str]:
    """Return the set of currently-loaded com.davidjcox.* labels."""
    try:
        out = subprocess.run(
            ["launchctl", "print", f"gui/{_gui_uid()}"],
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return set()
    if out.returncode != 0:
        return set()
    loaded: set[str] = set()
    for line in out.stdout.splitlines():
        # Lines look like '\t\t   31052      0 \tcom.davidjcox.foo' OR
        # 'com.davidjcox.foo => state' depending on which subsection
        # launchctl is printing.  Look for the prefix anywhere in the
        # line, then extract the token starting at the prefix.
        idx = line.find(AGENT_PREFIX)
        if idx < 0:
            continue
        token = line[idx:].split()[0].rstrip(",")
        if token.startswith(AGENT_PREFIX):
            loaded.add(token)
    return loaded


def _read_plist(path: Path) -> dict:
    try:
        return plistlib.loads(path.read_bytes()) or {}
    except Exception:
        return {}


def _latest_log_mtime(plist: dict) -> float | None:
    """Return the most-recent mtime across StandardOutPath / StandardErrorPath."""
    paths = []
    for key in ("StandardOutPath", "StandardErrorPath"):
        v = plist.get(key)
        if v:
            paths.append(Path(v))
    times = []
    for p in paths:
        try:
            times.append(p.stat().st_mtime)
        except FileNotFoundError:
            continue
    return max(times) if times else None


def check_all() -> list[dict]:
    """Return a list of {label, plist_path, loaded, last_seen_minutes,
    cadence_minutes, stale} dicts, one per discovered plist."""
    if not LAUNCH_AGENTS_DIR.exists():
        return []
    loaded = _list_loaded_agents()
    now = time.time()
    out = []
    for plist_path in sorted(LAUNCH_AGENTS_DIR.glob(f"{AGENT_PREFIX}*.plist")):
        plist = _read_plist(plist_path)
        label = plist.get("Label") or plist_path.stem
        mtime = _latest_log_mtime(plist)
        cadence = EXPECTED_CADENCE.get(label, 86400)
        last_seen_min = ((now - mtime) / 60) if mtime else None
        # Grace period for freshly-installed plists: if the plist itself was
        # modified within its cadence window, give it one cadence to fire
        # before flagging stale.
        try:
            plist_age_s = now - plist_path.stat().st_mtime
        except FileNotFoundError:
            plist_age_s = float("inf")
        in_grace = cadence is not None and plist_age_s < cadence
        if cadence is None:
            # Run-at-load only — staleness doesn't apply, but a never-loaded
            # plist (no logs ever) still counts as stale.
            is_stale = mtime is None
            cadence_min: float | None = None
        elif in_grace and mtime is None:
            is_stale = False
            cadence_min = cadence / 60
        else:
            is_stale = (
                last_seen_min is None
                or last_seen_min > (cadence / 60) * STALE_MULTIPLIER
            )
            cadence_min = cadence / 60
        out.append({
            "label": label,
            "plist_path": str(plist_path),
            "loaded": label in loaded,
            "last_seen_minutes": last_seen_min,
            "cadence_minutes": cadence_min,
            "stale": is_stale,
        })
    return out


def format_summary(checks: Iterable[dict]) -> str:
    """One-line-per-agent summary suitable for Ntfy."""
    lines = []
    for c in checks:
        if c["last_seen_minutes"] is None:
            recency = "no logs yet"
        elif c["last_seen_minutes"] < 60:
            recency = f"{c['last_seen_minutes']:.0f}m ago"
        elif c["last_seen_minutes"] < 1440:
            recency = f"{c['last_seen_minutes']/60:.1f}h ago"
        else:
            recency = f"{c['last_seen_minutes']/1440:.1f}d ago"
        loaded_mark = "[OK]  " if c["loaded"] else "[FAIL]"
        stale_mark = " [STALE]" if c["stale"] else ""
        short = c["label"].replace(AGENT_PREFIX, "")
        lines.append(f"{loaded_mark} {short}: {recency}{stale_mark}")
    return "\n".join(lines) if lines else "(no davidjcox.* agents found)"


def rebootstrap(plist_path: str) -> tuple[bool, str]:
    """bootout + bootstrap a single agent.  Returns (success, message)."""
    p = Path(plist_path)
    if not p.exists():
        return False, f"plist not found: {plist_path}"
    label = _read_plist(p).get("Label", p.stem)
    domain = f"gui/{_gui_uid()}"
    # bootout: harmless if not loaded (returns non-zero, we ignore).
    subprocess.run(
        ["launchctl", "bootout", f"{domain}/{label}"],
        capture_output=True, text=True, timeout=15,
    )
    r = subprocess.run(
        ["launchctl", "bootstrap", domain, str(p)],
        capture_output=True, text=True, timeout=15,
    )
    if r.returncode == 0:
        return True, f"rebootstrapped {label}"
    return False, f"bootstrap {label} failed: {r.stderr.strip() or r.stdout.strip()}"


def auto_heal(checks: Iterable[dict]) -> list[str]:
    """For every stale agent, attempt a rebootstrap.  Returns a list of
    human-readable messages describing what was done."""
    msgs = []
    for c in checks:
        if c["stale"]:
            ok, msg = rebootstrap(c["plist_path"])
            msgs.append(("[OK] " if ok else "[FAIL] ") + msg)
    return msgs


if __name__ == "__main__":
    import sys
    checks = check_all()
    print(format_summary(checks))
    if "--heal" in sys.argv:
        print()
        for m in auto_heal(checks):
            print(m)
