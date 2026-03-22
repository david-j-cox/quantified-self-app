#!/usr/bin/env python
"""Golf Entry Tool — PGA Tour watching form + GHIN score sync."""

import os
import sys
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, request
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

PORT = 8053

GHIN_EMAIL = os.getenv("GHIN_EMAIL")
GHIN_PASSWORD = os.getenv("GHIN_PASSWORD")
GHIN_NUMBER = os.getenv("GHIN_NUMBER")

# Sentinel file to prevent duplicate runs on the same day
SENTINEL = Path(__file__).resolve().parent.parent / "logs" / ".golf_entry_last_run"


def already_ran_today() -> bool:
    """Return True if the script already ran successfully today."""
    if SENTINEL.exists():
        return SENTINEL.read_text().strip() == str(date.today())
    return False


def mark_ran_today():
    """Write today's date to the sentinel file."""
    SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    SENTINEL.write_text(str(date.today()))


# ---------------------------------------------------------------------------
# ESPN PGA Tour API
# ---------------------------------------------------------------------------
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard?dates={year}"


def fetch_tournaments(year: int) -> list[dict]:
    """Fetch all PGA Tour events for a given year from ESPN."""
    resp = requests.get(ESPN_URL.format(year=year), timeout=30)
    resp.raise_for_status()
    data = resp.json()

    tournaments = []
    for event in data.get("events", []):
        start = datetime.fromisoformat(event["date"].replace("Z", "+00:00")).date()
        end = datetime.fromisoformat(event["endDate"].replace("Z", "+00:00")).date()
        status_name = event.get("status", {}).get("type", {}).get("name", "")
        tournaments.append({
            "id": event["id"],
            "name": event["name"],
            "start": start,
            "end": end,
            "status": status_name,
        })
    return tournaments


def find_last_weekend_tournaments() -> list[dict]:
    """Find tournaments that overlapped with the previous Thu-Sun window."""
    today = date.today()
    # Previous Thursday through Sunday
    days_since_monday = today.weekday()  # Monday=0
    last_thursday = today - timedelta(days=days_since_monday + 4)
    last_sunday = today - timedelta(days=days_since_monday + 1)

    year = last_thursday.year
    tournaments = fetch_tournaments(year)

    matches = []
    for t in tournaments:
        if t["status"] == "STATUS_CANCELED":
            continue
        # Check if tournament dates overlap with Thu-Sun window
        if t["start"] <= last_sunday and t["end"] >= last_thursday:
            matches.append(t)
    return matches


# ---------------------------------------------------------------------------
# GHIN API (unofficial — reverse-engineered from mobile app)
# ---------------------------------------------------------------------------
FIREBASE_URL = "https://firebaseinstallations.googleapis.com/v1/projects/ghin-mobile-app/installations"
FIREBASE_API_KEY = "AIzaSyBxgTOAWxiud0HuaE5tN-5NTlzFnrtyz-I"
GHIN_LOGIN_URL = "https://api2.ghin.com/api/v1/golfer_login.json"
GHIN_SCORES_URL = "https://api2.ghin.com/api/v1/scores.json"


def _get_firebase_token() -> str:
    """Get a Firebase Installation token for GHIN auth."""
    resp = requests.post(
        FIREBASE_URL,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": FIREBASE_API_KEY,
        },
        json={
            "appId": "1:884417644529:web:47fb315bc6c70242f72650",
            "authVersion": "FIS_v2",
            "sdkVersion": "w:0.5.7",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["authToken"]["token"]


def _get_ghin_token(firebase_token: str) -> str:
    """Login to GHIN and return the JWT."""
    resp = requests.post(
        GHIN_LOGIN_URL,
        headers={"Content-Type": "application/json"},
        json={
            "token": firebase_token,
            "user": {
                "email_or_ghin": GHIN_EMAIL,
                "password": GHIN_PASSWORD,
                "remember_me": True,
            },
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["golfer_user"]["golfer_user_token"]


def _fetch_ghin_scores(ghin_token: str) -> list[dict]:
    """Fetch recent scores from GHIN."""
    resp = requests.get(
        GHIN_SCORES_URL,
        headers={
            "Authorization": f"Bearer {ghin_token}",
            "Content-Type": "application/json",
        },
        params={
            "golfer_id": GHIN_NUMBER,
            "limit": 50,
            "page": 1,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("scores", [])


def pull_ghin_scores() -> str:
    """Pull scores from GHIN and insert new ones into the DB. Returns status message."""
    if not all([GHIN_EMAIL, GHIN_PASSWORD, GHIN_NUMBER]):
        return "GHIN credentials not configured"

    firebase_token = _get_firebase_token()
    ghin_token = _get_ghin_token(firebase_token)
    scores = _fetch_ghin_scores(ghin_token)

    if not scores:
        return "No scores found in GHIN"

    # Get existing GHIN score IDs to avoid duplicates
    with engine.connect() as conn:
        try:
            result = conn.execute(text("SELECT ghin_score_id FROM golf_scores"))
            existing_ids = {str(row[0]) for row in result}
        except Exception:
            # Table may not exist yet
            existing_ids = set()

    new_rows = []
    for s in scores:
        score_id = str(s["id"])
        if score_id in existing_ids:
            continue
        new_rows.append({
            "played_at": s["played_at"],
            "course_name": s.get("course_name", ""),
            "course_rating": s.get("course_rating"),
            "slope_rating": s.get("slope_rating"),
            "adjusted_gross_score": s.get("adjusted_gross_score"),
            "differential": s.get("differential"),
            "number_of_holes": s.get("number_of_holes", 18),
            "ghin_score_id": score_id,
        })

    if not new_rows:
        return f"GHIN: {len(scores)} scores found, all already synced"

    pd.DataFrame(new_rows).to_sql("golf_scores", engine, if_exists="append", index=False)
    return f"GHIN: {len(new_rows)} new score(s) synced"


# ---------------------------------------------------------------------------
# Check for existing entries
# ---------------------------------------------------------------------------
def existing_golf_entries(espn_event_id: str) -> int:
    with engine.connect() as conn:
        try:
            result = conn.execute(
                text("SELECT COUNT(*) FROM golf_watched WHERE espn_event_id = :eid"),
                {"eid": espn_event_id},
            )
            return result.scalar()
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
STYLE = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 600px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
  .card { background: #fff; border-radius: 8px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,.12); }
  h2 { margin-top: 0; }
  .tournament-row { padding: 12px 0; }
  .tournament-name { font-size: 1.1em; font-weight: 600; margin-bottom: 4px; }
  .tournament-dates { color: #666; font-size: 0.9em; margin-bottom: 12px; }
  .rounds { display: flex; gap: 16px; margin-bottom: 8px; }
  .round-item { display: flex; align-items: center; gap: 4px; }
  .round-label { font-size: 0.9em; color: #444; }
  .ip-label { color: #666; font-size: 0.9em; white-space: nowrap; }
  .btn { padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer;
         font-size: 1em; margin-right: 10px; }
  .btn-primary { background: #16a34a; color: #fff; }
  .btn-secondary { background: #e5e7eb; color: #333; }
  .btn:hover { opacity: 0.9; }
  .warning { background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px;
             padding: 10px 14px; margin-bottom: 16px; color: #92400e; }
  .info { background: #eff6ff; border: 1px solid #3b82f6; border-radius: 6px;
          padding: 10px 14px; margin-bottom: 16px; color: #1e40af; font-size: 0.9em; }
  .msg { text-align: center; color: #666; padding: 20px 0; }
  table.recent { width: 100%; border-collapse: collapse; font-size: 0.85em; margin-top: 16px; }
  table.recent th, table.recent td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #e5e7eb; }
  table.recent th { color: #666; font-weight: 600; }
  .section-label { color: #999; font-size: 0.85em; margin-top: 20px; margin-bottom: 4px; }
</style>
"""


def recent_entries_html() -> str:
    try:
        df = pd.read_sql(
            "SELECT date, tournament, r1, r2, r3, r4, in_person "
            "FROM golf_watched ORDER BY date DESC LIMIT 10",
            engine,
        )
    except Exception:
        return ""
    if df.empty:
        return ""
    rows = ""
    for _, r in df.iterrows():
        rows += (
            f"<tr><td>{r['date']}</td><td>{r['tournament']}</td>"
            f"<td>{r['r1']}</td><td>{r['r2']}</td>"
            f"<td>{r['r3']}</td><td>{r['r4']}</td>"
            f"<td>{r['in_person']}</td></tr>"
        )
    return (
        '<p class="section-label">Recent entries</p>'
        '<table class="recent"><thead><tr>'
        "<th>Date</th><th>Tournament</th><th>R1</th><th>R2</th><th>R3</th><th>R4</th><th>IP</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def page(title: str, body: str) -> str:
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title>{STYLE}</head><body>{body}</body></html>"


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def index():
    tournaments = find_last_weekend_tournaments()
    ghin_status = app.config.get("GHIN_STATUS", "")

    ghin_info = ""
    if ghin_status:
        ghin_info = f'<div class="info">{ghin_status}</div>'

    if not tournaments:
        body = f"""
        <div class="card">
          <h2>PGA Tour — Last Weekend</h2>
          {ghin_info}
          <p class="msg">No PGA Tour tournament was played last weekend.</p>
          <form method="post" action="/no-tournament">
            <button class="btn btn-secondary" type="submit">Close</button>
          </form>
        </div>"""
        return page("Golf Entry", body)

    tournament_rows = ""
    for i, t in enumerate(tournaments):
        existing = existing_golf_entries(t["id"])
        warning = ""
        if existing:
            warning = f'<div class="warning">&#9888; Entry already recorded for this tournament.</div>'

        date_range = f"{t['start'].strftime('%b %-d')} – {t['end'].strftime('%b %-d, %Y')}"
        tournament_rows += f"""
        <div class="tournament-row">
          {warning}
          <div class="tournament-name">{t['name']}</div>
          <div class="tournament-dates">{date_range}</div>
          <input type="hidden" name="event_id_{i}" value="{t['id']}">
          <input type="hidden" name="event_name_{i}" value="{t['name']}">
          <input type="hidden" name="event_start_{i}" value="{t['start']}">
          <div class="rounds">
            <div class="round-item">
              <input type="checkbox" name="r1_{i}" id="r1_{i}">
              <label for="r1_{i}" class="round-label">R1 (Thu)</label>
            </div>
            <div class="round-item">
              <input type="checkbox" name="r2_{i}" id="r2_{i}">
              <label for="r2_{i}" class="round-label">R2 (Fri)</label>
            </div>
            <div class="round-item">
              <input type="checkbox" name="r3_{i}" id="r3_{i}">
              <label for="r3_{i}" class="round-label">R3 (Sat)</label>
            </div>
            <div class="round-item">
              <input type="checkbox" name="r4_{i}" id="r4_{i}">
              <label for="r4_{i}" class="round-label">R4 (Sun)</label>
            </div>
          </div>
          <div class="round-item" style="margin-top: 4px;">
            <input type="checkbox" name="ip_{i}" id="ip_{i}">
            <label for="ip_{i}" class="ip-label">Attended In-Person</label>
          </div>
        </div>"""

    recent = recent_entries_html()
    body = f"""
    <div class="card">
      <h2>PGA Tour — Last Weekend</h2>
      {ghin_info}
      <form method="post" action="/submit">
        <input type="hidden" name="count" value="{len(tournaments)}">
        {tournament_rows}
        <hr style="margin:16px 0">
        <button class="btn btn-primary" type="submit">Submit</button>
        <button class="btn btn-secondary" type="submit" formaction="/no-tournament">Didn't Watch</button>
      </form>
      {recent}
    </div>"""

    return page("Golf Entry", body)


@app.route("/submit", methods=["POST"])
def submit():
    count = int(request.form.get("count", 0))

    rows = []
    for i in range(count):
        event_id = request.form.get(f"event_id_{i}")
        event_name = request.form.get(f"event_name_{i}")
        event_start = request.form.get(f"event_start_{i}")
        r1 = 1 if request.form.get(f"r1_{i}") else 0
        r2 = 1 if request.form.get(f"r2_{i}") else 0
        r3 = 1 if request.form.get(f"r3_{i}") else 0
        r4 = 1 if request.form.get(f"r4_{i}") else 0
        in_person = 1 if request.form.get(f"ip_{i}") else 0

        # Only save if at least one round was watched or attended in person
        if r1 or r2 or r3 or r4 or in_person:
            rows.append({
                "date": event_start,
                "tournament": event_name,
                "espn_event_id": event_id,
                "r1": r1,
                "r2": r2,
                "r3": r3,
                "r4": r4,
                "in_person": in_person,
            })

    if not rows:
        recent = recent_entries_html()
        body = f'<div class="card"><p class="msg">No rounds selected. No entries written.</p>{recent}</div>'
        mark_ran_today()
        _schedule_exit()
        return page("Golf Entry", body)

    pd.DataFrame(rows).to_sql("golf_watched", engine, if_exists="append", index=False)

    saved = len(rows)
    recent = recent_entries_html()
    body = f'<div class="card"><p class="msg">&#10003; {saved} tournament(s) saved. You can close this tab.</p>{recent}</div>'
    mark_ran_today()
    _schedule_exit()
    return page("Golf Entry", body)


@app.route("/no-tournament", methods=["POST"])
def no_tournament():
    recent = recent_entries_html()
    body = f'<div class="card"><p class="msg">No entries recorded. You can close this tab.</p>{recent}</div>'
    mark_ran_today()
    _schedule_exit()
    return page("Golf Entry", body)


def _schedule_exit():
    def _exit():
        import time
        time.sleep(1.5)
        os._exit(0)
    threading.Thread(target=_exit, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
AUTO_EXIT_MINUTES = 120
MAX_ATTEMPTS = 2
ATTEMPT_FILE = Path(__file__).resolve().parent.parent / "logs" / ".golf_entry_attempt"


def get_attempt() -> int:
    """Return the current attempt number for today (0 if none yet)."""
    if ATTEMPT_FILE.exists():
        parts = ATTEMPT_FILE.read_text().strip().split(",")
        if len(parts) == 2 and parts[0] == str(date.today()):
            return int(parts[1])
    return 0


def set_attempt(n: int):
    ATTEMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ATTEMPT_FILE.write_text(f"{date.today()},{n}")


if __name__ == "__main__":
    if already_ran_today():
        sys.exit(0)

    # Only run on Mondays unless --force is passed
    if "--force" not in sys.argv and date.today().weekday() != 0:
        sys.exit(0)

    import signal
    import socket
    import subprocess
    import time
    from threading import Timer

    attempt = get_attempt() + 1
    if attempt > MAX_ATTEMPTS:
        sys.exit(0)
    set_attempt(attempt)

    # Kill any leftover Flask process from a previous run still holding the port
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{PORT}"], capture_output=True, text=True
    )
    for pid in result.stdout.strip().split("\n"):
        if pid and int(pid) != os.getpid():
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    # Wait for port to be released
    for _ in range(10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", PORT)) != 0:
                break
        time.sleep(0.5)

    # Pull GHIN scores silently before opening the form
    try:
        ghin_msg = pull_ghin_scores()
        print(f"GHIN sync: {ghin_msg}")
    except Exception as e:
        ghin_msg = f"GHIN sync failed: {e}"
        print(ghin_msg, file=sys.stderr)
    app.config["GHIN_STATUS"] = ghin_msg

    # Auto-exit after 2 hours; re-launch once if this is the first attempt
    def _auto_exit():
        time.sleep(AUTO_EXIT_MINUTES * 60)
        if attempt < MAX_ATTEMPTS:
            subprocess.Popen(
                [sys.executable, __file__],
                cwd=str(Path(__file__).resolve().parent.parent),
            )
        os._exit(0)
    threading.Thread(target=_auto_exit, daemon=True).start()

    # Wait for Flask to be ready before opening browser
    def _open_browser():
        for _ in range(20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", PORT)) == 0:
                    subprocess.run(["/usr/bin/open", f"http://127.0.0.1:{PORT}"])
                    return
            time.sleep(0.25)
    Timer(0.5, _open_browser).start()

    app.run(port=PORT, debug=False)
