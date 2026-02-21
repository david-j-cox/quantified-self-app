#!/usr/bin/env python
"""MLB Game Entry Tool — fetch yesterday's games, present a form, insert selections into DB."""

import os
import threading
from datetime import date, timedelta

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

PORT = 8052
YESTERDAY = date.today() - timedelta(days=1)

# ---------------------------------------------------------------------------
# Team short-name lookup
# ---------------------------------------------------------------------------
TEAM_SHORT = {
    "Boston Red Sox": "Red Sox",
    "Chicago White Sox": "White Sox",
    "Toronto Blue Jays": "Blue Jays",
    "Tampa Bay Rays": "Rays",
    "New York Yankees": "Yankees",
    "New York Mets": "Mets",
    "San Francisco Giants": "Giants",
    "San Diego Padres": "Padres",
    "Los Angeles Dodgers": "Dodgers",
    "Los Angeles Angels": "Angels",
    "St. Louis Cardinals": "Cardinals",
    "Kansas City Royals": "Royals",
    "Minnesota Twins": "Twins",
    "Detroit Tigers": "Tigers",
    "Cleveland Guardians": "Guardians",
    "Cincinnati Reds": "Reds",
    "Milwaukee Brewers": "Brewers",
    "Chicago Cubs": "Cubs",
    "Pittsburgh Pirates": "Pirates",
    "Baltimore Orioles": "Orioles",
    "Washington Nationals": "Nationals",
    "Miami Marlins": "Marlins",
    "Philadelphia Phillies": "Phillies",
    "Atlanta Braves": "Braves",
    "Houston Astros": "Astros",
    "Texas Rangers": "Rangers",
    "Seattle Mariners": "Mariners",
    "Oakland Athletics": "Athletics",
    "Colorado Rockies": "Rockies",
    "Arizona Diamondbacks": "D-backs",
}


def short_name(full_name: str) -> str:
    return TEAM_SHORT.get(full_name, full_name.split()[-1])


# ---------------------------------------------------------------------------
# Fetch games from MLB Stats API
# ---------------------------------------------------------------------------
def fetch_games(game_date: date) -> list[dict]:
    url = f"https://statsapi.mlb.com/api/v1/schedule?date={game_date}&sportId=1"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            away_full = g["teams"]["away"]["team"]["name"]
            home_full = g["teams"]["home"]["team"]["name"]
            away = short_name(away_full)
            home = short_name(home_full)
            status = g["status"]["detailedState"]

            # Count games between same teams for doubleheader labeling
            game_num = g.get("gameNumber", 1)

            if status == "Postponed":
                box = f"{away} vs {home} (Postponed)"
            else:
                away_score = g["teams"]["away"].get("score", 0)
                home_score = g["teams"]["home"].get("score", 0)
                box = f"{away} {away_score}, {home} {home_score}"

            if game_num > 1:
                box += f" (Game {game_num})"
            # Tag first game of doubleheader only when there IS a second game
            elif any(
                gg.get("gameNumber", 1) > 1
                for gg in date_entry.get("games", [])
                if gg["teams"]["away"]["team"]["name"] == away_full
                and gg["teams"]["home"]["team"]["name"] == home_full
                and gg["gamePk"] != g["gamePk"]
            ):
                box += " (Game 1)"

            games.append({
                "box_score": box,
                "away_full": away_full,
                "home_full": home_full,
            })
    return games


# ---------------------------------------------------------------------------
# Auto-categorize
# ---------------------------------------------------------------------------
TRACKED = {
    "Pittsburgh Pirates": "pirates",
    "Cleveland Guardians": "guardians",
    "Atlanta Braves": "braves",
}


def categorize(away_full: str, home_full: str) -> dict:
    flags = {"pirates": 0, "guardians": 0, "braves": 0, "other": 0}
    matched = False
    for full_name, col in TRACKED.items():
        if full_name in (away_full, home_full):
            flags[col] = 1
            matched = True
    if not matched:
        flags["other"] = 1
    return flags


# ---------------------------------------------------------------------------
# Check for existing entries
# ---------------------------------------------------------------------------
def existing_entries(game_date: date) -> int:
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM baseball_watched_long WHERE date = :d"),
            {"d": game_date},
        )
        return result.scalar()


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
STYLE = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 600px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
  .card { background: #fff; border-radius: 8px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,.12); }
  h2 { margin-top: 0; }
  .game-row { padding: 8px 0; display: flex; align-items: center; gap: 12px; }
  .game-row label { flex: 1; }
  .ip-label { color: #666; font-size: 0.9em; white-space: nowrap; }
  .innings { width: 36px; text-align: center; border: 1px solid #d1d5db; border-radius: 4px;
             padding: 2px 4px; font-size: 0.9em; }
  .innings-label { color: #666; font-size: 0.9em; white-space: nowrap; }
  .btn { padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer;
         font-size: 1em; margin-right: 10px; }
  .btn-primary { background: #2563eb; color: #fff; }
  .btn-secondary { background: #e5e7eb; color: #333; }
  .btn:hover { opacity: 0.9; }
  .warning { background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px;
             padding: 10px 14px; margin-bottom: 16px; color: #92400e; }
  .msg { text-align: center; color: #666; padding: 20px 0; }
  table.recent { width: 100%; border-collapse: collapse; font-size: 0.85em; margin-top: 16px; }
  table.recent th, table.recent td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #e5e7eb; }
  table.recent th { color: #666; font-weight: 600; }
  .section-label { color: #999; font-size: 0.85em; margin-top: 20px; margin-bottom: 4px; }
</style>
"""


def recent_entries_html() -> str:
    df = pd.read_sql(
        "SELECT date, box_score, pirates, guardians, braves, other, in_person "
        "FROM baseball_watched_long ORDER BY date DESC, id DESC LIMIT 10",
        engine,
    )
    if df.empty:
        return ""
    rows = ""
    for _, r in df.iterrows():
        rows += (
            f"<tr><td>{r['date']}</td><td>{r['box_score']}</td>"
            f"<td>{r['pirates']}</td><td>{r['guardians']}</td>"
            f"<td>{r['braves']}</td><td>{r['other']}</td>"
            f"<td>{r['in_person']}</td></tr>"
        )
    return (
        '<p class="section-label">Recent entries</p>'
        '<table class="recent"><thead><tr>'
        "<th>Date</th><th>Box Score</th><th>PIT</th><th>CLE</th><th>ATL</th><th>Other</th><th>IP</th>"
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
    games = fetch_games(YESTERDAY)
    date_str = YESTERDAY.strftime("%A, %B %-d, %Y")

    if not games:
        body = f"""
        <div class="card">
          <h2>MLB Games &mdash; {date_str}</h2>
          <p class="msg">No games were played on this date.</p>
          <form method="post" action="/no-games">
            <button class="btn btn-secondary" type="submit">Close</button>
          </form>
        </div>"""
        return page("MLB Games", body)

    existing = existing_entries(YESTERDAY)
    warning = ""
    if existing:
        warning = f'<div class="warning">&#9888; {existing} entry(ies) already recorded for this date.</div>'

    rows = ""
    for i, g in enumerate(games):
        rows += f"""
        <div class="game-row">
          <input type="checkbox" name="game" value="{i}" id="g{i}">
          <label for="g{i}">{g['box_score']}</label>
          <input type="number" name="inn_{i}" class="innings" min="1" max="9" value="9">
          <span class="innings-label">/ 9</span>
          <input type="checkbox" name="ip_{i}" id="ip{i}">
          <label for="ip{i}" class="ip-label">In-Person</label>
        </div>"""

    body = f"""
    <div class="card">
      <h2>MLB Games &mdash; {date_str}</h2>
      {warning}
      <form method="post" action="/submit">
        {rows}
        <hr style="margin:16px 0">
        <button class="btn btn-primary" type="submit">Submit</button>
        <button class="btn btn-secondary" type="submit" formaction="/no-games">No Games Watched</button>
      </form>
    </div>"""

    # Stash game data for the submit handler
    app.config["GAMES"] = games
    return page("MLB Games", body)


@app.route("/submit", methods=["POST"])
def submit():
    games = app.config.get("GAMES", [])
    selected = request.form.getlist("game")

    if not selected:
        return page("MLB Games", '<div class="card"><p class="msg">No games selected. No entries written.</p></div>')

    rows = []
    for idx_str in selected:
        i = int(idx_str)
        g = games[i]
        cat = categorize(g["away_full"], g["home_full"])
        in_person = 1 if request.form.get(f"ip_{i}") else 0
        innings = int(request.form.get(f"inn_{i}", 9))
        proportion = round(innings / 9, 2)
        rows.append({
            "date": YESTERDAY,
            "box_score": g["box_score"],
            "pirates": cat["pirates"] * proportion,
            "guardians": cat["guardians"] * proportion,
            "braves": cat["braves"] * proportion,
            "other": cat["other"] * proportion,
            "in_person": in_person,
        })

    pd.DataFrame(rows).to_sql("baseball_watched_long", engine, if_exists="append", index=False)

    count = len(rows)
    recent = recent_entries_html()
    body = f'<div class="card"><p class="msg">&#10003; {count} game(s) saved. You can close this tab.</p>{recent}</div>'
    _schedule_exit()
    return page("MLB Games", body)


@app.route("/no-games", methods=["POST"])
def no_games():
    recent = recent_entries_html()
    body = f'<div class="card"><p class="msg">No entries recorded. You can close this tab.</p>{recent}</div>'
    _schedule_exit()
    return page("MLB Games", body)


def _schedule_exit():
    def _exit():
        import time
        time.sleep(1.5)
        os._exit(0)
    threading.Thread(target=_exit, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import webbrowser
    from threading import Timer

    Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    app.run(port=PORT, debug=False)
