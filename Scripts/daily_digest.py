#!/usr/bin/env python
"""Daily change-digest: how did yesterday compare to the trailing four weeks?

This is the deterministic stats core. It assembles a unified daily metric
matrix from every available source, then computes each metric's deviation
from its own trailing-28-day baseline for a given as-of date (default:
yesterday). The output is a ranked, structured list of changes that a
narration layer turns into a morning push -- the stats live here, never in
the LLM.

Sources (per the v1 "everything" scope):
  - df_numeric_for_clustering.csv : time-allocation levers + Whoop outcomes
  - Postgres food_entries/items   : daily nutrient totals (levers)
  - Postgres strava_activities    : running pace, min/mile (outcome)
  - Postgres golf_scores          : 18-hole differential (outcome)

Each metric is tagged lever vs. outcome and given a "good direction" so the
narrator can say better/worse, not just up/down. Levers are things you choose
today (food, hours); outcomes are results you cannot set directly (HRV, pace).
"""

import os
import shutil
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
CSV_PATH = DATA_DIR / "df_numeric_for_clustering.csv"

# Baseline window and minimum observations needed to trust a baseline.
BASELINE_DAYS = 28
MIN_OBS = 5
# A category counts as "tracked" only with at least this many nonzero days in
# the window. This is what makes untracked/sparse categories a non-issue: a
# metric a person never logs is silently excluded instead of reading as
# "dropped to 0". (Postgres sources store missing days as NaN; time-allocation
# stores them as 0 -- the nonzero gate handles both uniformly.)
MIN_ACTIVE = 3

# --------------------------------------------------------------------------
# Metric registry
# --------------------------------------------------------------------------
# kind:      'lever'  -> behavior you choose today (eligible for prospective focus)
#            'outcome'-> result you cannot set directly
# direction: +1 higher is better, -1 lower is better, 0 contextual/neutral
# Each entry: key -> (label, domain, kind, direction, unit)
METRICS = {
    # --- Food levers (Postgres daily nutrient totals) ---
    "calories":      ("Calories", "food", "lever", 0, "kcal"),
    "protein_g":     ("Protein", "food", "lever", +1, "g"),
    "carbs_g":       ("Carbs", "food", "lever", 0, "g"),
    "fat_g":         ("Total fat", "food", "lever", 0, "g"),
    "sat_fat_g":     ("Saturated fat", "food", "lever", -1, "g"),
    "fiber_g":       ("Fiber", "food", "lever", +1, "g"),
    "sugar_g":       ("Sugar", "food", "lever", -1, "g"),
    "sodium_mg":     ("Sodium", "food", "lever", -1, "mg"),
    "iron_mg":       ("Iron", "food", "lever", +1, "mg"),
    "calcium_mg":    ("Calcium", "food", "lever", +1, "mg"),
    "magnesium_mg":  ("Magnesium", "food", "lever", +1, "mg"),
    "potassium_mg":  ("Potassium", "food", "lever", +1, "mg"),
    # --- Time-allocation levers (CSV, minutes/day) ---
    "coding":          ("Coding", "time", "lever", +1, "min"),
    "writing":         ("Writing", "time", "lever", +1, "min"),
    "reading_books":   ("Reading books", "time", "lever", +1, "min"),
    "exercise":        ("Physical exercise", "time", "lever", +1, "min"),
    "family":          ("Family", "time", "lever", +1, "min"),
    "learning":        ("Learning", "time", "lever", +1, "min"),
    "research":        ("Research projects", "time", "lever", +1, "min"),
    "journal_articles":("Journal articles", "time", "lever", +1, "min"),
    "teaching":        ("Teaching", "time", "lever", +1, "min"),
    "hobbies":         ("Hobbies", "time", "lever", +1, "min"),
    # --- Whoop outcomes (CSV) ---
    "recovery":          ("Recovery", "whoop", "outcome", +1, "%"),
    "hrv":               ("HRV (RMSSD)", "whoop", "outcome", +1, "ms"),
    "resting_hr":        ("Resting HR", "whoop", "outcome", -1, "bpm"),
    "strain":            ("Strain", "whoop", "outcome", 0, ""),
    "sleep_hours":       ("Sleep duration", "whoop", "outcome", +1, "h"),
    "sleep_performance": ("Sleep performance", "whoop", "outcome", +1, "%"),
    "sleep_consistency": ("Sleep consistency", "whoop", "outcome", +1, "%"),
    "sleep_efficiency":  ("Sleep efficiency", "whoop", "outcome", +1, "%"),
    "respiratory_rate":  ("Respiratory rate", "whoop", "outcome", 0, "rpm"),
    # --- Activity / golf outcomes (Postgres) ---
    "run_pace":          ("Running pace", "running", "outcome", -1, "min/mi"),
    "golf_differential": ("Golf differential", "golf", "outcome", -1, ""),
}

# Mapping of CSV source columns -> metric keys (direct copies).
_CSV_COLUMN_MAP = {
    "Coding": "coding",
    "Writing": "writing",
    "Reading Books": "reading_books",
    "Physical Exercise": "exercise",
    "Family": "family",
    "Learning": "learning",
    "Research Projects": "research",
    "Journal Articles": "journal_articles",
    "Teaching": "teaching",
    "Hobbies": "hobbies",
    "score_recovery_score": "recovery",
    "score_hrv_rmssd_milli": "hrv",
    "score_resting_heart_rate": "resting_hr",
    "score_strain": "strain",
    "score_sleep_performance_percentage": "sleep_performance",
    "score_sleep_consistency_percentage": "sleep_consistency",
    "score_sleep_efficiency_percentage": "sleep_efficiency",
    "score_respiratory_rate": "respiratory_rate",
}


def _make_engine():
    """Build a Postgres engine from .env, or return None if unconfigured."""
    user, pw = os.getenv("DB_USER"), os.getenv("DB_PASSWORD")
    host, port, name = os.getenv("DB_HOST"), os.getenv("DB_PORT"), os.getenv("DB_NAME")
    if not all([user, pw, host, port, name]):
        return None
    return create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{name}")


# --------------------------------------------------------------------------
# Per-source daily frames (each indexed by a normalized date)
# --------------------------------------------------------------------------
def _csv_daily(csv_path=CSV_PATH):
    """Time-allocation levers + Whoop outcomes from the clustering CSV."""
    if not Path(csv_path).exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path, index_col=0)
    if "date_column" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date_column"]).dt.normalize()
    out = pd.DataFrame(index=df.index)
    for src, key in _CSV_COLUMN_MAP.items():
        if src in df.columns:
            out[key] = pd.to_numeric(df[src], errors="coerce")
    # Total sleep hours = light + slow-wave + REM (stored in milliseconds).
    sleep_parts = [
        "score_stage_summary_total_light_sleep_time_milli",
        "score_stage_summary_total_slow_wave_sleep_time_milli",
        "score_stage_summary_total_rem_sleep_time_milli",
    ]
    if all(c in df.columns for c in sleep_parts):
        total_milli = sum(pd.to_numeric(df[c], errors="coerce") for c in sleep_parts)
        out["sleep_hours"] = total_milli / 3.6e6
    out["date"] = df["date"].values
    return out.groupby("date").mean(numeric_only=True)


def _food_daily(engine):
    """Daily nutrient totals, scaled by serving quantity (mirrors the food plots)."""
    if engine is None:
        return pd.DataFrame()
    try:
        entries = pd.read_sql("SELECT * FROM food_entries", engine)
        items = pd.read_sql("SELECT * FROM food_items", engine)
    except Exception as e:
        print(f"  Warning: could not read food tables: {e}")
        return pd.DataFrame()
    if entries.empty or items.empty:
        return pd.DataFrame()

    merged = entries.merge(items, left_on="food_item_id", right_on="id",
                           suffixes=("", "_item"))
    merged["date"] = pd.to_datetime(merged["consumed_at"]).dt.normalize()
    qty = pd.to_numeric(merged["quantity"], errors="coerce").fillna(1)

    # Source nutrient column -> metric key.
    nutrient_map = {
        "calories": "calories", "protein_g": "protein_g",
        "carbohydrates_g": "carbs_g", "total_fat_g": "fat_g",
        "saturated_fat_g": "sat_fat_g", "fiber_g": "fiber_g",
        "sugar_g": "sugar_g", "sodium_mg": "sodium_mg", "iron_mg": "iron_mg",
        "calcium_mg": "calcium_mg", "magnesium_mg": "magnesium_mg",
        "potassium_mg": "potassium_mg",
    }
    out = pd.DataFrame({"date": merged["date"]})
    for src, key in nutrient_map.items():
        if src in merged.columns:
            out[key] = pd.to_numeric(merged[src], errors="coerce") * qty
    return out.groupby("date").sum(numeric_only=True)


def _running_daily(engine):
    """Mean running pace (min/mile) per day, plausible runs only."""
    if engine is None:
        return pd.DataFrame()
    try:
        act = pd.read_sql(
            "SELECT activitydate, activitytype, elapsedtime, distance "
            "FROM strava_activities", engine)
    except Exception as e:
        print(f"  Warning: could not read strava_activities: {e}")
        return pd.DataFrame()
    runs = act[act["activitytype"] == "Run"].copy()
    if runs.empty:
        return pd.DataFrame()
    runs["date"] = pd.to_datetime(runs["activitydate"]).dt.normalize()
    dist = pd.to_numeric(runs["distance"], errors="coerce")
    runs["run_pace"] = (pd.to_numeric(runs["elapsedtime"], errors="coerce") / 60) / dist
    runs = runs[(runs["run_pace"] >= 3.5) & (runs["run_pace"] <= 12)]
    return runs.groupby("date")[["run_pace"]].mean()


def _golf_daily(engine):
    """Mean 18-hole score differential per day."""
    if engine is None:
        return pd.DataFrame()
    try:
        golf = pd.read_sql(
            "SELECT played_at, differential, number_of_holes, adjusted_gross_score "
            "FROM golf_scores", engine)
    except Exception as e:
        print(f"  Warning: could not read golf_scores: {e}")
        return pd.DataFrame()
    golf["differential"] = pd.to_numeric(golf["differential"], errors="coerce")
    golf = golf[(golf["number_of_holes"] == 18)
                & golf["differential"].notna()
                & (pd.to_numeric(golf["adjusted_gross_score"], errors="coerce") > 50)]
    if golf.empty:
        return pd.DataFrame()
    golf["date"] = pd.to_datetime(golf["played_at"]).dt.normalize()
    return golf.groupby("date")[["differential"]].mean().rename(
        columns={"differential": "golf_differential"})


def build_daily_matrix(engine="auto", csv_path=CSV_PATH):
    """Assemble one daily-indexed frame with a column per metric key.

    engine='auto' builds a Postgres engine from .env; pass None to skip the
    Postgres sources (CSV-only), or pass an existing SQLAlchemy engine.
    """
    if engine == "auto":
        engine = _make_engine()

    frames = [
        _csv_daily(csv_path),
        _food_daily(engine),
        _running_daily(engine),
        _golf_daily(engine),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    matrix = pd.concat(frames, axis=1).sort_index()
    # Keep only known metric columns, in registry order.
    cols = [k for k in METRICS if k in matrix.columns]
    matrix = matrix[cols]
    matrix.index.name = "date"
    return matrix


# --------------------------------------------------------------------------
# Delta core: yesterday vs trailing baseline
# --------------------------------------------------------------------------
def compute_deltas(matrix, as_of=None, baseline_days=BASELINE_DAYS, min_obs=MIN_OBS,
                   min_active=MIN_ACTIVE):
    """Rank each metric's as-of value against its own trailing baseline.

    Returns a list of dicts (most deviant first). The baseline is the
    `baseline_days` calendar days immediately *before* as_of; a metric is
    only scored when as_of has a value and the baseline has >= min_obs
    observations. z is the standardized deviation; better/worse is read off
    the metric's good direction.
    """
    if matrix.empty:
        return []

    if as_of is None:
        as_of = date.today() - timedelta(days=1)
    as_of = pd.Timestamp(as_of).normalize()

    window_start = as_of - pd.Timedelta(days=baseline_days)
    baseline = matrix[(matrix.index >= window_start) & (matrix.index < as_of)]

    if as_of not in matrix.index:
        return []  # no data for the as-of day; nothing to compare
    today_row = matrix.loc[as_of]
    if isinstance(today_row, pd.DataFrame):  # dup dates -> average
        today_row = today_row.mean()

    results = []
    for key in matrix.columns:
        value = today_row.get(key, np.nan)
        if pd.isna(value):
            continue
        hist = baseline[key].dropna()
        if len(hist) < min_obs:
            continue
        # Skip categories that aren't really tracked, and those with no spread
        # to judge against -- both would otherwise produce noise.
        if int((hist != 0).sum()) < min_active:
            continue
        std = hist.std(ddof=0)
        if not std or np.isnan(std):
            continue

        mean = hist.mean()
        delta = value - mean
        z = delta / std
        pct = (delta / mean * 100) if mean else np.nan

        label, domain, kind, direction, unit = METRICS[key]
        if direction == 0:
            sentiment = "neutral"
        elif np.sign(delta) == 0:
            sentiment = "neutral"
        else:
            sentiment = "better" if np.sign(delta) == direction else "worse"

        results.append({
            "key": key, "label": label, "domain": domain, "kind": kind,
            "unit": unit, "direction": direction,
            "value": float(value), "baseline_mean": float(mean),
            "baseline_std": float(std), "delta": float(delta),
            "pct": float(pct) if not np.isnan(pct) else None,
            "z": float(z), "abs_z": float(abs(z)), "n_obs": int(len(hist)),
            "sentiment": sentiment,
        })

    results.sort(key=lambda r: r["abs_z"], reverse=True)
    return results


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]

# How far back the weekday analysis looks, and the minimum same-weekday
# observations needed before we trust a pattern.
FOCUS_LOOKBACK_DAYS = 12 * 7
FOCUS_MIN_OBS = 4
FOCUS_MIN_ACTIVE = 5


def weekday_focus(matrix, target_day=None, lookback_days=FOCUS_LOOKBACK_DAYS,
                  min_obs=FOCUS_MIN_OBS, min_active=FOCUS_MIN_ACTIVE):
    """What does this weekday historically slip on? -> what to focus on today.

    Only directional *levers* (things you choose today, with a clear good
    direction) are eligible. For each, we compare recent same-weekday history
    against your other recent days. A positive `shortfall` means this weekday
    tends to be worse on that lever -- you under-do a good thing or over-do a
    bad one -- which is exactly what's worth focusing on. Ranked by the
    standardized gap; most actionable first.
    """
    if matrix.empty:
        return []

    if target_day is None:
        target_day = date.today()
    target_dow = pd.Timestamp(target_day).weekday()

    cutoff = pd.Timestamp(target_day).normalize() - pd.Timedelta(days=lookback_days)
    recent = matrix[matrix.index >= cutoff]
    if recent.empty:
        return []

    results = []
    for key in matrix.columns:
        label, domain, kind, direction, unit = METRICS[key]
        if kind != "lever" or direction == 0:
            continue

        same = recent[recent.index.weekday == target_dow][key].dropna()
        other = recent[recent.index.weekday != target_dow][key].dropna()
        if len(same) < min_obs or len(other) < min_obs:
            continue
        # The metric must be genuinely tracked on other days; but the target
        # weekday is allowed to be low/zero -- that low is the insight.
        if int((other != 0).sum()) < min_active:
            continue
        other_std = other.std(ddof=0)
        if not other_std or np.isnan(other_std):
            continue

        same_mean, other_mean = same.mean(), other.mean()
        # gap: signed difference vs other days (+ = more on this weekday).
        # shortfall: same gap read through good-direction (+ = worse for you).
        gap = same_mean - other_mean
        gap_z = gap / other_std
        shortfall = -gap * direction
        z = shortfall / other_std

        results.append({
            "key": key, "label": label, "domain": domain, "unit": unit,
            "direction": direction, "weekday": WEEKDAYS[target_dow],
            "weekday_mean": float(same_mean), "other_mean": float(other_mean),
            "gap": float(gap), "gap_z": float(gap_z),
            "shortfall": float(shortfall), "z": float(z), "abs_z": float(abs(z)),
            "n_same": int(len(same)),
            "needs_focus": bool(shortfall > 0),
        })

    # Most-slipping levers first; these are the focus candidates.
    results.sort(key=lambda r: r["z"], reverse=True)
    return results


# How strong a weekday tilt must be before we call it a "more/less" pattern.
PATTERN_Z = 0.5


def weekday_patterns(matrix, target_day=None):
    """Split the weekday signal into an actionable nudge and a neutral pattern.

    - nutrition_focus: food levers this weekday tends to fall short on (low on a
      good nutrient, or high on a bad one). These are worth acting on today.
    - does_more / does_less: how this weekday's *activities* (time use) tilt vs
      your other days, stated neutrally so you can choose to replicate or avoid.
      Deliberately not prescriptive -- a low-coding Sunday is a fact, not a
      to-do.
    """
    rows = weekday_focus(matrix, target_day=target_day)
    weekday = rows[0]["weekday"] if rows else WEEKDAYS[
        pd.Timestamp(target_day or date.today()).weekday()]

    nutrition_focus = sorted(
        [r for r in rows if r["domain"] == "food" and r["needs_focus"]],
        key=lambda r: r["z"], reverse=True)
    does_more = sorted(
        [r for r in rows if r["domain"] == "time" and r["gap_z"] >= PATTERN_Z],
        key=lambda r: r["gap_z"], reverse=True)
    does_less = sorted(
        [r for r in rows if r["domain"] == "time" and r["gap_z"] <= -PATTERN_Z],
        key=lambda r: r["gap_z"])
    return {
        "weekday": weekday,
        "nutrition_focus": nutrition_focus,
        "does_more": does_more,
        "does_less": does_less,
    }


def _fmt(v, unit):
    s = f"{v:,.1f}" if abs(v) < 1000 else f"{v:,.0f}"
    return f"{s}{unit}" if unit and unit != "%" else (f"{s}%" if unit == "%" else s)


# --------------------------------------------------------------------------
# Narration: structured stats -> short morning digest
# --------------------------------------------------------------------------
# Only surface retrospective movers at least this many SDs from baseline.
NOTABLE_Z = 1.0
CLAUDE_TIMEOUT_S = 90

NARRATION_SYSTEM = (
    "You write a short daily digest for the person whose own quantified-self "
    "data this is. Use only the items and numbers provided; never invent any. "
    "\n\n"
    "Structure, two short paragraphs:\n"
    "1. What changed yesterday versus their typical last four weeks.\n"
    "2. Today: first, the nutrition worth hitting today (the listed nutrition "
    "focus items). Then, neutrally, how this weekday usually tends to go for "
    "their activities (what they do more and less of), stated as an observation "
    "they can act on or ignore. Do not tell them to work, block time, or be "
    "productive; just report the pattern.\n\n"
    "Voice: plain, everyday words and short sentences, like a knowledgeable "
    "friend texting you. Second person. Direct and honest, not a cheerleader. "
    "Hard rules: no em dashes, no semicolons, no emojis, no markdown, no bullet "
    "lists, no preamble. Avoid dramatic phrasing (no 'swung hard', 'ran hot', "
    "'on the upside'). Just say what happened. Under 120 words total."
)


def summarize_day(matrix, as_of=None, today=None):
    """Bundle the retrospective deltas + prospective focus into one payload."""
    if as_of is None:
        as_of = date.today() - timedelta(days=1)
    if today is None:
        today = date.today()

    deltas = compute_deltas(matrix, as_of=as_of)
    notable = [d for d in deltas if d["abs_z"] >= NOTABLE_Z]
    patterns = weekday_patterns(matrix, target_day=today)

    return {
        "as_of": str(as_of),
        "today_weekday": WEEKDAYS[pd.Timestamp(today).weekday()],
        "movers": notable,
        "nutrition_focus": patterns["nutrition_focus"],
        "does_more": patterns["does_more"],
        "does_less": patterns["does_less"],
    }


def _payload_to_prompt(payload):
    """Render the structured payload into a compact prompt for Claude."""
    wd = payload["today_weekday"]
    lines = [f"Yesterday: {payload['as_of']}",
             f"Today is: {wd}", "",
             "What changed yesterday vs the trailing 4 weeks "
             "(value | typical | direction):"]
    if payload["movers"]:
        for d in payload["movers"]:
            arrow = "up" if d["delta"] > 0 else "down"
            lines.append(
                f"- {d['label']}: {_fmt(d['value'], d['unit'])} "
                f"(typical {_fmt(d['baseline_mean'], d['unit'])}), "
                f"{arrow}, {d['sentiment']}")
    else:
        lines.append("- nothing notably different from typical")

    lines += ["", f"Nutrition worth hitting today ({wd}s tend to fall short here):"]
    if payload["nutrition_focus"]:
        for f in payload["nutrition_focus"][:5]:
            short = "tends low" if f["direction"] > 0 else "tends high"
            lines.append(
                f"- {f['label']}: {short} ({_fmt(f['weekday_mean'], f['unit'])} "
                f"vs {_fmt(f['other_mean'], f['unit'])} on other days)")
    else:
        lines.append("- nothing specific")

    lines += ["", f"How your {wd}s usually tilt for activities (neutral, just a "
              "pattern they can replicate or avoid):"]
    if payload["does_more"]:
        for f in payload["does_more"][:3]:
            lines.append(
                f"- more {f['label']}: {_fmt(f['weekday_mean'], f['unit'])} "
                f"vs {_fmt(f['other_mean'], f['unit'])} other days")
    if payload["does_less"]:
        for f in payload["does_less"][:3]:
            lines.append(
                f"- less {f['label']}: {_fmt(f['weekday_mean'], f['unit'])} "
                f"vs {_fmt(f['other_mean'], f['unit'])} other days")
    if not payload["does_more"] and not payload["does_less"]:
        lines.append("- no clear activity tilt")
    return "\n".join(lines)


def _find_claude():
    """Locate the claude binary without relying on PATH (LaunchAgents are bare)."""
    found = shutil.which("claude")
    if found:
        return found
    for cand in ("~/.local/bin/claude", "/usr/local/bin/claude",
                 "/opt/homebrew/bin/claude", "~/.claude/local/claude"):
        p = os.path.expanduser(cand)
        if os.path.exists(p):
            return p
    return None


def _claude_narrate(prompt, timeout=CLAUDE_TIMEOUT_S):
    """Narrate via the Claude Code CLI using subscription OAuth. None on failure.

    Uses plain `-p` (not `--bare`, which ignores CLAUDE_CODE_OAUTH_TOKEN) and
    disables tools so it's a pure text completion. Runs from the home dir so it
    doesn't load this repo's project MCP servers / skills. The OAuth token is
    inherited from the environment (load_dotenv puts CLAUDE_CODE_OAUTH_TOKEN
    from .env into os.environ), so no secret needs to live in the plist.
    """
    claude_bin = _find_claude()
    if not claude_bin:
        return None
    try:
        proc = subprocess.run(
            [claude_bin, "-p", "--output-format", "text",
             "--append-system-prompt", NARRATION_SYSTEM, "--allowedTools", ""],
            input=prompt, capture_output=True, text=True, timeout=timeout,
            cwd=os.path.expanduser("~"),
        )
    except Exception as e:
        print(f"  Claude narration failed ({e}); using template.")
        return None
    if proc.returncode != 0:
        print(f"  Claude narration exited {proc.returncode}; using template.")
        return None
    return proc.stdout.strip() or None


def _template_narrative(payload):
    """Deterministic fallback prose when Claude is unavailable.

    Kept in the same plain register as the Claude prompt: no em dashes, no
    semicolons, short sentences.
    """
    wd = payload["today_weekday"]
    movers = payload["movers"]

    if movers:
        bits = []
        for d in movers[:4]:
            arrow = "up" if d["delta"] > 0 else "down"
            tag = "" if d["sentiment"] == "neutral" else f" ({d['sentiment']})"
            bits.append(f"{d['label'].lower()} was {arrow} at "
                        f"{_fmt(d['value'], d['unit'])} from a typical "
                        f"{_fmt(d['baseline_mean'], d['unit'])}{tag}")
        para1 = "Yesterday vs your last four weeks: " + ", ".join(bits) + "."
    else:
        para1 = "Yesterday looked close to your usual four-week pattern."

    sentences = []
    focus = payload["nutrition_focus"]
    if focus:
        items = []
        for f in focus[:3]:
            verb = "more" if f["direction"] > 0 else "less"
            items.append(f"{verb} {f['label'].lower()}")
        sentences.append(f"On nutrition, {wd}s are a day to aim for "
                         + ", ".join(items) + ".")

    more = [f["label"].lower() for f in payload["does_more"][:3]]
    less = [f["label"].lower() for f in payload["does_less"][:3]]
    if more:
        sentences.append(f"You usually do more {', '.join(more)} on {wd}s.")
    if less:
        sentences.append(f"You usually do less {', '.join(less)}.")
    if not sentences:
        sentences.append(f"No clear {wd} pattern stands out.")
    para2 = " ".join(sentences)
    return f"{para1}\n\n{para2}"


def narrate(matrix, as_of=None, today=None, use_claude=True):
    """Produce the morning digest text (Claude when available, else template)."""
    payload = summarize_day(matrix, as_of=as_of, today=today)
    if use_claude:
        text = _claude_narrate(_payload_to_prompt(payload))
        if text:
            return text, payload, "claude"
    return _template_narrative(payload), payload, "template"


def main():
    matrix = build_daily_matrix()
    if matrix.empty:
        print("No data available to build the daily matrix.")
        return

    as_of = date.today() - timedelta(days=1)
    deltas = compute_deltas(matrix, as_of=as_of)

    print(f"\nDaily change-digest for {as_of} "
          f"(vs trailing {BASELINE_DAYS} days)")
    print(f"Matrix: {matrix.shape[0]} days x {matrix.shape[1]} metrics, "
          f"through {matrix.index.max().date()}")
    if not deltas:
        print("  No metrics had data for the as-of day with a usable baseline.")
        return

    print(f"\n{'metric':<22}{'kind':<9}{'value':>12}{'baseline':>12}"
          f"{'z':>7}  sentiment")
    print("-" * 78)
    for d in deltas:
        print(f"{d['label']:<22}{d['kind']:<9}"
              f"{_fmt(d['value'], d['unit']):>12}"
              f"{_fmt(d['baseline_mean'], d['unit']):>12}"
              f"{d['z']:>+7.1f}  {d['sentiment']}")

    # Prospective: today's weekday signal, split into nutrition focus + pattern.
    today = date.today()
    wd = WEEKDAYS[today.weekday()]
    pat = weekday_patterns(matrix, target_day=today)

    def _row(tag, f):
        print(f"{tag + ' ' + f['label']:<26}"
              f"{_fmt(f['weekday_mean'], f['unit']):>12}"
              f"{_fmt(f['other_mean'], f['unit']):>12}"
              f"{f.get('z', f.get('gap_z')):>+8.1f}")

    print(f"\n{wd}: nutrition worth hitting today:")
    if pat["nutrition_focus"]:
        for f in pat["nutrition_focus"][:5]:
            _row("low on" if f["direction"] > 0 else "high on", f)
    else:
        print("  nothing specific")

    print(f"\n{wd}: how your activities usually tilt (neutral):")
    if pat["does_more"] or pat["does_less"]:
        for f in pat["does_more"][:3]:
            _row("more", f)
        for f in pat["does_less"][:3]:
            _row("less", f)
    else:
        print("  no clear tilt")

    # Narrative digest (Claude via subscription OAuth, template fallback)
    use_claude = "--no-claude" not in sys.argv
    text, _, source = narrate(matrix, as_of=as_of, today=today, use_claude=use_claude)
    print(f"\n=== Morning digest ({source}) ===\n")
    print(text)

    if "--push" in sys.argv:
        import ntfy
        ok = ntfy.push(text, title=f"Daily digest, {wd}")
        print(f"\nPush sent: {ok}")


if __name__ == "__main__":
    main()
