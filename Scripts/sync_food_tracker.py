#!/usr/bin/env python
"""Sync food tracker data from Fly.io into local PostgreSQL."""

import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FOOD_TRACKER_URL = os.getenv("FOOD_TRACKER_URL", "").rstrip("/")
FOOD_TRACKER_API_KEY = os.getenv("FOOD_TRACKER_API_KEY", "")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# Tables to sync and their primary key columns
TABLES = {
    "food_items": "id",
    "food_entries": "id",
    "food_tags": "id",
    "symptom_entries": "id",
    "symptom_tags": "id",
}


# ---------------------------------------------------------------------------
# Ensure local tables exist
# ---------------------------------------------------------------------------
CREATE_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS food_items (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        brand TEXT,
        calories NUMERIC, total_fat_g NUMERIC, saturated_fat_g NUMERIC,
        protein_g NUMERIC, carbohydrates_g NUMERIC, fiber_g NUMERIC,
        sugar_g NUMERIC, sodium_mg NUMERIC, iron_mg NUMERIC,
        calcium_mg NUMERIC, magnesium_mg NUMERIC, potassium_mg NUMERIC,
        vitamin_b12_mcg NUMERIC, vitamin_d_mcg NUMERIC,
        serving_size TEXT,
        source TEXT NOT NULL DEFAULT 'manual',
        created_at TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS food_entries (
        id INTEGER PRIMARY KEY,
        food_item_id INTEGER REFERENCES food_items(id),
        meal_slot TEXT NOT NULL,
        quantity NUMERIC NOT NULL DEFAULT 1.0,
        logged_at TIMESTAMP,
        consumed_at TIMESTAMP NOT NULL,
        notes TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS food_tags (
        id INTEGER PRIMARY KEY,
        food_item_id INTEGER REFERENCES food_items(id),
        tag TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS symptom_entries (
        id INTEGER PRIMARY KEY,
        logged_at TIMESTAMP,
        notes TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS symptom_tags (
        id INTEGER PRIMARY KEY,
        symptom_entry_id INTEGER REFERENCES symptom_entries(id),
        tag TEXT NOT NULL,
        severity INTEGER NOT NULL
    )""",
]


def ensure_tables():
    """Create local mirror tables if they don't exist."""
    with engine.connect() as conn:
        for stmt in CREATE_STATEMENTS:
            conn.execute(text(stmt))
        conn.commit()


# ---------------------------------------------------------------------------
# Determine sync window
# ---------------------------------------------------------------------------
def get_since_date() -> str:
    """Find the most recent data we have locally, or default to 30 days ago."""
    with engine.connect() as conn:
        dates = []
        for table, ts_col in [("food_items", "created_at"), ("food_entries", "logged_at"),
                               ("symptom_entries", "logged_at")]:
            try:
                result = conn.execute(text(f"SELECT MAX({ts_col}) FROM {table}"))
                val = result.scalar()
                if val:
                    dates.append(val)
            except Exception:
                pass

    if dates:
        latest = max(dates)
        # Go back 1 day from latest to catch any stragglers
        since = latest - timedelta(days=1)
        return since.strftime("%Y-%m-%d")

    return (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------
def sync():
    """Pull data from the food tracker API and insert into local DB."""
    if not FOOD_TRACKER_URL or not FOOD_TRACKER_API_KEY:
        print("FOOD_TRACKER_URL or FOOD_TRACKER_API_KEY not set, skipping sync")
        return

    ensure_tables()
    since = get_since_date()
    print(f"  Syncing entries since {since}...")

    resp = requests.get(
        f"{FOOD_TRACKER_URL}/api/entries",
        params={"since": since},
        headers={"X-API-Key": FOOD_TRACKER_API_KEY},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    total_new = 0
    # Sync in order: food_items first (referenced by food_entries and food_tags),
    # then symptom_entries (referenced by symptom_tags), then the rest
    for table_name in ["food_items", "food_entries", "food_tags",
                       "symptom_entries", "symptom_tags"]:
        rows = data.get(table_name, [])
        if not rows:
            continue

        df = pd.DataFrame(rows)
        pk = TABLES[table_name]

        # Get existing IDs to skip duplicates
        with engine.connect() as conn:
            try:
                result = conn.execute(text(f"SELECT {pk} FROM {table_name}"))
                existing_ids = {row[0] for row in result}
            except Exception:
                existing_ids = set()

        new_df = df[~df[pk].isin(existing_ids)]
        if new_df.empty:
            print(f"  {table_name}: {len(rows)} rows fetched, all already synced")
            continue

        new_df.to_sql(table_name, engine, if_exists="append", index=False)
        print(f"  {table_name}: {len(new_df)} new row(s) synced")
        total_new += len(new_df)

    if total_new == 0:
        print("  All data already synced")
    else:
        print(f"  Total: {total_new} new row(s) synced")


if __name__ == "__main__":
    sync()
