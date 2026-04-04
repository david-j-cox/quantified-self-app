#!/usr/bin/env python
"""Generate food tracking dashboard plot images and save to plots/ directory."""

import os
import base64
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def read_table(table_name):
    try:
        return pd.read_sql(f"SELECT * FROM {table_name}", engine)
    except Exception as e:
        print(f"  Warning: could not read {table_name}: {e}")
        return pd.DataFrame()


def save_plot(fig, filename):
    """Save a matplotlib figure as a base64-encoded PNG file."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    (PLOTS_DIR / filename).write_text(b64)
    return b64


def generate_all():
    """Generate all food tracking plots."""
    food_items = read_table('food_items')
    food_entries = read_table('food_entries')
    food_tags = read_table('food_tags')
    symptom_entries = read_table('symptom_entries')
    symptom_tags = read_table('symptom_tags')

    if food_entries.empty or food_items.empty:
        print("  No food data yet, generating placeholder plots")
        _generate_placeholder()
        return

    # Merge entries with items for nutrient data
    merged = food_entries.merge(food_items, left_on='food_item_id', right_on='id',
                                 suffixes=('', '_item'))
    merged['consumed_at'] = pd.to_datetime(merged['consumed_at'])
    merged['date'] = merged['consumed_at'].dt.date

    # Scale nutrients by quantity
    nutrient_cols = ['calories', 'total_fat_g', 'saturated_fat_g', 'protein_g',
                     'carbohydrates_g', 'fiber_g', 'sugar_g', 'sodium_mg',
                     'iron_mg', 'calcium_mg', 'magnesium_mg', 'potassium_mg',
                     'vitamin_b12_mcg', 'vitamin_d_mcg']
    for col in nutrient_cols:
        if col in merged.columns:
            merged[f'{col}_scaled'] = pd.to_numeric(merged[col], errors='coerce') * \
                                      pd.to_numeric(merged['quantity'], errors='coerce').fillna(1)

    # Daily totals
    daily = merged.groupby('date').agg({
        'calories_scaled': 'sum',
        'total_fat_g_scaled': 'sum',
        'saturated_fat_g_scaled': 'sum',
        'protein_g_scaled': 'sum',
        'carbohydrates_g_scaled': 'sum',
        'fiber_g_scaled': 'sum',
        'sugar_g_scaled': 'sum',
        'sodium_mg_scaled': 'sum',
        'iron_mg_scaled': 'sum',
        'calcium_mg_scaled': 'sum',
        'magnesium_mg_scaled': 'sum',
        'potassium_mg_scaled': 'sum',
    }).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date')

    # 1. Daily calorie intake with rolling average
    _plot_daily_calories(daily)

    # 2. Daily macros stacked bar
    _plot_daily_macros(daily)

    # 3. Fat intake trend (recovery-focused)
    _plot_fat_trend(daily)

    # 4. Micronutrient trends
    _plot_micronutrients(daily)

    # 5. Symptom frequency
    _plot_symptom_frequency(symptom_tags)

    # 6. Symptom severity timeline
    _plot_symptom_timeline(symptom_entries, symptom_tags)

    print("  All food tracking plots generated")


def _generate_placeholder():
    """Generate a placeholder image when no data exists."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.text(0.5, 0.5, 'No food tracking data yet.\nStart logging at food-tracker.fly.dev',
            ha='center', va='center', fontsize=16, color='#888',
            transform=ax.transAxes)
    ax.set_axis_off()
    save_plot(fig, 'food_placeholder.b64')


def _plot_daily_calories(daily):
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.bar(daily['date'], daily['calories_scaled'], color='#93c5fd', alpha=0.7, label='Daily')

    if len(daily) >= 7:
        rolling = daily.set_index('date')['calories_scaled'].rolling(7, min_periods=1).mean()
        ax.plot(rolling.index, rolling.values, color='#2563eb', linewidth=3, label='7-day avg')

    ax.set_xlabel('Date', fontsize=18, labelpad=10)
    ax.set_ylabel('Calories (kcal)', fontsize=18, labelpad=10)
    ax.set_title('Daily Calorie Intake', fontsize=22, fontweight='bold', pad=15)
    ax.tick_params(labelsize=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.legend(fontsize=14)
    sns.despine(top=True, right=True)
    fig.autofmt_xdate()
    save_plot(fig, 'food_daily_calories.b64')


def _plot_daily_macros(daily):
    fig, ax = plt.subplots(figsize=(18, 8))
    dates = daily['date']
    protein = daily['protein_g_scaled'].fillna(0)
    carbs = daily['carbohydrates_g_scaled'].fillna(0)
    fat = daily['total_fat_g_scaled'].fillna(0)

    ax.bar(dates, protein, color='#2563eb', label='Protein', alpha=0.85)
    ax.bar(dates, carbs, bottom=protein, color='#16a34a', label='Carbs', alpha=0.85)
    ax.bar(dates, fat, bottom=protein + carbs, color='#dc2626', label='Fat', alpha=0.85)

    ax.set_xlabel('Date', fontsize=18, labelpad=10)
    ax.set_ylabel('Grams', fontsize=18, labelpad=10)
    ax.set_title('Daily Macronutrient Breakdown', fontsize=22, fontweight='bold', pad=15)
    ax.tick_params(labelsize=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.legend(fontsize=14)
    sns.despine(top=True, right=True)
    fig.autofmt_xdate()
    save_plot(fig, 'food_daily_macros.b64')


def _plot_fat_trend(daily):
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.fill_between(daily['date'], daily['total_fat_g_scaled'].fillna(0),
                     alpha=0.3, color='#dc2626', label='Total Fat')
    ax.plot(daily['date'], daily['total_fat_g_scaled'].fillna(0),
            color='#dc2626', linewidth=2)
    ax.fill_between(daily['date'], daily['saturated_fat_g_scaled'].fillna(0),
                     alpha=0.3, color='#f97316', label='Saturated Fat')
    ax.plot(daily['date'], daily['saturated_fat_g_scaled'].fillna(0),
            color='#f97316', linewidth=2)

    if len(daily) >= 7:
        rolling_fat = daily.set_index('date')['total_fat_g_scaled'].rolling(7, min_periods=1).mean()
        ax.plot(rolling_fat.index, rolling_fat.values, color='#991b1b',
                linewidth=3, linestyle='--', label='Total Fat 7-day avg')

    ax.set_xlabel('Date', fontsize=18, labelpad=10)
    ax.set_ylabel('Fat (g)', fontsize=18, labelpad=10)
    ax.set_title('Fat Intake Trend (Recovery Focus)', fontsize=22, fontweight='bold', pad=15)
    ax.tick_params(labelsize=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.legend(fontsize=14)
    sns.despine(top=True, right=True)
    fig.autofmt_xdate()
    save_plot(fig, 'food_fat_trend.b64')


def _plot_micronutrients(daily):
    micros = {
        'iron_mg_scaled': ('Iron', 'mg', 18),
        'calcium_mg_scaled': ('Calcium', 'mg', 1000),
        'magnesium_mg_scaled': ('Magnesium', 'mg', 400),
        'potassium_mg_scaled': ('Potassium', 'mg', 2600),
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    for ax, (col, (name, unit, rda)) in zip(axes.flat, micros.items()):
        vals = daily[col].fillna(0)
        ax.bar(daily['date'], vals, color='#6366f1', alpha=0.7)
        ax.axhline(y=rda, color='#dc2626', linestyle='--', linewidth=2, alpha=0.7, label=f'RDA ({rda}{unit})')
        ax.set_title(name, fontsize=16, fontweight='bold')
        ax.set_ylabel(unit, fontsize=12)
        ax.tick_params(labelsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.legend(fontsize=10)
        sns.despine(ax=ax, top=True, right=True)

    fig.suptitle('Micronutrient Intake vs. Recommended Daily Allowance', fontsize=22, fontweight='bold', y=1.02)
    fig.autofmt_xdate()
    plt.tight_layout()
    save_plot(fig, 'food_micronutrients.b64')


def _plot_symptom_frequency(symptom_tags):
    if symptom_tags.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No symptoms logged yet', ha='center', va='center',
                fontsize=16, color='#888', transform=ax.transAxes)
        ax.set_axis_off()
        save_plot(fig, 'food_symptom_frequency.b64')
        return

    counts = symptom_tags.groupby('tag').size().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(4, len(counts) * 0.5)))
    colors = ['#16a34a' if t in ('felt_great', 'high_energy', 'good_digestion')
              else '#dc2626' for t in counts.index]
    ax.barh(counts.index.str.replace('_', ' ').str.title(), counts.values, color=colors, alpha=0.85)
    ax.set_xlabel('Count', fontsize=16, labelpad=10)
    ax.set_title('Symptom Frequency', fontsize=22, fontweight='bold', pad=15)
    ax.tick_params(labelsize=13)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    save_plot(fig, 'food_symptom_frequency.b64')


def _plot_symptom_timeline(symptom_entries, symptom_tags):
    if symptom_entries.empty or symptom_tags.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No symptoms logged yet', ha='center', va='center',
                fontsize=16, color='#888', transform=ax.transAxes)
        ax.set_axis_off()
        save_plot(fig, 'food_symptom_timeline.b64')
        return

    merged = symptom_tags.merge(symptom_entries, left_on='symptom_entry_id', right_on='id',
                                 suffixes=('', '_entry'))
    merged['logged_at'] = pd.to_datetime(merged['logged_at'])

    # Filter to negative symptoms only
    positive = {'felt_great', 'high_energy', 'good_digestion'}
    neg = merged[~merged['tag'].isin(positive)]

    if neg.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No negative symptoms logged — great!', ha='center', va='center',
                fontsize=16, color='#16a34a', transform=ax.transAxes)
        ax.set_axis_off()
        save_plot(fig, 'food_symptom_timeline.b64')
        return

    fig, ax = plt.subplots(figsize=(18, 8))
    tags = neg['tag'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(tags)))
    for tag, color in zip(tags, colors):
        subset = neg[neg['tag'] == tag]
        ax.scatter(subset['logged_at'], subset['severity'], s=100, alpha=0.7,
                   label=tag.replace('_', ' ').title(), color=color, edgecolors='white')

    ax.set_xlabel('Date', fontsize=18, labelpad=10)
    ax.set_ylabel('Severity (1-5)', fontsize=18, labelpad=10)
    ax.set_title('Symptom Severity Over Time', fontsize=22, fontweight='bold', pad=15)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0.5, 5.5)
    ax.tick_params(labelsize=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.legend(fontsize=12, bbox_to_anchor=(1.02, 1), loc='upper left')
    sns.despine(top=True, right=True)
    fig.autofmt_xdate()
    plt.tight_layout()
    save_plot(fig, 'food_symptom_timeline.b64')


if __name__ == "__main__":
    print("Generating food tracking plots...")
    generate_all()
