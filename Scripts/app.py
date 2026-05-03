#!/usr/bin/env python
# coding: utf-8

# Data manipulation
from dotenv import load_dotenv
from sqlalchemy import create_engine
import os
import warnings
from IPython.display import display, HTML
import pandas as pd
import numpy as np

# System
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime, date
import random

# Data viz
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from matplotlib.ticker import FuncFormatter, ScalarFormatter, FixedFormatter

# Dashboard
from dash import Dash, dcc, html
from dash import dash_table
import webbrowser
from threading import Timer
from dash.dependencies import Input, Output

# Machine Learning
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import umap
from sklearn.decomposition import PCA
import phate
from kneed import KneeLocator
from itertools import combinations
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation, RPComputation
from scipy.stats import spearmanr

# Preferences
pd.options.display.max_columns = None
display(HTML("<style>.container { width:100% !important; }</style>"))
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# PostgreSQL database connection
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
engine = create_engine(DATABASE_URL)

# Function to read data from a table
def read_table(table_name):
    print(f"Reading table {table_name}...")
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        print(f"Successfully read {len(df)} rows from {table_name}")
        return df
    except Exception as e:
        print(f"Error reading table {table_name}: {str(e)}")
        return pd.DataFrame()


# Read in the raw data from the database
print("\n=== Debug: Loading Data ===")
print("Loading raw_data...")
all_df = read_table('raw_data')
print("Loading books_read...")
books = read_table('books_read')
print("Loading baseball_watched_long...")
baseball = read_table('baseball_watched_long')
print("Loading cv_additions...")
cv_adds = read_table('cv_additions')
print("Loading publication_stats...")
pubs = read_table('publication_stats')
print("Loading strava_activities...")
phys_act = read_table('strava_activities')
print("Loading golf_watched...")
golf_watched = read_table('golf_watched')
print("Loading golf_scores...")
golf_scores = read_table('golf_scores')
print("Loading modeling_ready_data...")
model_df = read_table('modeling_ready_data')
print("Loading whoop_recoveries...")
whoop_recoveries = read_table('whoop_recoveries')
whoop_recoveries['created_at'] = pd.to_datetime(whoop_recoveries['created_at'], errors='coerce', utc=True)
whoop_recoveries = whoop_recoveries.sort_values('created_at').reset_index(drop=True)

try:
    print("Loading Whoop sleep data...")
    whoop_sleep = read_table('whoop_sleep')
    whoop_sleep['sleep_start'] = pd.to_datetime(whoop_sleep['sleep_start'], utc=True)
    whoop_sleep = whoop_sleep.sort_values('sleep_start').reset_index(drop=True)
    
    # Preprocess WHOOP sleep data
    time_columns = [
        'score_stage_summary_total_in_bed_time_milli',
        'score_stage_summary_total_awake_time_milli',
        'score_stage_summary_total_light_sleep_time_milli',
        'score_stage_summary_total_slow_wave_sleep_time_milli',
        'score_stage_summary_total_rem_sleep_time_milli'
    ]
    
    # Convert milliseconds to hours
    for col in time_columns:
        whoop_sleep[col] = whoop_sleep[col] / (1000 * 60 * 60)
    
    # Calculate total time in bed
    whoop_sleep['total_in_bed'] = whoop_sleep['score_stage_summary_total_in_bed_time_milli']
    
    # Calculate proportions for each stage
    stage_columns = [
        'score_stage_summary_total_awake_time_milli',
        'score_stage_summary_total_light_sleep_time_milli',
        'score_stage_summary_total_slow_wave_sleep_time_milli',
        'score_stage_summary_total_rem_sleep_time_milli'
    ]
    
    for col in stage_columns:
        whoop_sleep[f'{col}_proportion'] = whoop_sleep[col] / whoop_sleep['total_in_bed']
    
    # Create consistent color mapping
    sleep_stages = ['Awake Time', 'Light Sleep Time', 'Slow Wave Sleep Time', 'Rem Sleep Time']
    sleep_colors = sns.color_palette('bright', n_colors=len(sleep_stages))
    sleep_color_dict = dict(zip(sleep_stages, sleep_colors))

except Exception as e:
    print("Error loading data:", str(e))
    import traceback
    traceback.print_exc()

# Load whoop_cycle_collection data
try:
    print("Loading whoop_cycle_collection...")
    whoop_cycle = read_table('whoop_cycle_collection')
    whoop_cycle['cycle_start'] = pd.to_datetime(whoop_cycle['cycle_start'], utc=True)
    whoop_cycle = whoop_cycle.sort_values('cycle_start').reset_index(drop=True)
except Exception as e:
    print("Error loading whoop_cycle_collection:", str(e))
    whoop_cycle = pd.DataFrame()

# Plotting functions for whoop_cycle_collection
def plot_whoop_cycle_strain():
    if whoop_cycle.empty:
        return ''
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=whoop_cycle, x='cycle_start', y='score_strain', color='black', marker='o', ax=ax)
    # Trendline
    import matplotlib.dates as mdates
    x = whoop_cycle['cycle_start'].map(mdates.date2num)
    y = whoop_cycle['score_strain']
    if len(whoop_cycle) > 1:
        z = np.polyfit(x, y, 4)
        p = np.poly1d(z)
        ax.plot(whoop_cycle['cycle_start'], p(x), linestyle='--', linewidth=2, alpha=0.7, color='black')
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Strain', fontsize=18)
    ax.tick_params(axis='x', labelsize=14, rotation=45)
    ax.tick_params(axis='y', labelsize=14)
    sns.despine(ax=ax)
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_base64

def plot_whoop_cycle_kilojoule():
    if whoop_cycle.empty:
        return ''
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=whoop_cycle, x='cycle_start', y='score_kilojoule', color='black', marker='o', ax=ax)
    # Trendline
    import matplotlib.dates as mdates
    x = whoop_cycle['cycle_start'].map(mdates.date2num)
    y = whoop_cycle['score_kilojoule']
    if len(whoop_cycle) > 1:
        z = np.polyfit(x, y, 4)
        p = np.poly1d(z)
        ax.plot(whoop_cycle['cycle_start'], p(x), linestyle='--', linewidth=2, alpha=0.7, color='black')
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Kilojoules', fontsize=18)
    ax.tick_params(axis='x', labelsize=14, rotation=45)
    ax.tick_params(axis='y', labelsize=14)
    sns.despine(ax=ax)
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_base64

def plot_whoop_cycle_heartrates():
    if whoop_cycle.empty:
        return ''
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=whoop_cycle, x='cycle_start', y='score_average_heart_rate', color='black', marker='o', label='Avg HR', ax=ax)
    sns.lineplot(data=whoop_cycle, x='cycle_start', y='score_max_heart_rate', color='red', marker='o', label='Max HR', ax=ax)
    # Trendlines
    import matplotlib.dates as mdates
    x = whoop_cycle['cycle_start'].map(mdates.date2num)
    y_avg = whoop_cycle['score_average_heart_rate']
    y_max = whoop_cycle['score_max_heart_rate']
    if len(whoop_cycle) > 1:
        z_avg = np.polyfit(x, y_avg, 4)
        p_avg = np.poly1d(z_avg)
        ax.plot(whoop_cycle['cycle_start'], p_avg(x), linestyle='--', linewidth=2, alpha=0.7, color='black')
        z_max = np.polyfit(x, y_max, 4)
        p_max = np.poly1d(z_max)
        ax.plot(whoop_cycle['cycle_start'], p_max(x), linestyle='--', linewidth=2, alpha=0.7, color='red')
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Heart Rate (bpm)', fontsize=18)
    ax.set_yscale('log')
    ax.tick_params(axis='y', labelsize=14)
    
    # Set y-limits and format y-axis to show standard notation
    ax.set_ylim(30, 250)
    # Manually set nice tick locations for the heart rate range and force no scientific notation
    ax.set_yticks([30, 50, 75, 100, 150, 200, 250])
    ax.set_yticklabels(['30', '50', '75', '100', '150', '200', '250'])
    # Force no minor ticks to prevent scientific notation
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.legend(fontsize=16)
    sns.despine(ax=ax)
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_base64

# OVR TIME SPENT PLOTS
def calculate_yearly_trendlines(df, column):
    trends = {}
    for year in df.Year.unique():
        yearly_data = df[df.Year == year]
        z = np.polyfit(yearly_data.DayOfYear, yearly_data[column], 1)
        p = np.poly1d(z)
        trends[year] = p
    return trends


def extract_slopes(trends):
    slopes = {year: trend[1] for year, trend in trends.items()}
    return slopes


def calculate_weekly_trendlines(all_data, column):
    trends = {}
    all_data['Week'] = all_data['date_column'].dt.to_period(
        'W').apply(lambda r: r.start_time)
    for week in all_data['Week'].unique():
        weekly_data = all_data[all_data['Week'] == week]
        if len(weekly_data) > 1:  # Ensure there's enough data to calculate a trendline
            z = np.polyfit(weekly_data['DayOfYear'], weekly_data[column], 1)
            p = np.poly1d(z)
            trends[week] = p
    return trends


def extract_weekly_slopes(trends):
    slopes = {week: trend[1] for week, trend in trends.items()}
    return slopes


def thousands_formatter(x, pos):
    return '{:,.0f}'.format(x)


# Prep the data
# Add in some info we'll use for analyses
all_data = all_df.set_index('date_column').reset_index()
all_data['date_column'] = pd.to_datetime(all_data['date_column'])
all_data['Year'] = all_data['date_column'].dt.year
all_data['Month_Num'] = all_data['date_column'].dt.month
all_data['Month'] = all_data['date_column'].dt.month_name()
all_data['Day'] = all_data['date_column'].dt.day_name()
all_data['DayOfYear'] = all_data['date_column'].dt.dayofyear
all_data = all_data.sort_values(by=['Year', 'DayOfYear'])
for col in list(all_data):
    if col in ['date_column', 'Year', 'Month_Num', 'Month', 'Day', 'DayOfYear']:
        continue
    else:
        try:
            all_data[col] = all_data[col].cumsum()
        except:
            continue

# Only include data from new year if it is at least three weeks into it
if (date.today().month == 1) & (date.today().day < 21):
    all_data = all_data[all_data['Year'] < date.today().year]

trends_all = calculate_yearly_trendlines(all_data, 'CR ALL')
trends_wo_job = calculate_yearly_trendlines(all_data, 'CR w/o Job')
trends_relationship = calculate_yearly_trendlines(all_data, 'Family')
trends_writing = calculate_yearly_trendlines(all_data, 'Writing')

slopes_all = extract_slopes(trends_all)
slopes_wo_job = extract_slopes(trends_wo_job)
slopes_relationship = extract_slopes(trends_relationship)
slopes_writing = extract_slopes(trends_writing)

# Create a DataFrame with the slopes
slopes_df = pd.DataFrame({
    'Year': slopes_all.keys(),
    'CR ALL': slopes_all.values(),
    'CR w/o Job': slopes_wo_job.values(),
    'Relationship': slopes_relationship.values(),
    'Writing': slopes_writing.values()
})
slopes_df = slopes_df.melt(id_vars=['Year'])

trends_all_weekly = calculate_weekly_trendlines(all_data, 'CR ALL')
trends_wo_job_weekly = calculate_weekly_trendlines(all_data, 'CR w/o Job')
trends_relationship_weekly = calculate_weekly_trendlines(all_data, 'Family')
trends_writing_weekly = calculate_weekly_trendlines(all_data, 'Writing')

weekly_slopes_all = extract_weekly_slopes(trends_all_weekly)
weekly_slopes_wo_job = extract_weekly_slopes(trends_wo_job_weekly)
weekly_slopes_relationship = extract_weekly_slopes(trends_relationship_weekly)
weekly_slopes_writing = extract_weekly_slopes(trends_writing_weekly)

# Create a DataFrame with the weekly slopes
weekly_slopes_all_data = pd.DataFrame({
    'Week': weekly_slopes_all.keys(),
    'CR ALL': weekly_slopes_all.values(),
    'CR w/o Job': weekly_slopes_wo_job.values(),
    'Family': weekly_slopes_relationship.values(),
    'Writing': weekly_slopes_writing.values()
})

# Calculate changes in slopes
weekly_slopes_all_data = weekly_slopes_all_data.sort_values(
    by='Week').reset_index(drop=True)

for col in ['CR ALL', 'CR w/o Job', 'Family', 'Writing']:
    weekly_slopes_all_data[
        f'{col} Change'
    ] = weekly_slopes_all_data[col].diff()

# Melting the original values
all_data_original = pd.melt(weekly_slopes_all_data, id_vars=['Week'], value_vars=['CR ALL', 'CR w/o Job', 'Family', 'Writing'],
                            var_name='Category', value_name='Value')

# Melting the change values
all_data_change = pd.melt(weekly_slopes_all_data, id_vars=['Week'], value_vars=['CR ALL Change', 'CR w/o Job Change', 'Family Change', 'Writing Change'],
                          var_name='Category Change', value_name='Change')

# Adjust the category names in the change DataFrame
all_data_change['Category'] = all_data_change['Category Change'].str.replace(
    ' Change', '')

# Merge the two DataFrames
all_data_melted = pd.merge(all_data_original, all_data_change[[
                           'Week', 'Category', 'Change']], on=['Week', 'Category'], how='left')


def create_cr_all_plot():
    # Functions
    def thousands_formatter(x, pos):
        return '{:,.0f}'.format(x)

    # Function for adding slope labels
    def annotate_slopes(trends, data, column_name, color):
        for year, trend in trends.items():
            slope = trend[1]
            last_date_of_year = data[data['Year'] == year]['date_column'].max()
            plt.annotate(f'{slope:.2f}', xy=(last_date_of_year, data.loc[data['date_column'] == last_date_of_year, column_name].values[0]),
                         xytext=(-10, 0), textcoords='offset points', ha='right', color=color)

    # Actual Plot
    fig, ax = plt.subplots(figsize=(18, 10))

    # Plot each category
    plt.plot(all_data['date_column'], all_data['CR ALL'],
             label='CR ALL', color='blue')
    plt.plot(all_data['date_column'], all_data['CR w/o Job'],
             label='CR w/o Job', color='red')
    plt.plot(all_data['date_column'], all_data['Family'],
             label='Relationship', color='orange')
    plt.plot(all_data['date_column'], all_data['Writing'],
             label='Writing', color='green')

    # Annotate the slopes using the function
    annotate_slopes(trends_all, all_data, 'CR ALL', 'blue')
    annotate_slopes(trends_wo_job, all_data, 'CR w/o Job', 'red')
    annotate_slopes(trends_relationship, all_data, 'Family', 'orange')
    annotate_slopes(trends_writing, all_data, 'Writing', 'green')

    # Draw vertical black lines at the beginning of each year
    for year in all_data['Year'].unique():
        plt.axvline(pd.Timestamp(f'{year}-01-01'),
                    color='black', linewidth=1, linestyle='--')

    plt.xlabel("Date", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Cumulative Minutes", fontsize=26, labelpad=12)
    plt.yticks(fontsize=16)

    ax.ticklabel_format(style='plain', axis='y')
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    sns.despine(top=True, right=True)
    legend = plt.legend(frameon=True, loc="upper center", fontsize=16,
                        facecolor='white', edgecolor='none', framealpha=1.0)

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


def create_avg_ovr_slopes():
    # Display the DataFrame
    fig, ax = plt.subplots(figsize=(8, 6))  # More compact figure for 45% width
    sns.lineplot(data=all_data_melted, x='Week', y='Value',
                 hue='Category')  # Corrected lineplot usage
    
    # Create twin axis for percentages
    ax2 = ax.twinx()
    
    # Set up the percentage scale
    # 100% = 960 minutes, so we need to calculate the percentage scale
    max_minutes = 1800  # Current y-axis limit
    max_percentage = (max_minutes / 960) * 100  # Calculate max percentage
    
    # Set the limits for both axes
    ax.set_ylim(0, max_minutes)
    ax2.set_ylim(0, max_percentage)
    
    # Set up percentage ticks on the right axis
    percentage_ticks = [0, 25, 50, 75, 100, 125, 150, 175]
    ax2.set_yticks(percentage_ticks)
    ax2.set_yticklabels([f'{p}%' for p in percentage_ticks])
    
    # Style the axes
    plt.xlabel("Week", fontsize=16, labelpad=8)
    plt.xticks(fontsize=12)
    ax.set_ylabel("Weekly AVG of Mins Tracked Daily", fontsize=14, labelpad=8)
    ax.yaxis.set_tick_params(labelsize=12)
    ax2.set_ylabel("Percentage of 16 Hr Day (960 min)", fontsize=14, labelpad=8)
    ax2.yaxis.set_tick_params(labelsize=12)
    
    # Add a horizontal line at 100% (960 minutes)
    ax.axhline(y=960, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    sns.despine(top=True, right=False)  # Keep right spine for percentage axis
    plt.legend(frameon=False, loc="best", fontsize=12)
    
    # Adjust layout to prevent right axis labels from being cut off
    plt.tight_layout(pad=2.0)  # More padding for compact layout

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


def create_avg_ovr_change():
    # Filter data for the last 30 days
    trim_weeks = all_data_melted[
        (all_data_melted['Week'] >= (all_data_melted['Week'].max() - pd.Timedelta(days=30))) &
        (all_data_melted['Week'] <= all_data_melted['Week'].max())
    ]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.lineplot(
        data=trim_weeks,
        x='Week',
        y='Change',
        hue='Category',
        markers=True,
        style='Category',
        palette='cividis', 
        markersize=12
    )

    # Add a zero baseline
    plt.axhline(0, color='k', linestyle='--', linewidth=1)

    # Add labels and styling
    plt.xlabel("Week", fontsize=26, labelpad=12)
    plt.xticks(fontsize=15.5, rotation=90)
    plt.ylabel("WoW Change Past Month", fontsize=16, labelpad=0)
    plt.yticks(fontsize=12)
    plt.ylim(trim_weeks['Change'].min() - 50, trim_weeks['Change'].max() + 50)
    sns.despine(top=True, right=True)
    plt.legend(frameon=False, loc="best", fontsize=16)

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


# LEISURE ACTIVITIES
def create_leisure_plot():
    # prep the data
    leisure_df = all_data[[
        'date_column',
        'Journal Articles',
        'Philosophy',
        'Reading Books',
        'Learning',
        'Writing',
        'Research Projects',
        'Teaching',
        'Language',
        'Ethics Work',
        'Presentations',
        'Physical Exercise',
        'Human Experience',
        'Coding',
        'Hobbies',
        'Art',
        'Year',
        'Month_Num',
        'Month',
        'Day',
        'DayOfYear',
        'Week'
    ]]
    leisure_df_plot = leisure_df.melt(
        id_vars=['date_column', 'Year', 'Month_Num', 'Month', 'Day', 'DayOfYear', 'Week'])
    leisure_df_plot = leisure_df_plot.sort_values(
        by=['date_column'], ascending=True).reset_index(drop=True)

    # Function to calculate the slope for each category
    def calculate_slope(df, category):
        x = (df['date_column'] - df['date_column'].min()).dt.days
        y = df[category]
        slope, intercept = np.polyfit(x, y, 1)
        return slope

    # Plot
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.lineplot(
        leisure_df_plot['date_column'], leisure_df_plot['value'], hue=leisure_df_plot['variable'])

    # Calculate slopes for each category
    slopes = {}
    categories = leisure_df_plot['variable'].unique()
    for category in categories:
        df_category = leisure_df_plot[leisure_df_plot['variable'] == category].dropna(
        ).reset_index(drop=True)
        slope = calculate_slope(df_category, 'value')
        slopes[category] = slope

    # Create custom legend
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [
        f'{label} (AVG Mins per day = {slopes[label]:.2f})' 
        if slopes[label] is not np.nan else label 
        for label in labels
    ]
    ax.legend(handles, new_labels, fontsize=12,
              frameon=False, loc='upper left')

    # Calculate and annotate the slope for each category
    categories = leisure_df_plot['variable'].unique()
    for category in categories:
        df_category = leisure_df_plot[leisure_df_plot['variable'] == category].dropna(
        ).reset_index(drop=True)
        slope = calculate_slope(df_category, 'value')
        max_value = df_category['value'].max()
        max_date = df_category['date_column'].max()
        ax.annotate(f'{slope:.2f}', xy=(max_date, max_value),
                    xytext=(10, 0), textcoords='offset points', ha='left', fontsize=12, color='black')

    plt.xlabel("Date", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Cumulative Minutes", fontsize=26, labelpad=12)
    plt.yticks(fontsize=16)

    ax.ticklabel_format(style='plain', axis='y')
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    sns.despine(top=True, right=True)

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


# JOURNAL ARTICLES
def create_journal_plot():
    # prep the data
    article_df = all_data[[
        'date_column',
        'EAB',
        'ABA',
        'OBM',
        'EAB Non-Research',
        'ABA Non-Research',
        'Behaviorism',
        'Ethics',
        'Non-BA Experimental',
        'Non-BA Non-Experimental',
        'Economics',
        'Behavioral Economics',
        'Behavioral Pharmacology',
        'Data & Analytics',
        'Computer Science',
        'Year',
        'Month_Num',
        'Month',
        'Day',
        'DayOfYear',
        'Week'
    ]]
    article_df_plot = article_df.melt(
        id_vars=['date_column', 'Year', 'Month_Num', 'Month', 'Day', 'DayOfYear', 'Week'])
    article_df_plot = article_df_plot.sort_values(
        by=['date_column'], ascending=True).reset_index(drop=True)

    # Function to calculate the slope for each category
    def calculate_slope(df, value_col):
        x = (df['date_column'] - df['date_column'].min()).dt.days
        y = df[value_col]
        slope, intercept = np.polyfit(x, y, 1)
        return slope

    # Plot
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.lineplot(data=article_df_plot, x='date_column', y='value', hue='variable')

    # Calculate slopes for each category
    slopes = {}
    categories = article_df_plot['variable'].unique()
    for category in categories:
        df_category = article_df_plot[article_df_plot['variable'] == category].dropna(
        ).reset_index(drop=True)
        slope = calculate_slope(df_category, 'value')
        slopes[category] = slope

    # Create custom legend
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [
        f'{label} (AVG Mins per day = {slopes[label]:.2f})' 
        if slopes[label] is not np.nan else label 
        for label in labels
    ]
    ax.legend(handles, new_labels, fontsize=12,
              frameon=False, loc='upper left')

    # Calculate and annotate the slope for each category
    categories = article_df_plot['variable'].unique()
    for category in categories:
        df_category = article_df_plot[article_df_plot['variable'] == category].dropna(
        ).reset_index(drop=True)
        slope = calculate_slope(df_category, 'value')
        max_value = df_category['value'].max()
        max_date = df_category['date_column'].max()
        ax.annotate(f'{slope:.2f}', xy=(max_date, max_value),
                    xytext=(10, 0), textcoords='offset points', ha='left', fontsize=12, color='black')

    plt.xlabel("Date", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Cumulative Minutes", fontsize=26, labelpad=12)
    plt.yticks(fontsize=16)

    ax.ticklabel_format(style='plain', axis='y')
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    sns.despine(top=True, right=True)

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


# PHYSICAL ACTIVITY TRACKING
def plot_phys_act_counts():
    # Prep the data
    phys_act['activitydate'] = pd.to_datetime(phys_act['activitydate'])
    plot_df = phys_act.groupby([pd.Grouper(
        key='activitydate', freq='M'), 'activitytype']).size().reset_index(name='Counts')
    plot_df = plot_df.replace({
        "Run": "Run",
        "Yoga": "Yoga",
        "Ride": "Other",
        "Swim": "Other",
        "Workout": "Other",
        "Walk": "Other",
    })

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(
        x=plot_df['activitydate'],
        y=plot_df['Counts'],
        hue=plot_df['activitytype'],
        markers=True,
        markersize=8,
        style=plot_df['activitytype'],
        palette='cividis'
    )
    plt.xlabel("Month", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Number of Activities", fontsize=26, labelpad=12)
    plt.yticks(fontsize=16)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    sns.despine(top=True, right=True)
    plt.legend(loc="best", fontsize=16, frameon=False)

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


def plot_yoga():
    # Prep the data
    phys_act['activitydate'] = pd.to_datetime(phys_act['activitydate'])
    plot_df = phys_act[['activitydate', 'activitytype', 'elapsedtime',]]
    plot_df = plot_df[plot_df['activitytype'].isin(['Yoga'])]
    plot_df = plot_df.pivot_table(index='activitydate', columns='activitytype', values=[
                                  'elapsedtime'], aggfunc='sum').reset_index().fillna(0)
    plot_df.columns = ['_'.join(col).strip() if col[1] else col[0]
                       for col in plot_df.columns.values]
    plot_df['elapsedtime_Yoga'] = (plot_df['elapsedtime_Yoga']/60).cumsum()

    def calculate_slope_runs(x, y):
        slope, intercept = np.polyfit(x, y, 1)
        return slope, intercept

    # Calculate overall slope
    x = (plot_df['activitydate'] - plot_df['activitydate'].min()).dt.days
    y = plot_df['elapsedtime_Yoga']
    overall_slope, intercept = calculate_slope_runs(x, y)

    # Calculate yearly slopes
    plot_df['Year'] = plot_df['activitydate'].dt.year
    yearly_slopes = {}
    for year in plot_df['Year'].unique():
        df_year = plot_df[plot_df['Year'] == year]
        x_year = (df_year['activitydate'] -
                  df_year['activitydate'].min()).dt.days
        y_year = df_year['elapsedtime_Yoga']
        slope_year, _ = calculate_slope_runs(x_year, y_year)
        yearly_slopes[year] = slope_year

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(x=plot_df['activitydate'], y=plot_df['elapsedtime_Yoga'],
                 marker='o', markersize=8, color='k')
    plt.xlabel("Year", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16, rotation=45)
    plt.ylabel("Minutes of Yoga", fontsize=26, labelpad=12)
    plt.yticks(fontsize=16)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    sns.despine(top=True, right=True)

    # Annotate overall slope
    plt.text(0.05, 0.95, f'Overall slope: {overall_slope:.2f} mins/day', 
             transform=ax.transAxes, fontsize=16, verticalalignment='top')

    # Annotate yearly slopes
    for year, slope in yearly_slopes.items():
        last_date_of_year = plot_df[plot_df['Year']
                                    == year]['activitydate'].max()
        y_value = plot_df.loc[plot_df['activitydate'] ==
                              last_date_of_year, 'elapsedtime_Yoga'].values[0]
        plt.annotate(f'{slope:.2f}', xy=(last_date_of_year, y_value),
                     xytext=(10, 0), textcoords='offset points', ha='left', fontsize=16, color='black')

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


def plot_run_miles():
    # Prep the data
    phys_act['activitydate'] = pd.to_datetime(phys_act['activitydate'])
    plot_df = phys_act[['activitydate', 'activitytype', 'distance']]
    plot_df = plot_df[plot_df['activitytype'].isin(['Run'])]
    plot_df = plot_df.pivot_table(index='activitydate', columns='activitytype',
                                  values='distance', aggfunc='sum').reset_index().fillna(0)

    # Flatten column names after pivot and compute cumulative miles
    plot_df.columns = [col[1] if col[1] else col[0] for col in plot_df.columns] if isinstance(plot_df.columns, pd.MultiIndex) else plot_df.columns
    run_col = 'Run' if 'Run' in plot_df.columns else 'distance_Run'
    plot_df['distance_Run'] = plot_df[run_col].cumsum()

    def calculate_slope_runs(x, y):
        slope, intercept = np.polyfit(x, y, 1)
        return slope, intercept

    # Calculate overall slope
    x = (plot_df['activitydate'] - plot_df['activitydate'].min()).dt.days
    y = plot_df['distance_Run']
    overall_slope, intercept = calculate_slope_runs(x, y)

    # Calculate yearly slopes
    plot_df['Year'] = plot_df['activitydate'].dt.year
    yearly_slopes = {}
    for year in plot_df['Year'].unique():
        df_year = plot_df[plot_df['Year'] == year]
        x_year = (df_year['activitydate'] -
                  df_year['activitydate'].min()).dt.days
        y_year = df_year['distance_Run']
        slope_year, _ = calculate_slope_runs(x_year, y_year)
        yearly_slopes[year] = slope_year

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(x=plot_df['activitydate'], y=plot_df['distance_Run'],
                 marker='o', markersize=8, color='k')
    plt.xlabel("Year", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Miles Run", fontsize=26, labelpad=0)
    plt.yticks(fontsize=16)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    sns.despine(top=True, right=True)

    # Annotate overall slope
    plt.text(0.05, 0.95, f'Overall slope: {overall_slope:.2f} miles/day', 
             transform=ax.transAxes, fontsize=16, verticalalignment='top')

    # Annotate yearly slopes
    for year, slope in yearly_slopes.items():
        last_date_of_year = plot_df[plot_df['Year']
                                    == year]['activitydate'].max()
        y_value = plot_df.loc[plot_df['activitydate'] ==
                              last_date_of_year, 'distance_Run'].values[0]
        plt.annotate(f'{slope:.2f}', xy=(last_date_of_year, y_value),
                     xytext=(10, 0), textcoords='offset points', ha='left', fontsize=16, color='black')

    # Save the plot to a bytes buffer
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


def plot_run_mins():
    # Prep the data
    phys_act['activitydate'] = pd.to_datetime(phys_act['activitydate'])
    plot_df = phys_act[['activitydate',
                        'activitytype', 'elapsedtime', 'distance']]
    plot_df = plot_df[plot_df['activitytype'].isin(['Run'])]
    plot_df = plot_df.pivot_table(index='activitydate', columns='activitytype', values=[
                                  'elapsedtime', 'distance'], aggfunc='sum').reset_index().fillna(0)
    plot_df.columns = ['_'.join(col).strip() if col[1] else col[0]
                       for col in plot_df.columns.values]
    plot_df['elapsedtime_Run'] = (plot_df['elapsedtime_Run']/60).cumsum()

    def calculate_slope_runs(x, y):
        slope, intercept = np.polyfit(x, y, 1)
        return slope, intercept

    # Calculate overall slope
    x = (plot_df['activitydate'] - plot_df['activitydate'].min()).dt.days
    y = plot_df['elapsedtime_Run']
    overall_slope, intercept = calculate_slope_runs(x, y)

    # Calculate yearly slopes
    plot_df['Year'] = plot_df['activitydate'].dt.year
    yearly_slopes = {}
    for year in plot_df['Year'].unique():
        df_year = plot_df[plot_df['Year'] == year]
        x_year = (df_year['activitydate'] -
                  df_year['activitydate'].min()).dt.days
        y_year = df_year['elapsedtime_Run']
        slope_year, _ = calculate_slope_runs(x_year, y_year)
        yearly_slopes[year] = slope_year

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(x=plot_df['activitydate'], y=plot_df['elapsedtime_Run'],
                 marker='o', markersize=8, color='k')
    plt.xlabel("Year", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Minutes Run", fontsize=26, labelpad=0)
    plt.yticks(fontsize=16)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    sns.despine(top=True, right=True)

    # Annotate overall slope
    plt.text(0.05, 0.95, f'Overall slope: {overall_slope:.2f} mins/day', 
             transform=ax.transAxes, fontsize=16, verticalalignment='top')

    # Annotate yearly slopes
    for year, slope in yearly_slopes.items():
        last_date_of_year = plot_df[plot_df['Year']
                                    == year]['activitydate'].max()
        y_value = plot_df.loc[plot_df['activitydate'] ==
                              last_date_of_year, 'elapsedtime_Run'].values[0]
        plt.annotate(f'{slope:.2f}', xy=(last_date_of_year, y_value),
                     xytext=(10, 0), textcoords='offset points', ha='left', fontsize=16, color='black')

    # Save the plot to a bytes buffer
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


def plot_run_mins_mile():
    # Prep the data
    phys_act['activitydate'] = pd.to_datetime(phys_act['activitydate'])
    plot_df = phys_act[['activitydate',
                        'activitytype', 'elapsedtime', 'distance']]
    plot_df = plot_df[plot_df['activitytype'].isin(['Run'])]
    plot_df['elapsedtime_Run'] = (plot_df['elapsedtime']/60)
    plot_df['distance_Run'] = (plot_df['distance'])
    plot_df['Min per Mile'] = (
        plot_df['elapsedtime_Run'] / plot_df['distance_Run'])
    plot_df = plot_df[(plot_df['Min per Mile'] <= 12) &
                      (plot_df['Min per Mile'] >= 3.5)]

    # Plotting
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.lineplot(x=plot_df['activitydate'], y=plot_df['Min per Mile'],
                 marker='o', markersize=8, color='k')
    plt.xlabel("Year", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Mins per Mile", fontsize=26, labelpad=12)
    plt.yticks(fontsize=16)
    plt.ylim(4.5, 12)
    ax.yaxis.grid()
    sns.despine(top=True, right=True)

    # Save the plot to a bytes buffer
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


def plot_run_mins_mile_month():
    # Prep the data
    phys_act['activitydate'] = pd.to_datetime(phys_act['activitydate'])
    plot_df = phys_act[['activitydate',
                        'activitytype', 'elapsedtime', 'distance']]
    plot_df = plot_df[plot_df['activitytype'].isin(['Run'])]
    plot_df['Year'] = phys_act['activitydate'].dt.year
    plot_df['Month'] = phys_act['activitydate'].dt.month
    plot_df['elapsedtime_Run'] = (plot_df['elapsedtime']/60)
    plot_df['distance_Run'] = (plot_df['distance'])
    plot_df['Min per Mile'] = (
        plot_df['elapsedtime_Run'] / plot_df['distance_Run'])
    plot_df = plot_df[(plot_df['Min per Mile'] <= 12) &
                      (plot_df['Min per Mile'] >= 3.5)]

    # Plotting
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.boxplot(
        x=plot_df['Month'],
        y=plot_df['Min per Mile'],
        hue=plot_df['Year'],
        showfliers=False,
    )
    plt.xlabel("Month", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Mins per Mile", fontsize=26, labelpad=12)
    plt.yticks(fontsize=16)
    plt.ylim(6, 10)
    sns.despine(top=True, right=True)
    plt.legend(frameon=False, loc="best", fontsize=12)

    # Save the plot to a bytes buffer
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


# BOOKS READ
def pages_per_day():
    current_year = date.today().year
    days_elapsed = (date.today() - date(current_year, 1, 1)).days or 1
    plot_books = books.groupby(by=['year_read'])['pages'].sum().reset_index()
    plot_books['pages'] = plot_books.apply(
        lambda row: row['pages'] / days_elapsed if row['year_read'] == current_year
        else row['pages'] / 365, axis=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(x='year_read', y='pages', data=plot_books,
                 color='k', marker='o', markersize=12)

    # Add data labels
    for i in range(len(plot_books)):
        ax.annotate(f'{plot_books["pages"].iloc[i]:.2f}',
                    (plot_books['year_read'].iloc[i],
                     plot_books['pages'].iloc[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=16)

    plt.xlabel("Year", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Pages per Day", fontsize=26, labelpad=12)
    plt.yticks(fontsize=16)
    plt.ylim(0, )
    sns.despine(top=True, right=True)

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


def books_annually():
    current_year = date.today().year
    day_of_year = (date.today() - date(current_year, 1, 1)).days or 1
    plot_books = books.groupby(by=['year_read'])['prop_read'].sum().reset_index()
    plot_books['prop_read'] = plot_books.apply(
        lambda row: row['prop_read'] * 365 / day_of_year if row['year_read'] == current_year
        else row['prop_read'], axis=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(x='year_read', y='prop_read', data=plot_books,
                 color='k', marker='o', markersize=12)

    # Add data labels
    for i in range(len(plot_books)):
        ax.annotate(f'{plot_books["prop_read"].iloc[i]:.2f}',
                    (plot_books['year_read'].iloc[i],
                     plot_books['prop_read'].iloc[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=16)

    plt.xlabel("Year", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Books per Year", fontsize=26, labelpad=12)
    plt.yticks(fontsize=16)
    plt.ylim(0, )
    sns.despine(top=True, right=True)

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


# Define a sample DataFrame to display as a table
table_data = books[['year_read', 'title', 'author']]

def plot_pages():
    # Prepare the data
    annual_pages = books.groupby('year_read')['pages'].sum().reset_index()
    
    # Calculate months passed for current year
    current_year = date.today().year
    current_month = date.today().month
    
    # Calculate monthly pages based on year-to-date for current year, full year for past years
    annual_pages['Monthly Pages'] = annual_pages.apply(
        lambda row: row['pages'] / current_month if row['year_read'] == current_year 
        else row['pages'] / 12, 
        axis=1
    )

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Plot annual pages
    ax1.plot(annual_pages['year_read'], annual_pages['pages'],
             'o-', color='blue', label='Annually')
    ax1.set_xlabel('Year', fontsize=26, labelpad=12)
    ax1.set_ylabel('Pages Annually', color='blue', fontsize=26)
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=16)
    ax1.tick_params(axis='x', labelcolor='k', labelsize=12)
    ax1.set_ylim(0, )

    # Add data labels for annual pages
    for i in range(len(annual_pages)):
        ax1.annotate(f"{annual_pages['pages'].iloc[i]:,.0f}",
                     (annual_pages['year_read'].iloc[i],
                      annual_pages['pages'].iloc[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=14, color='blue')

    # Create a secondary y-axis for monthly pages
    ax2 = ax1.twinx()
    ax2.plot(annual_pages['year_read'], annual_pages['Monthly Pages'],
             'o--', color='red', label='Per Month')
    ax2.set_ylabel('Pages Monthly', color='red', fontsize=26)
    ax2.tick_params(axis='y', labelcolor='red', labelsize=12)
    ax2.set_ylim(0, 2500)

    # Add data labels for monthly pages
    for i in range(len(annual_pages)):
        ax2.annotate(f"{annual_pages['Monthly Pages'].iloc[i]:,.0f}",
                     (annual_pages['year_read'].iloc[i],
                      annual_pages['Monthly Pages'].iloc[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=14, color='red')

    sns.despine(top=True, right=False)

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


# CV ADDITIONS
def total_cv_adds():
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(x='year', y='total_additions', data=cv_adds,
                 color='k', marker='o', markersize=12)

    # Add data labels
    for i in range(len(cv_adds)):
        ax.annotate(f'{cv_adds["total_additions"].iloc[i]:.0f}',
                    (cv_adds['year'].iloc[i],
                     cv_adds['total_additions'].iloc[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=16)

    plt.xlabel("Year", fontsize=26, labelpad=12)
    ticks = list(range(cv_adds['year'].min(), cv_adds['year'].max()+1, 2))
    plt.xticks(ticks=ticks, labels=ticks, fontsize=16)
    plt.ylabel("Total Additions", fontsize=26, labelpad=12)
    plt.yticks(fontsize=16)
    plt.ylim(0, )
    sns.despine(top=True, right=True)

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


def plot_research_cv():
    # Prep the data
    plot_df = cv_adds[['year', 'research', 'publications',
                       'editorial_decisions', 'peer_reviewer']].melt(id_vars=['year'])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(
        x=plot_df['year'],
        y=plot_df['value'],
        hue=plot_df['variable'],
        markers=True,
        markersize=12,
        style=plot_df['variable'],
        palette='cividis'
    )
    plt.xlabel("Year", fontsize=26, labelpad=12)
    ticks = list(range(cv_adds['year'].min(), cv_adds['year'].max()+1, 2))
    plt.xticks(ticks=ticks, labels=ticks, fontsize=16)
    plt.ylabel("Number of Additions", fontsize=26, labelpad=12)
    plt.yticks(fontsize=16)
    sns.despine(top=True, right=True)
    plt.legend(loc="best", fontsize=16, frameon=False)

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


def plot_dissem_cv():
    # Prep the data
    plot_df = cv_adds[[
        'year',
        'invited_presentations', 'international_national_presentations',
        'regional_state_presentations', 'local_presentations',
        'popular_media_podcasts']].melt(id_vars=['year'])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(
        x=plot_df['year'],
        y=plot_df['value'],
        hue=plot_df['variable'],
        markers=True,
        markersize=12,
        style=plot_df['variable'],
        palette='cividis'
    )
    plt.xlabel("Year", fontsize=26, labelpad=12)

    ticks = list(range(cv_adds['year'].min(), cv_adds['year'].max()+1, 2))
    plt.xticks(ticks=ticks, labels=ticks, fontsize=16)
    plt.ylabel("Number of Additions", fontsize=26, labelpad=12)

    ticks = list(range(0, plot_df['value'].max()+1, 2))
    plt.yticks(ticks=ticks, labels=ticks, fontsize=16)
    sns.despine(top=True, right=True)
    plt.legend(loc="best", fontsize=16, frameon=False)

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


def plot_service_cv():
    # Prep the data
    plot_df = cv_adds[['year', 'teaching', 'mentorship', 'volunteer']].melt(id_vars=[
                                                                            'year'])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(
        x=plot_df['year'],
        y=plot_df['value'],
        hue=plot_df['variable'],
        markers=True,
        markersize=12,
        style=plot_df['variable'],
        palette='cividis'
    )
    plt.xlabel("Year", fontsize=26, labelpad=12)
    ticks = list(range(cv_adds['year'].min(), cv_adds['year'].max()+1, 2))
    plt.xticks(ticks=ticks, labels=ticks, fontsize=16)
    plt.ylabel("Number of Additions", fontsize=26, labelpad=12)
    plt.yticks(fontsize=16)
    sns.despine(top=True, right=True)
    plt.legend(loc="best", fontsize=16, frameon=False)

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


# PUBLICATION STATS
def plot_cum_pages():
    # Prep the data
    pubs_data = pubs.groupby(by=['year'])['year'].count()

    def calculate_slope(x, y):
        slope, intercept = np.polyfit(x, y, 1)
        return slope, intercept

    # Calculate overall slope
    plot_df = pubs[['year', 'number_pages']]
    plot_df = plot_df.groupby(by=['year']).sum().reset_index()
    plot_df['number_pages'] = plot_df['number_pages'].cumsum()

    x = (plot_df['year'] - plot_df['year'].min())
    y = plot_df['number_pages']
    overall_slope, intercept = calculate_slope(x, y)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(x=plot_df['year'], y=plot_df['number_pages'],
                 marker='o', markersize=8, color='k')
    plt.xlabel("Year", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Cumulative Pages Published", fontsize=26, labelpad=0)
    plt.yticks(fontsize=16)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    sns.despine(top=True, right=True)

    # Add data labels
    for i in range(len(plot_df)):
        ax.annotate(f'{plot_df["number_pages"].iloc[i]:,.0f}',
                    (plot_df['year'].iloc[i], plot_df['number_pages'].iloc[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=16)

    # Annotate overall slope
    plt.text(0.05, 0.95, f'Overall slope: {overall_slope:.2f} Pages/Year', 
             transform=ax.transAxes, fontsize=16, verticalalignment='top')

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


def plot_cum_words():
    def calculate_slope(x, y):
        slope, intercept = np.polyfit(x, y, 1)
        return slope, intercept

    # Calculate overall slope
    plot_df = pubs[['year', 'number_words']]
    plot_df = plot_df.groupby(by=['year']).sum().reset_index()
    plot_df['number_words'] = plot_df['number_words'].cumsum()

    x = (plot_df['year'] - plot_df['year'].min())
    y = plot_df['number_words']
    overall_slope, intercept = calculate_slope(x, y)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(x=plot_df['year'], y=plot_df['number_words'],
                 marker='o', markersize=8, color='k')
    plt.xlabel("Year", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Cumulative Words Published", fontsize=26, labelpad=0)
    plt.yticks(fontsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    sns.despine(top=True, right=True)

    # Add data labels
    for i in range(len(plot_df)):
        ax.annotate(f'{plot_df["number_words"].iloc[i]:,.0f}',
                    (plot_df['year'].iloc[i], plot_df['number_words'].iloc[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=16)

    # Annotate overall slope
    plt.text(0.05, 0.95, f'Overall slope: {overall_slope:.2f} Words/Year', 
             transform=ax.transAxes, fontsize=16, verticalalignment='top')

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


def plot_pages_year():
    plot_df = pubs[['year', 'number_pages']]
    plot_df = plot_df.groupby(by=['year']).sum().reset_index()

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(x=plot_df['year'], y=plot_df['number_pages'],
                 marker='o', markersize=8, color='k')
    plt.xlabel("Year", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Total Pages Published", fontsize=26, labelpad=0)
    plt.yticks(fontsize=16)
    sns.despine(top=True, right=True)

    # Add data labels
    for i in range(len(plot_df)):
        ax.annotate(f'{plot_df["number_pages"].iloc[i]:,.0f}',
                    (plot_df['year'].iloc[i], plot_df['number_pages'].iloc[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=16)

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


def plot_words_year():
    plot_df = pubs[['year', 'number_words']]
    plot_df = plot_df.groupby(by=['year']).sum().reset_index()

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(x=plot_df['year'], y=plot_df['number_words'],
                 marker='o', markersize=8, color='k')
    plt.xlabel("Year", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Total Words Published", fontsize=26, labelpad=0)
    plt.yticks(fontsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    sns.despine(top=True, right=True)

    # Add data labels
    for i in range(len(plot_df)):
        ax.annotate(f'{plot_df["number_words"].iloc[i]:,.0f}',
                    (plot_df['year'].iloc[i], plot_df['number_words'].iloc[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=16)

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


def plot_cum_journals():
    def calculate_slope(x, y):
        slope, intercept = np.polyfit(x, y, 1)
        return slope, intercept

    # Calculate overall slope
    plot_df = pubs[['year', 'number_journals']]
    plot_df = plot_df.groupby(by=['year']).sum().reset_index()
    plot_df['number_journals'] = plot_df['number_journals'].cumsum()

    x = (plot_df['year'] - plot_df['year'].min())
    y = plot_df['number_journals']
    overall_slope, intercept = calculate_slope(x, y)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(x=plot_df['year'], y=plot_df['number_journals'],
                 marker='o', markersize=8, color='k')
    plt.xlabel("Year", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Cumulative Journals", fontsize=26, labelpad=0)
    plt.yticks(fontsize=16)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    sns.despine(top=True, right=True)

    # Add data labels
    for i in range(len(plot_df)):
        ax.annotate(f'{plot_df["number_journals"].iloc[i]:,.0f}',
                    (plot_df['year'].iloc[i],
                     plot_df['number_journals'].iloc[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=16)

    # Annotate overall slope
    plt.text(0.05, 0.95, f'Overall slope: {overall_slope:.2f} New Journals/Year', 
             transform=ax.transAxes, fontsize=16, verticalalignment='top')

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


# BASEBALL WATCHED
def create_cr_baseball():
    # Prep the data and ensure it's sorted by date
    plot_df = baseball[pd.to_datetime(baseball['date']) <= pd.to_datetime(
        date.today())].copy()
    plot_df['date'] = pd.to_datetime(plot_df['date'])
    plot_df = plot_df.sort_values('date').reset_index(drop=True)
    plot_df['Year'] = plot_df['date'].dt.year

    # Create cumulative counts for each team
    teams = ['pirates', 'guardians', 'braves', 'other']
    for team in teams:
        # Calculate cumulative sum of boolean values (already sorted by date)
        plot_df[team.capitalize()] = plot_df[team].cumsum()
    
    # Calculate total games watched (sum of all team games)
    plot_df['Total'] = plot_df[['Pirates', 'Guardians', 'Braves', 'Other']].sum(axis=1)

    def calculate_yearly_trendlines(df, column):
        trends = {}
        for year in df['Year'].unique():
            yearly_data = df[df['Year'] == year]
            yearly_data['DayOfYear'] = yearly_data['date'].dt.dayofyear
            z = np.polyfit(yearly_data['DayOfYear'], yearly_data[column], 1)
            p = np.poly1d(z)
            trends[year] = p
        return trends

    def extract_slopes(trends):
        slopes = {year: trend[1] for year, trend in trends.items()}
        return slopes

    # Function for adding slope labels
    def annotate_slopes(trends, data, column_name, color):
        for year, trend in trends.items():
            slope = trend[1]
            last_date_of_year = data[data['Year'] == year]['date'].max()
            y_value = data.loc[data['date'] == last_date_of_year, column_name].values[0]
            plt.annotate(f'{slope:.2f}', xy=(last_date_of_year, y_value),
                         xytext=(-10, 0), textcoords='offset points', ha='right', color=color, fontsize=14)

    # Calculate trendlines
    trends_pirates = calculate_yearly_trendlines(plot_df, 'Pirates')
    trends_guardians = calculate_yearly_trendlines(plot_df, 'Guardians')
    trends_braves = calculate_yearly_trendlines(plot_df, 'Braves')
    trends_other = calculate_yearly_trendlines(plot_df, 'Other')
    trends_total = calculate_yearly_trendlines(plot_df, 'Total')

    # Extract slopes
    slopes_pirates = extract_slopes(trends_pirates)
    slopes_guardians = extract_slopes(trends_guardians)
    slopes_braves = extract_slopes(trends_braves)
    slopes_other = extract_slopes(trends_other)
    slopes_total = extract_slopes(trends_total)

    # Create a DataFrame with the slopes
    slopes_df = pd.DataFrame({
        'Total': slopes_total.values(),
        'Year': slopes_pirates.keys(),
        'Pirates': slopes_pirates.values(),
        'Guardians': slopes_guardians.values(),
        'Braves': slopes_braves.values(),
        'Other': slopes_other.values(),
    })
    slopes_df = slopes_df.melt(id_vars=['Year'])

    fig, ax = plt.subplots(figsize=(18, 10))

    # Plot each category (data is already sorted by date)
    plt.plot(plot_df['date'], plot_df['Total'], label='Total', color='green')
    plt.plot(plot_df['date'], plot_df['Pirates'], label='Pirates', color='black')
    plt.plot(plot_df['date'], plot_df['Guardians'], label='Guardians', color='red')
    plt.plot(plot_df['date'], plot_df['Braves'], label='Braves', color='blue')
    plt.plot(plot_df['date'], plot_df['Other'], label='Other', color='purple')

    # Annotate the slopes using the function
    annotate_slopes(trends_pirates, plot_df, 'Pirates', 'black')
    annotate_slopes(trends_guardians, plot_df, 'Guardians', 'red')
    annotate_slopes(trends_braves, plot_df, 'Braves', 'blue')
    annotate_slopes(trends_other, plot_df, 'Other', 'purple')
    annotate_slopes(trends_total, plot_df, 'Total', 'green')

    plt.xlabel("Date", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Cumulative Games", fontsize=26, labelpad=12)
    plt.yticks(fontsize=16)

    ax.ticklabel_format(style='plain', axis='y')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
    sns.despine(top=True, right=True)
    plt.legend(loc="upper center", fontsize=20, frameon=False)

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


def plot_games_year():
    # Prep the data
    plot_df = baseball.copy()
    plot_df['Year'] = pd.to_datetime(plot_df['date']).dt.year
    
    # Sum boolean values for each team by year
    yearly_totals = plot_df.groupby('Year').agg({
        'pirates': 'sum',
        'guardians': 'sum',
        'braves': 'sum',
        'other': 'sum'
    }).reset_index()
    
    # Calculate total games
    yearly_totals['Total'] = yearly_totals[['pirates', 'guardians', 'braves', 'other']].sum(axis=1)
    
    # Melt the data for plotting
    plot_df = yearly_totals.melt(id_vars=['Year'])

    # Plotting
    fig, ax = plt.subplots(figsize=(13, 8))
    sns.lineplot(
        x=plot_df['Year'],
        y=plot_df['value'],
        hue=plot_df['variable'],
        markers=True,
        markersize=12,
        style=plot_df['variable'],
        palette='cividis'
    )
    plt.xlabel("Year", fontsize=26, labelpad=12)

    ticks = list(range(int(plot_df['Year'].min()),
                 int(plot_df['Year'].max())+1, 2))
    plt.xticks(ticks=ticks, labels=ticks, fontsize=16)
    plt.ylabel("Number of Games", fontsize=26, labelpad=12)

    ticks = list(range(0, int(plot_df['value'].max())+1, 50))
    plt.yticks(ticks=ticks, labels=ticks, fontsize=16)
    sns.despine(top=True, right=True)
    plt.legend(loc=(1.05, 0.5), fontsize=16, frameon=False)

    # Add data labels
    for i in range(len(plot_df)):
        ax.annotate(f'{plot_df["value"].iloc[i]:,.0f}',
                    (plot_df['Year'].iloc[i]+random.uniform(-0.3, 0.3),
                     plot_df['value'].iloc[i]+random.uniform(0, 5)),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=16)

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64

# Plot whoop sleep stages
def plot_whoop_sleep_stages():
    if whoop_sleep.empty:
        return ''
        
    plot_df = whoop_sleep.melt(
        id_vars=['sleep_start'],
        value_vars=time_columns,  # Include total_in_bed_time
        var_name='Sleep Stage',
        value_name='Hours'
    )
    plot_df['Sleep Stage'] = plot_df['Sleep Stage'].str.replace('score_stage_summary_total_', '').str.replace('_milli', '').str.replace('_', ' ').str.title()
    
    # Create a separate color for total time in bed
    total_color = 'black'
    sleep_color_dict_with_total = sleep_color_dict.copy()
    sleep_color_dict_with_total['In Bed Time'] = total_color
    
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.lineplot(
        data=plot_df,
        x='sleep_start',
        y='Hours',
        hue='Sleep Stage',
        style='Sleep Stage',
        markers='o',
        markersize=10,
        palette=sleep_color_dict_with_total,
        ax=ax
    )
    
    import matplotlib.dates as mdates
    for stage in plot_df['Sleep Stage'].unique():
        stage_df = plot_df[plot_df['Sleep Stage'] == stage].dropna()
        if len(stage_df) > 1:
            x = stage_df['sleep_start'].map(mdates.date2num)
            y = stage_df['Hours']
            z = np.polyfit(x, y, 4)  # degree 4 polynomial for smooth trend
            p = np.poly1d(z)
            color = total_color if stage == 'In Bed Time' else sleep_color_dict[stage]
            ax.plot(stage_df['sleep_start'], p(x), linestyle='--', linewidth=2, alpha=0.7, color=color)
    
    plt.xlabel("Date", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Hours in Stage", fontsize=30, labelpad=12)
    plt.yticks(fontsize=16)
    sns.despine(top=True, right=True)
    legend = plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=24, ncol=3)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.5)
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_base64

def plot_whoop_sleep_proportions():
    if whoop_sleep.empty:
        return ''
        
    plot_df = whoop_sleep.melt(
        id_vars=['sleep_start'],
        value_vars=[f'{col}_proportion' for col in stage_columns],
        var_name='Sleep Stage',
        value_name='Percentage'
    )
    plot_df['Sleep Stage'] = plot_df['Sleep Stage'].str.replace('score_stage_summary_total_', '').str.replace('_milli_proportion', '').str.replace('_', ' ').str.title()
    
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.lineplot(
        data=plot_df,
        x='sleep_start',
        y='Percentage',
        hue='Sleep Stage',
        style='Sleep Stage',
        markers='o',
        markersize=10,
        palette=sleep_color_dict,
        ax=ax
    )
    
    import matplotlib.dates as mdates
    for stage in plot_df['Sleep Stage'].unique():
        stage_df = plot_df[plot_df['Sleep Stage'] == stage].dropna()
        if len(stage_df) > 1:
            x = stage_df['sleep_start'].map(mdates.date2num)
            y = stage_df['Percentage']
            z = np.polyfit(x, y, 4)  # degree 4 polynomial for smooth trend
            p = np.poly1d(z)
            ax.plot(stage_df['sleep_start'], p(x), linestyle='--', linewidth=2, alpha=0.7, color=sleep_color_dict[stage])
    
    plt.xlabel("Date", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Percentage of Time in Bed", fontsize=30, labelpad=12)
    plt.yticks(fontsize=16)
    
    # Format y-axis ticks as percentages
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0%}'))
    
    sns.despine(top=True, right=True)
    legend = plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=24, ncol=2)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.5)
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_base64

def plot_whoop_sleep_needs():
    if whoop_sleep.empty:
        return ''
        
    need_columns = [
        'score_sleep_needed_need_from_sleep_debt_milli',
        'score_sleep_needed_need_from_recent_strain_milli',
    ]
    
    # Convert milliseconds to hours
    for col in need_columns:
        whoop_sleep[col] = whoop_sleep[col] / (1000 * 60 * 60)
    plot_df = whoop_sleep.melt(
        id_vars=['sleep_start'],
        value_vars=need_columns,
        var_name='Sleep Need',
        value_name='Hours'
    )
    plot_df['Sleep Need'] = plot_df['Sleep Need'].str.replace('score_sleep_needed_', '').str.replace('_milli', '').str.replace('_', ' ').str.title()
    fig, ax = plt.subplots(figsize=(18, 10))
    palette = sns.color_palette('bright', n_colors=plot_df['Sleep Need'].nunique())
    color_dict = dict(zip(plot_df['Sleep Need'].unique(), palette))
    sns.lineplot(
        data=plot_df,
        x='sleep_start',
        y='Hours',
        hue='Sleep Need',
        style='Sleep Need',
        markers='o',
        markersize=10,
        palette=color_dict,
        ax=ax
    )
    import matplotlib.dates as mdates
    for need in plot_df['Sleep Need'].unique():
        need_df = plot_df[plot_df['Sleep Need'] == need].dropna()
        if len(need_df) > 1:
            x = need_df['sleep_start'].map(mdates.date2num)
            y = need_df['Hours']
            z = np.polyfit(x, y, 4)  # degree 4 polynomial for smooth trend
            p = np.poly1d(z)
            ax.plot(need_df['sleep_start'], p(x), linestyle='--', linewidth=2, alpha=0.7, color=color_dict[need])
    plt.xlabel("Date", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Hours Needed", fontsize=30, labelpad=12)
    plt.yticks(fontsize=16)
    sns.despine(top=True, right=True)
    legend = plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=24, ncol=2)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.5)
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_base64

def plot_whoop_sleep_percentages():
    if whoop_sleep.empty:
        return ''
        
    percent_columns = [
        'score_sleep_performance_percentage',
        'score_sleep_consistency_percentage',
        'score_sleep_efficiency_percentage'
    ]
    plot_df = whoop_sleep[['sleep_start'] + percent_columns].melt(
        id_vars=['sleep_start'],
        value_vars=percent_columns,
        var_name='Metric',
        value_name='Percentage'
    )
    plot_df['Metric'] = plot_df['Metric'].str.replace('score_sleep_', '').str.replace('_percentage', '').str.replace('_', ' ').str.title()
    
    # Convert percentages to decimal format (0-1) for consistent plotting
    plot_df['Percentage'] = plot_df['Percentage'] / 100
    
    fig, ax = plt.subplots(figsize=(18, 10))
    palette = sns.color_palette('bright', n_colors=plot_df['Metric'].nunique())
    color_dict = dict(zip(plot_df['Metric'].unique(), palette))
    sns.lineplot(
        data=plot_df,
        x='sleep_start',
        y='Percentage',
        hue='Metric',
        style='Metric',
        markers='o',
        markersize=10,
        palette=color_dict,
        ax=ax
    )
    import matplotlib.dates as mdates
    for metric in plot_df['Metric'].unique():
        metric_df = plot_df[plot_df['Metric'] == metric].dropna()
        if len(metric_df) > 1:
            x = metric_df['sleep_start'].map(mdates.date2num)
            y = metric_df['Percentage']  # Already converted to decimal format above
            z = np.polyfit(x, y, 4)  # degree 4 polynomial for smooth trend
            p = np.poly1d(z)
            ax.plot(metric_df['sleep_start'], p(x), linestyle='--', linewidth=2, alpha=0.7, color=color_dict[metric])
    plt.xlabel("Date", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Sleep Percentage Scores", fontsize=30, labelpad=12)
    plt.yticks(fontsize=16)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0%}'))
    sns.despine(top=True, right=True)
    legend = plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=24, ncol=3)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.5)
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_base64

def plot_whoop_respiratory_rate():
    if whoop_sleep.empty:
        return ''
        
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.lineplot(
        data=whoop_sleep,
        x='sleep_start',
        y='score_respiratory_rate',
        marker='o',
        color='black',
        ax=ax, 
        markersize=10
    )
    # Add trendline
    import matplotlib.dates as mdates
    x = whoop_sleep['sleep_start'].map(mdates.date2num)
    y = whoop_sleep['score_respiratory_rate']
    if len(whoop_sleep) > 1:
        z = np.polyfit(x, y, 4)  # degree 4 polynomial for smooth trend
        p = np.poly1d(z)
        ax.plot(whoop_sleep['sleep_start'], p(x), linestyle='--', linewidth=2, alpha=0.7, color='black')
    plt.xlabel("Date", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Respiratory Rate", fontsize=30, labelpad=12)
    plt.yticks(fontsize=16)
    sns.despine(top=True, right=True)
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_base64

def plot_whoop_recovery_timeseries():
    cols = [
        ('score_recovery_score', 'Recovery Score', 'Higher is better (0-100)'),
        ('score_resting_heart_rate', 'Resting Heart Rate', 'Lower is better (bpm)'),
        ('score_hrv_rmssd_milli', 'HRV (RMSSD)', 'Higher is better (ms)'),
        ('score_spo2_percentage', 'SpO2 (%)', 'Higher is better (95-100%)'),
        ('score_skin_temp_celsius', 'Skin Temp (°F)', 'Normal range: 91-95°F')
    ]
    
    # Create a list to store the base64 encoded images
    plot_images = []
    
    # Create individual plots for each metric
    for col, label, subtitle in cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert temperature to Fahrenheit if needed
        if col == 'score_skin_temp_celsius':
            y_data = whoop_recoveries[col] * 9/5 + 32
        else:
            y_data = whoop_recoveries[col]
            
        sns.lineplot(
            data=whoop_recoveries,
            x='created_at',
            y=y_data,
            ax=ax,
            color='black'
        )
        
        # Add trendline
        import matplotlib.dates as mdates
        x = whoop_recoveries['created_at'].map(mdates.date2num)
        y = y_data
        if len(whoop_recoveries) > 1:
            z = np.polyfit(x, y, 4)  # degree 4 polynomial for smooth trend
            p = np.poly1d(z)
            ax.plot(whoop_recoveries['created_at'], p(x), linestyle='--', linewidth=2, alpha=0.7, color='black')
        
        ax.set_ylabel(label, fontsize=20, labelpad=12)
        ax.set_xlabel('Date', fontsize=20, labelpad=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_title(f'{label}\n{subtitle}', fontsize=20)
        sns.despine(ax=ax)
        
        # Save the plot to a bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plot_images.append(image_base64)
        plt.close()
    
    return plot_images


def find_optimal_n_components(df_scaled):
    """Use PCA explained variance with knee detection to determine optimal n_components."""
    max_components = min(len(df_scaled), len(df_scaled.columns), 15)
    pca_full = PCA(n_components=max_components, random_state=2225)
    pca_full.fit(df_scaled)

    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
    kneedle = KneeLocator(
        range(1, max_components + 1),
        cumulative_var,
        curve='concave',
        direction='increasing'
    )
    knee = kneedle.knee if kneedle.knee else 3
    knee = max(knee, 2)

    # Ensure at least 90% variance is explained
    min_90 = int(np.searchsorted(cumulative_var, 0.90)) + 1
    n_components = max(knee, min_90)
    n_components = min(n_components, max_components)

    # Generate elbow plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, max_components + 1), cumulative_var, 'bo-', linewidth=2, markersize=8)
    ax.axvline(x=n_components, color='r', linestyle='--', linewidth=2,
               label=f'n = {n_components} ({cumulative_var[n_components-1]:.0%} variance)')
    ax.axhline(y=0.90, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='90% threshold')
    ax.set_xlabel('Number of Components', fontsize=40, labelpad=12)
    ax.set_ylabel('Cumulative\nExplained\nVariance', fontsize=40, labelpad=12)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    ax.set_xticks(range(1, max_components + 1))
    sns.despine(ax=ax)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    elbow_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return n_components, pca_full, elbow_base64


def plot_2d_embedding_subplots(df, embedding_results, n_components, method_name='PHATE'):
    """
    Creates pairwise 2D projection subplots in a grid with 5 columns (no back views).
    """
    import math
    temp_df = df.copy()
    for i in range(n_components):
        temp_df[f'{method_name}{i+1}'] = embedding_results[:, i]

    pairs = list(combinations(range(n_components), 2))
    n_pairs = len(pairs)
    n_cols = 5
    n_rows = math.ceil(n_pairs / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for ax in axes.flatten():
        ax.set_visible(False)

    for idx, (i, j) in enumerate(pairs):
        x_col = f'{method_name}{i+1}'
        y_col = f'{method_name}{j+1}'
        row = idx // n_cols
        col = idx % n_cols

        ax = axes[row, col]
        ax.set_visible(True)
        sns.scatterplot(data=temp_df, x=x_col, y=y_col, hue='Year',
                        palette='viridis', alpha=0.7, ax=ax)
        ax.set_title(f'Dim {i+1} vs Dim {j+1}', fontsize=20)
        ax.set_xlabel(f'{method_name} {i+1}', fontsize=14)
        ax.set_ylabel(f'{method_name} {j+1}', fontsize=14)

    for ax in axes.flatten():
        if ax.get_visible():
            if ax.get_legend():
                ax.legend().remove()
            sns.despine(ax=ax)

    handles, labels = None, None
    for ax in axes.flatten():
        if ax.get_visible():
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
    if handles:
        fig.legend(handles, labels, title='Year', bbox_to_anchor=(1.02, 0.5),
                   loc='center left', fontsize=16, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return image_base64


def plot_feature_phate_correlations(df_features, phate_results, n_components):
    """Spearman correlation heatmap between original features and PHATE dimensions."""
    phate_df = pd.DataFrame(phate_results[:, :n_components],
                            columns=[f'PHATE {i+1}' for i in range(n_components)])
    n_features = len(df_features.columns)
    corr_matrix = np.zeros((n_features, n_components))
    pval_matrix = np.zeros((n_features, n_components))
    for i, col in enumerate(df_features.columns):
        for j in range(n_components):
            corr_matrix[i, j], pval_matrix[i, j] = spearmanr(df_features[col].values, phate_df.iloc[:, j].values)

    corr_df = pd.DataFrame(corr_matrix, index=df_features.columns,
                           columns=phate_df.columns)
    pval_df = pd.DataFrame(pval_matrix, index=df_features.columns,
                           columns=phate_df.columns)

    # Filter to features with at least one |correlation| >= 0.1
    mask = (corr_df.abs() >= 0.1).any(axis=1)
    corr_df_filtered = corr_df[mask]

    # Sort by max absolute correlation
    corr_df_filtered = corr_df_filtered.loc[corr_df_filtered.abs().max(axis=1).sort_values(ascending=False).index]

    fig_height = max(8, 0.4 * len(corr_df_filtered))
    fig_width = max(10, 3 * n_components)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    hm = sns.heatmap(corr_df_filtered, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, cbar_kws={'label': 'Spearman Correlation'},
                annot_kws={'size': 14})
    # Style the colorbar
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Spearman Correlation', fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=14)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return image_base64, corr_df, pval_df


_DOMAIN_BUCKETS = [
    ('sleep',    ['sleep', 'nap', 'bed', 'rem', 'slow_wave', 'asleep', 'wake']),
    ('recovery', ['recovery', 'hrv', 'rhr', 'resting_heart', 'spo2', 'skin_temp']),
    ('strain',   ['strain', 'kilojoule', 'calorie', 'max_heart', 'avg_heart']),
    ('exercise', ['run', 'ride', 'bike', 'walk', 'swim', 'workout', 'strava',
                  'distance', 'elapsedtime', 'activity', 'step', 'zone_duration']),
    ('baseball', ['pirate', 'guardian', 'brave', 'baseball', 'mlb', 'in_person']),
    ('work',     ['publication', 'cv_', 'book']),
    ('time',     ['day', 'month', 'year', 'dayofyear']),
]


def _bucket_for(feature_name):
    name = feature_name.lower()
    for bucket, needles in _DOMAIN_BUCKETS:
        if any(n in name for n in needles):
            return bucket
    return 'other'


def _summarize_buckets(feature_names):
    """Group feature names into domain buckets and return the dominant buckets."""
    if not feature_names:
        return []
    from collections import Counter
    counts = Counter(_bucket_for(f) for f in feature_names)
    return [b for b, _ in counts.most_common()]


def describe_phate_dimensions(corr_df, pval_df, corr_threshold=0.1, pval_threshold=0.05):
    """Generate interpretive descriptions for each PHATE dimension.

    For each dimension, keeps only features that are *distinctive* to that
    dimension (correlate more strongly here than with any other dimension),
    then builds a short table row describing what a high vs. low score on
    that axis looks like as a day-type.
    """
    descriptions = []

    for dim_col in corr_df.columns:
        dim_corrs = corr_df[dim_col]
        dim_pvals = pval_df[dim_col]

        sig_mask = (dim_corrs.abs() >= corr_threshold) & (dim_pvals < pval_threshold)
        sig_features = dim_corrs[sig_mask]

        other_cols = [c for c in corr_df.columns if c != dim_col]
        if other_cols and not sig_features.empty:
            max_other = corr_df.loc[sig_features.index, other_cols].abs().max(axis=1)
            distinctiveness = sig_features.abs() - max_other
        else:
            distinctiveness = sig_features.abs()

        distinctive = sig_features[distinctiveness > 0]
        if distinctive.empty:
            descriptions.append({
                'Dimension': dim_col,
                'High-score days': '(no distinctive features)',
                'Low-score days': '(no distinctive features)',
                'Narrative': 'No features load more strongly on this dimension than on the others — axis is not well-separated.'
            })
            continue

        ranked = distinctive.reindex(
            distinctiveness.loc[distinctive.index].sort_values(ascending=False).index
        )
        pos = ranked[ranked > 0]
        neg = ranked[ranked < 0]

        def fmt(series, k=4):
            return ', '.join(f"{f} ({r:+.2f})" for f, r in series.head(k).items()) or '—'

        pos_str = fmt(pos)
        neg_str = fmt(neg)

        pos_buckets = _summarize_buckets(pos.index.tolist())
        neg_buckets = _summarize_buckets(neg.index.tolist())

        def phrase(buckets):
            if not buckets:
                return 'no strong theme'
            if len(buckets) == 1:
                return f"{buckets[0]}-heavy"
            return f"{buckets[0]}- and {buckets[1]}-heavy"

        if pos_buckets and neg_buckets:
            narrative = (f"High end = {phrase(pos_buckets)} days; "
                         f"low end = {phrase(neg_buckets)} days.")
        elif pos_buckets:
            narrative = f"Axis picks out {phrase(pos_buckets)} days (no distinctive negative loadings)."
        else:
            narrative = f"Axis picks out the absence of {phrase(neg_buckets)} days."

        descriptions.append({
            'Dimension': dim_col,
            'High-score days': pos_str,
            'Low-score days': neg_str,
            'Narrative': narrative,
        })

    return descriptions


def _compute_multivariate_recurrence(df_features, percentile=10, dates=None):
    """Compute multivariate recurrence matrix using pairwise Euclidean distances.
    Only uses columns with data for at least 50% of rows, and drops rows where
    every usable column is zero (no-tracking days) so they don't create a
    spurious block of recurrences."""
    from scipy.spatial.distance import pdist, squareform
    nonzero_frac = (df_features != 0).mean()
    usable_cols = nonzero_frac[nonzero_frac >= 0.5].index
    usable = df_features[usable_cols]
    tracked_mask = (usable != 0).any(axis=1).values
    data = usable.values[tracked_mask]
    kept_dates = dates.reset_index(drop=True)[tracked_mask] if dates is not None else None
    dist_matrix = squareform(pdist(data, metric='euclidean'))
    threshold = np.percentile(dist_matrix[dist_matrix > 0], percentile)
    recurrence_matrix = (dist_matrix <= threshold).astype(int)
    np.fill_diagonal(recurrence_matrix, 0)
    return recurrence_matrix, kept_dates


def plot_recurrence(df_features, dates=None):
    """Generate a multivariate recurrence plot from the full feature space."""
    rp_matrix, kept_dates = _compute_multivariate_recurrence(df_features, dates=dates)

    n = rp_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rp_matrix, cmap='binary', origin='lower', interpolation='nearest')
    ax.set_xlabel('Date', fontsize=24, labelpad=12)
    ax.set_ylabel('Date', fontsize=24, labelpad=12)

    # Use actual dates for tick labels if provided
    tick_step = max(1, n // 10)
    tick_positions = list(range(0, n, tick_step))
    if kept_dates is not None:
        date_labels = [pd.to_datetime(kept_dates.iloc[t]).strftime('%-m/%d/%Y') for t in tick_positions]
    else:
        date_labels = [f'Day {t+1}' for t in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(date_labels, fontsize=12, rotation=45, ha='right')
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(date_labels, fontsize=12)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return image_base64


def compute_rqa_metrics(df_features):
    """Compute RQA metrics from the multivariate recurrence matrix."""
    rp_matrix, _ = _compute_multivariate_recurrence(df_features)
    n = rp_matrix.shape[0]
    total_possible = n * (n - 1)  # excluding diagonal

    # Recurrence rate
    n_recurrence = rp_matrix.sum()
    rr = n_recurrence / total_possible if total_possible > 0 else 0

    # Extract diagonal lines (parallel to main diagonal)
    diagonal_lengths = []
    for k in range(1, n):
        diag = np.diag(rp_matrix, k)
        length = 0
        for val in diag:
            if val:
                length += 1
            elif length > 1:
                diagonal_lengths.append(length)
                length = 0
        if length > 1:
            diagonal_lengths.append(length)

    # Extract vertical lines
    vertical_lengths = []
    for col in range(n):
        length = 0
        for row in range(n):
            if rp_matrix[row, col]:
                length += 1
            elif length > 1:
                vertical_lengths.append(length)
                length = 0
        if length > 1:
            vertical_lengths.append(length)

    # Determinism
    det_points = sum(diagonal_lengths)
    det = det_points / n_recurrence if n_recurrence > 0 else 0

    # Average diagonal line length
    avg_diag = np.mean(diagonal_lengths) if diagonal_lengths else 0

    # Longest diagonal line
    lmax = max(diagonal_lengths) if diagonal_lengths else 0

    # Entropy of diagonal line distribution
    if diagonal_lengths:
        counts = np.bincount(diagonal_lengths)
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs))
    else:
        entropy = 0

    # Laminarity
    lam_points = sum(vertical_lengths)
    lam = lam_points / n_recurrence if n_recurrence > 0 else 0

    # Trapping time
    tt = np.mean(vertical_lengths) if vertical_lengths else 0

    # Divergence
    div = 1.0 / lmax if lmax > 0 else 0

    return {
        'Recurrence Rate (RR)': rr,
        'Determinism (DET)': det,
        'Avg Diagonal Line (L)': avg_diag,
        'Max Diagonal Line (Lmax)': lmax,
        'Entropy (ENTR)': entropy,
        'Laminarity (LAM)': lam,
        'Trapping Time (TT)': tt,
        'Divergence (DIV)': div,
    }

def interpret_rqa(metrics):
    """Generate a plain-language interpretation of RQA metrics."""
    if not metrics:
        return "No RQA data available for interpretation."

    rr = metrics.get('Recurrence Rate (RR)', 0)
    det = metrics.get('Determinism (DET)', 0)
    lam = metrics.get('Laminarity (LAM)', 0)
    entr = metrics.get('Entropy (ENTR)', 0)
    tt = metrics.get('Trapping Time (TT)', 0)
    avg_l = metrics.get('Avg Diagonal Line (L)', 0)

    lines = []

    # Recurrence rate
    if rr < 0.05:
        lines.append(f"Recurrence Rate ({rr:.2%}): Very low — daily behavioral patterns rarely repeat. Each day tends to be distinct.")
    elif rr < 0.15:
        lines.append(f"Recurrence Rate ({rr:.2%}): Moderate — some days resemble past days, suggesting recurring routines.")
    else:
        lines.append(f"Recurrence Rate ({rr:.2%}): High — many days closely resemble previous days, indicating strong habitual patterns.")

    # Determinism
    if det < 0.3:
        lines.append(f"Determinism ({det:.2%}): Low — recurrences are mostly isolated (random), not forming predictable sequences.")
    elif det < 0.7:
        lines.append(f"Determinism ({det:.2%}): Moderate — some recurring day-sequences exist, suggesting semi-regular weekly or seasonal patterns.")
    else:
        lines.append(f"Determinism ({det:.2%}): High — recurring states tend to follow the same trajectory, indicating strong predictability in behavioral sequences.")

    # Laminarity
    if lam < 0.3:
        lines.append(f"Laminarity ({lam:.2%}): Low — the system rarely gets \"stuck\" in a single state. Behavior changes frequently.")
    elif lam < 0.7:
        lines.append(f"Laminarity ({lam:.2%}): Moderate — some periods of sustained similar behavior (e.g., consistent routines for stretches of days).")
    else:
        lines.append(f"Laminarity ({lam:.2%}): High — extended periods where daily patterns remain very similar, suggesting stable behavioral regimes.")

    # Trapping time
    lines.append(f"Trapping Time ({tt:.1f} days): On average, when behavior enters a recurring state, it stays there for ~{tt:.0f} consecutive days before shifting.")

    # Entropy
    if entr < 1.0:
        lines.append(f"Entropy ({entr:.2f}): Low complexity — recurring patterns are simple and repetitive.")
    elif entr < 2.5:
        lines.append(f"Entropy ({entr:.2f}): Moderate complexity — a healthy mix of pattern types across different time scales.")
    else:
        lines.append(f"Entropy ({entr:.2f}): High complexity — diverse set of recurring patterns at many scales, suggesting a rich behavioral repertoire.")

    # Avg diagonal line
    lines.append(f"Average Diagonal Line ({avg_l:.1f} days): Typical recurring behavioral sequence lasts ~{avg_l:.0f} days.")

    return lines


# Add this helper function near the top, after imports
def generate_placeholder_image(message="No data available"):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, message, fontsize=18, ha='center', va='center')
    ax.axis('off')
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return f'data:image/png;base64,{image_base64}'

def plot_pca_feature_importance(pca, feature_names, n_components=3, loading_threshold=0.01):
    """Returns a list of base64 images, one per PC."""
    from matplotlib.patches import Patch
    loadings_matrix = np.abs(pca.components_[:n_components])
    max_abs_loadings = loadings_matrix.max(axis=0)
    keep_idx = np.where(max_abs_loadings >= loading_threshold)[0]
    filtered_features = np.array(feature_names)[keep_idx]
    filtered_loadings = pca.components_[:, keep_idx]

    # Consistent feature ordering across all PCs: sort by max abs loading across all PCs
    global_sort_idx = np.argsort(max_abs_loadings[keep_idx])[::-1]
    filtered_features = filtered_features[global_sort_idx]
    filtered_loadings = filtered_loadings[:, global_sort_idx]

    n_features = len(filtered_features)
    fig_width = max(16, n_features * 0.3)
    images = []

    for i in range(n_components):
        fig, ax = plt.subplots(figsize=(fig_width, 4))
        loadings = filtered_loadings[i]
        bar_colors = ['green' if val >= 0 else 'red' for val in loadings]
        ax.bar(range(n_features), np.abs(loadings), color=bar_colors)
        ax.set_title(f'PC{i+1} Feature Loadings', fontsize=22)
        ax.set_ylabel('Abs(Loading)', fontsize=14)
        ax.set_xlabel('Feature', fontsize=14)
        ax.set_xticks(range(n_features))
        ax.set_xticklabels(filtered_features, rotation=90, fontsize=8, ha='center')
        ax.tick_params(axis='y', labelsize=9)
        sns.despine(ax=ax)
        if i == 0:
            legend_handles = [
                Patch(color='green', label='Positive Loading'),
                Patch(color='red', label='Negative Loading')
            ]
            ax.legend(handles=legend_handles, loc='upper right', fontsize=14)
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        images.append(base64.b64encode(buffer.read()).decode('utf-8'))
        plt.close()

    return images


# ---------------------------------------------------------------------------
# Golf plots
# ---------------------------------------------------------------------------
def plot_cumulative_rounds_watched():
    """Cumulative PGA Tour rounds watched with annual trend slopes (rounds/week)."""
    plot_df = golf_watched.copy()
    plot_df['date'] = pd.to_datetime(plot_df['date'])
    plot_df = plot_df.sort_values('date').reset_index(drop=True)
    plot_df['rounds'] = plot_df[['r1', 'r2', 'r3', 'r4']].sum(axis=1)
    plot_df['cumulative'] = plot_df['rounds'].cumsum()
    plot_df['Year'] = plot_df['date'].dt.year

    fig, ax = plt.subplots(figsize=(18, 10))
    plt.plot(plot_df['date'], plot_df['cumulative'], color='green', linewidth=2)

    # Annual trendline slopes (rounds per week = slope * 7)
    for year in plot_df['Year'].unique():
        yearly = plot_df[plot_df['Year'] == year].copy()
        yearly['DayOfYear'] = yearly['date'].dt.dayofyear
        if len(yearly) < 2:
            continue
        z = np.polyfit(yearly['DayOfYear'], yearly['cumulative'], 1)
        slope_per_week = z[0] * 7
        last_date = yearly['date'].max()
        y_value = yearly.loc[yearly['date'] == last_date, 'cumulative'].values[0]
        plt.annotate(f'{slope_per_week:.1f}/wk', xy=(last_date, y_value),
                     xytext=(-10, 0), textcoords='offset points', ha='right',
                     color='green', fontsize=14)

    plt.xlabel("Date", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Cumulative Rounds Watched", fontsize=26, labelpad=12)
    plt.yticks(fontsize=16)
    ax.ticklabel_format(style='plain', axis='y')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
    sns.despine(top=True, right=True)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_base64


def plot_rounds_played_monthly():
    """Rounds of golf played per month with lines per year."""
    plot_df = golf_scores.copy()
    plot_df['played_at'] = pd.to_datetime(plot_df['played_at'])
    plot_df['Year'] = plot_df['played_at'].dt.year
    plot_df['Month'] = plot_df['played_at'].dt.month

    monthly = plot_df.groupby(['Year', 'Month']).size().reset_index(name='Rounds')

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(
        data=monthly,
        x='Month',
        y='Rounds',
        hue='Year',
        markers=True,
        markersize=8,
        style='Year',
        palette={yr: 'red' if yr == 2026 else 'gray' for yr in monthly['Year'].unique()},
    )

    plt.xlabel("Month", fontsize=26, labelpad=12)
    plt.xticks(range(1, 13), fontsize=16)
    plt.ylabel("Rounds Played", fontsize=26, labelpad=12)
    plt.yticks(fontsize=16)
    sns.despine(top=True, right=True)
    plt.legend(loc="best", fontsize=16, frameon=False)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_base64


def plot_golf_scores_handicap():
    """Score differentials over time with rolling handicap index."""
    import matplotlib.dates as mdates

    plot_df = golf_scores.copy()
    plot_df['played_at'] = pd.to_datetime(plot_df['played_at'])
    plot_df['differential'] = pd.to_numeric(plot_df['differential'], errors='coerce')
    # Filter to 18-hole rounds with valid, plausible differentials
    plot_df = plot_df[
        (plot_df['number_of_holes'] == 18)
        & plot_df['differential'].notna()
        & (plot_df['adjusted_gross_score'] > 50)  # exclude mis-tagged 9-hole rounds
    ].sort_values('played_at').reset_index(drop=True)

    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No 18-hole scores available', transform=ax.transAxes,
                ha='center', fontsize=16)
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        return image_base64

    # WHS handicap lookup: number of scores → number of lowest differentials to use
    WHS_TABLE = {
        3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2,
        9: 3, 10: 3, 11: 3, 12: 4, 13: 4, 14: 4,
        15: 5, 16: 5, 17: 6, 18: 6, 19: 7, 20: 8,
    }

    # Calculate rolling handicap index per WHS rules
    handicaps = []
    for i in range(len(plot_df)):
        window = plot_df['differential'].iloc[max(0, i - 19):i + 1]
        n = len(window)
        if n < 3:
            handicaps.append(np.nan)
        else:
            k = WHS_TABLE.get(n, 8)  # 20+ scores → best 8
            best = window.nsmallest(k).mean()
            handicaps.append(round(best, 1))
    plot_df['handicap'] = handicaps

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Score differentials (scatter + trend)
    ax1.scatter(plot_df['played_at'], plot_df['differential'], color='black',
                alpha=0.6, s=40, zorder=3)

    # Trendline
    x_num = mdates.date2num(plot_df['played_at'])
    if len(plot_df) > 3:
        degree = min(3, len(plot_df) - 1)
        z = np.polyfit(x_num, plot_df['differential'], degree)
        p = np.poly1d(z)
        ax1.plot(plot_df['played_at'], p(x_num), linestyle='--', linewidth=2,
                 alpha=0.7, color='black', label='Differential trend')

    ax1.set_xlabel("Date", fontsize=20, labelpad=12)
    ax1.set_ylabel("Score Differential", fontsize=20, labelpad=12, color='black')
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12, colors='black')

    # Handicap index on secondary y-axis
    ax2 = ax1.twinx()
    valid_hc = plot_df.dropna(subset=['handicap'])
    if not valid_hc.empty:
        ax2.plot(valid_hc['played_at'], valid_hc['handicap'], color='#2563eb',
                 linewidth=2.5, label='Handicap Index')
        ax2.set_ylabel("Handicap Index", fontsize=20, labelpad=12, color='#2563eb')
        ax2.tick_params(axis='y', labelsize=12, colors='#2563eb')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
               fontsize=12, frameon=False)

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_base64


# CREATE DASH APP
app = Dash(__name__)

# After loading model_df, generate the clustering plot statically
if not model_df.empty:
    # Prep the data - use only numeric columns and exclude derived/redundant columns
    exclude_columns = ['date_column', 'Year', 'Month_Num', 'Day', 'DayOfYear', 'nap', 'total',
                       'ovr_pirates', 'ovr_guardians', 'ovr_other', 'cycle_id']
    # Exclude Whoop metadata/timing columns but keep score_ columns (sleep, recovery, strain, HR, HRV)
    exclude_columns += [col for col in model_df.columns if col.startswith('updated_at_')]
    exclude_columns += [col for col in model_df.columns if col.endswith('_minutes_into_day')]
    # Exclude duplicated zone duration columns (keep zone_duration_, drop zone_durations_)
    exclude_columns += [col for col in model_df.columns if 'zone_durations_' in col]
    # Exclude no-data and percent_recorded metadata
    exclude_columns += [col for col in model_df.columns if 'no_data' in col or 'percent_recorded' in col]
    numeric_columns = model_df.select_dtypes(include=['float64', 'int64']).columns
    df_numeric = model_df[numeric_columns].drop(columns=exclude_columns, errors='ignore')
    df_numeric = df_numeric.loc[:, (df_numeric != 0).any(axis=0)]
    df_numeric = df_numeric.loc[:, df_numeric.notna().any(axis=0)]
    df_numeric = df_numeric.fillna(0)
    save_df = df_numeric.copy()
    save_df['date_column'] = model_df['date_column']
    save_df.to_csv(os.path.join(os.path.dirname(__file__), '..', 'Data', 'df_numeric_for_clustering.csv'))
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

    # Data-driven n_components via PCA elbow detection
    n_components, pca_full, elbow_plot_base64 = find_optimal_n_components(df_scaled)
    print(f"Optimal n_components from elbow analysis: {n_components}")

    # PCA feature loadings (data-driven n_components)
    pca = PCA(n_components=n_components, random_state=2225)
    pca.fit(df_scaled)
    pca_feature_importance_images = plot_pca_feature_importance(pca, df_numeric.columns, n_components=n_components)

    # PHATE embedding
    phate_op = phate.PHATE(n_components=n_components, random_state=2225, n_jobs=-1)
    phate_results = phate_op.fit_transform(df_scaled)

    # Add Year for scatter plot coloring
    df_scaled['date_column'] = pd.to_datetime(model_df['date_column'], errors='coerce', utc=True)
    df_scaled['Year'] = df_scaled['date_column'].dt.year

    # PHATE scatter plots (pairwise projections)
    clustering_image_base64 = plot_2d_embedding_subplots(df_scaled, phate_results, n_components, method_name='PHATE')

    # Feature-PHATE correlation heatmap
    phate_correlation_image_base64, phate_corr_df, phate_pval_df = plot_feature_phate_correlations(
        df_scaled[df_numeric.columns], phate_results, n_components
    )

    unmatched_cols = [c for c in df_numeric.columns if _bucket_for(c) == 'other']
    if unmatched_cols:
        print(f"[PHATE bucketing] {len(unmatched_cols)} columns fell into 'other' — "
              f"add keywords to _DOMAIN_BUCKETS to classify them:")
        for c in unmatched_cols:
            print(f"  - {c}")

    # PHATE dimension descriptions
    phate_dimension_descriptions = describe_phate_dimensions(phate_corr_df, phate_pval_df)

    # Recurrence plot and RQA metrics
    recurrence_plot_base64 = plot_recurrence(df_scaled[df_numeric.columns], dates=model_df['date_column'].reset_index(drop=True))
    rqa_metrics = compute_rqa_metrics(df_scaled[df_numeric.columns])
    rqa_interpretation = interpret_rqa(rqa_metrics)

    # Drop columns with more than 40% missing data
    missing_fraction = df_numeric.isnull().mean()
    df_numeric = df_numeric.loc[:, missing_fraction <= 0.4]
else:
    clustering_image_base64 = generate_placeholder_image("No clustering data available.")
    pca_feature_importance_images = []
    elbow_plot_base64 = generate_placeholder_image("No elbow plot available.")
    phate_correlation_image_base64 = generate_placeholder_image("No PHATE correlation data available.")
    phate_dimension_descriptions = []
    recurrence_plot_base64 = generate_placeholder_image("No recurrence plot available.")
    rqa_metrics = {}
    rqa_interpretation = []

app.layout = html.Div(children=[
    dcc.Tabs(id='tabs', value='OVR Data', children=[
        # OVR Charts
        dcc.Tab(label='OVR Data', value='OVR Data', children=[
            # OVR Data
            html.Div(children='', style={'textAlign': 'center'}),
            html.Img(
                src=f'data:image/png;base64,{create_cr_all_plot()}',
                style={'display': 'block', 'width': '90%',
                       'margin-right': '2.5%', 'margin-left': '2.5%'}
            ),

            # Detailed Activities OVR and Week-Over-Week
            html.Div(children=[
                html.Img(
                    src=f'data:image/png;base64,{create_avg_ovr_slopes()}',
                    style={'display': 'inline-block',
                           'width': '45%', 'margin-right': '2.5%'}
                ),
                html.Img(
                    src=f'data:image/png;base64,{create_avg_ovr_change()}',
                    style={'display': 'inline-block',
                           'width': '45%', 'margin-left': '2.5%'}
                )
            ], style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center'}),

            # Leisure image centered
            html.Div(children='', style={'textAlign': 'center'}),
            html.Img(
                src=f'data:image/png;base64,{create_journal_plot()}',
                style={'display': 'block', 'width': '90%',
                       'margin-right': '2.5%', 'margin-left': '2.5%'}
            ),
        ]),

        # Clustering data
        dcc.Tab(label='Data Clustering', value='Data Clustering', children=[
            html.Div(children=[
                html.H3("PCA Elbow Analysis", style={'textAlign': 'center'}),
                html.Img(
                    src=f'data:image/png;base64,{elbow_plot_base64}',
                    style={'display': 'block', 'width': '60%', 'margin': '0 auto', 'margin-bottom': '4rem'}
                ),
                html.H3("PHATE Embedding Projections", style={'textAlign': 'center'}),
                html.Img(
                    src=f'data:image/png;base64,{clustering_image_base64}',
                    style={'display': 'block', 'width': '90%', 'margin': '0 auto', 'margin-bottom': '6rem'}
                ),
                html.H3("PCA Feature Loadings", style={'textAlign': 'center'}),
                *[html.Img(
                    src=f'data:image/png;base64,{img}',
                    style={'display': 'block', 'width': '100%', 'margin': '0 auto', 'margin-bottom': '3rem'}
                ) for img in pca_feature_importance_images],
                html.H3("Feature-PHATE Dimension Correlations", style={'textAlign': 'center'}),
                html.Img(
                    src=f'data:image/png;base64,{phate_correlation_image_base64}',
                    style={'display': 'block', 'width': '80%', 'margin': '0 auto', 'margin-bottom': '2rem'}
                ),
                html.H3("PHATE Dimension Descriptions", style={'textAlign': 'center'}),
                html.P("Only features that load more strongly on a given dimension than on any other (i.e., distinctive to it) are shown.",
                       style={'textAlign': 'center', 'fontSize': 14, 'color': 'gray', 'marginBottom': '1rem'}),
                dash_table.DataTable(
                    data=phate_dimension_descriptions,
                    columns=[
                        {'name': 'Dimension', 'id': 'Dimension'},
                        {'name': 'High-score days', 'id': 'High-score days'},
                        {'name': 'Low-score days', 'id': 'Low-score days'},
                        {'name': 'Narrative', 'id': 'Narrative'},
                    ],
                    style_table={'width': '90%', 'margin': '0 auto', 'marginBottom': '6rem'},
                    style_cell={'textAlign': 'left', 'fontSize': 14, 'padding': '10px',
                                'whiteSpace': 'normal', 'maxWidth': '300px'},
                    style_header={'fontWeight': 'bold', 'fontSize': 16, 'textAlign': 'center'},
                    style_data_conditional=[
                        {'if': {'column_id': 'Dimension'}, 'fontWeight': 'bold', 'textAlign': 'center', 'width': '80px'},
                        {'if': {'column_id': 'Narrative'}, 'fontStyle': 'italic'},
                    ],
                ) if phate_dimension_descriptions else html.P("No dimension descriptions available."),
                html.H3("Recurrence Plot", style={'textAlign': 'center'}),
                html.Img(
                    src=f'data:image/png;base64,{recurrence_plot_base64}',
                    style={'display': 'block', 'width': '80%', 'margin': '0 auto', 'margin-bottom': '4rem'}
                ),
                html.H3("Recurrence Quantification Analysis", style={'textAlign': 'center'}),
                dash_table.DataTable(
                    data=[{'Metric': k, 'Value': f'{v:.4f}'} for k, v in rqa_metrics.items()] if rqa_metrics else [],
                    columns=[{'name': 'Metric', 'id': 'Metric'}, {'name': 'Value', 'id': 'Value'}],
                    style_table={'width': '50%', 'margin': '0 auto', 'marginBottom': '2rem'},
                    style_cell={'textAlign': 'center', 'fontSize': 16, 'padding': '8px'},
                    style_header={'fontWeight': 'bold', 'fontSize': 18},
                ),
                html.Div(children=[
                    html.H4("Interpretation", style={'textAlign': 'left', 'marginBottom': '1rem'}),
                    html.Ul([
                        html.Li(line, style={'fontSize': 15, 'marginBottom': '0.5rem', 'textAlign': 'left'})
                        for line in rqa_interpretation
                    ]) if rqa_interpretation else html.P("No interpretation available."),
                ], style={'width': '70%', 'margin': '0 auto', 'marginBottom': '6rem'}),
            ], style={'textAlign': 'center'})
        ]),

        # Baseball Data
        dcc.Tab(label="Baseball Watched", children=[
            html.Div(children='', style={'textAlign': 'center'}),
            html.Img(
                src=f'data:image/png;base64,{create_cr_baseball()}',
                style={'display': 'block', 'width': '90%',
                       'margin-right': '2.5%', 'margin-left': '2.5%'}
            ),

            html.Div(children='', style={'textAlign': 'center'}),
            html.Img(
                src=f'data:image/png;base64,{plot_games_year()}',
                style={'display': 'block', 'width': '60%',
                       'margin-right': 'auto', 'margin-left': 'auto'}
            ),
        ]),

        # Golf Data
        dcc.Tab(label="Golf", value="Golf", children=[
            html.Img(
                src=f'data:image/png;base64,{plot_cumulative_rounds_watched()}',
                style={'display': 'block', 'width': '90%',
                       'margin-right': '2.5%', 'margin-left': '2.5%',
                       'margin-bottom': '3rem'}
            ),
            html.Img(
                src=f'data:image/png;base64,{plot_rounds_played_monthly()}',
                style={'display': 'block', 'width': '60%',
                       'margin-right': 'auto', 'margin-left': 'auto',
                       'margin-bottom': '3rem'}
            ),
            html.Img(
                src=f'data:image/png;base64,{plot_golf_scores_handicap()}',
                style={'display': 'block', 'width': '90%',
                       'margin-right': '2.5%', 'margin-left': '2.5%'}
            ),
        ]),

        # Food Tracking (loads pre-generated plot images from plots/ dir)
        dcc.Tab(label="Food Tracking", value="Food Tracking", children=[
            html.Div(children=[
                html.Img(
                    src=f'data:image/png;base64,{Path(Path(__file__).resolve().parent.parent / "plots" / "food_daily_calories.b64").read_text() if (Path(__file__).resolve().parent.parent / "plots" / "food_daily_calories.b64").exists() else ""}',
                    style={'display': 'block', 'width': '90%',
                           'margin-right': '2.5%', 'margin-left': '2.5%',
                           'margin-bottom': '3rem'}
                ) if (Path(__file__).resolve().parent.parent / "plots" / "food_daily_calories.b64").exists() else html.P("Run daily_food_update.py to generate plots.", style={'textAlign': 'center', 'color': '#888'}),
                html.Img(
                    src=f'data:image/png;base64,{Path(Path(__file__).resolve().parent.parent / "plots" / "food_daily_macros.b64").read_text() if (Path(__file__).resolve().parent.parent / "plots" / "food_daily_macros.b64").exists() else ""}',
                    style={'display': 'block', 'width': '90%',
                           'margin-right': '2.5%', 'margin-left': '2.5%',
                           'margin-bottom': '3rem'}
                ) if (Path(__file__).resolve().parent.parent / "plots" / "food_daily_macros.b64").exists() else html.Div(),
                html.Img(
                    src=f'data:image/png;base64,{Path(Path(__file__).resolve().parent.parent / "plots" / "food_fat_trend.b64").read_text() if (Path(__file__).resolve().parent.parent / "plots" / "food_fat_trend.b64").exists() else ""}',
                    style={'display': 'block', 'width': '90%',
                           'margin-right': '2.5%', 'margin-left': '2.5%',
                           'margin-bottom': '3rem'}
                ) if (Path(__file__).resolve().parent.parent / "plots" / "food_fat_trend.b64").exists() else html.Div(),
                html.Img(
                    src=f'data:image/png;base64,{Path(Path(__file__).resolve().parent.parent / "plots" / "food_micronutrients.b64").read_text() if (Path(__file__).resolve().parent.parent / "plots" / "food_micronutrients.b64").exists() else ""}',
                    style={'display': 'block', 'width': '90%',
                           'margin-right': '2.5%', 'margin-left': '2.5%',
                           'margin-bottom': '3rem'}
                ) if (Path(__file__).resolve().parent.parent / "plots" / "food_micronutrients.b64").exists() else html.Div(),
                html.Img(
                    src=f'data:image/png;base64,{Path(Path(__file__).resolve().parent.parent / "plots" / "food_symptom_frequency.b64").read_text() if (Path(__file__).resolve().parent.parent / "plots" / "food_symptom_frequency.b64").exists() else ""}',
                    style={'display': 'block', 'width': '60%',
                           'margin-right': 'auto', 'margin-left': 'auto',
                           'margin-bottom': '3rem'}
                ) if (Path(__file__).resolve().parent.parent / "plots" / "food_symptom_frequency.b64").exists() else html.Div(),
                html.Img(
                    src=f'data:image/png;base64,{Path(Path(__file__).resolve().parent.parent / "plots" / "food_symptom_timeline.b64").read_text() if (Path(__file__).resolve().parent.parent / "plots" / "food_symptom_timeline.b64").exists() else ""}',
                    style={'display': 'block', 'width': '90%',
                           'margin-right': '2.5%', 'margin-left': '2.5%'}
                ) if (Path(__file__).resolve().parent.parent / "plots" / "food_symptom_timeline.b64").exists() else html.Div(),
            ])
        ]),

        # Physical Activity
        dcc.Tab(label="Physical Activity", value="Physical Activity", children=[
            # Run and Yoga Data
            html.Div(children=[
                html.Img(
                    src=f'data:image/png;base64,{plot_phys_act_counts()}',
                    style={'display': 'inline-block',
                           'width': '45%', 'margin-right': '2.5%'}
                ),
                html.Img(
                    src=f'data:image/png;base64,{plot_yoga()}',
                    style={'display': 'inline-block',
                           'width': '45%', 'margin-left': '2.5%'}
                )
            ], style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center'}),
            html.Div(children=[
                html.Img(
                    src=f'data:image/png;base64,{plot_run_miles()}',
                    style={'display': 'inline-block',
                           'width': '45%', 'margin-right': '2.5%'}
                ),
                html.Img(
                    src=f'data:image/png;base64,{plot_run_mins()}',
                    style={'display': 'inline-block',
                           'width': '45%', 'margin-left': '2.5%'}
                )
            ], style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center'}),
            html.Div(children=[
                html.Img(
                    src=f'data:image/png;base64,{plot_run_mins_mile()}',
                    style={'display': 'inline-block',
                           'width': '90%', 'margin-right': '2.5%'}
                ),
            ], style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center'}),
            html.Div(children=[
                html.Img(
                    src=f'data:image/png;base64,{plot_run_mins_mile_month()}',
                    style={'display': 'inline-block',
                           'width': '90%', 'margin-left': '2.5%'}
                )
            ], style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center'}),
        ]),

        # Whoop Data
        dcc.Tab(label="Sleep Data", children=[
            html.Div(children=[
                html.Img(
                    src=f'data:image/png;base64,{plot_whoop_sleep_stages()}',
                    style={'display': 'inline-block', 'width': '45%', 'margin-right': '2.5%'}
                ),
                html.Img(
                    src=f'data:image/png;base64,{plot_whoop_sleep_proportions()}',
                    style={'display': 'inline-block', 'width': '45%', 'margin-left': '2.5%'}
                )
            ], style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center', 'margin-bottom': '50px'}),
            html.Div(children=[
                html.Img(
                    src=f'data:image/png;base64,{plot_whoop_sleep_needs()}',
                    style={'display': 'inline-block', 'width': '45%', 'margin-right': '2.5%'}
                ),
                html.Img(
                    src=f'data:image/png;base64,{plot_whoop_sleep_percentages()}',
                    style={'display': 'inline-block', 'width': '45%', 'margin-left': '2.5%'}
                )
            ], style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center', 'margin-bottom': '50px'}),
            html.Div(children=[
                html.Img(
                    src=f'data:image/png;base64,{plot_whoop_respiratory_rate()}',
                    style={'display': 'block', 'width': '45%', 'margin': '0 auto'}
                )
            ], style={'textAlign': 'center'}),
        ]),

        # Recovery Data
        dcc.Tab(label="Recovery Data", children=[
            html.Div(children=[
                # First row with two plots
                html.Div(children=[
                    html.Img(
                        src=f'data:image/png;base64,{plot_whoop_recovery_timeseries()[0]}',
                        style={'display': 'inline-block', 'width': '45%', 'margin-right': '2.5%'}
                    ),
                    html.Img(
                        src=f'data:image/png;base64,{plot_whoop_recovery_timeseries()[1]}',
                        style={'display': 'inline-block', 'width': '45%', 'margin-left': '2.5%'}
                    )
                ], style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center', 'margin-bottom': '20px'}),
                
                # Second row with two plots
                html.Div(children=[
                    html.Img(
                        src=f'data:image/png;base64,{plot_whoop_recovery_timeseries()[2]}',
                        style={'display': 'inline-block', 'width': '45%', 'margin-right': '2.5%'}
                    ),
                    html.Img(
                        src=f'data:image/png;base64,{plot_whoop_recovery_timeseries()[3]}',
                        style={'display': 'inline-block', 'width': '45%', 'margin-left': '2.5%'}
                    )
                ], style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center', 'margin-bottom': '20px'}),
                
                # Third row with one plot
                html.Div(children=[
                    html.Img(
                        src=f'data:image/png;base64,{plot_whoop_recovery_timeseries()[4]}',
                        style={'display': 'block', 'width': '45%', 'margin': '0 auto'}
                    )
                ], style={'textAlign': 'center'})
            ])
        ]),
        dcc.Tab(label="Cycle Data", children=[
            html.Div(children=[
                # First row: Strain and Kilojoules side by side
                html.Div(children=[
                    html.Img(
                        src=f'data:image/png;base64,{plot_whoop_cycle_strain()}',
                        style={'display': 'inline-block', 'width': '45%', 'margin-right': '2.5%'}
                    ),
                    html.Img(
                        src=f'data:image/png;base64,{plot_whoop_cycle_kilojoule()}',
                        style={'display': 'inline-block', 'width': '45%', 'margin-left': '2.5%'}
                    ),
                ], style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center', 'margin-bottom': '40px'}),
                # Second row: Heart Rates centered
                html.Div(children=[
                    html.Img(
                        src=f'data:image/png;base64,{plot_whoop_cycle_heartrates()}',
                        style={'display': 'block', 'width': '45%', 'margin': '0 auto'}
                    )
                ], style={'textAlign': 'center'})
            ], style={'textAlign': 'center'})
        ]),

        # Book Reading
        dcc.Tab(label="Books Read", children=[
            html.Div(children=[
                html.Img(
                    src=f'data:image/png;base64,{pages_per_day()}',
                    style={'display': 'inline-block',
                           'width': '45%', 'margin-right': '2.5%'}
                ),
                html.Img(
                    src=f'data:image/png;base64,{books_annually()}',
                    style={'display': 'inline-block',
                           'width': '45%', 'margin-left': '2.5%'}
                )
            ], style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center'}),
            html.Div(children='', style={'textAlign': 'center'}),
            html.Img(
                src=f'data:image/png;base64,{plot_pages()}',
                style={'display': 'block', 'width': '45%',
                       'margin-right': 'auto', 'margin-left': 'auto'}
            ),
            html.Div(children=[
                dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in table_data.columns],
                    data=table_data.to_dict('records'),
                    style_table={'height': 'auto', 'width': 'auto',
                                 'overflowY': 'auto', 'margin': 'auto'},
                    style_cell={
                        'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'fontSize': '12px'},
                    style_header={'fontWeight': 'bold', 'fontSize': '14px'}
                )
            ], style={'textAlign': 'center', 'display': 'flex', 'width': '90%', 'justify-content': 'center'}),
        ]),

        # CV Additions
        dcc.Tab(label="CV Additions", children=[
            html.Div(children=[
                html.Img(
                    src=f'data:image/png;base64,{total_cv_adds()}',
                    style={'display': 'inline-block',
                           'width': '45%', 'margin-right': '2.5%'}
                ),
                html.Img(
                    src=f'data:image/png;base64,{plot_research_cv()}',
                    style={'display': 'inline-block',
                           'width': '45%', 'margin-left': '2.5%'}
                )
            ], style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center'}),
            html.Div(children=[
                html.Img(
                    src=f'data:image/png;base64,{plot_dissem_cv()}',
                    style={'display': 'inline-block',
                           'width': '45%', 'margin-right': '2.5%'}
                ),
                html.Img(
                    src=f'data:image/png;base64,{plot_service_cv()}',
                    style={'display': 'inline-block',
                           'width': '45%', 'margin-left': '2.5%'}
                )
            ], style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center'}),
        ]),

        # Publication Data
        dcc.Tab(label="Publication Stats", children=[
            html.Div(children=[
                html.Img(
                    src=f'data:image/png;base64,{plot_cum_pages()}',
                    style={'display': 'inline-block',
                           'width': '45%', 'margin-right': '2.5%'}
                ),
                html.Img(
                    src=f'data:image/png;base64,{plot_cum_words()}',
                    style={'display': 'inline-block',
                           'width': '45%', 'margin-left': '2.5%'}
                )
            ], style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center'}),
            html.Div(children=[
                html.Img(
                    src=f'data:image/png;base64,{plot_pages_year()}',
                    style={'display': 'inline-block',
                           'width': '45%', 'margin-right': '2.5%'}
                ),
                html.Img(
                    src=f'data:image/png;base64,{plot_words_year()}',
                    style={'display': 'inline-block',
                           'width': '45%', 'margin-left': '2.5%'}
                )
            ], style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center'}),
            html.Div(children='', style={'textAlign': 'center'}),
            html.Img(
                src=f'data:image/png;base64,{plot_cum_journals()}',
                style={'display': 'block', 'width': '45%',
                       'margin-right': 'auto', 'margin-left': 'auto'}
            ),
        ]),

    ]),
])

def open_browser():
    webbrowser.open_new("http://localhost:8051/")

if __name__ == '__main__':
    Timer(1, open_browser).start()  # Open the browser after 1 second
    app.run(debug=False, port=8051)