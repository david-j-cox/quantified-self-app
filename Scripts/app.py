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
from datetime import datetime, date
import random

# Data viz
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from matplotlib.ticker import FuncFormatter

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
    ax.tick_params(axis='x', labelsize=14, rotation=45)
    ax.tick_params(axis='y', labelsize=14)
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
    legend = plt.legend(frameon=False, loc="upper center", fontsize=16)
    legend.get_frame().set_facecolor('white')

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    plt.close()

    return image_base64


def create_avg_ovr_slopes():
    # Display the DataFrame
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.lineplot(data=all_data_melted, x='Week', y='Value',
                 hue='Category')  # Corrected lineplot usage
    plt.xlabel("Week", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Weekly AVG of Mins Tracked Daily", fontsize=20, labelpad=12)
    plt.yticks(fontsize=16)
    plt.ylim(0, 1800)
    sns.despine(top=True, right=True)
    plt.legend(frameon=False, loc="best", fontsize=16)

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
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

    # Ensure the correct column name is used
    if 'distance_Run' in plot_df.columns:
        plot_df['distance_Run'] = plot_df['distance_Run']
    else:
        plot_df['distance_Run'] = plot_df['Run'].cumsum()

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
    fig, ax = plt.subplots(figsize=(10, 8))
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
    fig, ax = plt.subplots(figsize=(10, 8))
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
    plot_books = (books.groupby(by=['year_read'])[
                  'pages'].sum()/365).reset_index()
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
    plot_books = (books.groupby(by=['year_read'])[
                  'prop_read'].sum()).reset_index()
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
    plt.legend(loc="best", fontsize=16, frameon=False)

    # Add data labels
    for i in range(len(plot_df)):
        ax.annotate(f'{plot_df["value"].iloc[i]:,.0f}',
                    (plot_df['Year'].iloc[i]+random.uniform(-0.3, 0.3),
                     plot_df['value'].iloc[i]+random.uniform(0, 5)),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=16)

    # Save the plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
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
            y = metric_df['Percentage']
            z = np.polyfit(x, y, 4)  # degree 4 polynomial for smooth trend
            p = np.poly1d(z)
            ax.plot(metric_df['sleep_start'], p(x), linestyle='--', linewidth=2, alpha=0.7, color=color_dict[metric])
    plt.xlabel("Date", fontsize=26, labelpad=12)
    plt.xticks(fontsize=16)
    plt.ylabel("Sleep Percentage Scores", fontsize=26, labelpad=12)
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


def plot_2d_tsne_subplots(df, embedding_results, method_name='PCA'):
    """
    Creates six static 2D subplots showing different projections of the 3D embedding.
    First row: standard XY, XZ, YZ projections.
    Second row: same projections but with plotting order reversed to reveal hidden points.
    """
    temp_df = df.copy()
    temp_df[f'{method_name}1'] = embedding_results[:, 0]
    temp_df[f'{method_name}2'] = embedding_results[:, 1]
    temp_df[f'{method_name}3'] = embedding_results[:, 2]

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    # First row: standard projections
    sns.scatterplot(
        data=temp_df,
        x=f'{method_name}1',
        y=f'{method_name}2',
        hue='Year',
        palette='viridis',
        alpha=0.7,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('XY Projection', fontsize=24)
    axes[0, 0].set_xlabel(f'{method_name} Dimension 1', fontsize=18)
    axes[0, 0].set_ylabel(f'{method_name} Dimension 2', fontsize=18)

    sns.scatterplot(
        data=temp_df,
        x=f'{method_name}1',
        y=f'{method_name}3',
        hue='Year',
        palette='viridis',
        alpha=0.7,
        ax=axes[0, 1]
    )
    axes[0, 1].set_title('XZ Projection', fontsize=24)
    axes[0, 1].set_xlabel(f'{method_name} Dimension 1', fontsize=18)
    axes[0, 1].set_ylabel(f'{method_name} Dimension 3', fontsize=18)

    sns.scatterplot(
        data=temp_df,
        x=f'{method_name}2',
        y=f'{method_name}3',
        hue='Year',
        palette='viridis',
        alpha=0.7,
        ax=axes[0, 2]
    )
    axes[0, 2].set_title('YZ Projection', fontsize=24)
    axes[0, 2].set_xlabel(f'{method_name} Dimension 2', fontsize=18)
    axes[0, 2].set_ylabel(f'{method_name} Dimension 3', fontsize=18)

    # Second row: reversed plotting order to reveal hidden points
    for i, (x_dim, y_dim, title) in enumerate([
        (f'{method_name}1', f'{method_name}2', 'XY Projection (Back View)'),
        (f'{method_name}1', f'{method_name}3', 'XZ Projection (Back View)'),
        (f'{method_name}2', f'{method_name}3', 'YZ Projection (Back View)'),
    ]):
        # Sort by x_dim in descending order to reverse plotting order
        temp_df_sorted = temp_df.sort_values(by=x_dim, ascending=False).copy()
        sns.scatterplot(
            data=temp_df_sorted,
            x=x_dim,
            y=y_dim,
            hue='Year',
            palette='viridis',
            alpha=0.7,
            ax=axes[1, i]
        )
        axes[1, i].set_title(title, fontsize=24)
        axes[1, i].set_xlabel(f'{x_dim} (Back View)', fontsize=18)
        axes[1, i].set_ylabel(f'{y_dim} (Back View)', fontsize=18)

    # Remove duplicate legends
    for ax in axes.flatten():
        ax.legend().remove()
        sns.despine(ax=ax)

    # Add a single legend for all plots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Year', bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=20, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return image_base64

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
    import matplotlib.pyplot as plt
    import numpy as np
    from io import BytesIO
    import base64
    from matplotlib.patches import Patch
    # Compute max abs loading across all selected PCs for each feature
    loadings_matrix = np.abs(pca.components_[:n_components])
    max_abs_loadings = loadings_matrix.max(axis=0)
    # Filter features with at least one PC above threshold
    keep_idx = np.where(max_abs_loadings >= loading_threshold)[0]
    filtered_features = np.array(feature_names)[keep_idx]
    filtered_loadings = pca.components_[:, keep_idx]
    fig, axes = plt.subplots(n_components, 1, figsize=(max(14, len(filtered_features) * 0.4), 3.25 * n_components), sharex=True)
    if n_components == 1:
        axes = [axes]
    for i in range(n_components):
        loadings = filtered_loadings[i]
        sorted_idx = np.argsort(np.abs(loadings))[::-1]
        sorted_features = filtered_features[sorted_idx]
        sorted_loadings = loadings[sorted_idx]
        bar_colors = ['green' if val >= 0 else 'red' for val in sorted_loadings]
        axes[i].bar(sorted_features, np.abs(sorted_loadings), color=bar_colors)
        axes[i].set_title(f'PC{i+1} Feature Loadings', fontsize=22)
        axes[i].set_ylabel('Abs(Loading)', fontsize=18)
        axes[i].tick_params(axis='x', labelsize=18, rotation=90)
        axes[i].tick_params(axis='y', labelsize=18)
        if i == 1:
            legend_handles = [
                Patch(color='green', label='Positive Loading'),
                Patch(color='red', label='Negative Loading')
            ]
            axes[i].legend(handles=legend_handles, loc='upper right', fontsize=16)
    plt.xlabel('Feature', fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
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
                       'ovr_pirates', 'ovr_guardians', 'ovr_other']
    exclude_columns += [col for col in model_df.columns if col.startswith('score_')]
    exclude_columns += [col for col in model_df.columns if col.startswith('cycle_')]
    exclude_columns += [col for col in model_df.columns if col.startswith('sleep_')]
    exclude_columns += [col for col in model_df.columns if col.startswith('updated_at_')]
    exclude_columns += [col for col in model_df.columns if col.startswith('workout_')]
    numeric_columns = model_df.select_dtypes(include=['float64', 'int64']).columns
    df_numeric = model_df[numeric_columns].drop(columns=exclude_columns, errors='ignore')
    df_numeric = df_numeric.loc[:, (df_numeric != 0).any(axis=0)]
    df_numeric = df_numeric.loc[:, df_numeric.notna().any(axis=0)]
    df_numeric = df_numeric.fillna(0)
    save_df = df_numeric.copy()
    save_df['date_column'] = model_df['date_column']
    save_df.to_csv("../Data/df_numeric_for_clustering.csv")
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)
    # Use PCA instead of t-SNE or UMAP
    pca = PCA(n_components=3, random_state=2225)
    pca_results = pca.fit_transform(df_scaled)
    df_scaled['PCA1'] = pca_results[:, 0]
    df_scaled['PCA2'] = pca_results[:, 1]
    df_scaled['PCA3'] = pca_results[:, 2]
    df_scaled['Label'] = 'All Others'
    df_scaled.loc[df_scaled.tail(7).index, 'Label'] = 'Last 7 Days'
    df_scaled.loc[df_scaled.tail(30).head(23).index, 'Label'] = 'Last 30 Days'
    df_scaled['date_column'] = pd.to_datetime(model_df['date_column'], errors='coerce', utc=True)
    df_scaled['Year'] = df_scaled['date_column'].dt.year
    clustering_image_base64 = plot_2d_tsne_subplots(df_scaled, pca_results, method_name='PCA')
    # Add PCA feature importance plot
    pca_feature_importance_image_base64 = plot_pca_feature_importance(pca, df_numeric.columns, n_components=3)
    # Drop columns with more than 40% missing data
    missing_fraction = df_numeric.isnull().mean()
    df_numeric = df_numeric.loc[:, missing_fraction <= 0.4]
else:
    clustering_image_base64 = generate_placeholder_image("No clustering data available.")
    pca_feature_importance_image_base64 = generate_placeholder_image("No PCA feature importance available.")

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
                html.Img(
                    src=f'data:image/png;base64,{clustering_image_base64}',
                    style={'display': 'block', 'width': '90%', 'margin': '0 auto', 'margin-bottom': '6rem'}
                ),
                html.Img(
                    src=f'data:image/png;base64,{pca_feature_importance_image_base64}',
                    style={'display': 'block', 'width': '100%', 'margin': '0 auto', 'margin-top': '6rem'}
                )
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
                           'width': '45%', 'margin-right': '2.5%'}
                ),
                html.Img(
                    src=f'data:image/png;base64,{plot_run_mins_mile_month()}',
                    style={'display': 'inline-block',
                           'width': '45%', 'margin-left': '2.5%'}
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