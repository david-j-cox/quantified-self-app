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
import datetime
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
try:
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
    print("model_df shape:", model_df.shape if not model_df.empty else "Empty")
    print("model_df columns:", model_df.columns.tolist() if not model_df.empty else "Empty DataFrame")
except Exception as e:
    print("Error loading data:", str(e))
    import traceback
    traceback.print_exc()
    model_df = pd.DataFrame()  # Initialize as empty DataFrame if loading fails

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
if (datetime.date.today().month == 1) & (datetime.date.today().day < 21):
    all_data = all_data[all_data['Year'] < datetime.date.today().year]

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
    annual_pages['Monthly Pages'] = annual_pages['pages'] / 12

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
        datetime.date.today())].copy()
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


# CREATE DASH APP
app = Dash(__name__)

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

        # Clustering data
        dcc.Tab(label='Data Clustering', value='Data Clustering', children=[
            html.Div(children=[
                html.Img(
                    id='clustering-graph-xy',
                    style={'display': 'inline-block', 'width': '30%', 'margin': '1%'}
                ),
                html.Img(
                    id='clustering-graph-xz',
                    style={'display': 'inline-block', 'width': '30%', 'margin': '1%'}
                ),
                html.Img(
                    id='clustering-graph-yz',
                    style={'display': 'inline-block', 'width': '30%', 'margin': '1%'}
                )
            ], style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center'})
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
    ])
])

@app.callback(
    [Output('clustering-graph-xy', 'src'),
     Output('clustering-graph-xz', 'src'),
     Output('clustering-graph-yz', 'src')],
    Input('tabs', 'value')
)
def update_clustering_graphs(selected_tab):
    print("\n=== Debug: Clustering Tab ===")
    print("Selected tab:", selected_tab)
    
    if selected_tab == 'Data Clustering':
        print("Entering clustering tab section")
        
        # Debug model_df
        print("model_df exists:", 'model_df' in globals())
        print("model_df type:", type(model_df))
        print("model_df columns:", model_df.columns.tolist() if not model_df.empty else "Empty DataFrame")
        print("model_df shape:", model_df.shape if not model_df.empty else "Empty")
        
        # Check if model_df is populated
        if model_df.empty:
            print("model_df is empty")
            return '', '', ''
        
        try:
            # Prep the data - use only numeric columns and exclude derived/redundant columns
            exclude_columns = ['date_column', 'Year', 'Month_Num', 'Day', 'DayOfYear']
            numeric_columns = model_df.select_dtypes(include=['float64', 'int64']).columns
            print("Numeric columns:", numeric_columns.tolist())
            
            df_numeric = model_df[numeric_columns].drop(columns=exclude_columns, errors='ignore')
            print("df_numeric shape after exclusions:", df_numeric.shape)
            
            # Remove columns with all zeros or NaN
            df_numeric = df_numeric.loc[:, (df_numeric != 0).any(axis=0)]
            df_numeric = df_numeric.loc[:, df_numeric.notna().any(axis=0)]
            print("df_numeric shape after cleaning:", df_numeric.shape)
            print("Final features:", df_numeric.columns.tolist())

            # Fill NaN values with 0 (or you could use mean/median imputation)
            df_numeric = df_numeric.fillna(0)
            print("Shape after NaN handling:", df_numeric.shape)

            # Standardize using Min-Max Scaling
            scaler = MinMaxScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)
            print("Scaling completed")

            # t-SNE dimensionality reduction (3D)
            print("Starting t-SNE computation...")
            tsne = TSNE(
                n_components=3,
                random_state=2225,
                perplexity=30,
                n_iter=500,  # Reduced from 1000
                method='barnes_hut',  # Using the faster Barnes-Hut approximation
                angle=0.5,  # Default value for Barnes-Hut, higher values = faster but less accurate
                verbose=1
            )
            tsne_results = tsne.fit_transform(df_scaled)
            print("t-SNE computation completed")

            # Add t-SNE results back to the DataFrame
            df_scaled['t-SNE1'] = tsne_results[:, 0]
            df_scaled['t-SNE2'] = tsne_results[:, 1]
            df_scaled['t-SNE3'] = tsne_results[:, 2]

            # Label which days the data come from
            df_scaled['Label'] = 'All Others'
            df_scaled.loc[df_scaled.tail(7).index, 'Label'] = 'Last 7 Days'
            df_scaled.loc[df_scaled.tail(30).head(23).index, 'Label'] = 'Last 30 Days'

            # Add back in a date col for coloring
            df_scaled['date_column'] = pd.to_datetime(model_df['date_column'])
            df_scaled['Year'] = df_scaled['date_column'].dt.year

            def create_2d_projection(x, y, xlabel, ylabel, title):
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))

                    # Plot main scatter points colored by year
                    scatter = ax.scatter(
                        df_scaled[x],
                        df_scaled[y],
                        c=df_scaled['Year'],
                        cmap='viridis',
                        alpha=0.5,
                        s=50
                    )

                    # Add colorbar
                    plt.colorbar(scatter, label='Year')

                    # Plot last 30 days
                    last_30_days = df_scaled[df_scaled['Label'] == 'Last 30 Days']
                    if not last_30_days.empty:
                        ax.scatter(
                            last_30_days[x],
                            last_30_days[y],
                            color='blue',
                            alpha=0.8,
                            s=100,
                            marker='s',
                            label='Last 30 Days'
                        )

                    # Plot last 7 days
                    last_7_days = df_scaled[df_scaled['Label'] == 'Last 7 Days']
                    if not last_7_days.empty:
                        ax.scatter(
                            last_7_days[x],
                            last_7_days[y],
                            color='red',
                            alpha=0.8,
                            s=100,
                            marker='s',
                            label='Last 7 Days'
                        )

                    plt.title(title, fontsize=20, pad=20)
                    plt.xlabel(xlabel, fontsize=16)
                    plt.ylabel(ylabel, fontsize=16)
                    plt.legend(fontsize=12, frameon=False)
                    sns.despine(top=True, right=True)

                    # Save the plot to a bytes buffer
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                    plt.close()

                    return f'data:image/png;base64,{image_base64}'
                except Exception as e:
                    print(f"Error creating {title}:", str(e))
                    return ''

            print("Creating projections...")
            # Create three 2D projections
            xy_projection = create_2d_projection(
                't-SNE1', 't-SNE2', 
                't-SNE Dimension 1', 't-SNE Dimension 2',
                'XY Projection of Daily Activities'
            )
            print("XY projection created")
            
            xz_projection = create_2d_projection(
                't-SNE1', 't-SNE3',
                't-SNE Dimension 1', 't-SNE Dimension 3',
                'XZ Projection of Daily Activities'
            )
            print("XZ projection created")
            
            yz_projection = create_2d_projection(
                't-SNE2', 't-SNE3',
                't-SNE Dimension 2', 't-SNE Dimension 3',
                'YZ Projection of Daily Activities'
            )
            print("YZ projection created")

            return xy_projection, xz_projection, yz_projection
        
        except Exception as e:
            print("Error in clustering:", str(e))
            import traceback
            traceback.print_exc()
            return '', '', ''

    return '', '', ''

def open_browser():
    webbrowser.open_new("http://localhost:8051/")

if __name__ == '__main__':
    Timer(1, open_browser).start()  # Open the browser after 1 second
    app.run(debug=False, port=8051)