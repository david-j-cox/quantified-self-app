#!/usr/bin/env python
# coding: utf-8

# Data manipulation
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os
import warnings
from IPython.display import display, HTML
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
import umap
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler

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

# Preferences
pd.options.display.max_columns = None
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# PostgreSQL database credentials from .env
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Create a database engine
DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
engine = create_engine(DATABASE_URL)

# Function to read data from a table
def read_table(table_name):
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql_query(query, con=engine)

# Read in the raw data from the database
baseball = read_table('baseball_watched')
books = read_table('books_read')
cv_adds = read_table('cv_additions')
pubs = read_table('publication_stats')
all_df = read_table('raw_data')

# Whoop
cycle_collection = read_table('whoop_cycle_collection')
recoveries = read_table('whoop_recoveries')
sleep_collection = read_table('whoop_sleep')
workouts = read_table('whoop_workouts')

# Strava
phys_act = read_table('strava_activities')
phys_act = phys_act[['activitydate', 'activitytype', 'elapsedtime', 'distance']]

# Keep only numeric baseball cols
baseball = baseball[
    [col for col in baseball.columns if "tv" not in col and "in_person" not in col and "ovr" not in col and "season" not in col]
]
baseball = baseball.sort_values(by='date', ascending=True).reset_index(drop=True)

# Convert cumulative counts to binary indicators (1 if increased, 0 otherwise)
def convert_cumulative_to_binary(df, columns_to_convert):
    """
    Convert cumulative columns to binary indicators showing when values increase.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        columns_to_convert (list): List of column names to convert
    
    Returns:
        pd.DataFrame: DataFrame with binary columns
    """
    df = df.copy()
    
    for col in columns_to_convert:
        if col in df.columns:
            # Create binary column: 1 if value increased from previous row, 0 otherwise
            df[f'{col}'] = (df[col] > df[col].shift(1)).astype(int)
            # First row will be NaN, set to 0
            df[f'{col}'] = df[f'{col}'].fillna(0)
    
    return df

# Convert the cumulative columns to binary
columns_to_convert = ['total', 'pirates', 'guardians', 'other']
baseball = convert_cumulative_to_binary(baseball, columns_to_convert)
baseball['total'] = baseball[['pirates', 'guardians', 'other']].sum(axis=1)

# Cleanup Whoop data
def convert_to_minutes(df, datetime_cols):
    """
    Converts specified datetime columns in a DataFrame to the number of minutes into the day.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        datetime_cols (list): List of column names containing datetime values.
    
    Returns:
        pd.DataFrame: Modified DataFrame with columns converted to minutes into the day.
    """
    df = df.copy()
    for col in datetime_cols:
        df = df.dropna(subset=[col]).reset_index(drop=True)
        try:
            # Handle timezone-aware datetime objects properly
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')  # Convert to datetime, coercing errors
            df[col] = df[col].dt.tz_localize(None)  # Remove timezone info
            df = df.dropna(subset=[col])  # Drop rows where conversion failed
            df[f"{col}_minutes_into_day"] = df[col].dt.hour * 60 + df[col].dt.minute
            df[f"{col}_minutes_into_day"] = df[f"{col}_minutes_into_day"].astype(float).round(2)
        except Exception as e:
            print(f"Error converting column {col}: {e}")
    
    return df

# Cycle collections
datetime_cols = ['updated_at', 'cycle_start', 'cycle_end']
cycle_collection = convert_to_minutes(cycle_collection, datetime_cols)
cycle_collection = cycle_collection.drop(datetime_cols + ['id', 'user_id', 'timezone_offset', 'score_state'], axis=1)
cycle_collection['created_at'] = pd.to_datetime(cycle_collection['created_at'], utc=True).dt.tz_localize(None).dt.date

# Recoveries
datetime_cols = ['updated_at']
recoveries = convert_to_minutes(recoveries, datetime_cols)
recoveries = recoveries.drop(datetime_cols + ['cycle_id', 'sleep_id', 'user_id', 'score_state', 'score_user_calibrating'], axis=1)
recoveries['created_at'] = pd.to_datetime(recoveries['created_at'], utc=True).dt.tz_localize(None).dt.date

# Sleep Collections
datetime_cols = ['updated_at', 'sleep_start', 'sleep_end', ]
sleep_collection = convert_to_minutes(sleep_collection, datetime_cols)
sleep_collection = sleep_collection.drop(datetime_cols + ['id', 'user_id', 'timezone_offset', 'score_state'], axis=1)
sleep_collection['nap'] = [0 if val=="false" else 1 for val in sleep_collection['nap']]
sleep_collection['created_at'] = pd.to_datetime(sleep_collection['created_at'], utc=True).dt.tz_localize(None).dt.date

# Workouts
datetime_cols = ['updated_at', 'workout_start', 'workout_end', ]
workouts = convert_to_minutes(workouts, datetime_cols)
workouts = workouts.drop(datetime_cols + ['id', 'user_id', 'timezone_offset', 'score_state'], axis=1)
workouts['created_at'] = pd.to_datetime(workouts['created_at'], utc=True).dt.tz_localize(None).dt.date

# Normalize date to remove varied time listings
phys_act = phys_act.drop_duplicates().reset_index(drop=True)
phys_act['year'] = pd.to_datetime(phys_act['activitydate']).dt.year.astype(str)
phys_act['month'] = pd.to_datetime(phys_act['activitydate']).dt.month.astype(str)
phys_act['day'] = pd.to_datetime(phys_act['activitydate']).dt.day.astype(str)
phys_act['date'] = pd.to_datetime(phys_act['year']+"-"+phys_act['month']+"-"+phys_act['day'])
phys_act = phys_act.drop(['activitydate', 'year', 'month', 'day'], axis=1)
phys_act = phys_act.drop_duplicates(subset=['date', 'activitytype', 'elapsedtime']).reset_index(drop=True)

# Ensure 'date' is in datetime format and normalized (if not already)
phys_act['date'] = pd.to_datetime(phys_act['date']).dt.normalize()

# Pivot the table to wide format for all dates at once
wide_phys_act = (
    phys_act
    .pivot_table(
        index='date',                  # Use 'date' as the index
        columns='activitytype',       # Pivot on 'Activity Type'
        aggfunc='sum',                 # Aggregate by summing values
    )
)

# Flatten MultiIndex columns and clean column names
wide_phys_act.columns = [(f"{col[0]}_{col[1]}").replace(" ", "_").lower() for col in wide_phys_act.columns]

# Reset index to move 'date' back as a column
wide_phys_act = wide_phys_act.reset_index()

# Replace NaNs with 0
wide_phys_act = wide_phys_act.fillna(0)

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

# Only include data from new year if it is at least three weeks into it
# if (datetime.date.today().month == 1) & (datetime.date.today().day < 21):
#     all_data = all_data[all_data['Year'] < datetime.date.today().year]

# Copy all_data into model_df
model_df = all_data.copy()
model_df['date_column'] = model_df['date_column'].astype(str)

# Define date columns for different dataframes
date_cols = {
    'baseball': ('date', baseball), 
    'cycle_collection': ('created_at', cycle_collection),
    'recoveries': ('created_at', recoveries),
    'sleep_collection': ('created_at', sleep_collection),
    'workouts': ('created_at', workouts),
    'phys_act': ('date', wide_phys_act)
}

# Process each dataframe: convert to datetime, remove time, and set index
cleaned_dfs = pd.DataFrame()
for name, (date_col, df) in date_cols.items():
    temp_df = df.copy()
    
    # Handle timezone-aware datetime objects properly
    if name in ['cycle_collection', 'recoveries', 'sleep_collection', 'workouts']:
        # For Whoop data, convert to UTC first, then normalize
        temp_df['date_column'] = pd.to_datetime(temp_df[date_col], utc=True).dt.tz_localize(None).dt.normalize()
    else:
        # For other data, convert normally
        temp_df['date_column'] = pd.to_datetime(temp_df[date_col]).dt.normalize()
    
    temp_df['date_column'] = temp_df['date_column'].astype(str)
    temp_df = temp_df.drop(date_col, axis=1)
    
    if len(cleaned_dfs)==0:
        cleaned_dfs = pd.concat([cleaned_dfs, temp_df])
    else:
        cleaned_dfs = pd.merge(
            left=cleaned_dfs, 
            right=temp_df, 
            left_on='date_column', 
            right_on='date_column', 
            how='outer', 
            suffixes=['', f'_{name}']
        )

# Merge with all_data
model_df = pd.merge(
    left=model_df, 
    right=cleaned_dfs, 
    left_on='date_column', 
    right_on='date_column', 
    how='outer', 
)

# Reset index if needed
model_df = model_df.drop(['Month'], axis=1).reset_index(drop=True)

# Convert day of the week to numeric
day_to_number = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}
model_df['Day'] = model_df['Day'].map(day_to_number)

# Drop duplicates
model_df = model_df.drop_duplicates(subset=['date_column'], keep="first").reset_index(drop=True)

# Push to Database
model_df.to_sql('modeling_ready_data', engine, if_exists='replace', index=False)
model_df.to_csv('../Data/modeling_data.csv', index=False)