# import os
# from whoop import WhoopClient
# from dotenv import load_dotenv
# import pandas as pd
# from sqlalchemy import create_engine, text
# import requests
# import psycopg2
# import json
# from datetime import datetime
# load_dotenv()

# # Retrieve variables from environment
# WHOOP_EMAIL = os.getenv("WHOOP_EMAIL")
# WHOOP_PASSWORD = os.getenv("WHOOP_PASSWORD")
# DATABASE_URL = os.getenv('DB_URL')

# # Debug: Check if environment variables are loaded
# print(f"WHOOP_EMAIL loaded: {'Yes' if WHOOP_EMAIL else 'No'}")
# print(f"WHOOP_PASSWORD loaded: {'Yes' if WHOOP_PASSWORD else 'No'}")
# print(f"DATABASE_URL loaded: {'Yes' if DATABASE_URL else 'No'}")

# if not WHOOP_EMAIL or not WHOOP_PASSWORD:
#     print("ERROR: WHOOP_EMAIL or WHOOP_PASSWORD not found in environment variables!")
#     print("Please check your .env file contains:")
#     print("WHOOP_EMAIL=your_email@example.com")
#     print("WHOOP_PASSWORD=your_password")
#     exit(1)

# # Initialize database connection
# engine = create_engine(DATABASE_URL)

# # Function to get the latest `created_at` from a table
# def get_latest_created_at(table_name):
#     try:
#         query = f"SELECT MAX(created_at) AS latest_date FROM {table_name}"
#         with engine.connect() as conn:
#             result = conn.execute(text(query)).fetchone()
#             return result[0]
#     except:
#         return '2024-09-01'

# # Function to remove duplicates from a table
# def remove_duplicates(table_name, unique_column):
#     query = f"""
#     DELETE FROM {table_name}
#     WHERE {unique_column} IN (
#         SELECT {unique_column} FROM (
#             SELECT {unique_column}, ROW_NUMBER() OVER (
#                 PARTITION BY {unique_column} 
#                 ORDER BY created_at DESC
#             ) AS row_num
#             FROM {table_name}
#         ) subquery
#         WHERE row_num > 1
#     );
#     """
#     with engine.connect() as conn:
#         conn.execute(text(query))

# # List of table names to get latest `created_at` dates from
# table_names = {
#     'workouts': 'whoop_workouts',
#     'sleep_collection': 'whoop_sleep',
#     'recoveries': 'whoop_recoveries',
#     'cycle_collection': 'whoop_cycle_collection'
# }

# # Get latest `created_at` dates from the tables
# latest_dates = {
#     name: get_latest_created_at(table_name)
#     for name, table_name in table_names.items()
# }

# # Set default start_date if no data exists
# default_start_date = "2024-09-01"

# # Convert latest_dates[name] to a datetime object if it's a string
# start_dates = {
#     name: (pd.to_datetime(latest_dates[name]).strftime("%Y-%m-%d") if latest_dates[name] else default_start_date)
#     for name in table_names.keys()
# }

# print(f"Start dates for data collection: {start_dates}")

# # Initialize WhoopClient and fetch data since the latest `created_at`
# try:
#     print("Attempting to authenticate with Whoop...")
#     with WhoopClient(WHOOP_EMAIL, WHOOP_PASSWORD) as client:
#         print(f"\n\n{client}\n\n")
#         print("Successfully authenticated with Whoop!")
        
#         print("Fetching workout data...")
#         workouts = client.get_workout_collection(start_date=start_dates['workouts'])
#         print(f"Retrieved {len(workouts)} workouts")
        
#         print("Fetching sleep data...")
#         sleep_collection = client.get_sleep_collection(start_date=start_dates['sleep_collection'])
#         print(f"Retrieved {len(sleep_collection)} sleep records")
        
#         print("Fetching recovery data...")
#         recoveries = client.get_recovery_collection(start_date=start_dates['recoveries'])
#         print(f"Retrieved {len(recoveries)} recovery records")
        
#         print("Fetching cycle data...")
#         cycle_collection = client.get_cycle_collection(start_date=start_dates['cycle_collection'])
#         print(f"Retrieved {len(cycle_collection)} cycle records")

# except Exception as e:
#     print(f"ERROR during Whoop authentication or data fetching: {str(e)}")
#     print(f"Error type: {type(e).__name__}")
#     import traceback
#     traceback.print_exc()
#     exit(1)

# # Tweak cols to match SQL DB field names
# for i, df in enumerate([workouts, sleep_collection, recoveries, cycle_collection]):
#     df = pd.json_normalize(df)  # Normalize data
#     df.columns = [val.replace(".", "_") for val in list(df)]
    
#     # Adjust column names for each df
#     if i == 0: 
#         df.columns = [val.replace("start", "workout_start") if val == "start" else val for val in df.columns]
#         df.columns = [val.replace("end", "workout_end") if val == "end" else val for val in df.columns]
#     elif i ==1:
#         df.columns = [val.replace("start", "sleep_start") if val == "start" else val for val in df.columns]
#         df.columns = [val.replace("end", "sleep_end") if val == "end" else val for val in df.columns]
#     elif i ==3:
#         df.columns = [val.replace("start", "cycle_start") if val == "start" else val for val in df.columns]
#         df.columns = [val.replace("end", "cycle_end") if val == "end" else val for val in df.columns]
    
#     # Update the original DataFrame
#     if i == 0:
#         workouts = df
#     elif i == 1:
#         sleep_collection = df
#     elif i == 2:
#         recoveries = df
#     elif i == 3:
#         cycle_collection = df

# # TRANSFORM DATA BEFORE LOADING
# # Convert columns to match the schema for workouts
# workouts["created_at"] = pd.to_datetime(workouts["created_at"], utc=True)
# workouts["updated_at"] = pd.to_datetime(workouts["updated_at"], utc=True)
# workouts["workout_start"] = pd.to_datetime(workouts["workout_start"], utc=True)
# workouts["workout_end"] = pd.to_datetime(workouts["workout_end"], utc=True)
# workouts = workouts.astype({
#     "id": "string",
#     "user_id": "string",
#     "timezone_offset": "string",
#     "sport_id": "string",
#     "score_state": "string",
#     "score_strain": "float",
#     "score_average_heart_rate": "int",
#     "score_max_heart_rate": "int",
#     "score_kilojoule": "float",
#     "score_percent_recorded": "int",
#     "score_distance_meter": "float",
#     "score_altitude_gain_meter": "float",
#     "score_altitude_change_meter": "float",
#     "score_zone_duration_zone_zero_milli": "int",
#     "score_zone_duration_zone_one_milli": "int",
#     "score_zone_duration_zone_two_milli": "int",
#     "score_zone_duration_zone_three_milli": "int",
#     "score_zone_duration_zone_four_milli": "int",
#     "score_zone_duration_zone_five_milli": "int"
# })

# # Convert columns for matching schema for sleep
# sleep_collection["created_at"] = pd.to_datetime(sleep_collection["created_at"], errors="coerce", utc=True)
# sleep_collection["updated_at"] = pd.to_datetime(sleep_collection["updated_at"], errors="coerce", utc=True)
# sleep_collection["sleep_end"] = pd.to_datetime(sleep_collection["sleep_end"], errors="coerce", utc=True)
# sleep_collection["sleep_start"] = pd.to_datetime(sleep_collection["sleep_start"], errors="coerce", utc=True)

# # Fill missing values for integer columns with 0 and cast
# int_columns = [
#     "score_stage_summary_total_in_bed_time_milli",
#     "score_stage_summary_total_awake_time_milli",
#     "score_stage_summary_total_no_data_time_milli",
#     "score_stage_summary_total_light_sleep_time_milli",
#     "score_stage_summary_total_slow_wave_sleep_time_milli",
#     "score_stage_summary_total_rem_sleep_time_milli",
#     "score_stage_summary_sleep_cycle_count",
#     "score_stage_summary_disturbance_count",
#     "score_sleep_needed_baseline_milli",
#     "score_sleep_needed_need_from_sleep_debt_milli",
#     "score_sleep_needed_need_from_recent_strain_milli",
#     "score_sleep_needed_need_from_recent_nap_milli",
#     "score_sleep_performance_percentage",
#     "score_sleep_consistency_percentage",
# ]

# for col in int_columns:
#     sleep_collection[col] = pd.to_numeric(
#         sleep_collection[col], 
#         errors="coerce"
#     ).fillna(0).astype("int")

# # Handle float columns
# float_columns = [
#     "score_respiratory_rate",
#     "score_sleep_efficiency_percentage"
# ]

# for col in float_columns:
#     sleep_collection[col] = pd.to_numeric(
#         sleep_collection[col], 
#         errors="coerce"
#     ).fillna(0.0).astype("float")

# # Convert string columns
# string_columns = [
#     "id", 
#     "user_id", 
#     "timezone_offset", 
#     "nap", 
#     "score_state"
# ]

# sleep_collection[string_columns] = sleep_collection[string_columns].astype("string")

# # Convert cols for recoveries
# recoveries["created_at"] = pd.to_datetime(recoveries["created_at"], utc=True)
# recoveries["updated_at"] = pd.to_datetime(recoveries["updated_at"], utc=True)

# # Convert other columns
# recoveries = recoveries.astype({
#     "cycle_id": "string",
#     "sleep_id": "string",
#     "user_id": "string",
#     "score_state": "string",
#     "score_user_calibrating": "string",
#     "score_recovery_score": "int",
#     "score_resting_heart_rate": "int",
#     "score_hrv_rmssd_milli": "float",
#     "score_spo2_percentage": "float",
#     "score_skin_temp_celsius": "float"
# })

# # Convert cols for cycle_collection
# cycle_collection["created_at"] = pd.to_datetime(cycle_collection["created_at"], utc=True)
# cycle_collection["updated_at"] = pd.to_datetime(cycle_collection["updated_at"], utc=True)
# cycle_collection["cycle_start"] = pd.to_datetime(cycle_collection["cycle_start"], utc=True)
# cycle_collection["cycle_end"] = pd.to_datetime(cycle_collection["cycle_end"], utc=True)
# cycle_collection = cycle_collection.astype({
#     "id": "string",
#     "user_id": "string",
#     "timezone_offset": "string",
#     "score_state": "string",
#     "score_strain": "float",
#     "score_kilojoule": "float",
#     "score_average_heart_rate": "int",
#     "score_max_heart_rate": "int"
# })

# # LOAD DATA INTO THE SQL DATABASE
# table_names = {
#     'whoop_workouts': workouts, 
#     'whoop_sleep': sleep_collection, 
#     'whoop_recoveries': recoveries, 
#     'whoop_cycle_collection': cycle_collection
# }

# for name, df in table_names.items():
#     # Append data to the table
#     df.to_sql(name, engine, if_exists="append", index=False)
    
#     # Define unique columns for each table
#     if name == 'whoop_workouts':
#         unique_columns = 'id'
#     elif name == 'whoop_sleep':
#         unique_columns = 'created_at'  
#     elif name == 'whoop_recoveries':
#         unique_columns = 'cycle_id'  
#     elif name == 'whoop_cycle_collection':
#         unique_columns = 'id'  

#     # Remove duplicates from the table after insertion
#     remove_duplicates(name, unique_columns)
#     print(f"Duplicates removed from {name}")

import os
import requests
from whoopy import WhoopClient
from dotenv import load_dotenv

load_dotenv()

WHOOP_CLIENT_ID = os.getenv("WHOOP_CLIENT_ID")
WHOOP_CLIENT_SECRET = os.getenv("WHOOP_CLIENT_SECRET")
WHOOP_EMAIL = os.getenv("WHOOP_EMAIL")
WHOOP_PASSWORD = os.getenv("WHOOP_PASSWORD")

# Step 1: Get access token using Resource Owner Password Credentials (ROPC) flow
def get_access_token():
    url = "https://api.prod.whoop.com/oauth/oauth2/token"
    payload = {
        "grant_type": "password",
        "client_id": WHOOP_CLIENT_ID,
        "client_secret": WHOOP_CLIENT_SECRET,
        "username": WHOOP_EMAIL,
        "password": WHOOP_PASSWORD,
    }
    response = requests.post(url, data=payload)
    response.raise_for_status()
    return response.json()

# Step 2: Initialize client
token_data = get_access_token()

access_token = token_data["access_token"]
expires_in = token_data["expires_in"]
scopes = token_data["scope"]

client = WhoopClient(access_token, expires_in, scopes)
