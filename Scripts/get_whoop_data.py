import os
from whoop import WhoopClient
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text
import requests
import psycopg2
import json
from datetime import datetime

load_dotenv()

# Retrieve variables from environment
WHOOP_EMAIL = os.getenv("WHOOP_EMAIL")
WHOOP_PASSWORD = os.getenv("WHOOP_PASSWORD")
DATABASE_URL = os.getenv('DATABASE_URL')

# Initialize database connection
engine = create_engine(DATABASE_URL)

# Function to get the latest `created_at` from a table
def get_latest_created_at(table_name):
    query = f"SELECT MAX(created_at) AS latest_date FROM {table_name}"
    with engine.connect() as conn:
        result = conn.execute(text(query)).fetchone()
        return result["latest_date"]

# Function to remove duplicates from a table
def remove_duplicates(table_name, unique_columns):
    query = f"""
    DELETE FROM {table_name}
    WHERE id IN (
        SELECT id FROM (
            SELECT id, ROW_NUMBER() OVER (
                PARTITION BY {', '.join(unique_columns)} 
                ORDER BY created_at DESC
            ) AS row_num
            FROM {table_name}
        ) subquery
        WHERE row_num > 1
    );
    """
    with engine.connect() as conn:
        conn.execute(text(query))

# Get latest `created_at` dates from the tables
latest_dates = {
    name: get_latest_created_at(table_name)
    for name, table_name in table_names.items()
}

# Set default start_date if no data exists
default_start_date = "2024-09-01"
start_dates = {
    name: (latest_dates[name].strftime("%Y-%m-%d") if latest_dates[name] else default_start_date)
    for name in table_names.keys()
}

# Initialize WhoopClient and fetch data since the latest `created_at`
with WhoopClient(WHOOP_EMAIL, WHOOP_PASSWORD) as client:
    workouts = client.get_workout_collection(start_date=start_dates[workouts])
    sleep_collection = client.get_sleep_collection(start_date=start_dates[sleep_collection])
    recoveries = client.get_recovery_collection(start_date=start_dates[recoveries])
    cycle_collection = client.get_cycle_collection(start_date=start_dates[cycle_collection])

# Tweak cols to match SQL DB field names
for i, df in enumerate([workouts, sleep_collection, recoveries, cycle_collection]):
    df = pd.json_normalize(df)  # Normalize data
    df.columns = [val.replace(".", "_") for val in list(df)]
    
    # Adjust column names for each df
    if i == 0: 
        df.columns = [val.replace("start", "workout_start") if val == "start" else val for val in df.columns]
        df.columns = [val.replace("end", "workout_end") if val == "end" else val for val in df.columns]
    elif i ==1:
        df.columns = [val.replace("start", "sleep_start") if val == "start" else val for val in df.columns]
        df.columns = [val.replace("end", "sleep_end") if val == "end" else val for val in df.columns]
    elif i ==3:
        df.columns = [val.replace("start", "cycle_start") if val == "start" else val for val in df.columns]
        df.columns = [val.replace("end", "cycle_end") if val == "end" else val for val in df.columns]
    
    # Update the original DataFrame
    if i == 0:
        workouts = df
    elif i == 1:
        sleep_collection = df
    elif i == 2:
        recoveries = df
    elif i == 3:
        cycle_collection = df

# TRANSFORM DATA BEFORE LOADING
# Convert columns to match the schema for workouts
workouts["created_at"] = pd.to_datetime(workouts["created_at"], utc=True)
workouts["updated_at"] = pd.to_datetime(workouts["updated_at"], utc=True)
workouts["workout_start"] = pd.to_datetime(workouts["workout_start"], utc=True)
workouts["workout_end"] = pd.to_datetime(workouts["workout_end"], utc=True)
workouts = workouts.astype({
    "id": "string",
    "user_id": "string",
    "timezone_offset": "string",
    "sport_id": "string",
    "score_state": "string",
    "score_strain": "float",
    "score_average_heart_rate": "int",
    "score_max_heart_rate": "int",
    "score_kilojoule": "float",
    "score_percent_recorded": "int",
    "score_distance_meter": "float",
    "score_altitude_gain_meter": "float",
    "score_altitude_change_meter": "float",
    "score_zone_duration_zone_zero_milli": "int",
    "score_zone_duration_zone_one_milli": "int",
    "score_zone_duration_zone_two_milli": "int",
    "score_zone_duration_zone_three_milli": "int",
    "score_zone_duration_zone_four_milli": "int",
    "score_zone_duration_zone_five_milli": "int"
})

# Convert columns for matching schema for sleep
sleep_collection["created_at"] = pd.to_datetime(sleep_collection["created_at"], errors="coerce", utc=True)
sleep_collection["updated_at"] = pd.to_datetime(sleep_collection["updated_at"], errors="coerce", utc=True)
sleep_collection["sleep_end"] = pd.to_datetime(sleep_collection["sleep_end"], errors="coerce", utc=True)
sleep_collection["sleep_start"] = pd.to_datetime(sleep_collection["sleep_start"], errors="coerce", utc=True)

# Fill missing values for integer columns with 0 and cast
int_columns = [
    "score_stage_summary_total_in_bed_time_milli",
    "score_stage_summary_total_awake_time_milli",
    "score_stage_summary_total_no_data_time_milli",
    "score_stage_summary_total_light_sleep_time_milli",
    "score_stage_summary_total_slow_wave_sleep_time_milli",
    "score_stage_summary_total_rem_sleep_time_milli",
    "score_stage_summary_sleep_cycle_count",
    "score_stage_summary_disturbance_count",
    "score_sleep_needed_baseline_milli",
    "score_sleep_needed_need_from_sleep_debt_milli",
    "score_sleep_needed_need_from_recent_strain_milli",
    "score_sleep_needed_need_from_recent_nap_milli",
    "score_sleep_performance_percentage",
    "score_sleep_consistency_percentage",
]

for col in int_columns:
    sleep_collection[col] = pd.to_numeric(
        sleep_collection[col], 
        errors="coerce"
    ).fillna(0).astype("int")

# Handle float columns
float_columns = [
    "score_respiratory_rate",
    "score_sleep_efficiency_percentage"
]

for col in float_columns:
    sleep_collection[col] = pd.to_numeric(
        sleep_collection[col], 
        errors="coerce"
    ).fillna(0.0).astype("float")

# Convert string columns
string_columns = [
    "id", 
    "user_id", 
    "timezone_offset", 
    "nap", 
    "score_state"
]

sleep_collection[string_columns] = sleep_collection[string_columns].astype("string")

# Convert cols for recoveries
recoveries["created_at"] = pd.to_datetime(recoveries["created_at"], utc=True)
recoveries["updated_at"] = pd.to_datetime(recoveries["updated_at"], utc=True)

# Convert other columns
recoveries = recoveries.astype({
    "cycle_id": "string",
    "sleep_id": "string",
    "user_id": "string",
    "score_state": "string",
    "score_user_calibrating": "string",
    "score_recovery_score": "int",
    "score_resting_heart_rate": "int",
    "score_hrv_rmssd_milli": "float",
    "score_spo2_percentage": "float",
    "score_skin_temp_celsius": "float"
})

# Convert cols for cycle_collection
cycle_collection["created_at"] = pd.to_datetime(cycle_collection["created_at"], utc=True)
cycle_collection["updated_at"] = pd.to_datetime(cycle_collection["updated_at"], utc=True)
cycle_collection["cycle_start"] = pd.to_datetime(cycle_collection["cycle_start"], utc=True)
cycle_collection["cycle_end"] = pd.to_datetime(cycle_collection["cycle_end"], utc=True)
cycle_collection = cycle_collection.astype({
    "id": "string",
    "user_id": "string",
    "timezone_offset": "string",
    "score_state": "string",
    "score_strain": "float",
    "score_kilojoule": "float",
    "score_average_heart_rate": "int",
    "score_max_heart_rate": "int"
})

# LOAD DATA INTO THE SQL DATABASE
table_names = {
    workouts : 'whoop_workouts', 
    sleep_collection: 'whoop_sleep', 
    recoveries: 'whoop_recoveries', 
    cycle_collection: 'whoop_cycle_collection'
}

for df, table_name in table_names.items():
    # Append data to the table
    df.to_sql(table_name, engine, if_exists="append", index=False)
    
    # Remove duplicates from the table after insertion
    unique_columns = ["id", "id", "cycle_id", "id"]
    remove_duplicates(table_name, unique_columns)

class WhoopDataPuller:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # WHOOP API credentials
        self.whoop_email = os.getenv("WHOOP_EMAIL")
        self.whoop_password = os.getenv("WHOOP_PASSWORD")
        
        # PostgreSQL database credentials
        self.db_name = os.getenv("DB_NAME")
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")
        self.db_host = os.getenv("DB_HOST")
        self.db_port = os.getenv("DB_PORT")

    def authenticate_with_whoop(self):
        """
        Authenticate with the WHOOP API and return the passport token.
        """
        url = "https://api.whoop.com/oauth/token"
        payload = {
            "grant_type": "password",
            "username": self.whoop_email,
            "password": self.whoop_password
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, data=json.dumps(payload), headers=headers)

        if response.status_code == 200:
            access_token = response.json()["access_token"]
            
            # Use the access token to get the passport token
            passport_url = "https://api.whoop.com/developer/v1/oauth/passport"
            passport_headers = {"Authorization": f"Bearer {access_token}"}
            passport_response = requests.get(passport_url, headers=passport_headers)
            
            if passport_response.status_code == 200:
                return passport_response.json()["passport_token"]
            else:
                print("Failed to obtain passport token:", passport_response.json())
                return None
        else:
            print("Failed to authenticate with WHOOP API:", response.json())
            return None

    def fetch_daily_data(self, passport_token):
        """
        Fetch daily data from the WHOOP API using the passport token.
        """
        url = "https://api.prod.whoop.com/users/me/daily"
        headers = {"Authorization": f"Bearer {passport_token}"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            print("Failed to fetch daily data:", response.json())
            return None

    def save_to_postgresql(self, data):
        """
        Save WHOOP data to a PostgreSQL table.
        """
        conn = psycopg2.connect(
            dbname=self.db_name, user=self.db_user, password=self.db_password, host=self.db_host, port=self.db_port
        )
        cur = conn.cursor()

        # Create table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS whoop_daily_data (
            date DATE PRIMARY KEY,
            strain FLOAT,
            recovery INT,
            sleep_performance FLOAT,
            total_sleep_hours FLOAT,
            resting_heart_rate INT,
            heart_rate_variability FLOAT
        );
        """
        cur.execute(create_table_query)

        # Insert or update the data
        insert_query = """
        INSERT INTO whoop_daily_data (date, strain, recovery, sleep_performance, total_sleep_hours, resting_heart_rate, heart_rate_variability)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (date) DO UPDATE SET
            strain = EXCLUDED.strain,
            recovery = EXCLUDED.recovery,
            sleep_performance = EXCLUDED.sleep_performance,
            total_sleep_hours = EXCLUDED.total_sleep_hours,
            resting_heart_rate = EXCLUDED.resting_heart_rate,
            heart_rate_variability = EXCLUDED.heart_rate_variability;
        """
        for day in data:
            cur.execute(insert_query, (
                day["days"][0]["date"],
                day["strain"],
                day["recovery"],
                day["sleepPerformance"],
                day["totalSleepHours"],
                day["restingHeartRate"],
                day["heartRateVariability"]
            ))

        conn.commit()
        cur.close()
        conn.close()
        print("Data successfully saved to PostgreSQL.")

    def run(self):
        # Step 1: Authenticate with WHOOP
        passport_token = self.authenticate_with_whoop()
        if not passport_token:
            return

        # Step 2: Fetch daily data
        daily_data = self.fetch_daily_data(passport_token)
        if not daily_data:
            return

        # Step 3: Save data to PostgreSQL
        self.save_to_postgresql(daily_data)

if __name__ == "__main__":
    whoop_data_puller = WhoopDataPuller()
    whoop_data_puller.run()