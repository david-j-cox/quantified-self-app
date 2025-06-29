import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve variables from environment
STRAVA_ID = os.getenv('STRAVA_ID')
STRAVA_SECRET = os.getenv('STRAVA_SECRET')
STRAVA_ACCESS = os.getenv('STRAVA_ACCESS')
STRAVA_REFRESH = os.getenv('STRAVA_REFRESH')
STRAVA_AUTHORIZATION_CODE = os.getenv('STRAVA_AUTHORIZATION_CODE')
DATABASE_URL = os.getenv('DB_URL')
TABLE_NAME = 'strava_activities'

def refresh_access_token():
    global STRAVA_ACCESS
    global STRAVA_REFRESH
    url = 'https://www.strava.com/oauth/token'
    payload = {
        'client_id': STRAVA_ID,
        'client_secret': STRAVA_SECRET,
        'grant_type': 'refresh_token',
        'refresh_token': STRAVA_REFRESH
    }

    response = requests.post(url, data=payload)
    if response.status_code == 200:
        tokens = response.json()
        STRAVA_ACCESS = tokens['access_token']
        STRAVA_REFRESH = tokens['refresh_token']
        print("Access token refreshed successfully.")
    else:
        print("Failed to refresh access token:", response.json())
        exit()

def fetch_activities(after_timestamp):
    url = 'https://www.strava.com/api/v3/athlete/activities'
    headers = {
        'Authorization': f'Bearer {STRAVA_ACCESS}'
    }

    params = {
        'per_page': 200,
        'page': 1,
        'after': after_timestamp
    }

    all_activities = []
    while True:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            activities = response.json()
            if not activities:
                break
            all_activities.extend(activities)
            params['page'] += 1
        else:
            print("Error fetching activities:", response.json())
            exit()

    return all_activities

def append_to_csv(file_path, df):
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, df]).drop_duplicates(
            subset=['activitydate', 'activitytype', 'elapsedtime']
        ).reset_index(drop=True)
    else:
        combined_df = df
    combined_df.to_csv(file_path, index=False)
    print(f"Data appended to {file_path}.")

def append_to_database(df, engine):
    try:
        df.to_sql(TABLE_NAME, con=engine, if_exists='append', index=False)
        print(f"Data appended to the '{TABLE_NAME}' table in the database.")
    except Exception as e:
        print(f"Error appending data to the database: {e}")

def format_date(date_str):
    try:
        original_date = pd.to_datetime(date_str)
        return original_date.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"Error formatting date: {e}")
        return None

if __name__ == "__main__":
    # Refresh token before fetching activities
    refresh_access_token()

    # Determine the latest date in activities.csv
    csv_file_path = '../Data/activities.csv'
    if os.path.exists(csv_file_path):
        existing_df = pd.read_csv(csv_file_path)
        if 'activitydate' in existing_df.columns:
            latest_date = pd.to_datetime(existing_df['activitydate']).max()
            after_timestamp = int(latest_date.timestamp()) if not pd.isna(latest_date) else 0
        else:
            after_timestamp = 0
    else:
        after_timestamp = 0

    # Fetch activities from Strava API
    activities = fetch_activities(after_timestamp)

    if activities:
        df = pd.DataFrame(activities)

        # Convert activity date to datetime and sort
        df['start_date_local'] = pd.to_datetime(df['start_date_local'], errors='coerce')
        df.sort_values('start_date_local', ascending=False, inplace=True)

        # Select only the cols we want to keep
        df = df[[
            'start_date_local', 
            'sport_type',
            'elapsed_time',
            'distance',
            'max_speed',
            'average_speed',
            'total_elevation_gain',
            'elev_low',
            'elev_high',
            'average_cadence'
        ]]
        df.columns = [
            'activitydate', 
            'activitytype',
            'elapsedtime', 
            'distance', 
            'maxspeed', 
            'averagespeed', 
            'elevationgain', 
            'elevationlow',
            'elevationhigh', 
            'averagecadence'
        ]

        # Setup the df so it matches the activities data
        df = df.sort_values(by=['activitydate'], ascending=True)
        df['activitydate'] = df['activitydate'].apply(format_date)

        # Convert distance to miles if the activity type is 'Run'
        df.loc[df['activitytype'] == 'Run', 'distance'] = df['distance'] / 1609.34

        # Save to CSV
        append_to_csv(csv_file_path, df)

        # Save to database
        engine = create_engine(DATABASE_URL)
        append_to_database(df, engine)

        # Remove duplicates from the database
        remove_duplicates_query = """
        WITH cte AS (
            SELECT 
                ctid AS row_id,
                ROW_NUMBER() OVER (
                    PARTITION BY "activitydate", "activitytype", "elapsedtime" 
                    ORDER BY "activitydate"
                ) AS row_num
            FROM strava_activities
        )
        DELETE FROM strava_activities
        WHERE ctid IN (
            SELECT row_id
            FROM cte
            WHERE row_num > 1
        );
        """
        with engine.connect() as connection:
            connection.execute(text(remove_duplicates_query))
        print("Duplicates removed from the database.")
    else:
        print("No new activities to append.")