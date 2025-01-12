import requests
import psycopg2
import json
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# WHOOP API credentials from .env
WHOOP_EMAIL = os.getenv("WHOOP_EMAIL")
WHOOP_PASSWORD = os.getenv("WHOOP_PASSWORD")

# PostgreSQL database credentials from .env
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

def authenticate_with_whoop():
    """
    Authenticate with the WHOOP API and return the access token.
    """
    url = "https://api.whoop.com/oauth/token"
    payload = {
        "grant_type": "password",
        "username": WHOOP_EMAIL,
        "password": WHOOP_PASSWORD
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        print("Failed to authenticate with WHOOP API:", response.json())
        return None

def fetch_daily_data(token):
    """
    Fetch daily data from the WHOOP API.
    """
    url = "https://api.prod.whoop.com/users/me/daily"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch daily data:", response.json())
        return None

def save_to_postgresql(data):
    """
    Save WHOOP data to a PostgreSQL table.
    """
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
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

def main():
    # Step 1: Authenticate with WHOOP
    token = authenticate_with_whoop()
    if not token:
        return

    # Step 2: Fetch daily data
    daily_data = fetch_daily_data(token)
    if not daily_data:
        return

    # Step 3: Save data to PostgreSQL
    save_to_postgresql(daily_data)

if __name__ == "__main__":
    main()
