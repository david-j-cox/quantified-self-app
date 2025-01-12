import os
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import shutil

# Load environment variables from .env file
load_dotenv()

# PostgreSQL database credentials from .env
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Paths
folder_path = '../Data/Time Tracking/'  # Folder with the data files
raw_data_path = '../Data/raw_data.csv'  # Path to the raw_data.csv file

# Read existing data from raw_data.csv
if os.path.exists(raw_data_path):
    all_data = pd.read_csv(raw_data_path)
else:
    all_data = pd.DataFrame()

# Data processing
new_df = pd.DataFrame()

# Iterate over each .csv file in the folder
for filename in os.listdir(folder_path):
    if filename.startswith("Stopwatch Export") and filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)

        # Extract the date from the filename
        date_str = filename.replace(
            'Stopwatch Export ', '').replace('.csv', '')
        date_str = date_str.split("-")
        date_str = f"{date_str[1]}-{date_str[0]}-{date_str[2]}"
        date = pd.to_datetime(date_str, format='%m-%d-%Y')
        date = date - timedelta(days=1)

        # Read the .csv file into a DataFrame, skipping the first two rows
        new_data = pd.read_csv(file_path, skiprows=3)

        # Use the "Seconds" column, converting it to total minutes
        new_data['Minutes'] = new_data['Lap']/60
        new_data = new_data.reset_index()
        new_data = new_data[['index', 'Minutes']].T
        new_data.columns = new_data.iloc[0]
        new_data = new_data.reset_index(drop=True)
        new_data = new_data[1:].reset_index(drop=True)
        new_data = new_data[[
            "Journal Articles",
            "Philosophy",
            "Reading Books",
            "Learning",
            "Writing",
            "Research Projects",
            "Teaching",
            "Language",
            "Ethics Work",
            "Presentations",
            "Physical Exercise",
            "Human Experience",
            "Coding",
            "Hobbies",
            "Art",
            "Family",
            "Job",
            "EAB",
            "ABA",
            "OBM",
            "EAB Non-Research",
            "ABA Non-Research",
            "Behaviorism",
            "Ethics",
            "Non-BA Experimental",
            "Non-BA Non-Experimental",
            "Economics",
            "Behavioral Economics",
            "Data & Analytics",
            "Computer Science",
            "Behavioral Pharmacology"
        ]].copy()

        # Add the date column
        new_data['Date'] = date

        # Append the new data to the new_df DataFrame
        new_df = pd.concat([new_df, new_data], ignore_index=True)

# Ensure new_df has unique rows and reset the index
new_df = new_df.drop_duplicates().sort_values(
    by=['Date'], ascending=True).reset_index(drop=True)

# Add new data to the existing raw_data.csv data
all_data['Date'] = pd.to_datetime(all_data['Date'])
all_data = pd.concat([all_data, new_df]).drop_duplicates(
    subset='Date', keep='first').reset_index(drop=True)

# Sort by date and calculate additional columns
all_data['Date'] = pd.to_datetime(all_data['Date'])
all_data = all_data.sort_values(
    by=['Date'], ascending=True).reset_index(drop=True)

# Save the updated data back to raw_data.csv
all_data = all_data.drop_duplicates(subset=['Date'], keep='last')
all_data.to_csv(raw_data_path, index=False)
print("Data has been successfully appended to the raw_data.csv file.")

# Database connection details
DATABASE_URL = f'postgresql://{DB_USER}:password@{DB_HOST}:{DB_PORT}/{DB_NAME}'
TABLE_NAME = 'raw_data'

# Create a database engine
engine = create_engine(DATABASE_URL)

# Save the new data (new_df) to the PostgreSQL database
try:
    new_df.to_sql(TABLE_NAME, con=engine, if_exists='append', index=False)
    print(f"New data has been successfully appended to the '{
          TABLE_NAME}' table in the database.")
except Exception as e:
    print(f"Error saving new data to the database: {e}")

# MOVE DATA TO AN ARCHIVE FOLDER
# Get the directory of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))

# The Archive folder is one directory up from script_dir
archive_path = os.path.join(script_dir, '..', 'Archive')

# Ensure the archive directory exists
if not os.path.exists(archive_path):
    os.makedirs(archive_path)

# Move the processed files to the archive folder
for filename in os.listdir(folder_path):
    if filename.startswith("Stopwatch Export") and filename.endswith('.csv'):
        src_path = os.path.join(folder_path, filename)
        dst_path = os.path.join(archive_path, filename)
        try:
            shutil.move(src_path, dst_path)
            print(f"Moved file: {src_path} to {dst_path}")
        except OSError as e:
            print(f"Error moving file {src_path} to {dst_path}: {e}")


# Create a database engine
engine = create_engine(DATABASE_URL)

# SQL query to remove duplicates
remove_duplicates_query = """
WITH cte AS (
    SELECT 
        ctid AS row_id,
        ROW_NUMBER() OVER (PARTITION BY "Date", "Journal Articles", "Philosophy", "Reading Books", "Learning",
                                       "Writing", "Research Projects", "Teaching", "Language",
                                       "Ethics Work", "Presentations", "Physical Exercise",
                                       "Human Experience", "Coding"
                          ORDER BY "Date") AS row_num
    FROM raw_data
)
DELETE FROM raw_data
WHERE ctid IN (
    SELECT row_id
    FROM cte
    WHERE row_num > 1
);
"""

# Execute the query
with engine.connect() as connection:
    # Use text() to wrap the query
    connection.execute(text(remove_duplicates_query))

print("Duplicates removed successfully.")
