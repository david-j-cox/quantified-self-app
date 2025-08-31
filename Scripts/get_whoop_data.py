import os
import requests
import pandas as pd
import json
import uuid
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import time

load_dotenv()

# Retrieve variables from environment
WHOOP_CLIENT_ID = os.getenv("WHOOP_CLIENT_ID")
WHOOP_CLIENT_SECRET = os.getenv("WHOOP_CLIENT_SECRET") 
WHOOP_ACCESS_TOKEN = os.getenv("WHOOP_ACCESS_TOKEN")  # You'll need to obtain this via OAuth
DATABASE_URL = os.getenv('DB_URL')

# Debug: Check if environment variables are loaded
print(f"WHOOP_CLIENT_ID loaded: {'Yes' if WHOOP_CLIENT_ID else 'No'}")
print(f"WHOOP_CLIENT_SECRET loaded: {'Yes' if WHOOP_CLIENT_SECRET else 'No'}")
print(f"WHOOP_ACCESS_TOKEN loaded: {'Yes' if WHOOP_ACCESS_TOKEN else 'No'}")
print(f"DATABASE_URL loaded: {'Yes' if DATABASE_URL else 'No'}")

if not WHOOP_ACCESS_TOKEN:
    print("ERROR: WHOOP_ACCESS_TOKEN not found in environment variables!")
    print("Please obtain an access token using OAuth 2.0 flow and add it to your .env file:")
    print("WHOOP_ACCESS_TOKEN=your_access_token_here")
    print("See: https://developer.whoop.com/docs/developing/user-data-access")
    exit(1)

# Initialize database connection
engine = create_engine(DATABASE_URL)

# Whoop API v2 Base URL - confirmed working from endpoint tests
BASE_URL = "https://api.prod.whoop.com/developer"

class WhoopAPIv2Client:
    def __init__(self, access_token):
        self.access_token = access_token
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
    
    def _make_request(self, endpoint, params=None):
        """Make a request to the Whoop API v2"""
        url = f"{BASE_URL}{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed for {endpoint}: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            raise
    
    def get_user_profile(self):
        """Get user profile information"""
        return self._make_request("/v2/user/profile/basic")
    
    def get_workouts(self, start_date=None, end_date=None, limit=25):
        """Get workout data from Whoop API v2"""
        params = {"limit": limit}
        if start_date:
            params["start"] = start_date
        if end_date:
            params["end"] = end_date
        
        all_workouts = []
        next_token = None
        
        while True:
            if next_token:
                params["nextToken"] = next_token
            
            response = self._make_request("/v2/activity/workout", params)
            workouts = response.get("records", [])
            all_workouts.extend(workouts)
            
            next_token = response.get("next_token")
            if not next_token:
                break
            
            # Remove nextToken from params for next iteration if it was added
            if "nextToken" in params:
                del params["nextToken"]
        
        return all_workouts
    
    def get_sleep(self, start_date=None, end_date=None, limit=25):
        """Get sleep data from Whoop API v2"""
        params = {"limit": limit}
        if start_date:
            params["start"] = start_date
        if end_date:
            params["end"] = end_date
            
        all_sleep = []
        next_token = None
        
        while True:
            if next_token:
                params["nextToken"] = next_token
            
            response = self._make_request("/v2/activity/sleep", params)
            sleep_records = response.get("records", [])
            all_sleep.extend(sleep_records)
            
            next_token = response.get("next_token")
            if not next_token:
                break
                
            if "nextToken" in params:
                del params["nextToken"]
        
        return all_sleep
    
    def get_recovery(self, start_date=None, end_date=None, limit=25):
        """Get recovery data from Whoop API v2"""
        params = {"limit": limit}
        if start_date:
            params["start"] = start_date
        if end_date:
            params["end"] = end_date
            
        all_recovery = []
        next_token = None
        
        while True:
            if next_token:
                params["nextToken"] = next_token
            
            response = self._make_request("/v2/recovery", params)
            recovery_records = response.get("records", [])
            all_recovery.extend(recovery_records)
            
            next_token = response.get("next_token")
            if not next_token:
                break
                
            if "nextToken" in params:
                del params["nextToken"]
        
        return all_recovery
    
    def get_cycles(self, start_date=None, end_date=None, limit=25):
        """Get physiological cycle data from Whoop API v2"""
        params = {"limit": limit}
        if start_date:
            params["start"] = start_date
        if end_date:
            params["end"] = end_date
            
        all_cycles = []
        next_token = None
        
        while True:
            if next_token:
                params["nextToken"] = next_token
            
            response = self._make_request("/v2/cycle", params)
            cycle_records = response.get("records", [])
            all_cycles.extend(cycle_records)
            
            next_token = response.get("next_token")
            if not next_token:
                break
                
            if "nextToken" in params:
                del params["nextToken"]
        
        return all_cycles

# Function to get the latest `created_at` from a table
def get_latest_created_at(table_name):
    try:
        query = f"SELECT MAX(created_at) AS latest_date FROM {table_name}"
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()
            return result[0]
    except:
        return '2024-09-01'

# Function to remove duplicates from a table
def remove_duplicates(table_name, unique_column):
    query = f"""
    DELETE FROM {table_name}
    WHERE {unique_column} IN (
        SELECT {unique_column} FROM (
            SELECT {unique_column}, ROW_NUMBER() OVER (
                PARTITION BY {unique_column} 
                ORDER BY created_at DESC
            ) AS row_num
            FROM {table_name}
        ) subquery
        WHERE row_num > 1
    );
    """
    with engine.connect() as conn:
        conn.execute(text(query))
        conn.commit()

# List of table names to get latest `created_at` dates from
table_names = {
    'workouts': 'whoop_workouts',
    'sleep_collection': 'whoop_sleep',
    'recoveries': 'whoop_recoveries',
    'cycle_collection': 'whoop_cycle_collection'
}

# Get latest `created_at` dates from the tables
latest_dates = {
    name: get_latest_created_at(table_name)
    for name, table_name in table_names.items()
}

# Set default start_date if no data exists
default_start_date = "2024-09-01"

# Convert latest_dates[name] to ISO 8601 datetime format for API v2
start_dates = {}
for name in table_names.keys():
    if latest_dates[name]:
        # Convert to ISO 8601 datetime format (API v2 requirement)
        dt = pd.to_datetime(latest_dates[name])
        start_dates[name] = dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    else:
        # Convert default date to ISO 8601 datetime format
        dt = pd.to_datetime(default_start_date)
        start_dates[name] = dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

print(f"Start dates for data collection: {start_dates}")

# Initialize Whoop API v2 client and fetch data
try:
    print("Initializing Whoop API v2 client...")
    client = WhoopAPIv2Client(WHOOP_ACCESS_TOKEN)
    
    # Test authentication by getting user profile
    print("Testing authentication...")
    user_profile = client.get_user_profile()
    print(f"Successfully authenticated! User ID: {user_profile.get('user_id', 'Unknown')}")
    
    print("Fetching workout data...")
    workouts = client.get_workouts(start_date=start_dates['workouts'])
    print(f"Retrieved {len(workouts)} workouts")
    
    print("Fetching sleep data...")
    sleep_collection = client.get_sleep(start_date=start_dates['sleep_collection'])
    print(f"Retrieved {len(sleep_collection)} sleep records")
    
    print("Fetching recovery data...")
    recoveries = client.get_recovery(start_date=start_dates['recoveries'])
    print(f"Retrieved {len(recoveries)} recovery records")
    
    print("Fetching cycle data...")
    cycle_collection = client.get_cycles(start_date=start_dates['cycle_collection'])
    print(f"Retrieved {len(cycle_collection)} cycle records")

except Exception as e:
    print(f"ERROR during Whoop API access or data fetching: {str(e)}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    exit(1)

# Process and normalize API v2 data
def process_v2_data(data_list, data_type):
    """Process Whoop API v2 data and normalize for database storage"""
    if not data_list:
        return pd.DataFrame()
    
    # Convert list of dicts to DataFrame
    df = pd.json_normalize(data_list)
    
    # Replace dots with underscores in column names for SQL compatibility
    df.columns = [col.replace(".", "_") for col in df.columns]
    
    # Data type specific column renaming
    if data_type == 'workouts':
        # Rename start/end columns to be more specific
        if 'start' in df.columns:
            df = df.rename(columns={'start': 'workout_start'})
        if 'end' in df.columns:
            df = df.rename(columns={'end': 'workout_end'})
    elif data_type == 'sleep':
        if 'start' in df.columns:
            df = df.rename(columns={'start': 'sleep_start'})
        if 'end' in df.columns:
            df = df.rename(columns={'end': 'sleep_end'})
    elif data_type == 'cycles':
        if 'start' in df.columns:
            df = df.rename(columns={'start': 'cycle_start'})
        if 'end' in df.columns:
            df = df.rename(columns={'end': 'cycle_end'})
    
    return df

def filter_columns_for_existing_schema(df, table_name):
    """Filter DataFrame to only include columns that exist in the existing database schema"""
    
    # Define columns that exist in your current database tables (from screenshots)
    existing_columns = {
        'whoop_workouts': [
            'id', 'user_id', 'created_at', 'updated_at', 'workout_start', 'workout_end',
            'timezone_offset', 'sport_id', 'score_state', 'score_strain', 
            'score_average_heart_rate', 'score_max_heart_rate', 'score_kilojoule',
            'score_percent_recorded', 'score_distance_meter', 'score_altitude_gain_meter',
            'score_altitude_change_meter', 
            # Include the new zone duration columns we're adding
            'score_zone_durations_zone_zero_milli', 'score_zone_durations_zone_one_milli',
            'score_zone_durations_zone_two_milli', 'score_zone_durations_zone_three_milli',
            'score_zone_durations_zone_four_milli', 'score_zone_durations_zone_five_milli'
        ],
        'whoop_sleep': [
            'id', 'user_id', 'created_at', 'updated_at', 'sleep_start', 'sleep_end',
            'timezone_offset', 'nap', 'score_state', 'score_stage_summary_total_in_bed_time_milli',
            'score_stage_summary_total_awake_time_milli', 'score_stage_summary_total_no_data_time_milli',
            'score_stage_summary_total_light_sleep_time_milli', 'score_stage_summary_total_slow_wave_sleep_time_milli',
            'score_stage_summary_total_rem_sleep_time_milli', 'score_stage_summary_sleep_cycle_count',
            'score_stage_summary_disturbance_count', 'score_sleep_needed_baseline_milli',
            'score_sleep_needed_need_from_sleep_debt_milli', 'score_sleep_needed_need_from_recent_strain_milli',
            'score_sleep_needed_need_from_recent_nap_milli', 'score_respiratory_rate',
            'score_sleep_performance_percentage', 'score_sleep_consistency_percentage',
            'score_sleep_efficiency_percentage',
            # Include the new cycle_id column we're adding
            'cycle_id'
        ],
        'whoop_recoveries': [
            'cycle_id', 'sleep_id', 'user_id', 'created_at', 'updated_at',
            'score_state', 'score_user_calibrating', 'score_recovery_score',
            'score_resting_heart_rate', 'score_hrv_rmssd_milli', 'score_spo2_percentage',
            'score_skin_temp_celsius'
        ],
        'whoop_cycle_collection': [
            'id', 'user_id', 'created_at', 'updated_at', 'cycle_start', 'cycle_end',
            'timezone_offset', 'score_state', 'score_strain', 'score_kilojoule',
            'score_average_heart_rate', 'score_max_heart_rate'
        ]
    }
    
    if table_name not in existing_columns:
        print(f"Warning: No column mapping defined for table {table_name}")
        return df
    
    # Get the allowed columns for this table
    allowed_columns = existing_columns[table_name]
    
    # Filter the DataFrame to only include columns that exist in the current schema
    available_columns = [col for col in allowed_columns if col in df.columns]
    filtered_df = df[available_columns].copy()
    
    # Report what was filtered out
    dropped_columns = [col for col in df.columns if col not in available_columns]
    if dropped_columns:
        print(f"Filtered out columns from {table_name}: {dropped_columns}")
    
    return filtered_df

# Process each dataset
print("Processing workout data...")
workouts_df = process_v2_data(workouts, 'workouts')

print("Processing sleep data...")
sleep_df = process_v2_data(sleep_collection, 'sleep')

print("Processing recovery data...")
recoveries_df = process_v2_data(recoveries, 'recovery')

print("Processing cycle data...")
cycles_df = process_v2_data(cycle_collection, 'cycles')

# TRANSFORM DATA BEFORE LOADING FOR API v2
def transform_workouts_v2(df):
    """Transform workout data for v2 API schema"""
    if df.empty:
        return df
    
    # Handle datetime columns
    datetime_cols = ['created_at', 'updated_at', 'workout_start', 'workout_end']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
    
    # Handle mixed ID types per OpenAPI spec
    # id is UUID, user_id is integer
    uuid_cols = ['id']
    for col in uuid_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    int_id_cols = ['user_id']
    for col in int_id_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    # Handle numeric columns with proper type conversion (based on OpenAPI spec)
    numeric_mapping = {
        'sport_id': 'Int64',  # Will be deprecated 09/01/2025 per spec
        'score_strain': 'float64',
        'score_average_heart_rate': 'Int64',
        'score_max_heart_rate': 'Int64',
        'score_kilojoule': 'float64',
        'score_percent_recorded': 'float64',  # Changed to float per spec
        'score_distance_meter': 'float64',
        'score_altitude_gain_meter': 'float64',
        'score_altitude_change_meter': 'float64'
    }
    
    # Handle zone duration columns (OpenAPI spec shows different naming)
    # These are millisecond values and should be integers
    zone_cols = [
        'score_zone_durations_zone_zero_milli',
        'score_zone_durations_zone_one_milli', 
        'score_zone_durations_zone_two_milli',
        'score_zone_durations_zone_three_milli',
        'score_zone_durations_zone_four_milli',
        'score_zone_durations_zone_five_milli'
    ]
    for zone_col in zone_cols:
        if zone_col in df.columns:
            numeric_mapping[zone_col] = 'Int64'  # Nullable integer for millisecond values
    
    # Apply type conversions
    for col, dtype in numeric_mapping.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
    
    return df

def transform_sleep_v2(df):
    """Transform sleep data for v2 API schema"""
    if df.empty:
        return df
    
    # Handle datetime columns
    datetime_cols = ['created_at', 'updated_at', 'sleep_start', 'sleep_end']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
    
    # Handle mixed ID types per OpenAPI spec  
    # id is UUID, user_id and cycle_id are integers
    uuid_cols = ['id']
    for col in uuid_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    int_id_cols = ['user_id', 'cycle_id']
    for col in int_id_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    # Handle integer columns with nullables
    int_cols = [
        'score_stage_summary_total_in_bed_time_milli',
        'score_stage_summary_total_awake_time_milli', 
        'score_stage_summary_total_no_data_time_milli',
        'score_stage_summary_total_light_sleep_time_milli',
        'score_stage_summary_total_slow_wave_sleep_time_milli',
        'score_stage_summary_total_rem_sleep_time_milli',
        'score_stage_summary_sleep_cycle_count',
        'score_stage_summary_disturbance_count',
        'score_sleep_needed_baseline_milli',
        'score_sleep_needed_need_from_sleep_debt_milli',
        'score_sleep_needed_need_from_recent_strain_milli',
        'score_sleep_needed_need_from_recent_nap_milli',
        'score_sleep_performance_percentage',
        'score_sleep_consistency_percentage'
    ]
    
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    # Handle float columns
    float_cols = ['score_respiratory_rate', 'score_sleep_efficiency_percentage']
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
    
    # Handle string/boolean columns
    str_cols = ['timezone_offset', 'nap', 'score_state']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df

def transform_recovery_v2(df):
    """Transform recovery data for v2 API schema"""
    if df.empty:
        return df
    
    # Handle datetime columns
    datetime_cols = ['created_at', 'updated_at']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
    
    # Handle mixed ID types per OpenAPI spec
    # cycle_id and user_id are integers, sleep_id is UUID
    int_id_cols = ['cycle_id', 'user_id']
    for col in int_id_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    uuid_cols = ['sleep_id']
    for col in uuid_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Handle string columns
    str_cols = ['score_state', 'score_user_calibrating']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Handle numeric columns
    numeric_mapping = {
        'score_recovery_score': 'Int64',
        'score_resting_heart_rate': 'Int64',
        'score_hrv_rmssd_milli': 'float64',
        'score_spo2_percentage': 'float64',
        'score_skin_temp_celsius': 'float64'
    }
    
    for col, dtype in numeric_mapping.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
    
    return df

def transform_cycles_v2(df):
    """Transform cycle data for v2 API schema"""
    if df.empty:
        return df
    
    # Handle datetime columns
    datetime_cols = ['created_at', 'updated_at', 'cycle_start', 'cycle_end']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
    
    # Handle integer ID columns (cycles use int64, not UUIDs per OpenAPI spec)
    int_id_cols = ['id', 'user_id']
    for col in int_id_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    # Handle string columns
    str_cols = ['timezone_offset', 'score_state']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Handle numeric columns
    numeric_mapping = {
        'score_strain': 'float64',
        'score_kilojoule': 'float64',
        'score_average_heart_rate': 'Int64',
        'score_max_heart_rate': 'Int64'
    }
    
    for col, dtype in numeric_mapping.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
    
    return df

# Apply transformations
print("Transforming workout data...")
workouts_df = transform_workouts_v2(workouts_df)

print("Transforming sleep data...")
sleep_df = transform_sleep_v2(sleep_df)

print("Transforming recovery data...")
recoveries_df = transform_recovery_v2(recoveries_df)

print("Transforming cycle data...")
cycles_df = transform_cycles_v2(cycles_df)

# Filter data to match existing database schema (keeping only valuable new columns)
print("Filtering data to match existing database schema...")
workouts_df = filter_columns_for_existing_schema(workouts_df, 'whoop_workouts')
sleep_df = filter_columns_for_existing_schema(sleep_df, 'whoop_sleep') 
recoveries_df = filter_columns_for_existing_schema(recoveries_df, 'whoop_recoveries')
cycles_df = filter_columns_for_existing_schema(cycles_df, 'whoop_cycle_collection')

# LOAD DATA INTO THE SQL DATABASE using API v2 data
print("Loading data into database...")

# Define the mapping between dataframes and table names
data_mapping = {
    'whoop_workouts': workouts_df,
    'whoop_sleep': sleep_df, 
    'whoop_recoveries': recoveries_df,
    'whoop_cycle_collection': cycles_df
}

# Define unique columns for deduplication (v2 uses UUIDs)
unique_columns_mapping = {
    'whoop_workouts': 'id',        # UUID in v2
    'whoop_sleep': 'id',           # Use ID instead of created_at for v2
    'whoop_recoveries': 'cycle_id', # UUID in v2  
    'whoop_cycle_collection': 'id'  # UUID in v2
}

for table_name, df in data_mapping.items():
    if df.empty:
        print(f"No new data to load for {table_name}")
        continue
    
    try:
        print(f"Loading {len(df)} records into {table_name}...")
        
        # Use the original working approach with better error handling
        df.to_sql(table_name, engine, if_exists="append", index=False)
        
        # Get the unique column for this table
        unique_col = unique_columns_mapping[table_name]
        
        # Remove duplicates from the table after insertion
        remove_duplicates(table_name, unique_col)
        print(f"Successfully loaded {len(df)} records into {table_name}")
        
    except Exception as e:
        print(f"ERROR loading data into {table_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"Data was fetched but NOT saved to database for {table_name}")

print("Data loading completed!")

# Optional: Print summary statistics
print("\n=== Data Summary ===")
for table_name, df in data_mapping.items():
    if not df.empty:
        print(f"{table_name}: {len(df)} records processed")
        if 'created_at' in df.columns:
            print(f"  Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    else:
        print(f"{table_name}: No data")

print("\nScript completed successfully!")
