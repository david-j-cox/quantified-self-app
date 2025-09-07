#!/usr/bin/env python
# coding: utf-8

import subprocess
import re
import os
from pathlib import Path

def run_script(script_path):
    """Function to run a script."""
    subprocess.run(['python', script_path], check=True)

def get_fresh_whoop_token():
    """Run OAuth helper and return the fresh access token."""
    try:
        # Run oauth_helper.py and capture output
        result = subprocess.run(['python', 'oauth_helper.py'], 
                              capture_output=True, text=True, check=True)
        
        # Extract access token from output using regex
        token_pattern = r'WHOOP_ACCESS_TOKEN=([^\s]+)'
        match = re.search(token_pattern, result.stdout)
        
        if match:
            new_token = match.group(1)
            print(f"New access token obtained: {new_token[:20]}...")
            return new_token
        else:
            print("Could not extract access token from OAuth output")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running oauth_helper.py: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error getting token: {e}")
        return None

def run_script_with_env(script_path, env_vars=None):
    """Function to run a script with additional environment variables."""
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    subprocess.run(['python', script_path], env=env, check=True)

# Run time allocation script
print("\n\nRunning time allocation aggregation...")
try:
    run_script('aggregate_time_allocation_data.py')
    print("Time allocation data aggregated")
except subprocess.CalledProcessError as e:
    print(f"Error running {e.cmd}: {e}")
except Exception as e:
    if "NoDataToAggregateException" in str(e):
        print("No data to aggregate")
    else:
        print(f"An unexpected error occurred allocating time data: {e}")

# Run the Strava data extract
print("\n\nRunning strava extraction...")
try:
    run_script('strava_data_pull.py')
    print("Strava data extracted")
except subprocess.CalledProcessError as e:
    print(f"Error running {e.cmd}: {e}")
except Exception as e:
    print(f"An unexpected error occurred extracting strava data: {e}")

# Run Whoop data extract
print("Getting fresh access token for whoop...")
fresh_token = get_fresh_whoop_token()

if fresh_token:
    print("\n\nRunning whoop extraction...")
    try:
        # Pass the fresh token as an environment variable
        run_script_with_env('get_whoop_data.py', {'WHOOP_ACCESS_TOKEN': fresh_token})
        print("Whoop data extracted")
    except subprocess.CalledProcessError as e:
        print(f"Error running {e.cmd}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred extracting whoop data: {e}")
else:
    print("Skipping Whoop data extraction due to token retrieval failure")

# Prep the data for unsupervised learning
print("\n\nPrepping data for unsupervised learning...")
try:
    run_script('unsupervised_ml_data_prep.py')
    print("Data prepped for unsupervised learning")
except subprocess.CalledProcessError as e:
    print(f"Error running {e.cmd}: {e}")
except Exception as e:
    print(f"An unexpected error occurred prepping data for unsupervised learning: {e}")

# Run the dashboard script
print("\n\nRunning dashboard creation...")
run_script('app.py')