#!/usr/bin/env python
# coding: utf-8

import subprocess

def run_script(script_path):
    """Function to run a script."""
    subprocess.run(['python', script_path], check=True)

# Run time allocation script
print("\n\nRunning time allocation aggregation...")
try:
    run_script('aggregate_time_allocation_data.py')
    print("Time allocation data aggregated")
except subprocess.CalledProcessError as e:
    print(f"Error running {e.cmd}: {e}")
except NoDataToAggregateException:
    print("No data to aggregate")
except Exception as e:
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
print("\n\nRunning whoop extraction...")
try:
    run_script('get_whoop_data.py')
    print("Whoop data extracted")
except subprocess.CalledProcessError as e:
    print(f"Error running {e.cmd}: {e}")
except Exception as e:
    print(f"An unexpected error occurred extracting whoop data: {e}")

# Prep the data for unsupervised learning
print("\n\Prepping data for unsupervised learning...")
try:
    run_script('unsupervised_learning_ml_data_prep.py')
    print("Data prepped for unsupervised learning")
except subprocess.CalledProcessError as e:
    print(f"Error running {e.cmd}: {e}")
except Exception as e:
    print(f"An unexpected error occurred prepping data for unsupervised learning: {e}")

# Run the dashboard script
print("\n\nRunning dashboard creation...")
run_script('app.py')