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
except subprocess.CalledProcessError as e:
    print(f"Error running {e.cmd}: {e}")
except NoDataToAggregateException:
    print("No data to aggregate")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Run the Strava data extract
print("\n\nRunning strava extraction...")
run_script('strava_data_pull.py')

# Run the dashboard script
print("\n\nRunning dashboard creation...")
run_script('app.py')