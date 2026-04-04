#!/usr/bin/env python
"""Daily food tracker update — syncs data and regenerates food plots only.

Run by Launch Agent at 7:00 AM daily. Does NOT regenerate the full
dashboard or other data sources — those run via run_all.py on Sundays.
"""

import subprocess
import sys


def run_script(script_path):
    subprocess.run([sys.executable, script_path], check=True)


print("=== Daily Food Tracker Update ===")

# Step 1: Sync food tracker data from Fly.io
print("\nSyncing food tracker data...")
try:
    run_script('sync_food_tracker.py')
    print("Food tracker data synced")
except subprocess.CalledProcessError as e:
    print(f"Error syncing food tracker: {e}")
except Exception as e:
    print(f"Error syncing food tracker: {e}")

# Step 2: Regenerate food tracking plots
print("\nGenerating food tracking plots...")
try:
    run_script('generate_food_plots.py')
    print("Food tracking plots generated")
except subprocess.CalledProcessError as e:
    print(f"Error generating food plots: {e}")
except Exception as e:
    print(f"Error generating food plots: {e}")

print("\n=== Daily Food Tracker Update Complete ===")
