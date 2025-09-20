#!/usr/bin/env python3
"""
Spotify Weekly Data Collection Script

This script fetches the last week's listening data from Spotify's Web API
and saves it in a format compatible with your existing streaming history data.

Setup required:
1. Create a Spotify app at https://developer.spotify.com/dashboard/
2. Get Client ID and Client Secret
3. Set redirect URI to http://localhost:8888/callback
4. Run initial setup to get refresh token
"""

import requests
import json
import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
from urllib.parse import urlencode
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import sys
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpotifyAPI:
    def __init__(self, client_id, client_secret, redirect_uri="https://open.spotify.com/user/cox.david.j"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        
        # API endpoints
        self.token_url = "https://accounts.spotify.com/api/token"
        self.auth_url = "https://accounts.spotify.com/authorize"
        self.recently_played_url = "https://api.spotify.com/v1/me/player/recently-played"
    
    def get_auth_url(self):
        """Generate the authorization URL for initial setup."""
        params = {
            'client_id': self.client_id,
            'response_type': 'code',
            'redirect_uri': self.redirect_uri,
            'scope': 'user-read-recently-played',
            'show_dialog': 'true'
        }
        return f"{self.auth_url}?{urlencode(params)}"
    
    def exchange_code_for_tokens(self, auth_code):
        """Exchange authorization code for access and refresh tokens."""
        payload = {
            'grant_type': 'authorization_code',
            'code': auth_code,
            'redirect_uri': self.redirect_uri,
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        
        response = requests.post(self.token_url, data=payload)
        
        if response.status_code == 200:
            data = response.json()
            self.access_token = data['access_token']
            self.refresh_token = data['refresh_token']
            self.token_expires_at = datetime.now() + timedelta(seconds=data['expires_in'])
            logger.info("Successfully obtained tokens")
            return True
        else:
            logger.error(f"Failed to get tokens: {response.text}")
            return False
    
    def refresh_access_token(self):
        """Refresh the access token using the refresh token."""
        if not self.refresh_token:
            logger.error("No refresh token available")
            return False
        
        payload = {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token,
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        
        response = requests.post(self.token_url, data=payload)
        
        if response.status_code == 200:
            data = response.json()
            self.access_token = data['access_token']
            self.token_expires_at = datetime.now() + timedelta(seconds=data['expires_in'])
            # Refresh token might be updated
            if 'refresh_token' in data:
                self.refresh_token = data['refresh_token']
            logger.info("Successfully refreshed access token")
            return True
        else:
            logger.error(f"Failed to refresh token: {response.text}")
            return False
    
    def ensure_valid_token(self):
        """Ensure we have a valid access token."""
        if not self.access_token or datetime.now() >= self.token_expires_at:
            return self.refresh_access_token()
        return True
    
    def get_recently_played(self, after_timestamp=None, limit=50):
        """
        Get recently played tracks.
        
        Args:
            after_timestamp: Unix timestamp in milliseconds (optional)
            limit: Number of items to return (max 50)
        """
        if not self.ensure_valid_token():
            logger.error("Cannot get valid access token")
            return None
        
        headers = {
            'Authorization': f'Bearer {self.access_token}'
        }
        
        params = {'limit': limit}
        if after_timestamp:
            params['after'] = after_timestamp
        
        response = requests.get(self.recently_played_url, headers=headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get recently played: {response.status_code} - {response.text}")
            return None
    
    def get_all_recently_played(self, after_timestamp=None):
        """Get all recently played tracks, handling pagination."""
        all_items = []
        next_url = None
        
        while True:
            if next_url:
                # Use the next URL for pagination
                headers = {'Authorization': f'Bearer {self.access_token}'}
                response = requests.get(next_url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                else:
                    logger.error(f"Failed to get paginated data: {response.status_code}")
                    break
            else:
                # First request
                data = self.get_recently_played(after_timestamp)
                if not data:
                    break
            
            all_items.extend(data['items'])
            
            # Check if there are more items
            next_url = data.get('next')
            if not next_url:
                break
            
            # Add a small delay to respect rate limits
            time.sleep(0.1)
        
        logger.info(f"Retrieved {len(all_items)} tracks")
        return all_items

class AuthServer(BaseHTTPRequestHandler):
    """Simple HTTP server to handle OAuth callback."""
    
    def do_GET(self):
        if self.path.startswith('/callback'):
            # Extract the authorization code from the query parameters
            from urllib.parse import urlparse, parse_qs
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)
            
            if 'code' in query_params:
                self.server.auth_code = query_params['code'][0]
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'<html><body><h1>Authorization successful!</h1><p>You can close this window.</p></body></html>')
            else:
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'<html><body><h1>Authorization failed!</h1></body></html>')
        else:
            self.send_response(404)
            self.end_headers()

def run_auth_server():
    """Run a simple HTTP server to capture the OAuth callback."""
    server = HTTPServer(('localhost', 8888), AuthServer)
    server.auth_code = None
    
    # Run server in a separate thread
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    return server

def get_last_timestamp_from_csv(csv_file):
    """Get the last timestamp from existing spotify_history.csv file."""
    try:
        if not os.path.exists(csv_file):
            logger.info(f"No existing CSV file found at {csv_file}")
            return None
        
        df = pd.read_csv(csv_file, low_memory=False)
        if df.empty:
            logger.info("CSV file exists but is empty")
            return None
        
        # Convert timestamp column to datetime with flexible parsing
        df['ts'] = pd.to_datetime(df['ts'], format='mixed')
        last_timestamp = df['ts'].max()
        
        logger.info(f"Last timestamp in CSV: {last_timestamp}")
        return last_timestamp
        
    except Exception as e:
        logger.error(f"Error reading existing CSV file: {e}")
        return None

def convert_spotify_to_extended_format(items):
    """Convert Spotify API response to match Extended Streaming History format."""
    converted_items = []
    
    for item in items:
        track = item['track']
        played_at = item['played_at']
        
        # Convert to Extended Streaming History format
        converted_item = {
            'ts': played_at,
            'date': pd.to_datetime(played_at).date().isoformat(),
            'hour': pd.to_datetime(played_at).hour,
            'master_metadata_track_name': track['name'],
            'master_metadata_album_artist_name': track['artists'][0]['name'] if track['artists'] else None,
            'master_metadata_album_album_name': track['album']['name'],
            'ms_played': track.get('duration_ms', 0),  # Full duration - API doesn't provide actual play time
            'seconds_played': track.get('duration_ms', 0) / 1000 if track.get('duration_ms') else 0,
            'minutes_played': track.get('duration_ms', 0) / 60000 if track.get('duration_ms') else 0,
            'platform': 'api',  # Mark as API source
            'conn_country': None,  # Not available in API
            'spotify_track_uri': track['uri'],
            'episode_name': None,
            'episode_show_name': None,
            'spotify_episode_uri': None,
            'reason_start': 'api_data',  # Mark as API source
            'reason_end': 'api_data',
            'shuffle': None,  # Not available in API
            'skipped': None,  # Not available in API
            'offline': False,
            'incognito_mode': None,  # Not available in API
            'offline_timestamp': None,
            'offline_datetime': None,
            'ip_addr': None,  # Not available in API
            'source_file': 'api_incremental_fetch'
        }
        
        converted_items.append(converted_item)
    
    return converted_items

def append_to_csv(new_data, csv_file):
    """Append new data to existing CSV file, removing duplicates."""
    if not new_data:
        logger.info("No new data to append")
        return
    
    # Convert new data to DataFrame
    new_df = pd.DataFrame(new_data)
    new_df['ts'] = pd.to_datetime(new_df['ts'])
    
    if os.path.exists(csv_file):
        # Read existing data
        existing_df = pd.read_csv(csv_file, low_memory=False)
        existing_df['ts'] = pd.to_datetime(existing_df['ts'], format='mixed')
        
        # Combine and remove duplicates based on timestamp and track URI
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Remove duplicates - keep the last occurrence (API data over historical data if same timestamp)
        combined_df = combined_df.drop_duplicates(
            subset=['ts', 'spotify_track_uri'], 
            keep='last'
        ).sort_values('ts')
        
        logger.info(f"Combined {len(existing_df)} existing + {len(new_df)} new = {len(combined_df)} total records (after deduplication)")
        
    else:
        # No existing file, just use new data
        combined_df = new_df.sort_values('ts')
        logger.info(f"Creating new CSV with {len(combined_df)} records")
    
    # Save the combined data
    combined_df.to_csv(csv_file, index=False)
    logger.info(f"Updated {csv_file}")
    
    return len(new_df)

def save_credentials(spotify_api, config_file):
    """Save API credentials to file."""
    config = {
        'client_id': spotify_api.client_id,
        'client_secret': spotify_api.client_secret,
        'redirect_uri': spotify_api.redirect_uri,
        'refresh_token': spotify_api.refresh_token,
        'access_token': spotify_api.access_token,
        'token_expires_at': spotify_api.token_expires_at.isoformat() if spotify_api.token_expires_at else None
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Credentials saved to {config_file}")

def load_credentials(config_file):
    """Load API credentials from file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        spotify_api = SpotifyAPI(
            config['client_id'],
            config['client_secret'],
            config['redirect_uri']
        )
        
        spotify_api.refresh_token = config.get('refresh_token')
        spotify_api.access_token = config.get('access_token')
        
        if config.get('token_expires_at'):
            spotify_api.token_expires_at = datetime.fromisoformat(config['token_expires_at'])
        
        return spotify_api
        
    except FileNotFoundError:
        logger.info(f"Config file {config_file} not found")
        return None
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return None

def setup_spotify_auth():
    """Interactive setup for Spotify API authentication."""
    print("=== Spotify API Setup ===")
    
    # Load environment variables
    load_dotenv()
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("ERROR: Missing Spotify credentials in .env file")
        print("Please add the following to your .env file:")
        print("SPOTIFY_CLIENT_ID=your_client_id_here")
        print("SPOTIFY_CLIENT_SECRET=your_client_secret_here")
        return None
    
    print(f"Found Spotify credentials in .env file")
    spotify_api = SpotifyAPI(client_id, client_secret)
    
    print("\n=== Manual Authorization Process ===")
    print("1. A browser window will open for Spotify authorization")
    print("2. Log in to Spotify and authorize the application")
    print("3. You'll be redirected to your profile page")
    print("4. Copy the authorization code from the URL")
    print()
    input("Press Enter to continue...")
    
    # Open browser for authorization
    auth_url = spotify_api.get_auth_url()
    print(f"Opening browser to: {auth_url}")
    webbrowser.open(auth_url)
    
    print("\n=== After Authorization ===")
    print("After you authorize the app, you'll be redirected to a URL like:")
    print("https://open.spotify.com/user/cox.david.j?code=AQA...")
    print()
    print("Look for '?code=' in the URL and copy everything after it (up to any '&' symbol)")
    print()
    
    auth_code = input("Enter the authorization code from the URL: ").strip()
    
    if auth_code:
        print("Authorization code received!")
        if spotify_api.exchange_code_for_tokens(auth_code):
            print("Setup completed successfully!")
            return spotify_api
        else:
            print("Failed to exchange authorization code for tokens")
            return None
    else:
        print("No authorization code provided")
        return None

def fetch_incremental_data(data_dir, config_file):
    """Fetch new Spotify data since the last entry in spotify_history.csv."""
    # Load environment variables
    load_dotenv()
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        logger.error("Missing Spotify credentials in .env file")
        logger.error("Please add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to your .env file")
        return False
    
    # Define the main CSV file path
    csv_file = Path(data_dir) / "spotify_history.csv"
    
    # Check last timestamp in existing data
    last_timestamp = get_last_timestamp_from_csv(csv_file)
    
    if last_timestamp:
        # Add a small buffer (1 minute) to avoid missing data due to timing
        start_time = last_timestamp + timedelta(minutes=1)
        after_timestamp = int(start_time.timestamp() * 1000)  # Convert to milliseconds
        logger.info(f"Fetching data from {start_time.strftime('%Y-%m-%d %H:%M:%S')} onwards...")
    else:
        # No existing data, fetch last week
        start_time = datetime.now() - timedelta(days=7)
        after_timestamp = int(start_time.timestamp() * 1000)
        logger.info(f"No existing data found. Fetching data from {start_time.strftime('%Y-%m-%d %H:%M:%S')} onwards...")
    
    # Try to load existing credentials
    spotify_api = load_credentials(config_file)
    
    if not spotify_api:
        print("No saved credentials found. Running setup...")
        spotify_api = setup_spotify_auth()
        if not spotify_api:
            logger.error("Failed to set up Spotify API")
            return False
        
        # Save credentials for future use
        save_credentials(spotify_api, config_file)
    
    # Fetch recently played tracks
    items = spotify_api.get_all_recently_played(after_timestamp)
    
    if not items:
        logger.info("No new data retrieved since last run (this is normal for frequent automated runs)")
        return True  # Not an error, just no new data
    
    # Convert to Extended Streaming History format
    converted_items = convert_spotify_to_extended_format(items)
    
    # Append to main CSV file
    new_records = append_to_csv(converted_items, csv_file)
    
    # Also save as timestamped backup JSON in the Spotify Extended Streaming History folder
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    spotify_history_dir = Path(data_dir) / "Spotify Extended Streaming History"
    backup_file = spotify_history_dir / f"API_Fetch_{timestamp_str}.json"
    
    # Ensure the directory exists
    spotify_history_dir.mkdir(exist_ok=True)
    
    with open(backup_file, 'w') as f:
        json.dump(converted_items, f, indent=2)
    
    logger.info(f"Backup saved to {backup_file}")
    
    # Update saved credentials in case tokens were refreshed
    save_credentials(spotify_api, config_file)
    
    # Show summary
    if converted_items:
        df = pd.DataFrame(converted_items)
        df['ts'] = pd.to_datetime(df['ts'])
        logger.info("=== INCREMENTAL UPDATE SUMMARY ===")
        logger.info(f"New data range: {df['ts'].min()} to {df['ts'].max()}")
        logger.info(f"New tracks fetched: {len(df)}")
        logger.info(f"Records added to CSV: {new_records}")
        logger.info(f"Unique new tracks: {df['master_metadata_track_name'].nunique()}")
        logger.info(f"Unique new artists: {df['master_metadata_album_artist_name'].nunique()}")
        
        # Show current dataset info
        current_df = pd.read_csv(csv_file, low_memory=False)
        current_df['ts'] = pd.to_datetime(current_df['ts'], format='mixed')
        logger.info(f"Total dataset now spans: {current_df['ts'].min()} to {current_df['ts'].max()}")
        logger.info(f"Total records in dataset: {len(current_df):,}")
    
    return True

def main():
    """Main function."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "Data"
    config_file = project_root / "Scripts" / "spotify_config.json"
    
    logger.info("Starting Spotify incremental data collection (4-hour automated run)...")
    
    if len(sys.argv) > 1 and sys.argv[1] == 'setup':
        logger.info("Running setup mode...")
        spotify_api = setup_spotify_auth()
        if spotify_api:
            save_credentials(spotify_api, config_file)
            logger.info("Setup completed! You can now run the script without 'setup' to fetch data.")
        else:
            logger.error("Setup failed")
        return
    
    success = fetch_incremental_data(data_dir, config_file)
    
    if success:
        logger.info("Incremental data collection completed successfully!")
    else:
        logger.error("Incremental data collection failed")

if __name__ == "__main__":
    main()

