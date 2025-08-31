#!/usr/bin/env python3
"""
OAuth Helper for Whoop API v2

This script helps you obtain an access token using OAuth 2.0 Authorization Code flow.
This is the recommended authentication method for Whoop API v2.

Instructions:
1. Register your application at https://developer.whoop.com/
2. Get your CLIENT_ID and CLIENT_SECRET
3. Set the redirect URI in your app registration (can be http://localhost:8080 for development)
4. Run this script to get your access token
"""

import os
import requests
import webbrowser
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from dotenv import load_dotenv

load_dotenv()

# Configuration
CLIENT_ID = os.getenv("WHOOP_CLIENT_ID")
CLIENT_SECRET = os.getenv("WHOOP_CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8080"  # Must match your app registration
AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"

# Required scopes for data access
SCOPES = [
    "read:profile",
    "read:workout", 
    "read:sleep",
    "read:recovery",
    "read:cycles",
    "read:body_measurement"
]

class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/'):
            query_components = parse_qs(urlparse(self.path).query)
            
            if 'code' in query_components:
                self.server.auth_code = query_components['code'][0]
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'<html><body><h1>Authorization successful!</h1><p>You can close this window.</p></body></html>')
            elif 'error' in query_components:
                self.server.auth_error = query_components['error'][0]
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(f'<html><body><h1>Authorization failed!</h1><p>Error: {query_components["error"][0]}</p></body></html>'.encode())
    
    def log_message(self, format, *args):
        # Suppress log messages
        return

def start_callback_server():
    """Start a local server to handle OAuth callback"""
    server = HTTPServer(('localhost', 8080), CallbackHandler)
    server.auth_code = None
    server.auth_error = None
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    return server

def get_authorization_url():
    """Generate the authorization URL"""
    params = {
        'response_type': 'code',
        'client_id': CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'scope': ' '.join(SCOPES),
        'state': 'random_state_string'  # Use a secure random string in production
    }
    
    # URL encode the parameters properly
    from urllib.parse import urlencode
    query_string = urlencode(params)
    full_url = f"{AUTH_URL}?{query_string}"
    
    # Debug output
    print(f"\n🔍 DEBUG: Authorization URL components:")
    print(f"   Client ID: {CLIENT_ID}")
    print(f"   Redirect URI: {REDIRECT_URI}")
    print(f"   Full URL: {full_url}")
    print()
    
    return full_url

def exchange_code_for_token(auth_code):
    """Exchange authorization code for access token"""
    data = {
        'grant_type': 'authorization_code',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': auth_code,
        'redirect_uri': REDIRECT_URI
    }
    
    response = requests.post(TOKEN_URL, data=data)
    response.raise_for_status()
    return response.json()

def main():
    if not CLIENT_ID or not CLIENT_SECRET:
        print("ERROR: WHOOP_CLIENT_ID and WHOOP_CLIENT_SECRET must be set in your .env file")
        print("Please register your app at https://developer.whoop.com/ and add:")
        print("WHOOP_CLIENT_ID=your_client_id")
        print("WHOOP_CLIENT_SECRET=your_client_secret")
        return
    
    print("Starting OAuth 2.0 flow for Whoop API v2...")
    print(f"Client ID: {CLIENT_ID}")
    print(f"Redirect URI: {REDIRECT_URI}")
    print(f"Scopes: {', '.join(SCOPES)}")
    
    # Start callback server
    print("\nStarting callback server on localhost:8080...")
    server = start_callback_server()
    
    # Generate authorization URL and open browser
    auth_url = get_authorization_url()
    print(f"\nOpening browser to: {auth_url}")
    webbrowser.open(auth_url)
    
    print("\nWaiting for authorization callback...")
    print("Please authorize the application in your browser.")
    
    # Wait for callback
    while server.auth_code is None and server.auth_error is None:
        import time
        time.sleep(1)
    
    server.shutdown()
    
    if server.auth_error:
        print(f"Authorization failed: {server.auth_error}")
        return
    
    if server.auth_code:
        print("Authorization code received! Exchanging for access token...")
        try:
            token_data = exchange_code_for_token(server.auth_code)
            access_token = token_data['access_token']
            expires_in = token_data.get('expires_in', 'unknown')
            
            print(f"\nSuccess! Access token obtained:")
            print(f"Access Token: {access_token}")
            print(f"Expires in: {expires_in} seconds")
            
            print(f"\nAdd this to your .env file:")
            print(f"WHOOP_ACCESS_TOKEN={access_token}")
            
            # Optionally save to .env file
            env_path = '.env'
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    content = f.read()
                
                if 'WHOOP_ACCESS_TOKEN=' in content:
                    # Replace existing token
                    lines = content.split('\n')
                    new_lines = []
                    for line in lines:
                        if line.startswith('WHOOP_ACCESS_TOKEN='):
                            new_lines.append(f'WHOOP_ACCESS_TOKEN={access_token}')
                        else:
                            new_lines.append(line)
                    content = '\n'.join(new_lines)
                else:
                    # Add new token
                    content += f'\nWHOOP_ACCESS_TOKEN={access_token}'
                
                with open(env_path, 'w') as f:
                    f.write(content)
                
                print(f"Token saved to {env_path}")
            else:
                print(f".env file not found. Please create one with the access token.")
                
        except Exception as e:
            print(f"Error exchanging code for token: {e}")

if __name__ == "__main__":
    main()
