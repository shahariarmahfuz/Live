# -*- coding: utf-8 -*-
import requests
from flask import Flask, render_template, request, redirect, url_for, send_file, Response, abort, flash, session, jsonify, current_app
import io
import os
import json
import html
from urllib.parse import urljoin, urlparse
from collections import defaultdict # ক্যাটেগরি গ্রুপিংয়ের জন্য
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import time

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Flask Application Setup ---
app = Flask(__name__)

# Secret key for session management (used by main.py's admin login)
# Change this key in a production environment!
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_key_for_multichannel_v2_merged')

# Admin password for the main admin panel (/admin)
# Change this password in a production environment!
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'password123')

# --- API Endpoints & Configuration ---

# Data server base URL (Used by both sets of API calls)
# এটি আপনার actual PythonAnywhere Web App URL হতে হবে
# যেমন: https://your-username.pythonanywhere.com
PYTHONANYWHERE_BASE_URL = os.environ.get(
    'DATABASE_API_URL', # Use the same env var name as code 2 for consistency if possible
    "https://itachi321.pythonanywhere.com" # আপনার আসল URL দিয়ে প্রতিস্থাপন করুন
)

# API Endpoint for the main channel database (used by IPTV proxy/admin logic)
DATABASE_API_URL = f"{PYTHONANYWHERE_BASE_URL}/api/channels"

# API Endpoint for M3U channel data (used by M3U playlist generator/admin1 logic)
API_M3U_CHANNELS_URL = f"{PYTHONANYWHERE_BASE_URL}/api/m3u/channels"

# API Endpoints for Toffee data synchronization (used by scheduler logic)
TOFFEE_UPDATE_API_URL = f"{PYTHONANYWHERE_BASE_URL}/api/toffee/update"
TOFFEE_CHANNELS_API_URL = f"{PYTHONANYWHERE_BASE_URL}/api/toffee/channels"

# GitHub source for Toffee data
GITHUB_JSON_URL = os.environ.get(
    'GITHUB_JSON_URL',
    'https://raw.githubusercontent.com/byte-capsule/Toffee-Channels-Link-Headers/refs/heads/main/toffee_NS_Player.m3u'
)

# --- Constants ---

# Predefined categories for the main admin panel (/admin)
PREDEFINED_CATEGORIES = [
    "News", "Entertainment", "Movies", "Sports", "Music",
    "Kids", "Lifestyle", "Knowledge", "Religious", "International",
    "Regional (BD)", "Regional (IN)", "Regional (PK)", "Regional (Other)",
    "Documentary", "Drama", "Comedy", "Action", "Sci-Fi", "Other"
]
PREDEFINED_CATEGORIES.sort() # বর্ণানুক্রমে সাজান

# --- Helper Functions for DATABASE_API_URL (from original code 1) ---

def get_remote_channels():
    """ডাটাবেস সার্ভার থেকে সব চ্যানেল ডেটা আনে (নাম ও ক্যাটাগরি সহ)"""
    try:
        logger.info(f"Fetching channels from {DATABASE_API_URL}")
        response = requests.get(DATABASE_API_URL, timeout=15) # Increased timeout slightly
        response.raise_for_status()
        channels_list = response.json()
        # API থেকে পাওয়া লিস্টকে ডিকশনারিতে রূপান্তর করুন (channel_id কে কী হিসাবে)
        channels_dict = {ch.get('channel_id'): ch for ch in channels_list if ch.get('channel_id')} # Ensure channel_id exists
        logger.info(f"Successfully fetched {len(channels_dict)} channels from {DATABASE_API_URL}")
        return channels_dict
    except requests.exceptions.Timeout:
        logger.error(f"Timeout error fetching channels from remote API {DATABASE_API_URL}.")
        flash(f"Timeout connecting to database server.", "error")
        return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching channels from remote API {DATABASE_API_URL}: {e}")
        flash(f"Error connecting to database server: {e}", "error")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON response from database server {DATABASE_API_URL}.")
        flash("Received invalid data from database server.", "error")
        return {}
    except Exception as e: # অন্য কোনো এরর ধরতে
        logger.error(f"An unexpected error occurred while fetching channels from {DATABASE_API_URL}: {e}")
        flash("An unexpected error occurred while fetching channels.", "error")
        return {}

def get_remote_channel(channel_id):
    """ডাটাবেস সার্ভার থেকে নির্দিষ্ট চ্যানেল ডেটা আনে (নাম ও ক্যাটাগরি সহ)"""
    try:
        url = f"{DATABASE_API_URL}/{channel_id.lower()}"
        logger.info(f"Fetching channel '{channel_id}' from {url}")
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            logger.warning(f"Channel '{channel_id}' not found at {url}")
            return None
        response.raise_for_status()
        channel_data = response.json()
        logger.info(f"Successfully fetched channel '{channel_id}' from {url}")
        return channel_data
    except requests.exceptions.Timeout:
        logger.error(f"Timeout error fetching channel '{channel_id}' from remote API {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching channel '{channel_id}' from remote API {url}: {e}")
        return None
    except json.JSONDecodeError:
         logger.error(f"Error decoding JSON for channel '{channel_id}' from {url}.")
         return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching channel {channel_id} from {url}: {e}")
        return None

def save_channel_remote(channel_id, source_url, name, category):
    """ডাটাবেস সার্ভারে চ্যানেল যোগ বা আপডেট করে (নাম ও ক্যাটাগরি সহ)"""
    payload = {
        "channel_id": channel_id.strip().lower(),
        "source_url": html.unescape(source_url.strip()),
        "name": name.strip(), # নাম যোগ করুন
        "category": category # ক্যাটাগরি যোগ করুন
    }
    try:
        logger.info(f"Saving channel '{payload['channel_id']}' to {DATABASE_API_URL} with payload: {payload}")
        # POST রিকোয়েস্ট পাঠান
        response = requests.post(DATABASE_API_URL, json=payload, timeout=15) # Increased timeout slightly
        response.raise_for_status() # HTTP errors (4xx, 5xx) ধরবে

        # API সফল হলে 201 বা 200 কোড ফেরত দেবে
        if response.status_code in [200, 201]:
            api_response_data = response.json()
            message = api_response_data.get("message", "Success")
            logger.info(f"Channel '{payload['channel_id']}' saved successfully via API. Message: {message}")
            return True, message
        else:
            # অপ্রত্যাশিত সফল স্ট্যাটাস কোড (যদিও raise_for_status এটি ধরতে পারে)
            error_message = f"Unexpected success status code {response.status_code} from API {DATABASE_API_URL} during save."
            logger.error(error_message)
            return False, error_message

    except requests.exceptions.Timeout:
        error_message = f"Timeout error saving channel '{payload['channel_id']}' via API {DATABASE_API_URL}."
        logger.error(error_message)
        return False, "API communication timeout during save."
    except requests.exceptions.RequestException as e:
        error_message = f"Error saving channel '{payload['channel_id']}' via API {DATABASE_API_URL}: {e}"
        api_error_details = ""
        # API থেকে আসা এরর মেসেজ দেখানোর চেষ্টা করুন
        try:
            if e.response is not None:
                api_error_details = e.response.json().get("error", str(e.response.text))
                error_message += f" (API Status: {e.response.status_code}, API Error: {api_error_details})"
        except (json.JSONDecodeError, AttributeError):
             if e.response is not None:
                 error_message += f" (API Status: {e.response.status_code}, Response: {e.response.text[:100]}...)"
             else:
                 error_message += " (No response received from API)"

        logger.error(error_message)
        return False, f"API communication error: {api_error_details or str(e)}" # ব্যবহারকারীকে দেখানোর জন্য সংক্ষিপ্ত বার্তা

    except json.JSONDecodeError:
         # POST রিকোয়েস্টের Response ডিকোড করতে সমস্যা হলে
         error_message = f"Error decoding JSON response from API {DATABASE_API_URL} after save attempt."
         logger.error(error_message)
         return False, error_message
    except Exception as e:
        error_message = f"An unexpected error occurred during remote save to {DATABASE_API_URL}: {e}"
        logger.error(error_message)
        return False, error_message

def delete_channel_remote(channel_id):
    """ডাটাবেস সার্ভার থেকে চ্যানেল মুছে ফেলে"""
    channel_id = channel_id.lower()
    try:
        url = f"{DATABASE_API_URL}/{channel_id}"
        logger.info(f"Deleting channel '{channel_id}' from {url}")
        response = requests.delete(url, timeout=15) # Increased timeout slightly

        if response.status_code == 404:
            logger.warning(f"Channel '{channel_id}' not found on remote server {url} for deletion.")
            return False, "Channel not found on database server."

        response.raise_for_status() # অন্যান্য HTTP Error (যেমন 500) চেক করুন

        # সফলভাবে ডিলিট হলে (সাধারণত 200 বা 204 No Content)
        logger.info(f"Channel '{channel_id}' delete request successful via API {url}.")
        message = "Deleted successfully"
        try:
             # API যদি মেসেজ পাঠায়, সেটা ব্যবহার করুন
             message = response.json().get("message", message)
        except json.JSONDecodeError:
             # যদি কোনো JSON বডি না থাকে (যেমন 204 No Content)
             pass
        return True, message

    except requests.exceptions.Timeout:
        error_message = f"Timeout error deleting channel '{channel_id}' via API {url}."
        logger.error(error_message)
        return False, "API communication timeout during delete."
    except requests.exceptions.RequestException as e:
        error_message = f"Error deleting channel '{channel_id}' via API {url}: {e}"
        api_error_details = ""
        try:
            if e.response is not None:
                 api_error_details = e.response.json().get("error", str(e.response.text))
                 error_message += f" (API Status: {e.response.status_code}, API Error: {api_error_details})"
        except (json.JSONDecodeError, AttributeError):
             if e.response is not None:
                 error_message += f" (API Status: {e.response.status_code}, Response: {e.response.text[:100]}...)"
             else:
                 error_message += " (No response received from API)"
        logger.error(error_message)
        return False, f"API communication error during delete: {api_error_details or str(e)}"
    except Exception as e:
        error_message = f"An unexpected error occurred during remote delete from {url}: {e}"
        logger.error(error_message)
        return False, error_message

# --- Helper Functions for Toffee Data Sync (from original code 2) ---

def fetch_from_github():
    """GitHub থেকে Toffee JSON ডেটা fetch করে।"""
    logger.info(f"Attempting to fetch data from GitHub source: {GITHUB_JSON_URL}")
    try:
        response = requests.get(GITHUB_JSON_URL, timeout=20)
        response.raise_for_status()
        try:
            data = response.json()
            if isinstance(data, list):
                 logger.info(f"Successfully fetched {len(data)} channel entries from GitHub.")
                 return data
            else:
                 logger.error(f"Fetched data from GitHub is not a JSON list. Type: {type(data)}")
                 return None
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to decode JSON from GitHub URL. Error: {json_err}")
            logger.error(f"Response text (first 500 chars): {response.text[:500]}") # Log response text on decode error
            return None
    except requests.exceptions.Timeout:
        logger.error(f"Timeout error while fetching data from GitHub.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from GitHub source: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during GitHub fetch: {e}")
        return None

def update_database_toffee(channel_data):
    """প্রাপ্ত Toffee চ্যানেল ডেটা PythonAnywhere ডাটাবেস সার্ভারে পাঠায়।"""
    if not channel_data:
        logger.warning("No Toffee channel data provided to update database.")
        return False

    update_endpoint = TOFFEE_UPDATE_API_URL # Use defined constant
    logger.info(f"Attempting to send Toffee data to database API endpoint: {update_endpoint}")
    try:
        headers = {'Content-Type': 'application/json'}
        # Longer timeout for potentially large data uploads
        response = requests.post(update_endpoint, json=channel_data, headers=headers, timeout=60)
        response.raise_for_status()
        try:
             result = response.json()
             logger.info(f"Toffee Database update API response: {result}")
             # Check for success based on API response structure
             if response.ok and ("message" in result or result.get("processed_count", 0) > 0 or result.get("added_count", 0) > 0):
                 logger.info("Toffee Database update reported success.")
                 return True
             else:
                 logger.warning(f"Toffee Database update might have issues based on response: {result}")
                 return False
        except json.JSONDecodeError:
             # Handle successful responses that might not return JSON (e.g., 204 No Content)
             if response.ok:
                 logger.info(f"Toffee Database update successful (Status: {response.status_code}, Non-JSON response).")
                 return True
             else:
                 logger.warning(f"Toffee Database update response was not JSON and status was not OK. Status: {response.status_code}, Text: {response.text[:200]}")
                 return False
    except requests.exceptions.Timeout:
        logger.error(f"Timeout error while sending Toffee data to database API: {update_endpoint}")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending Toffee data to database API {update_endpoint}: {e}")
        if e.response is not None:
            logger.error(f"Database API Response Status: {e.response.status_code}")
            logger.error(f"Database API Response Body (first 500 chars): {e.response.text[:500]}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during Toffee database update: {e}")
        return False

def sync_toffee_data_task(task_name="Scheduled"):
    """GitHub থেকে Toffee ডেটা fetch করে ডাটাবেস আপডেট করার scheduled টাস্ক।"""
    logger.info(f"Starting Toffee data sync task ({task_name})...")
    # Need app context for potential Flask operations within the task, although not strictly needed here yet
    with app.app_context():
        channel_data = fetch_from_github()
        if channel_data:
            success = update_database_toffee(channel_data)
            if success:
                logger.info(f"Toffee data sync task ({task_name}) completed successfully.")
            else:
                logger.error(f"Toffee data sync task ({task_name}) failed during database update.")
        else:
            logger.error(f"Toffee data sync task ({task_name}) failed: Could not fetch data from GitHub.")

def get_toffee_channels_from_db():
    """PythonAnywhere ডাটাবেস সার্ভার থেকে Toffee চ্যানেলের তালিকা আনে (JSON list হিসেবে)।"""
    get_endpoint = TOFFEE_CHANNELS_API_URL # Use defined constant
    logger.info(f"Attempting to fetch Toffee channels from database API endpoint: {get_endpoint}")
    try:
        response = requests.get(get_endpoint, timeout=30) # Increased timeout
        response.raise_for_status()
        channels = response.json()
        if isinstance(channels, list):
             logger.info(f"Successfully fetched {len(channels)} Toffee channels as JSON list from database API.")
             return channels
        else:
             logger.error(f"Received unexpected data format (not a list) from Toffee database API. Type: {type(channels)}")
             return None
    except requests.exceptions.Timeout:
         logger.error(f"Timeout error while fetching Toffee channels from database API: {get_endpoint}")
         return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Toffee channels from database API {get_endpoint}: {e}")
        if e.response is not None:
             logger.error(f"DB API Response Status: {e.response.status_code}")
             logger.error(f"DB API Response Body (first 500 chars): {e.response.text[:500]}")
        return None
    except json.JSONDecodeError as json_err:
        logger.error(f"Failed to decode JSON response from Toffee database API. Error: {json_err}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching Toffee channels from DB: {e}")
        return None

# --- Scheduler Setup (from original code 2) ---
scheduler = BackgroundScheduler(daemon=True, timezone='Asia/Dhaka') # Use appropriate timezone
# Schedule the Toffee data sync task
scheduler.add_job(sync_toffee_data_task, 'interval', hours=1, id='hourly_toffee_sync_job', replace_existing=True)

# Start the scheduler within a try-except block
try:
    scheduler.start()
    logger.info("Scheduler started successfully for hourly Toffee data updates.")
except Exception as e:
    logger.error(f"CRITICAL: Failed to start scheduler: {e}", exc_info=True)

# --- Initial Data Sync on Startup ---
# Perform an initial sync when the application starts
# Run this within the application context
def perform_initial_sync():
     with app.app_context():
         logger.info("Performing initial Toffee data sync on application startup...")
         sync_toffee_data_task(task_name="Startup")
         logger.info("Initial Toffee data sync task initiated.")

# Call the initial sync function. It's better to run it once the app context is available.
# We can trigger this before the first request or keep it within the scheduler startup block.
# Let's keep it simple and run it after scheduler starts but before server runs.
# Using app.app_context() ensures it runs correctly even if called outside a request.
perform_initial_sync()


# --- Routes from main.py / code 1 (Primary IPTV Proxy & Admin) ---

@app.route('/admin', methods=['GET'])
def admin_panel():
    """
    Main Admin Panel - Handles login and displays channels fetched from DATABASE_API_URL.
    Uses admin_multi.html template.
    """
    if not session.get('logged_in'):
        # লগইন না থাকলে শুধু লগইন ফরম দেখান, ক্যাটেগরি পাঠানোর দরকার নেই
        return render_template('admin_multi.html', channels=None, categories=None) # categories=None যোগ করুন

    # লগইন করা থাকলে চ্যানেল এবং ক্যাটেগরি লোড করুন
    current_channels = get_remote_channels() # Fetches from DATABASE_API_URL
    # টেমপ্লেটে ক্যাটেগরি লিস্ট পাঠান
    return render_template('admin_multi.html', channels=current_channels, categories=PREDEFINED_CATEGORIES)

@app.route('/admin/login', methods=['POST'])
def admin_login():
    """Handles login for the main admin panel (/admin)."""
    password_attempt = request.form.get('password')
    if password_attempt == ADMIN_PASSWORD:
        session['logged_in'] = True
        flash('Login successful!', 'success')
        logger.info("Admin login successful.")
    else:
        session.pop('logged_in', None)
        flash('Incorrect password.', 'error')
        logger.warning("Failed admin login attempt.")
    return redirect(url_for('admin_panel'))

@app.route('/admin/logout', methods=['POST'])
def admin_logout():
    """Handles logout for the main admin panel (/admin)."""
    session.pop('logged_in', None)
    flash('You have been logged out.', 'success')
    logger.info("Admin logout successful.")
    return redirect(url_for('admin_panel'))

@app.route('/admin/channels', methods=['POST'])
def add_or_update_channel_main():
    """
    Adds or updates a channel using the DATABASE_API_URL.
    Used by the /admin panel. Requires login.
    """
    if not session.get('logged_in'):
        logger.warning("Unauthorized attempt to access /admin/channels")
        abort(403)

    # ফরম থেকে ডেটা সংগ্রহ করুন
    channel_id = request.form.get('channel_id', '').strip().lower()
    source_url_raw = request.form.get('source_url', '').strip()
    channel_name = request.form.get('channel_name', '').strip()
    category = request.form.get('category', '').strip()

    # বেসিক ভ্যালিডেশন
    if not channel_id or not source_url_raw or not channel_name or not category:
        flash('Channel ID, Source URL, Channel Name, and Category cannot be empty.', 'error')
        logger.warning(f"Validation failed for adding/updating channel '{channel_id}': Missing fields.")
        return redirect(url_for('admin_panel'))

    # চ্যানেল আইডি ভ্যালিডেশন
    if not channel_id.replace('-', '').replace('_', '').isalnum() or ' ' in channel_id: # Allow hyphen and underscore
         flash('Channel ID should only contain letters, numbers, hyphens (-), or underscores (_), and no spaces.', 'error')
         logger.warning(f"Validation failed for adding/updating channel '{channel_id}': Invalid characters in ID.")
         return redirect(url_for('admin_panel'))

    # ক্যাটাগরি ভ্যালিডেশন (প্রিডিফাইন্ড লিস্টে আছে কিনা)
    if category not in PREDEFINED_CATEGORIES:
         flash(f'Invalid category selected: "{category}". Please choose from the list.', 'error')
         logger.warning(f"Validation failed for adding/updating channel '{channel_id}': Invalid category '{category}'.")
         return redirect(url_for('admin_panel'))

    # URL ভ্যালিডেশন (বেসিক)
    try:
        parsed_url = urlparse(html.unescape(source_url_raw))
        if not all([parsed_url.scheme, parsed_url.netloc]):
             raise ValueError("Invalid URL format")
    except ValueError:
         flash('Invalid Source URL format. Please provide a full URL (e.g., http://...).', 'error')
         logger.warning(f"Validation failed for adding/updating channel '{channel_id}': Invalid Source URL '{source_url_raw}'.")
         return redirect(url_for('admin_panel'))

    # রিমোট API তে সেভ করার চেষ্টা করুন (এখন নাম ও ক্যাটাগরি সহ)
    success, message = save_channel_remote(channel_id, source_url_raw, channel_name, category)

    if success:
        flash(f'Channel "{channel_id}" (Name: {channel_name}) processed successfully. API Message: {message}', 'success')
        logger.info(f"Successfully saved channel '{channel_id}' via /admin panel.")
    else:
        flash(f'Failed to save channel "{channel_id}". Error: {message}', 'error')
        logger.error(f"Failed to save channel '{channel_id}' via /admin panel. Error: {message}")

    return redirect(url_for('admin_panel'))


@app.route('/admin/channels/delete/<channel_id>', methods=['POST'])
def delete_channel_main(channel_id):
    """
    Deletes a channel using the DATABASE_API_URL.
    Used by the /admin panel. Requires login.
    """
    if not session.get('logged_in'):
        logger.warning(f"Unauthorized attempt to delete channel '{channel_id}' via /admin")
        abort(403)

    channel_id_lower = channel_id.lower()

    # রিমোট API ব্যবহার করে ডিলিট করুন
    success, message = delete_channel_remote(channel_id_lower)

    if success:
        flash(f'Channel "{channel_id_lower}" delete request sent. API Message: {message}', 'success')
        logger.info(f"Successfully sent delete request for channel '{channel_id_lower}' via /admin panel.")
    else:
        flash(f'Failed to delete channel "{channel_id_lower}". Error: {message}', 'error')
        logger.error(f"Failed to delete channel '{channel_id_lower}' via /admin panel. Error: {message}")

    return redirect(url_for('admin_panel'))


@app.route('/live/<channel_id>.m3u8')
def serve_m3u8(channel_id):
    """
    Serves the M3U8 playlist by fetching it from the source URL
    obtained from DATABASE_API_URL and rewriting relative paths.
    """
    channel_id = channel_id.lower()
    channel_info = get_remote_channel(channel_id) # Fetches from DATABASE_API_URL

    if channel_info is None:
        logger.warning(f"M3U8 request failed: Channel '{channel_id}' not found or database connection failed.")
        abort(404, description=f"Channel '{channel_id}' not found or database connection failed.")

    source_url = channel_info.get('source_url')
    # Base URL should ideally come from the API per channel if sources differ significantly
    # If API provides 'base_url', use it, otherwise derive from source_url
    base_url = channel_info.get('base_url')
    if not base_url and source_url:
        parsed_source = urlparse(source_url)
        base_url = f"{parsed_source.scheme}://{parsed_source.netloc}{os.path.dirname(parsed_source.path)}/"
        base_url = base_url.rstrip('/') + '/' # Ensure it ends with a slash

    channel_name = channel_info.get('name', channel_id) # নাম থাকলে ব্যবহার করুন, না হলে আইডি

    if not source_url or not base_url:
         logger.error(f"Configuration error from API: Source or Base URL missing/unusable for channel '{channel_id}'. API Data: {channel_info}, Derived Base: {base_url}")
         abort(500, description=f"Configuration error for channel '{channel_id}'. Missing source or base URL.")

    logger.info(f"Fetching M3U8 for channel '{channel_name}' ({channel_id}) from: {source_url} with base: {base_url}")

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(source_url, timeout=15, headers=headers, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '').lower()
        if 'mpegurl' not in content_type and 'x-mpegurl' not in content_type and 'octet-stream' not in content_type: # Allow octet-stream too
             logger.warning(f"Warning: Content-Type for channel '{channel_id}' is '{content_type}', not M3U8. Trying to process anyway.")

        original_m3u8_content = response.text
        modified_lines = []
        effective_base_url = base_url # Use the determined base_url

        for line in original_m3u8_content.splitlines():
            line = line.strip()
            if not line:
                continue

            # EXT-X-KEY URI rewriting
            if line.startswith('#EXT-X-KEY'):
                uri_match = requests.utils.parse_dict_header(line.split(':', 1)[1]).get('uri') # More robust parsing
                if uri_match:
                    uri_part = uri_match.strip('"') # Remove potential quotes
                    if effective_base_url and not urlparse(uri_part).scheme:
                        full_key_uri = urljoin(effective_base_url, uri_part)
                        # Reconstruct the line carefully
                        line = line.replace(f'URI="{uri_part}"', f'URI="{full_key_uri}"') # Simple replacement, might need adjustment for complex cases
                        # logger.debug(f"Rewriting KEY URI: '{uri_part}' -> '{full_key_uri}'")
                    modified_lines.append(line)
                else:
                     modified_lines.append(line)
                     logger.warning(f"Warning: Could not parse URI from EXT-X-KEY line for channel {channel_id}: {line}")

            # Segment or sub-playlist URI rewriting
            elif not line.startswith('#'):
                uri_part = line
                parsed_uri = urlparse(uri_part)

                if parsed_uri.scheme:
                    modified_lines.append(line) # Absolute URL, keep as is
                    # logger.debug(f"Keeping absolute URI: {line}")
                elif uri_part.lower().endswith(('.m3u8', '.m3u')): # Sub-playlist
                     if effective_base_url:
                         absolute_sublist_url = urljoin(effective_base_url, uri_part)
                         modified_lines.append(absolute_sublist_url)
                         # logger.debug(f"Rewriting nested playlist URI: '{uri_part}' -> '{absolute_sublist_url}' (Absolute)")
                     else:
                        modified_lines.append(line)
                        logger.warning(f"Warning: Keeping relative nested playlist URI (no effective base_url): {line} for channel {channel_id}")
                else: # Segment (.ts, .aac, etc.)
                    # Proxy the segment through this server using relative URL
                    # Use _external=False for internal URL generation relative to current server
                    proxy_segment_url = url_for('serve_segment', channel_id=channel_id, segment=uri_part, _external=False)
                    modified_lines.append(proxy_segment_url)
                    # logger.debug(f"Rewriting segment URI: '{uri_part}' -> '{proxy_segment_url}' (Proxied)")

            # Other M3U8 tags (keep unchanged)
            else:
                modified_lines.append(line)

        modified_m3u8_content = "\n".join(modified_lines)
        # logger.debug(f"Modified M3U8 content for {channel_id}:\n{modified_m3u8_content[:500]}...") # Log first 500 chars for debugging
        return Response(modified_m3u8_content, mimetype='application/vnd.apple.mpegurl')

    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching manifest for channel '{channel_id}' from {source_url}")
        abort(504, description=f"Timeout fetching manifest for channel '{channel_id}'")
    except requests.exceptions.RequestException as e:
        status = e.response.status_code if e.response is not None else 502
        logger.error(f"Error fetching original M3U8 for channel '{channel_id}' ({source_url}). Status: {status}. Error: {e}")
        abort(status if status in [403, 404] else 502, description=f"Could not fetch manifest for channel '{channel_id}'. Origin error: {status}")
    except Exception as e:
        logger.error(f"Unexpected error processing manifest for channel '{channel_id}': {e}", exc_info=True)
        abort(500, description="Internal server error processing manifest.")


@app.route('/stream/<channel_id>/<path:segment>')
def serve_segment(channel_id, segment):
    """
    Proxies segment requests, fetching segment data from the original source
    using the base_url obtained from DATABASE_API_URL.
    """
    channel_id = channel_id.lower()
    channel_info = get_remote_channel(channel_id) # Fetches from DATABASE_API_URL

    if channel_info is None:
        logger.warning(f"Segment request failed: Channel '{channel_id}' not found or DB connection failed.")
        abort(404, description=f"Channel '{channel_id}' not found for segment request or DB connection failed.")

    # Determine base URL again, consistent with serve_m3u8 logic
    source_url = channel_info.get('source_url')
    base_url = channel_info.get('base_url')
    if not base_url and source_url:
        parsed_source = urlparse(source_url)
        base_url = f"{parsed_source.scheme}://{parsed_source.netloc}{os.path.dirname(parsed_source.path)}/"
        base_url = base_url.rstrip('/') + '/'

    if not base_url:
        logger.error(f"Error: Base URL not found/determined for channel '{channel_id}' during segment request. API Data: {channel_info}")
        abort(500, description=f"Configuration error: Base URL missing or unusable for channel '{channel_id}'.")

    try:
        # Ensure segment path is correctly joined, handling potential leading slashes in segment
        original_segment_url = urljoin(base_url, segment.lstrip('/'))
        # logger.debug(f"Fetching segment '{segment}' for channel '{channel_id}' from: {original_segment_url}")
    except ValueError as e:
        logger.error(f"Error creating segment URL for channel '{channel_id}': base='{base_url}', segment='{segment}'. Error: {e}")
        abort(500, description="Internal error creating segment URL.")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': base_url # Referer is often important
        }
        # Using stream=True for efficient large file transfer
        response = requests.get(original_segment_url, stream=True, timeout=20, headers=headers, allow_redirects=True)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Get content type, default to application/octet-stream if not provided or unsure
        content_type = response.headers.get('Content-Type', 'application/octet-stream')
        # Often segments are video/MP2T, but could be audio or other types
        if '.ts' in segment.lower():
            content_type = 'video/MP2T'
        elif '.aac' in segment.lower():
            content_type = 'audio/aac'
        # Add more specific types if needed

        # Generator function to stream chunks
        def generate_stream():
            try:
                # logger.debug(f"Streaming segment for {channel_id}...")
                for chunk in response.iter_content(chunk_size=1024*512): # 512KB chunk size
                    if chunk:
                        yield chunk
            except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as stream_err:
                 logger.warning(f"Network error while streaming segment for channel {channel_id}: {stream_err}")
            except Exception as stream_err:
                 logger.error(f"Error during segment streaming for {channel_id}: {stream_err}", exc_info=True)
            finally:
                response.close() # Ensure the connection is closed
                # logger.debug(f"Finished streaming segment for {channel_id}")

        # Prepare response headers
        resp_headers = {k: v for k, v in response.headers.items() if k.lower() in ['content-length', 'content-range', 'accept-ranges']}
        # Use origin server's Content-Type if available and seems reasonable
        resp_headers['Content-Type'] = content_type
        # resp_headers['Cache-Control'] = 'no-cache' # Optional: control client caching

        # Return a streaming response
        return Response(generate_stream(),
                        status=response.status_code, # Use status code from the origin server
                        headers=resp_headers)

    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching segment '{segment}' for channel '{channel_id}' from {original_segment_url}")
        abort(504, description=f"Timeout fetching segment: {segment}")
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 502
        logger.error(f"Error fetching segment '{segment}' for channel '{channel_id}' from {original_segment_url}. Status: {status_code}. Error: {e}")
        abort(status_code if 400 <= status_code < 500 else 502, description=f"Failed to fetch segment '{segment}' for channel '{channel_id}'. Origin error: {status_code}")
    except Exception as e:
        logger.error(f"Unexpected error serving segment '{segment}' for channel '{channel_id}': {e}", exc_info=True)
        abort(500, description="Internal server error fetching segment.")


@app.route('/')
def index():
    """
    Homepage displaying channels grouped by category, fetched from DATABASE_API_URL.
    Uses index.html template.
    """
    # API থেকে চ্যানেল ডেটা আনুন (main channel data)
    current_channels_data = get_remote_channels() # Returns dict: {'channel_id': {details}}

    # ক্যাটেগরি অনুযায়ী গ্রুপ করুন
    grouped_channels = defaultdict(list)
    if current_channels_data:
         # Sort channels by name before grouping
         try:
             # Ensure we handle potential missing 'name' or 'channel_id' safely
             sorted_channels = sorted(
                 current_channels_data.values(),
                 key=lambda ch: ch.get('name', ch.get('channel_id', 'zzz')).lower() # Use zzz to sort unnamed last
             )
             for channel_info in sorted_channels:
                 # Ensure channel_info is a dict and has necessary keys
                 if isinstance(channel_info, dict) and 'channel_id' in channel_info:
                     category = channel_info.get('category', 'Uncategorized') # Default if category is missing
                     # Add the full channel info dictionary to the list for its category
                     grouped_channels[category].append(channel_info)
                 else:
                      logger.warning(f"Skipping invalid channel data structure from API: {channel_info}")
         except Exception as e:
             logger.error(f"Error processing channel data for index page: {e}", exc_info=True)
             # grouped_channels will remain empty or partially filled

    # ক্যাটেগরিগুলো বর্ণানুক্রমে সাজান (defaultdict কে সাধারণ dict এ রূপান্তর করে)
    sorted_grouped_channels = dict(sorted(grouped_channels.items()))

    # টেমপ্লেটে গ্রুপ করা ডেটা পাঠান
    return render_template('index.html', grouped_channels=sorted_grouped_channels)

# --- Routes from app.py / code 1 (M3U Playlist & Admin1) ---

@app.route('/admin1')
def admin1_panel():
    """
    Admin Panel 1 (M3U data) - Displays channels fetched from API_M3U_CHANNELS_URL.
    NOTE: This panel does NOT have login protection in the original design.
    Consider adding protection if needed.
    Uses admin.html template.
    """
    channels = []
    error = None
    logger.info(f"Fetching channels for /admin1 from {API_M3U_CHANNELS_URL}")
    try:
        response = requests.get(API_M3U_CHANNELS_URL, timeout=15)
        response.raise_for_status() # HTTP errors এর জন্য Exception Raise করবে
        channels = response.json()
        logger.info(f"Successfully fetched {len(channels)} channels for /admin1")
    except requests.exceptions.Timeout:
        error = f"Timeout fetching channels from data server {API_M3U_CHANNELS_URL}."
        logger.error(error)
    except requests.exceptions.RequestException as e:
        error = f"Failed to fetch channels from data server {API_M3U_CHANNELS_URL}: {e}"
        logger.error(f"Error fetching channels for /admin1: {e}")
    except json.JSONDecodeError:
        error = f"Failed to decode JSON from data server {API_M3U_CHANNELS_URL}."
        logger.error(error)
    except Exception as e:
        error = f"An unexpected error occurred fetching channels for /admin1: {e}"
        logger.error(error)

    # Pass the error to the template to display it
    if error:
        flash(error, "error")

    return render_template('admin.html', channels=channels) # Removed error=error, using flash instead

@app.route('/admin1/add_or_update', methods=['POST'])
def add_or_update_channel_m3u():
    """
    Adds or updates M3U channel data using the API_M3U_CHANNELS_URL.
    Used by the /admin1 panel. (No login protection)
    """
    channel_id = request.form.get('channel_id', '').strip()
    name = request.form.get('name', '').strip()
    logo_link = request.form.get('logo_link', '').strip()
    group_title = request.form.get('group_title', '').strip()
    stream_link = request.form.get('stream_link', '').strip()

    # Basic server-side validation before sending to data server
    if not channel_id or not name or not stream_link:
         logger.warning("Validation error for /admin1/add_or_update: Missing required fields.")
         flash('Channel ID, Name, and Stream Link are required.', 'error')
         return redirect(url_for('admin1_panel'))

    channel_data = {
        "channel_id": channel_id,
        "name": name,
        "logo_link": logo_link,
        "group_title": group_title,
        "stream_link": stream_link
    }

    logger.info(f"Sending add/update request for M3U channel '{channel_id}' to {API_M3U_CHANNELS_URL}")
    try:
        # ডাটা সার্ভারের API তে POST রিকোয়েস্ট পাঠান
        response = requests.post(API_M3U_CHANNELS_URL, json=channel_data, timeout=15)
        response.raise_for_status() # HTTP errors এর জন্য Exception Raise করবে
        logger.info(f"Successfully added/updated M3U channel: {channel_id}")
        flash(f"Channel '{channel_id}' added/updated successfully.", "success")

    except requests.exceptions.Timeout:
        error = f"Timeout adding/updating channel on data server {API_M3U_CHANNELS_URL}."
        logger.error(error)
        flash(error, "error")
    except requests.exceptions.RequestException as e:
        error = f"Failed to add/update channel on data server {API_M3U_CHANNELS_URL}: {e}"
        # Try to get more details from the API response
        api_error_details = ""
        if e.response is not None:
            try:
                api_error_details = e.response.json().get("error", str(e.response.text[:200]))
            except (json.JSONDecodeError, AttributeError):
                api_error_details = str(e.response.text[:200])
            error += f" (API Status: {e.response.status_code}, Details: {api_error_details})"
        logger.error(f"Error adding/updating M3U channel {channel_id}: {error}")
        flash(error, "error")
    except Exception as e:
        error = f"An unexpected error occurred during M3U channel add/update: {e}"
        logger.error(error)
        flash(error, "error")

    # ডেটা সেভ করার পর অ্যাডমিন পেজে রিডাইরেক্ট করুন
    return redirect(url_for('admin1_panel'))


@app.route('/admin1/delete/<string:channel_id>', methods=['POST'])
def delete_m3u_channel(channel_id):
    """
    Deletes an M3U channel using the API_M3U_CHANNELS_URL.
    Used by the /admin1 panel. (No login protection)
    """
    # Ensure channel_id is clean (though data server should handle this too)
    channel_id_clean = channel_id.strip().lower()
    if not channel_id_clean:
        flash("Invalid Channel ID for deletion.", "error")
        return redirect(url_for('admin1_panel'))


    delete_url = f"{API_M3U_CHANNELS_URL}/{channel_id_clean}"
    logger.info(f"Sending delete request for M3U channel '{channel_id_clean}' to {delete_url}")

    try:
        # ডাটা সার্ভারের API তে DELETE রিকোয়েস্ট পাঠান
        response = requests.delete(delete_url, timeout=15)
        response.raise_for_status() # HTTP errors এর জন্য Exception Raise করবে
        logger.info(f"Successfully deleted M3U channel: {channel_id_clean}")
        flash(f"Channel '{channel_id_clean}' deleted successfully.", "success")

    except requests.exceptions.Timeout:
        error = f"Timeout deleting channel on data server {delete_url}."
        logger.error(error)
        flash(error, "error")
    except requests.exceptions.RequestException as e:
        error = f"Failed to delete channel on data server {delete_url}: {e}"
        api_error_details = ""
        if e.response is not None:
             # Handle 404 (Not Found) specifically
             if e.response.status_code == 404:
                 api_error_details = "Channel not found on server."
             else:
                 try:
                     api_error_details = e.response.json().get("error", str(e.response.text[:200]))
                 except (json.JSONDecodeError, AttributeError):
                     api_error_details = str(e.response.text[:200])
             error += f" (API Status: {e.response.status_code}, Details: {api_error_details})"
        logger.error(f"Error deleting M3U channel {channel_id_clean}: {error}")
        flash(error, "error")
    except Exception as e:
        error = f"An unexpected error occurred during M3U channel deletion: {e}"
        logger.error(error)
        flash(error, "error")


    # ডেটা মোছার পর অ্যাডমিন পেজে রিডাইরেক্ট করুন
    return redirect(url_for('admin1_panel'))


@app.route('/channel.m3u')
def generate_m3u_playlist():
    """
    Generates an M3U playlist file by fetching M3U-specific channel data
    from API_M3U_CHANNELS_URL.
    Removes the 'Data fetched from' comment as requested.
    """
    channels = []
    logger.info(f"Fetching channels for M3U playlist from {API_M3U_CHANNELS_URL}")
    try:
        response = requests.get(API_M3U_CHANNELS_URL, timeout=15)
        response.raise_for_status()
        channels = response.json()
        logger.info(f"Successfully fetched {len(channels)} channels for M3U playlist")
    except requests.exceptions.Timeout:
        error_msg = f"Timeout fetching channels from {API_M3U_CHANNELS_URL}"
        logger.error(error_msg)
        m3u_content = f"#EXTM3U\n# Error: {error_msg}\n"
        return Response(m3u_content, status=504, mimetype='application/vnd.apple.mpegurl')
    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to fetch channels from {API_M3U_CHANNELS_URL}: {e}"
        logger.error(error_msg)
        m3u_content = f"#EXTM3U\n# Error: {error_msg}\n"
        return Response(m3u_content, status=502, mimetype='application/vnd.apple.mpegurl')
    except json.JSONDecodeError:
        error_msg = f"Failed to decode JSON from {API_M3U_CHANNELS_URL}"
        logger.error(error_msg)
        m3u_content = f"#EXTM3U\n# Error: {error_msg}\n"
        return Response(m3u_content, status=500, mimetype='application/vnd.apple.mpegurl')
    except Exception as e:
        error_msg = f"An unexpected error occurred generating M3U: {e}"
        logger.error(error_msg)
        m3u_content = f"#EXTM3U\n# Error: {error_msg}\n"
        return Response(m3u_content, status=500, mimetype='application/vnd.apple.mpegurl')


    # Generate M3U content
    m3u_content = "#EXTM3U\n"
    m3u_content += f"# Generated by IPTV Server ({request.host_url}).\n" # Use current host URL
    # m3u_content += "# Data fetched from: " + API_M3U_CHANNELS_URL + "\n\n" # --- This line is REMOVED as requested ---
    m3u_content += "\n" # Add a blank line for separation

    # Sort channels by name for a cleaner M3U, handle missing names/ids
    try:
        sorted_channels = sorted(channels, key=lambda ch: ch.get('name', ch.get('channel_id', 'zzz')).lower())
    except TypeError: # Handle case where channels might not be a list of dicts
        logger.error("Error sorting channels, data format might be incorrect.")
        m3u_content += "# Error: Could not process channel data from API.\n"
        return Response(m3u_content, status=500, mimetype='application/vnd.apple.mpegurl')


    for channel in sorted_channels:
        # Ensure channel is a dictionary before accessing keys
        if not isinstance(channel, dict):
            logger.warning(f"Skipping non-dictionary item in channel list: {channel}")
            continue

        # Construct EXTINF line
        extinf_parts = ["#EXTINF:-1"]
        channel_id = channel.get('channel_id')
        logo_link = channel.get('logo_link')
        group_title = channel.get('group_title')
        name = channel.get('name', channel_id or 'Unnamed') # Use name, fallback to id, then 'Unnamed'

        if channel_id:
             extinf_parts.append(f'tvg-id="{channel_id}"')
        if logo_link:
            extinf_parts.append(f'tvg-logo="{html.escape(logo_link)}"') # Escape logo URL just in case
        if group_title:
             extinf_parts.append(f'group-title="{html.escape(group_title)}"') # Escape group title

        extinf_line = " ".join(extinf_parts)
        m3u_content += extinf_line

        # Add channel name (after comma)
        m3u_content += f",{name}\n"

        # Add stream link
        stream_link = channel.get('stream_link', '')
        if stream_link:
             # Basic check if stream link needs escaping (rare in M3U but safe)
             # M3U generally expects raw URLs
             m3u_content += f"{stream_link.strip()}\n"
        else:
             m3u_content += "# Error: Missing stream link\n" # Indicate missing link clearly
        m3u_content += "\n" # Add extra newline between entries for readability


    # Return the M3U file content
    return Response(m3u_content, mimetype='application/vnd.apple.mpegurl', headers={
        "Content-Disposition": "inline; filename=playlist.m3u" # Suggest filename, display inline
    })


# --- Routes from code 2 (Toffee Data & Health Check) ---

@app.route('/health')
def health_check():
    """Health check endpoint for Render or other monitoring."""
    logger.debug("Health check endpoint accessed.")
    return jsonify({"status": "OK"}), 200

# --- /toffee.m3u রুট (Plain JSON Array আউটপুট) ---
@app.route('/toffee.m3u') # Note: This endpoint name might be confusing as it serves JSON, not M3U format.
def serve_toffee_data_as_json():
    """
    ডাটাবেস থেকে Toffee চ্যানেল ডেটা এনে সরাসরি JSON অ্যারে হিসেবে সার্ভ করে।
    Fetches Toffee channel data from the database and serves it directly as a JSON array.
    Endpoint kept as /toffee.m3u based on original code 2, but consider renaming to /api/toffee/playlist.json for clarity.
    """
    logger.info("Request received for /toffee.m3u (serving plain JSON array)")
    channels = get_toffee_channels_from_db() # ডাটাবেস থেকে Toffee চ্যানেলের তালিকা পান (list of dicts)

    if channels is None:
        # ডেটা আনতে ব্যর্থ হলে JSON ত্রুটি বার্তা দিন
        logger.error("Failed to get Toffee channels from DB for JSON response.")
        return jsonify({"error": "Failed to retrieve Toffee channel list from the database server."}), 503 # Service Unavailable

    # সরাসরি প্রাপ্ত list of dictionaries (channels) কে JSON হিসেবে রিটার্ন করুন
    logger.info(f"Returning {len(channels)} Toffee channels as plain JSON array.")
    # jsonify() automatically sets Content-Type: application/json
    return jsonify(channels), 200 # Return the list directly


# --- Server Runner (Using Waitress for Production) ---

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000)) # Use PORT environment variable if available (common on platforms like Render)
    logger.info("Starting Combined IPTV Proxy, M3U Server, and Toffee Sync...")
    logger.info(f"Primary Channel Database API: {DATABASE_API_URL}")
    logger.info(f"M3U Data API: {API_M3U_CHANNELS_URL}")
    logger.info(f"Toffee Update API: {TOFFEE_UPDATE_API_URL}")
    logger.info(f"Toffee Channels API: {TOFFEE_CHANNELS_API_URL}")
    logger.info(f"GitHub Toffee Source: {GITHUB_JSON_URL}")

    # Use Waitress for a production-ready WSGI server
    try:
        from waitress import serve
        logger.info(f"Running with Waitress WSGI server on http://0.0.0.0:{port}")
        # Use 0.0.0.0 to listen on all available network interfaces
        # threads=10 might be a reasonable starting point, adjust based on load
        serve(app, host='0.0.0.0', port=port, threads=10)
    except ImportError:
        logger.warning("Waitress not found. Falling back to Flask's development server.")
        logger.warning("Flask's development server is NOT suitable for production.")
        # Run Flask's development server (ensure debug=False for production-like testing)
        # Set debug=True only for local development if needed
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.critical(f"Failed to start the server: {e}", exc_info=True)
