import requests
from flask import Flask, render_template, request, redirect, url_for, send_file, Response, abort, flash, session
import io
import os
import json
import html
from urllib.parse import urljoin, urlparse
from collections import defaultdict # ক্যাটেগরি গ্রুপিংয়ের জন্য

# --- Flask Application Setup ---
app = Flask(__name__)

# Secret key for session management (used by main.py's admin login)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_key_for_multichannel_v2') # কী পরিবর্তন করা ভালো

# Admin password for the main admin panel (/admin)
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'password123')

# --- API Endpoints ---

# Data server base URL (for both APIs)
# এটি আপনার actual PythonAnywhere Web App URL হতে হবে
# যেমন: https://your-username.pythonanywhere.com
PYTHONANYWHERE_BASE_URL = "https://itachi321.pythonanywhere.com" # আপনার আসল URL দিয়ে প্রতিস্থাপন করুন

# API Endpoint for the main channel database (used by main.py logic)
DATABASE_API_URL = f"{PYTHONANYWHERE_BASE_URL}/api/channels"

# API Endpoint for M3U channel data (used by app.py logic)
API_M3U_CHANNELS_URL = f"{PYTHONANYWHERE_BASE_URL}/api/m3u/channels"


# --- Constants ---

# Predefined categories for the main admin panel (/admin)
PREDEFINED_CATEGORIES = [
    "News", "Entertainment", "Movies", "Sports", "Music",
    "Kids", "Lifestyle", "Knowledge", "Religious", "International",
    "Regional (BD)", "Regional (IN)", "Regional (PK)", "Regional (Other)",
    "Documentary", "Drama", "Comedy", "Action", "Sci-Fi", "Other"
]
PREDEFINED_CATEGORIES.sort() # বর্ণানুক্রমে সাজান

# --- Helper Functions for DATABASE_API_URL (from main.py) ---

def get_remote_channels():
    """ডাটাবেস সার্ভার থেকে সব চ্যানেল ডেটা আনে (এখন নাম ও ক্যাটাগরি সহ)"""
    try:
        print(f"Fetching channels from {DATABASE_API_URL}")
        response = requests.get(DATABASE_API_URL, timeout=10)
        response.raise_for_status()
        channels_list = response.json()
        # API থেকে পাওয়া লিস্টকে ডিকশনারিতে রূপান্তর করুন (channel_id কে কী হিসাবে)
        channels_dict = {ch.get('channel_id'): ch for ch in channels_list if ch.get('channel_id')} # Ensure channel_id exists
        print(f"Successfully fetched {len(channels_dict)} channels from {DATABASE_API_URL}")
        return channels_dict
    except requests.exceptions.RequestException as e:
        print(f"Error fetching channels from remote API {DATABASE_API_URL}: {e}")
        flash(f"Error connecting to database server: {e}", "error")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON response from database server {DATABASE_API_URL}.")
        flash("Received invalid data from database server.", "error")
        return {}
    except Exception as e: # অন্য কোনো এরর ধরতে
        print(f"An unexpected error occurred while fetching channels from {DATABASE_API_URL}: {e}")
        flash("An unexpected error occurred while fetching channels.", "error")
        return {}


def get_remote_channel(channel_id):
    """ডাটাবেস সার্ভার থেকে নির্দিষ্ট চ্যানেল ডেটা আনে (এখন নাম ও ক্যাটাগরি সহ)"""
    try:
        url = f"{DATABASE_API_URL}/{channel_id.lower()}"
        print(f"Fetching channel '{channel_id}' from {url}")
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            print(f"Channel '{channel_id}' not found at {url}")
            return None
        response.raise_for_status()
        channel_data = response.json()
        print(f"Successfully fetched channel '{channel_id}' from {url}")
        return channel_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching channel '{channel_id}' from remote API {url}: {e}")
        return None
    except json.JSONDecodeError:
         print(f"Error decoding JSON for channel '{channel_id}' from {url}.")
         return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching channel {channel_id} from {url}: {e}")
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
        print(f"Saving channel '{payload['channel_id']}' to {DATABASE_API_URL} with payload: {payload}")
        # POST রিকোয়েস্ট পাঠান
        response = requests.post(DATABASE_API_URL, json=payload, timeout=10)
        response.raise_for_status() # HTTP errors (4xx, 5xx) ধরবে

        # API সফল হলে 201 বা 200 কোড ফেরত দেবে
        if response.status_code in [200, 201]:
            api_response_data = response.json()
            message = api_response_data.get("message", "Success")
            print(f"Channel '{payload['channel_id']}' saved successfully via API. Message: {message}")
            return True, message
        else:
            # অপ্রত্যাশিত সফল স্ট্যাটাস কোড (যদিও raise_for_status এটি ধরতে পারে)
            error_message = f"Unexpected success status code {response.status_code} from API {DATABASE_API_URL} during save."
            print(error_message)
            return False, error_message

    except requests.exceptions.RequestException as e:
        error_message = f"Error saving channel '{payload['channel_id']}' via API {DATABASE_API_URL}: {e}"
        api_error_details = ""
        # API থেকে আসা এরর মেসেজ দেখানোর চেষ্টা করুন
        try:
            if e.response is not None:
                api_error_details = e.response.json().get("error", str(e.response.text))
                error_message += f" (API Status: {e.response.status_code}, API Error: {api_error_details})"
        except (json.JSONDecodeError, AttributeError):
             # যদি response JSON না হয় বা response অবজেক্ট না থাকে
             if e.response is not None:
                 error_message += f" (API Status: {e.response.status_code}, Response: {e.response.text[:100]}...)" # প্রথম 100 অক্ষর
             else:
                 error_message += " (No response received from API)"

        print(error_message)
        return False, f"API communication error: {api_error_details or str(e)}" # ব্যবহারকারীকে দেখানোর জন্য সংক্ষিপ্ত বার্তা

    except json.JSONDecodeError:
         # POST রিকোয়েস্টের Response ডিকোড করতে সমস্যা হলে
         error_message = f"Error decoding JSON response from API {DATABASE_API_URL} after save attempt."
         print(error_message)
         return False, error_message
    except Exception as e:
        error_message = f"An unexpected error occurred during remote save to {DATABASE_API_URL}: {e}"
        print(error_message)
        return False, error_message


def delete_channel_remote(channel_id):
    """ডাটাবেস সার্ভার থেকে চ্যানেল মুছে ফেলে"""
    channel_id = channel_id.lower()
    try:
        url = f"{DATABASE_API_URL}/{channel_id}"
        print(f"Deleting channel '{channel_id}' from {url}")
        response = requests.delete(url, timeout=10)

        if response.status_code == 404:
            print(f"Channel '{channel_id}' not found on remote server {url} for deletion.")
            return False, "Channel not found on database server."

        response.raise_for_status() # অন্যান্য HTTP Error (যেমন 500) চেক করুন

        # সফলভাবে ডিলিট হলে (সাধারণত 200 বা 204 No Content)
        print(f"Channel '{channel_id}' delete request successful via API {url}.")
        message = "Deleted successfully"
        try:
             # API যদি মেসেজ পাঠায়, সেটা ব্যবহার করুন
             message = response.json().get("message", message)
        except json.JSONDecodeError:
             # যদি কোনো JSON বডি না থাকে (যেমন 204 No Content)
             pass
        return True, message

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
        print(error_message)
        return False, f"API communication error during delete: {api_error_details or str(e)}"
    except Exception as e:
        error_message = f"An unexpected error occurred during remote delete from {url}: {e}"
        print(error_message)
        return False, error_message

# --- Routes from main.py (Primary IPTV Proxy & Admin) ---

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
    current_channels = get_remote_channels()
    # টেমপ্লেটে ক্যাটেগরি লিস্ট পাঠান
    return render_template('admin_multi.html', channels=current_channels, categories=PREDEFINED_CATEGORIES)

@app.route('/admin/login', methods=['POST'])
def admin_login():
    """Handles login for the main admin panel (/admin)."""
    password_attempt = request.form.get('password')
    if password_attempt == ADMIN_PASSWORD:
        session['logged_in'] = True
        flash('Login successful!', 'success')
    else:
        session.pop('logged_in', None)
        flash('Incorrect password.', 'error')
    return redirect(url_for('admin_panel'))

@app.route('/admin/logout', methods=['POST'])
def admin_logout():
    """Handles logout for the main admin panel (/admin)."""
    session.pop('logged_in', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('admin_panel'))

@app.route('/admin/channels', methods=['POST'])
def add_or_update_channel_main(): # Renamed to distinguish from app.py's function
    """
    Adds or updates a channel using the DATABASE_API_URL.
    Used by the /admin panel.
    """
    if not session.get('logged_in'):
        abort(403)

    # ফরম থেকে ডেটা সংগ্রহ করুন
    channel_id = request.form.get('channel_id', '').strip().lower()
    source_url_raw = request.form.get('source_url', '').strip()
    channel_name = request.form.get('channel_name', '').strip() # নতুন: নাম পান
    category = request.form.get('category', '').strip() # নতুন: ক্যাটাগরি পান

    # বেসিক ভ্যালিডেশন
    if not channel_id or not source_url_raw or not channel_name or not category:
        flash('Channel ID, Source URL, Channel Name, and Category cannot be empty.', 'error')
        return redirect(url_for('admin_panel'))

    # চ্যানেল আইডি ভ্যালিডেশন
    if not channel_id.isalnum() or ' ' in channel_id:
         flash('Channel ID should only contain letters and numbers (no spaces).', 'error')
         return redirect(url_for('admin_panel'))

    # ক্যাটাগরি ভ্যালিডেশন (প্রিডিফাইন্ড লিস্টে আছে কিনা)
    if category not in PREDEFINED_CATEGORIES:
         flash(f'Invalid category selected: "{category}". Please choose from the list.', 'error')
         return redirect(url_for('admin_panel'))

    # URL ভ্যালিডেশন (বেসিক)
    try:
        parsed_url = urlparse(html.unescape(source_url_raw))
        if not all([parsed_url.scheme, parsed_url.netloc]):
             raise ValueError("Invalid URL format")
    except ValueError:
         flash('Invalid Source URL format. Please provide a full URL (e.g., http://...).', 'error')
         return redirect(url_for('admin_panel'))

    # রিমোট API তে সেভ করার চেষ্টা করুন (এখন নাম ও ক্যাটাগরি সহ)
    success, message = save_channel_remote(channel_id, source_url_raw, channel_name, category)

    if success:
        flash(f'Channel "{channel_id}" (Name: {channel_name}) processed successfully. API Message: {message}', 'success')
    else:
        flash(f'Failed to save channel "{channel_id}". Error: {message}', 'error')

    return redirect(url_for('admin_panel'))


@app.route('/admin/channels/delete/<channel_id>', methods=['POST'])
def delete_channel_main(channel_id): # Renamed to distinguish from app.py's function
    """
    Deletes a channel using the DATABASE_API_URL.
    Used by the /admin panel.
    """
    if not session.get('logged_in'):
        abort(403)

    channel_id_lower = channel_id.lower()

    # রিমোট API ব্যবহার করে ডিলিট করুন
    success, message = delete_channel_remote(channel_id_lower)

    if success:
        flash(f'Channel "{channel_id_lower}" delete request sent. API Message: {message}', 'success')
    else:
        flash(f'Failed to delete channel "{channel_id_lower}". Error: {message}', 'error')

    return redirect(url_for('admin_panel'))


@app.route('/live/<channel_id>.m3u8')
def serve_m3u8(channel_id):
    """
    Serves the M3U8 playlist by fetching it from the source URL
    obtained from DATABASE_API_URL and rewriting relative paths.
    """
    channel_id = channel_id.lower()
    channel_info = get_remote_channel(channel_id)

    if channel_info is None:
        abort(404, description=f"Channel '{channel_id}' not found or database connection failed.")

    source_url = channel_info.get('source_url')
    base_url = channel_info.get('base_url') # base_url should come from the API now
    channel_name = channel_info.get('name', channel_id) # নাম থাকলে ব্যবহার করুন, না হলে আইডি

    if not source_url or not base_url:
         print(f"Configuration error from API: Source or Base URL missing for channel '{channel_id}'. API Data: {channel_info}")
         abort(500, description=f"Configuration error for channel '{channel_id}'. Missing source or base URL.")

    print(f"Fetching M3U8 for channel '{channel_name}' ({channel_id}) from: {source_url}")

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(source_url, timeout=15, headers=headers, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '').lower()
        if 'mpegurl' not in content_type and 'x-mpegurl' not in content_type:
             print(f"Warning: Content-Type for channel '{channel_id}' is '{content_type}', not M3U8. Trying to process anyway.")

        original_m3u8_content = response.text
        modified_lines = []
        effective_base_url = base_url # Use the base_url from the API

        for line in original_m3u8_content.splitlines():
            line = line.strip()
            if not line:
                continue

            # EXT-X-KEY URI rewriting
            if line.startswith('#EXT-X-KEY'):
                if 'URI="' in line:
                    parts = line.split('URI="')
                    uri_part_and_rest = parts[1].split('"', 1)
                    uri_part = uri_part_and_rest[0]
                    rest_of_line = uri_part_and_rest[1] if len(uri_part_and_rest) > 1 else ""

                    if effective_base_url and not urlparse(uri_part).scheme:
                        full_key_uri = urljoin(effective_base_url, uri_part)
                        new_key_line = f'{parts[0]}URI="{full_key_uri}"{rest_of_line}'
                        modified_lines.append(new_key_line)
                        # print(f"Rewriting KEY URI: '{uri_part}' -> '{full_key_uri}'")
                    else:
                         modified_lines.append(line)
                else: # Handle cases where URI is not quoted or present in a strange format
                     modified_lines.append(line)
                     # print(f"Warning: Unusual EXT-X-KEY format for channel {channel_id}: {line}")


            # Segment or sub-playlist URI rewriting
            elif not line.startswith('#'):
                segment_uri_part = line
                parsed_segment_uri = urlparse(segment_uri_part)

                if parsed_segment_uri.scheme:
                    modified_lines.append(line) # Absolute URL, keep as is
                    # print(f"Keeping absolute URI: {line}")
                elif segment_uri_part.lower().endswith(('.m3u8', '.m3u')): # Check for both extensions
                     if effective_base_url:
                         absolute_sublist_url = urljoin(effective_base_url, segment_uri_part)
                         modified_lines.append(absolute_sublist_url)
                         # print(f"Rewriting nested playlist URI: '{segment_uri_part}' -> '{absolute_sublist_url}' (Absolute)")
                     else:
                        modified_lines.append(line)
                        print(f"Warning: Keeping relative nested playlist URI (no effective base_url): {line} for channel {channel_id}")

                else: # Segment (.ts, .aac, etc.)
                    # Proxy the segment through this server
                    # Use _external=False for internal URL generation
                    proxy_segment_url = url_for('serve_segment', channel_id=channel_id, segment=segment_uri_part, _external=False)
                    modified_lines.append(proxy_segment_url)
                    # print(f"Rewriting segment URI: '{segment_uri_part}' -> '{proxy_segment_url}' (Proxied)")


            # Other M3U8 tags (keep unchanged)
            else:
                modified_lines.append(line)

        modified_m3u8_content = "\n".join(modified_lines)
        # print(f"Modified M3U8 content for {channel_id}:\n{modified_m3u8_content[:500]}...") # Print first 500 chars
        return Response(modified_m3u8_content, mimetype='application/vnd.apple.mpegurl')

    except requests.exceptions.Timeout:
        print(f"Timeout fetching manifest for channel '{channel_id}' from {source_url}")
        abort(504, description=f"Timeout fetching manifest for channel '{channel_id}'")
    except requests.exceptions.RequestException as e:
        status = e.response.status_code if e.response is not None else 502
        print(f"Error fetching original M3U8 for channel '{channel_id}' ({source_url}). Status: {status}. Error: {e}")
        abort(status if status in [403, 404] else 502, description=f"Could not fetch manifest for channel '{channel_id}'. Origin error: {status}")
    except Exception as e:
        print(f"Unexpected error processing manifest for channel '{channel_id}': {e}")
        abort(500, description="Internal server error processing manifest.")


@app.route('/stream/<channel_id>/<path:segment>')
def serve_segment(channel_id, segment):
    """
    Proxies segment requests, fetching segment data from the original source
    using the base_url obtained from DATABASE_API_URL.
    """
    channel_id = channel_id.lower()
    channel_info = get_remote_channel(channel_id)

    if channel_info is None:
        print(f"Channel '{channel_id}' not found for segment request or DB connection failed.")
        abort(404, description=f"Channel '{channel_id}' not found for segment request or DB connection failed.")

    base_url = channel_info.get('base_url')

    if not base_url:
        print(f"Error: Base URL not found in API response for channel '{channel_id}' during segment request. API Data: {channel_info}")
        abort(500, description=f"Configuration error from API: Base URL missing for channel '{channel_id}'.")

    try:
        original_segment_url = urljoin(base_url, segment)
        # print(f"Fetching segment '{segment}' for channel '{channel_id}' from: {original_segment_url}")
    except ValueError as e:
        print(f"Error creating segment URL for channel '{channel_id}': base='{base_url}', segment='{segment}'. Error: {e}")
        abort(500, description="Internal error creating segment URL.")

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': base_url # Referer যোগ করা
        }
        # Using stream=True for efficient large file transfer
        response = requests.get(original_segment_url, stream=True, timeout=20, headers=headers, allow_redirects=True)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Get content type, default to video/MP2T if not provided
        content_type = response.headers.get('Content-Type', 'video/MP2T')

        # Generator function to stream chunks
        def generate_stream():
            try:
                # print(f"Streaming segment for {channel_id}...")
                for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunk size
                    if chunk:
                        yield chunk
            except requests.exceptions.ChunkedEncodingError:
                 print(f"ChunkedEncodingError while streaming segment for channel {channel_id}")
            except Exception as stream_err:
                 print(f"Error during segment streaming for {channel_id}: {stream_err}")
            finally:
                # Ensure the response connection is closed
                response.close()
                # print(f"Finished streaming segment for {channel_id}")


        # Prepare response headers, including Content-Length if available
        resp_headers = {}
        if 'Content-Length' in response.headers:
            resp_headers['Content-Length'] = response.headers['Content-Length']
        # resp_headers['Cache-Control'] = 'no-cache' # Optional: control client caching

        # Return a streaming response
        return Response(generate_stream(),
                        content_type=content_type,
                        status=response.status_code, # Use status code from the origin server
                        headers=resp_headers)

    except requests.exceptions.Timeout:
        print(f"Timeout fetching segment '{segment}' for channel '{channel_id}' from {original_segment_url}")
        abort(504, description=f"Timeout fetching segment: {segment}")
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 502
        print(f"Error fetching segment '{segment}' for channel '{channel_id}' from {original_segment_url}. Status: {status_code}. Error: {e}")
        # Return origin server status code if it's a client error (4xx) or generic 502
        abort(status_code if 400 <= status_code < 500 else 502, description=f"Failed to fetch segment '{segment}' for channel '{channel_id}'. Origin error: {status_code}")
    except Exception as e:
        print(f"Unexpected error serving segment '{segment}' for channel '{channel_id}': {e}")
        abort(500, description="Internal server error fetching segment.")


@app.route('/')
def index():
    """
    Homepage displaying channels grouped by category, fetched from DATABASE_API_URL.
    Uses index.html template.
    """
    # API থেকে চ্যানেল ডেটা আনুন
    current_channels_data = get_remote_channels() # এটি এখন {'channel_id': {details}} ফরম্যাটে

    # ক্যাটেগরি অনুযায়ী গ্রুপ করুন
    grouped_channels = defaultdict(list)
    # চ্যানেল ডেটা না থাকলে বা খালি হলে এটি কাজ করবে
    if current_channels_data:
         # Sort channels by name before grouping, ensure required fields exist
         sorted_channels = sorted(
             current_channels_data.values(),
             key=lambda ch: ch.get('name', ch.get('channel_id', '')).lower() # নাম বা আইডি দিয়ে সর্ট করুন
         )
         for channel_info in sorted_channels:
             # নিশ্চিত করুন channel_info একটি ডিকশনারি এবং প্রয়োজনীয় কী আছে
             if isinstance(channel_info, dict) and 'category' in channel_info and 'channel_id' in channel_info:
                 category = channel_info.get('category', 'Uncategorized') # ক্যাটাগরি না থাকলে ডিফল্ট
                 # পুরো চ্যানেল তথ্য যোগ করুন লিস্টে
                 grouped_channels[category].append(channel_info)
             else:
                  print(f"Skipping invalid channel data from API: {channel_info}")


    # ক্যাটেগরিগুলো বর্ণানুক্রমে সাজান (defaultdict কে সাধারণ dict এ রূপান্তর করে)
    sorted_grouped_channels = dict(sorted(grouped_channels.items()))

    # টেমপ্লেটে গ্রুপ করা ডেটা পাঠান
    return render_template('index.html', grouped_channels=sorted_grouped_channels)

# --- Routes from app.py (M3U Playlist & Admin1) ---

@app.route('/admin1')
def admin1_panel():
    """
    Admin Panel 1 (from original app.py) - Displays channels fetched from API_M3U_CHANNELS_URL.
    Does NOT have built-in login protection in the original code.
    Uses admin.html template.
    """
    channels = []
    error = None
    print(f"Fetching channels for /admin1 from {API_M3U_CHANNELS_URL}")
    try:
        response = requests.get(API_M3U_CHANNELS_URL)
        response.raise_for_status() # HTTP errors এর জন্য Exception Raise করবে
        channels = response.json()
        print(f"Successfully fetched {len(channels)} channels for /admin1")
    except requests.exceptions.RequestException as e:
        error = f"Failed to fetch channels from data server {API_M3U_CHANNELS_URL}: {e}"
        print(f"Error fetching channels for /admin1: {e}") # সার্ভার কনসোলে লগিং
    except json.JSONDecodeError:
        error = f"Failed to decode JSON from data server {API_M3U_CHANNELS_URL}."
        print(f"Error decoding JSON for /admin1: {error}")


    return render_template('admin.html', channels=channels, error=error)

@app.route('/admin1/add_or_update', methods=['POST'])
def add_or_update_channel_m3u(): # Renamed to distinguish
    """
    Adds or updates M3U channel data using the API_M3U_CHANNELS_URL.
    Used by the /admin1 panel.
    """
    channel_id = request.form.get('channel_id', '').strip()
    name = request.form.get('name', '').strip()
    logo_link = request.form.get('logo_link', '').strip()
    group_title = request.form.get('group_title', '').strip()
    stream_link = request.form.get('stream_link', '').strip()

    # Basic server-side validation before sending to data server
    if not channel_id or not name or not stream_link:
         print("Validation error for /admin1/add_or_update: Missing required fields.")
         # You might want to flash an error here
         return redirect(url_for('admin1_panel'))

    channel_data = {
        "channel_id": channel_id,
        "name": name,
        "logo_link": logo_link,
        "group_title": group_title,
        "stream_link": stream_link
    }

    error = None
    print(f"Sending add/update request for M3U channel '{channel_id}' to {API_M3U_CHANNELS_URL}")
    try:
        # ডাটা সার্ভারের API তে POST রিকোয়েস্ট পাঠান
        response = requests.post(API_M3U_CHANNELS_URL, json=channel_data)
        response.raise_for_status() # HTTP errors এর জন্য Exception Raise করবে
        print(f"Successfully added/updated M3U channel: {channel_id}")

    except requests.exceptions.RequestException as e:
        error = f"Failed to add/update channel on data server {API_M3U_CHANNELS_URL}: {e}"
        print(f"Error adding/updating M3U channel {channel_id}: {e}") # সার্ভার কনসোলে লগিং
        # You might want to pass this error to the admin page

    # ডেটা সেভ করার পর অ্যাডমিন পেজে রিডাইরেক্ট করুন
    return redirect(url_for('admin1_panel'))


@app.route('/admin1/delete/<string:channel_id>', methods=['POST'])
def delete_m3u_channel(channel_id):
    """
    Deletes an M3U channel using the API_M3U_CHANNELS_URL.
    Used by the /admin1 panel.
    """
    # Ensure channel_id is clean (though data server does this too)
    channel_id = channel_id.strip().lower()

    error = None
    delete_url = f"{API_M3U_CHANNELS_URL}/{channel_id}"
    print(f"Sending delete request for M3U channel '{channel_id}' to {delete_url}")

    try:
        # ডাটা সার্ভারের API তে DELETE রিকোয়েস্ট পাঠান
        response = requests.delete(delete_url)
        response.raise_for_status() # HTTP errors এর জন্য Exception Raise করবে
        print(f"Successfully deleted M3U channel: {channel_id}")

    except requests.exceptions.RequestException as e:
        error = f"Failed to delete channel on data server {delete_url}: {e}"
        print(f"Error deleting M3U channel {channel_id}: {e}") # সার্ভার কনসোলে লগিং
        # You might want to pass this error to the admin page

    # ডেটা মোছার পর অ্যাডমিন পেজে রিডাইরেক্ট করুন
    return redirect(url_for('admin1_panel'))


@app.route('/channel.m3u')
def generate_m3u_playlist():
    """
    Generates an M3U playlist file by fetching channel data from API_M3U_CHANNELS_URL.
    """
    channels = []
    print(f"Fetching channels for M3U playlist from {API_M3U_CHANNELS_URL}")
    try:
        response = requests.get(API_M3U_CHANNELS_URL)
        response.raise_for_status() # HTTP errors এর জন্য Exception Raise করবে
        channels = response.json()
        print(f"Successfully fetched {len(channels)} channels for M3U playlist")
    except requests.exceptions.RequestException as e:
        # If fetching fails, return an M3U with an error comment
        print(f"Error fetching channels for M3U: {e}")
        m3u_content = f"#EXTM3U\n# Error fetching channels: {e}\n"
        return m3u_content, 500, {'Content-Type': 'application/vnd.apple.mpegurl'}
    except json.JSONDecodeError:
        print(f"Error decoding JSON for M3U playlist from {API_M3U_CHANNELS_URL}")
        m3u_content = f"#EXTM3U\n# Error decoding data from server.\n"
        return m3u_content, 500, {'Content-Type': 'application/vnd.apple.mpegurl'}


    # Generate M3U content
    m3u_content = "#EXTM3U\n"
    m3u_content += f"# Generated from your playlist server ({request.host_url}).\n" # Use current host URL
    m3u_content += "# Data fetched from: " + API_M3U_CHANNELS_URL + "\n\n"

    # Sort channels by name for a cleaner M3U
    sorted_channels = sorted(channels, key=lambda ch: ch.get('name', ch.get('channel_id', '')).lower())

    for channel in sorted_channels:
        # Construct EXTINF line
        extinf_parts = ["#EXTINF:-1"]
        # Add tvg-id using channel_id if available
        if channel.get('channel_id'):
             extinf_parts.append(f'tvg-id="{channel["channel_id"]}"')
        if channel.get('logo_link'):
            extinf_parts.append(f'tvg-logo="{channel["logo_link"]}"')
        if channel.get('group_title'):
             extinf_parts.append(f'group-title="{channel["group_title"]}"')

        extinf_line = " ".join(extinf_parts)
        m3u_content += extinf_line

        # Add channel name (after comma)
        m3u_content += f",{channel.get('name', channel.get('channel_id', 'Unnamed'))}\n" # Use name or id or Unnamed

        # Add stream link
        stream_link = channel.get('stream_link', '')
        if stream_link:
             # Ensure stream_link is escaped if necessary (though M3U generally doesn't need it like HTML)
             m3u_content += f"{stream_link}\n\n" # Add extra newline for readability
        else:
             m3u_content += "# Missing stream link\n\n" # Indicate missing link


    # Return the M3U file
    # Use send_file with BytesIO to serve the string content as a file
    m3u_file = io.BytesIO(m3u_content.encode('utf-8'))
    m3u_file.seek(0) # Go back to the beginning of the stream

    return send_file(
        m3u_file,
        mimetype='application/vnd.apple.mpegurl', # Standard M3U MIME type
        as_attachment=False, # Set to True if you want it downloaded instead
        download_name='playlist.m3u' # Suggested filename if downloaded
    )


# --- Server Runner (from main.py, using Waitress) ---

if __name__ == '__main__':
    print("Starting Combined IPTV Proxy and M3U Playlist Server...")
    print(f"Primary Database API: {DATABASE_API_URL}")
    print(f"M3U Data API: {API_M3U_CHANNELS_URL}")
    # Waitress ব্যবহার
    try:
        from waitress import serve
        print(f"Running with Waitress WSGI server on http://0.0.0.0:5000")
        # Use 0.0.0.0 to listen on all interfaces, allowing external access (e.g., from Render.com)
        serve(app, host='0.0.0.0', port=5000, threads=10) # থ্রেড সংখ্যা বাড়ানো যেতে পারে
    except ImportError:
        print("Waitress not found, using Flask's development server (NOT recommended for production).")
        # For development, you might use debug=True
        # For production, ensure debug=False
        app.run(host='0.0.0.0', port=5000, debug=False)                                                                              
