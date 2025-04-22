import os
import requests
import json
from flask import Flask, Response, abort, request, render_template, redirect, url_for, flash, session
from urllib.parse import urljoin, urlparse, parse_qs, urlencode

app = Flask(__name__)

# সিক্রেট কী (অ্যাডমিন পাসওয়ার্ডের মতই এনভায়রনমেন্ট ভেরিয়েবল থেকে নেওয়া ভালো)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_key_for_multichannel_fixed')

# অ্যাডমিন পাসওয়ার্ড (এনভায়রনমেন্ট ভেরিয়েবল ব্যবহার করুন!)
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'password123') # এটি পরিবর্তন করুন!

# যে JSON ফাইলে চ্যানেলগুলোর তথ্য (ID -> URL) সংরক্ষণ করা হবে
CHANNELS_STORAGE_FILE = "channels.json"

# --- Helper Functions ---

def load_channels():
    """JSON ফাইল থেকে চ্যানেল ডেটা লোড করে এবং বেস URL ক্যালকুলেট করে।"""
    if not os.path.exists(CHANNELS_STORAGE_FILE):
        return {} # ফাইল না থাকলে খালি ডিকশনারি রিটার্ন করুন
    try:
        with open(CHANNELS_STORAGE_FILE, 'r') as f:
            data = json.load(f)
            # নিশ্চিত করুন ডেটা সঠিক ফরম্যাটে আছে (dict of dicts)
            if isinstance(data, dict):
                updated_data = {}
                needs_save = False
                for channel_id, info in data.items():
                    source_url = None
                    base_url = None
                    if isinstance(info, str): # পুরাতন ফরম্যাট (শুধু URL স্ট্রিং)
                        source_url = info
                        needs_save = True # পুরাতন ফরম্যাট পেলে আপডেট করে সেভ করতে হবে
                    elif isinstance(info, dict) and 'source_url' in info:
                        source_url = info['source_url']
                        base_url = info.get('base_url') # আগের বেস ইউআরএল (যদি থাকে)
                    else:
                         print(f"Warning: Invalid data format for channel '{channel_id}'. Skipping.")
                         continue

                    # যদি source_url থাকে এবং base_url না থাকে বা আপডেট দরকার হয়
                    if source_url:
                         try:
                             # Base URL হলো স্কিম + নেটলোক + পাথ (শেষ অংশ বাদে)
                             parsed_original = urlparse(source_url)
                             # Ensure path ends with a slash if it has segments
                             path_part = parsed_original.path
                             if '/' in path_part:
                                 base_path = path_part.rsplit('/', 1)[0] + '/'
                             else:
                                 base_path = '/' # Or handle cases with no path?
                             calculated_base_url = urlunparse((parsed_original.scheme, parsed_original.netloc, base_path, '', '', ''))

                             # যদি ক্যালকুলেট করা বেস ইউআরএল আগের থেকে ভিন্ন হয় বা আগে না থাকে
                             if calculated_base_url != base_url:
                                 base_url = calculated_base_url
                                 needs_save = True

                             updated_data[channel_id] = {'source_url': source_url, 'base_url': base_url}

                         except Exception as e:
                             print(f"Warning: Could not process URL '{source_url}' for channel '{channel_id}'. Error: {e}. Skipping.")
                             continue # অবৈধ হলে বাদ দিন
                    else:
                        # যদি info একটি dict হয় কিন্তু source_url না থাকে
                        print(f"Warning: Missing 'source_url' for channel '{channel_id}'. Skipping.")

                # যদি কোনো পরিবর্তন হয়ে থাকে (যেমন পুরাতন ফরম্যাট আপডেট বা বেস ইউআরএল গণনা)
                if needs_save:
                    save_channels(updated_data)
                return updated_data
            else:
                print("Warning: channels.json is not a valid dictionary. Starting fresh.")
                # Consider backing up the invalid file here
                return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {CHANNELS_STORAGE_FILE}. Please check the file format. Starting fresh.")
         # Consider backing up the invalid file here
        return {}
    except Exception as e:
        print(f"Error loading channels from file '{CHANNELS_STORAGE_FILE}': {e}")
        return {} # ত্রুটি হলে খালি ডিকশনারি

def save_channels(channels_data):
    """চ্যানেল ডেটা JSON ফাইলে সংরক্ষণ করে।"""
    try:
        with open(CHANNELS_STORAGE_FILE, 'w') as f:
            json.dump(channels_data, f, indent=4) # সুন্দর ফরম্যাটিং এর জন্য indent ব্যবহার করুন
        return True
    except Exception as e:
        print(f"Error saving channels to file '{CHANNELS_STORAGE_FILE}': {e}")
        return False

# --- Global State ---
# অ্যাপ্লিকেশন শুরু হওয়ার সময় চ্যানেল ডেটা লোড করুন
channels_data = load_channels()

# --- Admin Routes ---
@app.route('/admin', methods=['GET'])
def admin_panel():
    """অ্যাডমিন পেজ রেন্ডার করে।"""
    if not session.get('logged_in'):
        # Use the multi-channel admin template
        return render_template('admin_multi.html')
    # If logged in, show the admin page with the channel list
    return render_template('admin_multi.html', channels=channels_data)

@app.route('/admin/login', methods=['POST'])
def admin_login():
    """অ্যাডমিন লগইন হ্যান্ডেল করে।"""
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
    """অ্যাডমিন লগআউট হ্যান্ডেল করে।"""
    session.pop('logged_in', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('admin_panel'))

@app.route('/admin/channels', methods=['POST'])
def add_channel():
    """নতুন চ্যানেল যোগ করে।"""
    global channels_data
    if not session.get('logged_in'):
        abort(403) # Forbidden

    channel_id = request.form.get('channel_id', '').strip().lower()
    source_url = request.form.get('source_url', '').strip()

    # Basic validation
    if not channel_id or not source_url:
        flash('Channel ID and Source URL cannot be empty.', 'error')
        return redirect(url_for('admin_panel'))

    # Allow more flexible channel IDs (e.g., with hyphens) but prevent slashes
    if not all(c.isalnum() or c == '-' for c in channel_id) or '/' in channel_id:
         flash('Channel ID should only contain letters, numbers, and hyphens (no spaces or slashes).', 'error')
         return redirect(url_for('admin_panel'))

    if channel_id in channels_data:
        flash(f'Channel ID "{channel_id}" already exists.', 'error')
        return redirect(url_for('admin_panel'))

    # Validate and calculate base URL
    try:
        parsed_url = urlparse(source_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise ValueError("Invalid URL format (missing scheme or domain)")

        # Calculate base URL correctly
        path_part = parsed_url.path
        if '/' in path_part:
             base_path = path_part.rsplit('/', 1)[0] + '/'
        else:
             base_path = '/'
        base_url = urlunparse((parsed_url.scheme, parsed_url.netloc, base_path, '', '', ''))

    except ValueError as e:
        flash(f'Invalid Source URL format: {e}. Please enter a valid M3U8 URL (e.g., http://.../playlist.m3u8)', 'error')
        return redirect(url_for('admin_panel'))

    # Update data and save
    channels_data[channel_id] = {'source_url': source_url, 'base_url': base_url}
    if save_channels(channels_data):
        flash(f'Channel "{channel_id}" added successfully.', 'success')
    else:
        # Rollback if save fails
        channels_data.pop(channel_id, None)
        flash('Failed to save channel data. Check server logs.', 'error')

    return redirect(url_for('admin_panel'))

@app.route('/admin/channels/delete/<channel_id>', methods=['POST'])
def delete_channel(channel_id):
    """নির্দিষ্ট চ্যানেল মুছে ফেলে।"""
    global channels_data
    if not session.get('logged_in'):
        abort(403)

    channel_id_lower = channel_id.lower() # Ensure lowercase for lookup and delete
    if channel_id_lower in channels_data:
        removed_data = channels_data.pop(channel_id_lower) # Remove from dictionary
        if save_channels(channels_data):
            flash(f'Channel "{channel_id}" deleted successfully.', 'success')
        else:
            # Rollback: If save fails, put the data back
            channels_data[channel_id_lower] = removed_data
            flash('Failed to save changes after deletion. Check server logs.', 'error')
    else:
        flash(f'Channel ID "{channel_id}" not found.', 'error')

    return redirect(url_for('admin_panel'))


# --- Stream Proxy Routes (Dynamic) ---

# M3U8 Manifest Route
@app.route('/live/<channel_id>.m3u8')
def serve_m3u8(channel_id):
    """নির্দিষ্ট চ্যানেলের জন্য পরিবর্তিত M3U8 ফাইল পরিবেশন করে।"""
    channel_id_lower = channel_id.lower()
    if channel_id_lower not in channels_data:
        abort(404, description=f"Channel '{channel_id}' not found.")

    channel_info = channels_data[channel_id_lower]
    source_url = channel_info['source_url']

    try:
        # Fetch the original manifest, passing its own query parameters correctly
        response = requests.get(source_url, timeout=10)
        response.raise_for_status()

        original_m3u8_content = response.text
        modified_lines = []
        # Segment path prefix for this specific channel
        segment_proxy_prefix = f"/stream/{channel_id_lower}" # Use lowercase ID

        for line in original_m3u8_content.splitlines():
            line = line.strip()
            if not line:
                continue
            if not line.startswith('#'):
                # --- FIX 1: Preserve query parameters in segment URIs ---
                # Assume 'line' contains the relative path + query string (e.g., "segment1.ts?p=1")
                # Or it could be an absolute URL, handle that later if needed.
                # Prepend the proxy path correctly.
                # Avoid double slashes if line starts with /
                segment_part = line.lstrip('/')
                modified_line = f"{segment_proxy_prefix}/{segment_part}"
                # --- End FIX 1 ---
                modified_lines.append(modified_line)
            else:
                # Keep comment or tag lines as they are
                modified_lines.append(line)

        modified_m3u8_content = "\n".join(modified_lines)
        return Response(modified_m3u8_content, mimetype='application/vnd.apple.mpegurl')

    except requests.exceptions.Timeout:
        abort(504, description=f"Timeout fetching manifest for channel '{channel_id}' from {source_url}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching original M3U8 for channel '{channel_id}' ({source_url}): {e}")
        status_code = e.response.status_code if e.response is not None else 502
        abort(status_code if status_code in [401, 403, 404] else 502,
              description=f"Could not fetch manifest for channel '{channel_id}'. Origin error: {status_code}")
    except Exception as e:
        print(f"Unexpected error processing manifest for channel '{channel_id}': {e}")
        abort(500, description="Internal error processing manifest.")


# Segment Route
@app.route('/stream/<channel_id>/<path:segment>')
def serve_segment(channel_id, segment):
    """নির্দিষ্ট চ্যানেলের জন্য সেগমেন্ট ফাইল রিলে করে, কুয়েরি প্যারামিটার সহ।"""
    channel_id_lower = channel_id.lower()
    if channel_id_lower not in channels_data:
        abort(404, description=f"Channel '{channel_id}' not found for segment request.")

    channel_info = channels_data[channel_id_lower]
    base_url = channel_info.get('base_url')
    source_url = channel_info.get('source_url') # Need this for its query parameters

    if not base_url or not source_url:
         print(f"Error: Configuration missing for channel '{channel_id}'")
         abort(500, description=f"Configuration error for channel '{channel_id}'.")

    # --- FIX 2: Forward necessary query parameters ---
    params_to_forward = {}

    # 1. Extract query parameters from the original source_url (admin provided)
    try:
        original_source_params_str = urlparse(source_url).query
        if original_source_params_str:
             # Handle potential HTML entities like &amp;
             decoded_params_str = original_source_params_str.replace('&amp;', '&')
             # Use parse_qs which handles multiple values for the same key correctly
             original_params_dict = parse_qs(decoded_params_str)
             # Convert list values back to simple values if only one item exists
             # Requests library handles lists correctly anyway, but this simplifies logging
             params_to_forward.update({k: v[0] if len(v) == 1 else v for k, v in original_params_dict.items()})
    except Exception as e:
        print(f"Warning: Could not parse query parameters from source URL for channel {channel_id}: {source_url}. Error: {e}")

    # 2. Extract query parameters from the client's request URL to this proxy segment
    # These might have been added by the manifest (Fix 1)
    client_request_params = request.args.to_dict(flat=False) # Use flat=False to handle multi-value keys
    # Merge client params, potentially overwriting source params if keys clash
    # Convert list values back to simple values if only one item exists
    params_to_forward.update({k: v[0] if len(v) == 1 else v for k, v in client_request_params.items()})
    # --- End FIX 2 ---


    # Construct the target segment URL path using the channel's base URL
    # urljoin handles relative paths correctly (e.g., ../ in segment path)
    target_segment_url_path = urljoin(base_url, segment)

    # Log the target URL and parameters for debugging
    print(f"Channel '{channel_id}': Fetching segment '{segment}' from '{target_segment_url_path}' with params {params_to_forward}")

    try:
        # Make the request using the constructed path and combined parameters
        response = requests.get(
            target_segment_url_path,
            params=params_to_forward, # Pass parameters dict to requests
            stream=True,
            timeout=10 # Timeout for segment fetching
        )
        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

        # Stream the response back to the client
        return Response(response.iter_content(chunk_size=1024*1024), # 1MB chunks
                        content_type=response.headers.get('Content-Type', 'video/MP2T')) # Use original content type

    except requests.exceptions.Timeout:
        print(f"Timeout fetching segment {segment} for channel '{channel_id}' from {target_segment_url_path}")
        abort(504, description=f"Timeout fetching segment: {segment}")
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 502
        print(f"Error fetching segment {segment} for channel '{channel_id}' from {target_segment_url_path}. Status: {status_code}. Error: {e}")
        # Pass through specific client errors like 404, 403, 401, otherwise use 502
        abort(status_code if status_code in [401, 403, 404] else 502,
              description=f"Failed to fetch segment '{segment}' for channel '{channel_id}'. Origin error: {status_code}")
    except Exception as e:
        print(f"Unexpected error serving segment {segment} for channel '{channel_id}': {e}")
        abort(500, description="Internal error fetching segment.")


if __name__ == '__main__':
    # Run on 0.0.0.0 to be accessible within Docker, specify port 5000
    app.run(host='0.0.0.0', port=5000)                                                                                                          
