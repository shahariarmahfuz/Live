import os
import requests
import json
import html # <-- URL পরিষ্কার করার জন্য ইম্পোর্ট করা হয়েছে
from flask import Flask, Response, abort, request, render_template, redirect, url_for, flash, session
from urllib.parse import urljoin, urlparse

app = Flask(__name__)

# সিক্রেট কী (পরিবেশ ভেরিয়েবল থেকে নেওয়া ভালো)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_key_for_multichannel')

# অ্যাডমিন পাসওয়ার্ড (পরিবেশ ভেরিয়েবল ব্যবহার করুন!)
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'password123') # এটি পরিবর্তন করুন!

# যে JSON ফাইলে চ্যানেলগুলোর তথ্য (ID -> URL Info) সংরক্ষণ করা হবে
CHANNELS_STORAGE_FILE = "channels.json"

# --- Helper Functions ---

def load_channels():
    """JSON ফাইল থেকে চ্যানেল ডেটা লোড করে এবং URL আনএসকেপ করে।"""
    if not os.path.exists(CHANNELS_STORAGE_FILE):
        return {} # ফাইল না থাকলে খালি ডিকশনারি রিটার্ন করুন
    try:
        with open(CHANNELS_STORAGE_FILE, 'r') as f:
            data = json.load(f)
            # নিশ্চিত করুন ডেটা সঠিক ফরম্যাটে আছে (dict of dicts)
            if isinstance(data, dict):
                updated_data = {}
                made_changes = False
                for channel_id, info in data.items():
                    normalized_id = channel_id.lower() # আইডি কে লোয়ারকেস করুন

                    if isinstance(info, str): # পুরাতন ফরম্যাট (শুধু URL স্ট্রিং)
                        source_url_raw = info
                        source_url = html.unescape(source_url_raw) # নিশ্চিত করুন URL আনএসকেপ করা আছে
                        try:
                            base_url = urljoin(source_url, '.')
                            updated_data[normalized_id] = {'source_url': source_url, 'base_url': base_url}
                            # যদি মূল আইডি লোয়ারকেস না হয় বা URL আনএসকেপ করা হয়
                            if channel_id != normalized_id or source_url != source_url_raw:
                                made_changes = True
                        except ValueError:
                            print(f"Warning: Invalid source URL '{source_url}' for channel '{channel_id}'. Skipping.")
                            continue
                    elif isinstance(info, dict) and 'source_url' in info:
                        source_url_raw = info['source_url']
                        source_url = html.unescape(source_url_raw) # নিশ্চিত করুন URL আনএসকেপ করা আছে

                        # যদি পরিবর্তন দরকার হয় (কেস, আনএসকেপ, বেস URL)
                        if source_url != source_url_raw:
                            info['source_url'] = source_url # আপডেটেড URL সংরক্ষণ করুন
                            made_changes = True
                        if channel_id != normalized_id:
                             made_changes = True

                        if 'base_url' not in info or not info['base_url']:
                             try:
                                 info['base_url'] = urljoin(source_url, '.')
                                 made_changes = True
                             except ValueError:
                                 print(f"Warning: Invalid source URL '{source_url}' for channel '{channel_id}'. Skipping base_url update.")
                                 info['base_url'] = None
                                 made_changes = True # বেস URL সেট না হলেও পরিবর্তন ধরা হবে
                        elif info.get('base_url') != urljoin(source_url, '.'):
                            # যদি সোর্স URL পরিবর্তন হয়ে থাকে, বেস URL আবার ক্যালকুলেট করুন
                             try:
                                 info['base_url'] = urljoin(source_url, '.')
                                 made_changes = True
                             except ValueError:
                                 print(f"Warning: Invalid source URL '{source_url}' for channel '{channel_id}'. Skipping base_url update.")
                                 info['base_url'] = None

                        updated_data[normalized_id] = info # লোয়ারকেস আইডি দিয়ে সংরক্ষণ করুন
                    else:
                         print(f"Warning: Invalid data format for channel '{channel_id}'. Skipping.")

                if made_changes:
                    print("Updating channel data format, IDs to lowercase, base URLs, or unescaping URLs...")
                    # আপডেটেড ডেটা দিয়ে ফাইল সেভ করুন
                    save_channels(updated_data)
                return updated_data
            else:
                print(f"Warning: {CHANNELS_STORAGE_FILE} does not contain a valid dictionary. Returning empty.")
                return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {CHANNELS_STORAGE_FILE}. Returning empty.")
        return {}
    except Exception as e:
        print(f"Error loading channels from file: {e}. Returning empty.")
        return {} # ত্রুটি হলে খালি ডিকশনারি

def save_channels(channels_data):
    """চ্যানেল ডেটা JSON ফাইলে সংরক্ষণ করে।"""
    try:
        # নিশ্চিত করুন সব আইডি লোয়ারকেস আছে সেভ করার আগে
        normalized_data = {k.lower(): v for k, v in channels_data.items()}
        with open(CHANNELS_STORAGE_FILE, 'w') as f:
            json.dump(normalized_data, f, indent=4) # সুন্দর ফরম্যাটিং এর জন্য indent ব্যবহার করুন
        print(f"Channel data saved to {CHANNELS_STORAGE_FILE}")
        return True
    except Exception as e:
        print(f"Error saving channels to file: {e}")
        return False

# --- Global State ---
# অ্যাপ্লিকেশন শুরু হওয়ার সময় চ্যানেল ডেটা লোড করুন (এবং প্রয়োজনে আপডেট করুন)
channels_data = load_channels()

# --- Admin Routes ---
@app.route('/admin', methods=['GET'])
def admin_panel():
    """অ্যাডমিন পেজ রেন্ডার করে। চ্যানেল লিস্ট সহ।"""
    if not session.get('logged_in'):
        # লগইন করা না থাকলে শুধু লগইন ফর্ম দেখান
        return render_template('admin_multi.html', channels=None)
    # লগইন করা থাকলে চ্যানেল লিস্ট সহ অ্যাডমিন পেজ দেখান
    # সবসময় লেটেস্ট ডেটা লোড করে পাঠান
    current_channels = load_channels()
    return render_template('admin_multi.html', channels=current_channels)

@app.route('/admin/login', methods=['POST'])
def admin_login():
    """অ্যাডমিন লগইন হ্যান্ডেল করে।"""
    password_attempt = request.form.get('password')
    if password_attempt == ADMIN_PASSWORD:
        session['logged_in'] = True
        flash('Login successful!', 'success')
    else:
        session.pop('logged_in', None) # নিশ্চিত করুন ভুল পাসওয়ার্ড দিলে লগইন অবস্থা রিমুভ হয়
        flash('Incorrect password.', 'error')
    return redirect(url_for('admin_panel'))

@app.route('/admin/logout', methods=['POST'])
def admin_logout():
    """অ্যাডমিন লগআউট হ্যান্ডেল করে।"""
    session.pop('logged_in', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('admin_panel'))

@app.route('/admin/channels', methods=['POST'])
def add_or_update_channel():
    """নতুন চ্যানেল যোগ করে বা বিদ্যমান চ্যানেল আপডেট করে।"""
    global channels_data # গ্লোবাল স্টেট আপডেট করার জন্য
    if not session.get('logged_in'):
        abort(403) # Forbidden

    channel_id = request.form.get('channel_id', '').strip().lower() # আইডি কে লোয়ারকেস করুন
    source_url_raw = request.form.get('source_url', '').strip() # Raw URL নিন
    action = request.form.get('action', 'add') # অ্যাকশন (add বা update)

    # --- URL পরিষ্কার করুন ---
    source_url = html.unescape(source_url_raw) # HTML entities (&amp;, etc.) ঠিক করুন
    # ------------------------

    # বেসিক ভ্যালিডেশন
    if not channel_id or not source_url:
        flash('Channel ID and Source URL cannot be empty.', 'error')
        return redirect(url_for('admin_panel'))

    if not channel_id.isalnum() or ' ' in channel_id: # আইডি তে শুধু অক্ষর ও সংখ্যা রাখুন
         flash('Channel ID should only contain letters and numbers (no spaces).', 'error')
         return redirect(url_for('admin_panel'))

    # লোড করা ডেটা ব্যবহার করে চেক করুন চ্যানেল আগে থেকে আছে কিনা
    current_channels = load_channels()
    is_update = channel_id in current_channels and action == 'update'
    is_add = not is_update and action == 'add'

    if is_add and channel_id in current_channels:
         flash(f'Channel ID "{channel_id}" already exists. Use edit to update.', 'error')
         return redirect(url_for('admin_panel'))

    try:
        parsed_url = urlparse(source_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise ValueError("Invalid URL format (Requires scheme and netloc)")
        # Base URL ক্যালকুলেট করুন source_url থেকে (পরিষ্কার করা URL)
        base_url = urljoin(source_url, '.')
    except ValueError as e:
        flash(f'Invalid Source URL format: {e}. Please provide a full URL (e.g., http://...).', 'error')
        return redirect(url_for('admin_panel'))

    # ডেটা আপডেট করুন (প্রথমে মেমরিতে, তারপর ফাইলে)
    current_channels[channel_id] = {'source_url': source_url, 'base_url': base_url}
    if save_channels(current_channels):
        # গ্লোবাল স্টেট আপডেট করুন
        channels_data = current_channels
        if is_update:
             flash(f'Channel "{channel_id}" updated successfully.', 'success')
        elif is_add:
             flash(f'Channel "{channel_id}" added successfully.', 'success')
        else: # যদি action=add কিন্তু is_update=true হয় (বা উল্টো), যা হওয়া উচিত নয়
             flash(f'Channel "{channel_id}" processed. Check list.', 'warning')

    else:
        flash('Failed to save channel data. Check server logs.', 'error')

    return redirect(url_for('admin_panel'))

@app.route('/admin/channels/delete/<channel_id>', methods=['POST'])
def delete_channel(channel_id):
    """নির্দিষ্ট চ্যানেল মুছে ফেলে।"""
    global channels_data # গ্লোবাল স্টেট আপডেট করার জন্য
    if not session.get('logged_in'):
        abort(403)

    channel_id = channel_id.lower() # নিশ্চিত করুন আইডি লোয়ারকেস আছে

    # ফাইল থেকে লোড করে নিশ্চিত হন সর্বশেষ ডেটা আছে
    current_channels = load_channels()

    if channel_id in current_channels:
        current_channels.pop(channel_id) # ডিকশনারি থেকে রিমুভ করুন
        if save_channels(current_channels):
             # গ্লোবাল স্টেট আপডেট করুন
            channels_data = current_channels
            flash(f'Channel "{channel_id}" deleted successfully.', 'success')
        else:
            flash('Failed to save changes after deletion. Check server logs.', 'error')
    else:
        flash(f'Channel ID "{channel_id}" not found.', 'error')

    return redirect(url_for('admin_panel'))

# --- Stream Proxy Routes (Dynamic) ---

# M3U8 Manifest Route
@app.route('/live/<channel_id>.m3u8')
def serve_m3u8(channel_id):
    """নির্দিষ্ট চ্যানেলের জন্য পরিবর্তিত M3U8 ফাইল পরিবেশন করে।"""
    channel_id = channel_id.lower()
    # সবসময় লেটেস্ট ডেটা ব্যবহার করুন
    current_channels = load_channels()
    if channel_id not in current_channels:
        abort(404, description=f"Channel '{channel_id}' not found.")

    channel_info = current_channels[channel_id]
    source_url = channel_info.get('source_url')
    base_url = channel_info.get('base_url') # কী URI বা Absolute Path এর জন্য দরকার হতে পারে

    if not source_url:
         abort(500, description=f"Configuration error: Source URL missing for channel '{channel_id}'.")

    print(f"Fetching M3U8 for channel '{channel_id}' from: {source_url}")

    try:
        # User-Agent সেট করা অনেক সময় সাহায্য করে ব্লক এড়াতে
        # কিছু সার্ভার নির্দিষ্ট User-Agent ব্লক করতে পারে, তাই একটি সাধারণ ব্রাউজারের মতো সেট করা ভালো
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(source_url, timeout=15, headers=headers, allow_redirects=True) # Allow redirects
        response.raise_for_status() # HTTP error থাকলে exception raise করবে

        # Check if the content type is actually an M3U8 playlist
        content_type = response.headers.get('Content-Type', '').lower()
        if 'mpegurl' not in content_type and 'x-mpegurl' not in content_type:
             print(f"Warning: Content-Type for channel '{channel_id}' is '{content_type}', not M3U8. Trying to process anyway.")
             # Optionally, you could abort here if strict checking is needed:
             # abort(502, description=f"Origin server did not return an M3U8 playlist (Content-Type: {content_type})")

        original_m3u8_content = response.text
        modified_lines = []

        # Determine the effective base URL after potential redirects
        effective_base_url = urljoin(response.url, '.') if base_url else None # Use final URL from response

        for line in original_m3u8_content.splitlines():
            line = line.strip()
            if not line:
                continue

            if line.startswith('#EXT-X-KEY'):
                if 'URI="' in line:
                    parts = line.split('URI="')
                    # URI এবং বাকি অংশ আলাদা করুন (Split by " only once)
                    uri_part_and_rest = parts[1].split('"', 1)
                    uri_part = uri_part_and_rest[0]
                    # Check if there is content after the closing quote
                    rest_of_line = uri_part_and_rest[1] if len(uri_part_and_rest) > 1 else ""

                    # যদি URI আপেক্ষিক হয় এবং একটি কার্যকর base_url থাকে
                    if effective_base_url and not urlparse(uri_part).scheme:
                        full_key_uri = urljoin(effective_base_url, uri_part)
                        # --- f-string সিনট্যাক্স এরর সমাধান করা হয়েছে ---
                        new_key_line = f'{parts[0]}URI="{full_key_uri}"{rest_of_line}'
                        modified_lines.append(new_key_line)
                        # ---------------------------------------------
                        print(f"Rewriting KEY URI: '{uri_part}' -> '{full_key_uri}' (Absolute)")
                    else:
                         # URI অ্যাবসোলিউট অথবা base_url নেই, তাই লাইন অপরিবর্তিত
                         modified_lines.append(line)
                else:
                     modified_lines.append(line) # URI অ্যাট্রিবিউট নেই

            elif not line.startswith('#'):
                # এটি একটি সেগমেন্ট বা প্লেলিস্ট URI
                segment_uri_part = line
                parsed_segment_uri = urlparse(segment_uri_part)

                if parsed_segment_uri.scheme:
                    # Absolute URL - অপরিবর্তিত রাখা হচ্ছে (প্রক্সি করা হচ্ছে না)
                    modified_lines.append(line)
                    # print(f"Keeping absolute URI: {line}") # ডিবাগিং এর জন্য
                elif segment_uri_part.lower().endswith('.m3u8'):
                     # Relative Nested playlist - এটিকে প্রক্সি করা দরকার
                     # একটি সাব-প্লেলিস্ট প্রক্সি রুট তৈরি করা যেতে পারে, অথবা এটিকে Absolute করা যেতে পারে
                     if effective_base_url:
                         absolute_sublist_url = urljoin(effective_base_url, segment_uri_part)
                         modified_lines.append(absolute_sublist_url)
                         print(f"Rewriting nested playlist URI: '{segment_uri_part}' -> '{absolute_sublist_url}' (Absolute)")
                     else:
                        # Base URL না থাকলে প্রক্সি করা কঠিন, আপাতত Relative রাখছি
                        # অথবা একটি Relative Proxy URL তৈরি করা যেতে পারে
                        proxy_sublist_url = url_for('serve_m3u8_relative', channel_id=channel_id, path=segment_uri_part) # উদাহরণ, এই রুটটি তৈরি করতে হবে
                        # modified_lines.append(proxy_sublist_url) # আপাতত Relative রাখছি
                        modified_lines.append(line)
                        print(f"Keeping relative nested playlist URI (no base_url): {line}")

                else:
                    # এটি একটি Relative সেগমেন্ট URI (.ts, .aac, etc.)
                    # সম্পূর্ণ URI (পাথ + query) কে segment প্যারামিটার হিসাবে পাস করুন
                    # url_for ব্যবহার করে প্রক্সি URL তৈরি করুন
                    proxy_segment_url = url_for('serve_segment', channel_id=channel_id, segment=segment_uri_part, _external=False)
                    modified_lines.append(proxy_segment_url)
                    # print(f"Rewriting segment URI: '{segment_uri_part}' -> '{proxy_segment_url}'") # ডিবাগিং এর জন্য

            else:
                # অন্য সব # ট্যাগ লাইন (#EXTINF, #EXT-X-VERSION, etc.) அப்படியே রাখুন
                modified_lines.append(line)

        modified_m3u8_content = "\n".join(modified_lines)
        # সঠিক MIME টাইপ সেট করুন
        return Response(modified_m3u8_content, mimetype='application/vnd.apple.mpegurl') #'audio/mpegurl' ও হতে পারে

    except requests.exceptions.Timeout:
        print(f"Timeout fetching manifest for channel '{channel_id}' from {source_url}")
        abort(504, description=f"Timeout fetching manifest for channel '{channel_id}'")
    except requests.exceptions.RequestException as e:
        status = e.response.status_code if e.response is not None else 502 # 502 Bad Gateway default
        print(f"Error fetching original M3U8 for channel '{channel_id}' ({source_url}). Status: {status}. Error: {e}")
        # ক্লায়েন্টকে মূল সার্ভারের দেওয়া এরর কোড দেখানো ভালো (যেমন 403, 404)
        abort(status if status in [403, 404] else 502, description=f"Could not fetch manifest for channel '{channel_id}'. Origin error: {status}")
    except Exception as e:
        print(f"Unexpected error processing manifest for channel '{channel_id}': {e}")
        # traceback.print_exc() # More detailed error for debugging
        abort(500, description="Internal server error processing manifest.")


# Segment Route
@app.route('/stream/<channel_id>/<path:segment>')
def serve_segment(channel_id, segment):
    """নির্দিষ্ট চ্যানেলের জন্য সেগমেন্ট ফাইল রিলে করে।"""
    channel_id = channel_id.lower()
    # সবসময় লেটেস্ট ডেটা ব্যবহার করুন
    current_channels = load_channels()
    if channel_id not in current_channels:
        abort(404, description=f"Channel '{channel_id}' not found for segment request.")

    channel_info = current_channels[channel_id]
    # চ্যানেলের নিজস্ব বেস ইউআরএল ব্যবহার করুন
    base_url = channel_info.get('base_url')

    if not base_url:
         # যদি M3U8 রুট রিডাইরেক্ট করে থাকে, তাহলে সেই ফাইনাল URL থেকে বেস URL নেওয়া উচিত,
         # কিন্তু এই রুটে সেই 정보 নেই। তাই configuration error দেওয়াই যুক্তিযুক্ত।
         print(f"Error: Base URL not found in configuration for channel '{channel_id}'")
         abort(500, description=f"Configuration error: Base URL missing for channel '{channel_id}'.")

    # urljoin ব্যবহার করে আপেক্ষিক পাথ ('segment') সঠিকভাবে হ্যান্ডেল করুন
    try:
        original_segment_url = urljoin(base_url, segment)
    except ValueError as e:
        print(f"Error creating segment URL for channel '{channel_id}': base='{base_url}', segment='{segment}'. Error: {e}")
        abort(500, description="Internal error creating segment URL.")

    # print(f"Channel '{channel_id}': Fetching segment '{segment}' -> Resolved URL: {original_segment_url}") # Debug log

    try:
        # User-Agent এবং অন্যান্য হেডার যোগ করা ভালো
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        # কিছু সার্ভারের জন্য Referer প্রয়োজন হতে পারে
        # headers['Referer'] = base_url

        # stream=True ব্যবহার করুন বড় ফাইলের জন্য মেমরি বাঁচাতে
        response = requests.get(original_segment_url, stream=True, timeout=15, headers=headers, allow_redirects=True)
        response.raise_for_status() # 4xx বা 5xx স্ট্যাটাস কোডের জন্য exception raise করবে

        # Content-Type সঠিকভাবে সেট করুন (মূল সার্ভার থেকে নিন)
        content_type = response.headers.get('Content-Type', 'video/MP2T') # ডিফল্ট video/MP2T

        # ডেটা স্ট্রিম করুন ক্লায়েন্টের কাছে
        # iter_content ব্যবহার করে মেমরি সাশ্রয় করুন
        def generate_stream():
            try:
                for chunk in response.iter_content(chunk_size=1024*1024): # 1MB চাঙ্ক সাইজ
                    yield chunk
            except requests.exceptions.ChunkedEncodingError:
                 print(f"ChunkedEncodingError while streaming segment for channel {channel_id}")
                 # Optionally handle this, maybe abort or return partial content status?

        return Response(generate_stream(), # ডেটা স্ট্রিম করার জন্য জেনারেটর ব্যবহার করুন
                        content_type=content_type,
                        status=response.status_code) # মূল স্ট্যাটাস কোড পাস করুন (e.g., 200 OK)

    except requests.exceptions.Timeout:
        print(f"Timeout fetching segment '{segment}' for channel '{channel_id}' from {original_segment_url}")
        abort(504, description=f"Timeout fetching segment: {segment}")
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 502
        print(f"Error fetching segment '{segment}' for channel '{channel_id}' from {original_segment_url}. Status: {status_code}. Error: {e}")
        # ক্লায়েন্টকে মূল সার্ভারের দেওয়া এরর কোড দেখানো ভালো (যেমন 403, 404)
        abort(status_code if status_code in [403, 404] else 502, description=f"Failed to fetch segment '{segment}' for channel '{channel_id}'. Origin error: {status_code}")
    except Exception as e:
        print(f"Unexpected error serving segment '{segment}' for channel '{channel_id}': {e}")
        # traceback.print_exc() # More detailed error for debugging
        abort(500, description="Internal server error fetching segment.")


# --- Relative M3U8 Route (উদাহরণ - প্রয়োজন অনুযায়ী তৈরি করতে হবে) ---
# @app.route('/live_relative/<channel_id>/<path:path>')
# def serve_m3u8_relative(channel_id, path):
#     """Relative nested M3U8 playlist পরিবেশন করে"""
#     channel_id = channel_id.lower()
#     current_channels = load_channels()
#     if channel_id not in current_channels:
#         abort(404)
#
#     channel_info = current_channels[channel_id]
#     base_url = channel_info.get('base_url')
#     if not base_url:
#         abort(500, "Base URL not configured")
#
#     # মূল Relative playlist URL তৈরি করুন
#     original_relative_m3u8_url = urljoin(base_url, path)
#
#     # এখন এই URL থেকে ডেটা ফেচ করে serve_m3u8 এর মতো প্রসেস করতে হবে
#     # ... (এখানে serve_m3u8 এর লজিক প্রয়োগ করতে হবে) ...
#     print(f"Fetching relative playlist: {original_relative_m3u8_url}")
#     # আপাতত শুধু একটি placeholder দিচ্ছি
#     return Response(f"#EXTM3U\n# Placeholder for relative playlist: {path}\n", mimetype='application/vnd.apple.mpegurl')


# --- রুট পেজ (হোমপেজ) ---
@app.route('/')
def index():
    """হোমপেজে উপলব্ধ চ্যানেলগুলির একটি তালিকা দেখায়।"""
    # ফাইল থেকে লোড করে লেটেস্ট তালিকা দেখান
    current_channels = load_channels()
    available_channels = {
        cid: url_for('serve_m3u8', channel_id=cid, _external=True)
        for cid in current_channels.keys()
    }
    return render_template('index.html', available_channels=available_channels)

# --- মেইন এক্সিকিউশন ---
if __name__ == '__main__':
    print("Starting IPTV Proxy Server...")
    # প্রোডাকশনের জন্য Waitress ব্যবহার করার চেষ্টা করুন
    try:
        from waitress import serve
        print("Running with Waitress WSGI server on http://0.0.0.0:5000")
        serve(app, host='0.0.0.0', port=5000, threads=8) # থ্রেড সংখ্যা প্রয়োজন অনুযায়ী পরিবর্তন করুন
    except ImportError:
        print("Waitress not found, using Flask's development server (NOT recommended for production).")
        # ডেভেলপমেন্ট সার্ভার ব্যবহার করুন যদি Waitress ইনস্টল করা না থাকে
        # debug=True মোড প্রোডাকশনে ব্যবহার করবেন না!
        app.run(host='0.0.0.0', port=5000, debug=False)
