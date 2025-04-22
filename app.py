import os
import requests
import json
import html # <-- এই লাইনটি যোগ করা হয়েছে
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
    """JSON ফাইল থেকে চ্যানেল ডেটা লোড করে।"""
    if not os.path.exists(CHANNELS_STORAGE_FILE):
        return {} # ফাইল না থাকলে খালি ডিকশনারি রিটার্ন করুন
    try:
        with open(CHANNELS_STORAGE_FILE, 'r') as f:
            data = json.load(f)
            # নিশ্চিত করুন ডেটা সঠিক ফরম্যাটে আছে (dict of dicts)
            if isinstance(data, dict):
                 # base_url ক্যালকুলেট করে নিন যদি না থাকে (পুরাতন ফরম্যাট আপগ্রেড)
                updated_data = {}
                made_changes = False
                for channel_id, info in data.items():
                    # আইডি কে লোয়ারকেস করা ভালো সামঞ্জস্য রাখার জন্য
                    normalized_id = channel_id.lower()

                    if isinstance(info, str): # পুরাতন ফরম্যাট (শুধু URL স্ট্রিং)
                        source_url_raw = info
                        source_url = html.unescape(source_url_raw) # নিশ্চিত করুন URL আনএসকেপ করা আছে
                        try:
                            base_url = urljoin(source_url, '.')
                            updated_data[normalized_id] = {'source_url': source_url, 'base_url': base_url}
                            made_changes = True # পরিবর্তন হয়েছে
                        except ValueError:
                            print(f"Warning: Invalid source URL '{source_url}' for channel '{channel_id}'. Skipping.")
                            continue # অবৈধ হলে বাদ দিন
                    elif isinstance(info, dict) and 'source_url' in info:
                        # নতুন ফরম্যাট, শুধু নিশ্চিত করুন base_url আছে এবং আইডি লোয়ারকেস
                        source_url_raw = info['source_url']
                        source_url = html.unescape(source_url_raw) # নিশ্চিত করুন URL আনএসকেপ করা আছে
                        info['source_url'] = source_url # আপডেটেড URL সংরক্ষণ করুন

                        if 'base_url' not in info or not info['base_url']:
                             try:
                                 info['base_url'] = urljoin(source_url, '.')
                                 made_changes = True
                             except ValueError:
                                 print(f"Warning: Invalid source URL '{source_url}' for channel '{channel_id}'. Skipping base_url update.")
                                 info['base_url'] = None # বা খালি রাখুন

                        if channel_id != normalized_id: # যদি আইডি কেস ভিন্ন থাকে
                            made_changes = True

                        updated_data[normalized_id] = info # লোয়ারকেস আইডি দিয়ে সংরক্ষণ করুন

                    else:
                         print(f"Warning: Invalid data format for channel '{channel_id}'. Skipping.")

                if made_changes:
                    print("Updating channel data format, IDs to lowercase, or unescaping URLs...")
                    save_channels(updated_data) # যদি কোনো পরিবর্তন হয়ে থাকে তবে সেভ করুন
                return updated_data
            else:
                print(f"Warning: {CHANNELS_STORAGE_FILE} does not contain a valid dictionary. Starting fresh.")
                return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {CHANNELS_STORAGE_FILE}. Starting fresh.")
        return {}
    except Exception as e:
        print(f"Error loading channels from file: {e}")
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
# অ্যাপ্লিকেশন শুরু হওয়ার সময় চ্যানেল ডেটা লোড করুন
channels_data = load_channels()

# --- Admin Routes ---
@app.route('/admin', methods=['GET'])
def admin_panel():
    """অ্যাডমিন পেজ রেন্ডার করে। চ্যানেল লিস্ট সহ।"""
    if not session.get('logged_in'):
        return render_template('admin_multi.html') # লগইন পেজ দেখাবে
    # লগইন করা থাকলে চ্যানেল লিস্ট সহ অ্যাডমিন পেজ দেখান
    # নিশ্চিত করুন চ্যানেলের তথ্য টেমপ্লেটে পাঠানোর আগে আপডেট করা আছে
    return render_template('admin_multi.html', channels=load_channels()) # লোড করা ডেটা পাঠান

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
    global channels_data
    if not session.get('logged_in'):
        abort(403) # Forbidden

    channel_id = request.form.get('channel_id', '').strip().lower() # আইডি কে লোয়ারকেস করুন
    source_url_raw = request.form.get('source_url', '').strip() # Raw URL নিন
    action = request.form.get('action', 'add') # অ্যাকশন (add বা update)

    # --- সংশোধন: ইনপুট URL পরিষ্কার করুন ---
    source_url = html.unescape(source_url_raw) # HTML entities (&amp;, etc.) ঠিক করুন
    # ------------------------------------

    # বেসিক ভ্যালিডেশন
    if not channel_id or not source_url:
        flash('Channel ID and Source URL cannot be empty.', 'error')
        return redirect(url_for('admin_panel'))

    if not channel_id.isalnum() or ' ' in channel_id: # আইডি তে শুধু অক্ষর ও সংখ্যা রাখুন
         flash('Channel ID should only contain letters and numbers (no spaces).', 'error')
         return redirect(url_for('admin_panel'))

    # লোড করা ডেটা ব্যবহার করে চেক করুন চ্যানেল আগে থেকে আছে কিনা
    current_channels = load_channels()
    is_update = channel_id in current_channels

    try:
        parsed_url = urlparse(source_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise ValueError("Invalid URL format (Requires scheme and netloc)")
        # Base URL ক্যালকুলেট করুন source_url থেকে (পরিষ্কার করা URL)
        base_url = urljoin(source_url, '.')
    except ValueError as e:
        flash(f'Invalid Source URL format: {e}. Please provide a full URL (e.g., http://...).', 'error')
        return redirect(url_for('admin_panel'))

    # ডেটা আপডেট করুন (গ্লোবাল ভ্যারিয়েবলে এবং ফাইলে)
    current_channels[channel_id] = {'source_url': source_url, 'base_url': base_url}
    if save_channels(current_channels):
        # গ্লোবাল স্টেট আপডেট করুন
        channels_data = current_channels
        if is_update:
             flash(f'Channel "{channel_id}" updated successfully.', 'success')
        else:
             flash(f'Channel "{channel_id}" added successfully.', 'success')
    else:
        # এখানে রোলব্যাক করার দরকার নেই কারণ ফাইল সেভ হয়নি
        flash('Failed to save channel data. Check server logs.', 'error')

    return redirect(url_for('admin_panel'))

@app.route('/admin/channels/delete/<channel_id>', methods=['POST'])
def delete_channel(channel_id):
    """নির্দিষ্ট চ্যানেল মুছে ফেলে।"""
    global channels_data
    if not session.get('logged_in'):
        abort(403)

    channel_id = channel_id.lower() # নিশ্চিত করুন আইডি লোয়ারকেস আছে

    # ফাইল থেকে লোড করে নিশ্চিত হন সর্বশেষ ডেটা আছে
    current_channels = load_channels()

    if channel_id in current_channels:
        removed_data = current_channels.pop(channel_id) # ডিকশনারি থেকে রিমুভ করুন
        if save_channels(current_channels):
             # গ্লোবাল স্টেট আপডেট করুন
            channels_data = current_channels
            flash(f'Channel "{channel_id}" deleted successfully.', 'success')
        else:
            # এখানে রোলব্যাক করার দরকার নেই কারণ ফাইল সেভ হয়নি
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
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'} # আরও বাস্তবসম্মত User-Agent
        response = requests.get(source_url, timeout=15, headers=headers) # টাইমআউট বাড়ানো যেতে পারে
        response.raise_for_status() # HTTP error থাকলে exception raise করবে

        original_m3u8_content = response.text
        modified_lines = []

        # --- সংশোধন: উন্নত URL Rewrite ---
        for line in original_m3u8_content.splitlines():
            line = line.strip()
            if not line:
                continue

            if line.startswith('#EXT-X-KEY'):
                # কী URI যদি Relative হয়, তাহলে Base URL দিয়ে Absolute করতে হবে
                # এবং প্রক্সি করতে হলে নতুন রুট লাগবে। আপাতত অপরিবর্তিত রাখা হচ্ছে।
                # এখানে URI এক্সট্র্যাক্ট করে urljoin ব্যবহার করা যেতে পারে।
                if 'URI="' in line:
                    parts = line.split('URI="')
                    uri_part = parts[1].split('"')[0]
                    # যদি URI আপেক্ষিক হয় এবং base_url থাকে
                    if base_url and not urlparse(uri_part).scheme:
                        # সম্পূর্ণ কী URI তৈরি করুন
                        full_key_uri = urljoin(base_url, uri_part)
                        # একটি কী প্রক্সি রুট তৈরি করতে হবে (যেমন /key_proxy/<channel_id>/<key_path>)
                        # proxy_key_uri = url_for('serve_key', channel_id=channel_id, key_path=uri_part) # উদাহরণ
                        # modified_lines.append(f'{parts[0]}URI="{proxy_key_uri}"{parts[1].split("\"", 1)[1]}')
                        # আপাতত Absolute URI ব্যবহার করছি
                        modified_lines.append(f'{parts[0]}URI="{full_key_uri}"{parts[1].split("\"", 1)[1]}')
                        print(f"Rewriting KEY URI: '{uri_part}' -> '{full_key_uri}' (Absolute)")
                    else:
                         modified_lines.append(line) # Absolute বা base_url নেই
                else:
                     modified_lines.append(line) # URI অ্যাট্রিবিউট নেই

            elif not line.startswith('#'):
                # এটি একটি সেগমেন্ট বা প্লেলিস্ট URI
                segment_uri_part = line # মূল M3U8 থেকে পাওয়া URI

                parsed_segment_uri = urlparse(segment_uri_part)
                if parsed_segment_uri.scheme:
                    # Absolute URL - অপরিবর্তিত রাখা হচ্ছে (প্রক্সি করা হচ্ছে না)
                    # কারণ serve_segment ভিন্ন ডোমেইন হ্যান্ডেল করার জন্য তৈরি নয়
                    modified_lines.append(line)
                    print(f"Keeping absolute URI: {line}")
                elif segment_uri_part.lower().endswith('.m3u8'):
                    # Relative Nested playlist - এটিকে প্রক্সি করা দরকার
                    # একটি সাব-প্লেলিস্ট প্রক্সি রুট তৈরি করা যেতে পারে
                    # proxy_sublist_url = url_for('serve_m3u8_relative', channel_id=channel_id, path=segment_uri_part) # উদাহরণ
                    # আপাতত Absolute করার চেষ্টা করা হচ্ছে, যদি Base URL থাকে
                    if base_url:
                         absolute_sublist_url = urljoin(base_url, segment_uri_part)
                         modified_lines.append(absolute_sublist_url)
                         print(f"Rewriting nested playlist URI: '{segment_uri_part}' -> '{absolute_sublist_url}' (Absolute)")
                    else:
                        modified_lines.append(line) # Base URL না থাকলে অপরিবর্তিত
                        print(f"Keeping relative nested playlist URI (no base_url): {line}")

                else:
                    # এটি একটি Relative সেগমেন্ট URI (.ts, .aac, etc.)
                    # সম্পূর্ণ URI (পাথ + query) কে segment প্যারামিটার হিসাবে পাস করুন
                    proxy_segment_url = url_for('serve_segment', channel_id=channel_id, segment=segment_uri_part, _external=False)
                    modified_lines.append(proxy_segment_url)
                    # print(f"Rewriting segment URI: '{segment_uri_part}' -> '{proxy_segment_url}'") # ডিবাগিং এর জন্য

            else:
                # অন্য সব # ট্যাগ লাইন (#EXTINF, #EXT-X-VERSION, etc.) அப்படியே রাখুন
                modified_lines.append(line)
        # -------------------------------------

        modified_m3u8_content = "\n".join(modified_lines)
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
    # এখন চ্যানেলের নিজস্ব বেস ইউআরএল ব্যবহার করুন
    base_url = channel_info.get('base_url') # লোড করার সময় ক্যালকুলেট করা হয়েছে

    if not base_url:
         print(f"Error: Base URL not found for channel '{channel_id}'")
         abort(500, description=f"Configuration error: Base URL missing for channel '{channel_id}'.")

    # urljoin ব্যবহার করে আপেক্ষিক পাথ সঠিকভাবে হ্যান্ডেল করুন
    # 'segment' ভ্যারিয়েবলে M3U8 থেকে পাওয়া সম্পূর্ণ পাথ থাকবে (e.g., "path/to/seg.ts?key=val")
    original_segment_url = urljoin(base_url, segment)
    print(f"Channel '{channel_id}': Fetching segment '{segment}' -> Resolved URL: {original_segment_url}") # Debug log

    try:
        # এখানেও User-Agent এবং অন্যান্য হেডার যোগ করা ভালো
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        # রেফারার বা অন্যান্য হেডার প্রয়োজন হতে পারে কিছু সার্ভারের জন্য
        # headers['Referer'] = base_url or source_url

        # stream=True ব্যবহার করুন বড় ফাইলের জন্য মেমরি বাঁচাতে
        response = requests.get(original_segment_url, stream=True, timeout=15, headers=headers) # টাইমআউট বাড়ানো যেতে পারে
        response.raise_for_status() # 4xx বা 5xx স্ট্যাটাস কোডের জন্য exception raise করবে

        # Content-Type সঠিকভাবে সেট করুন
        content_type = response.headers.get('Content-Type', 'video/MP2T') # ডিফল্ট video/MP2T

        # ডেটা স্ট্রিম করুন ক্লায়েন্টের কাছে
        return Response(response.iter_content(chunk_size=1024*1024), # 1MB চাঙ্ক সাইজ
                        content_type=content_type,
                        status=response.status_code) # মূল স্ট্যাটাস কোড পাস করুন

    except requests.exceptions.Timeout:
        print(f"Timeout fetching segment {segment} for channel '{channel_id}' from {original_segment_url}")
        abort(504, description=f"Timeout fetching segment: {segment}")
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 502
        print(f"Error fetching segment '{segment}' for channel '{channel_id}' from {original_segment_url}. Status: {status_code}. Error: {e}")
        abort(status_code if status_code in [403, 404] else 502, description=f"Failed to fetch segment '{segment}' for channel '{channel_id}'. Origin error: {status_code}")
    except Exception as e:
        print(f"Unexpected error serving segment {segment} for channel '{channel_id}': {e}")
        abort(500, description="Internal server error fetching segment.")


# --- রুট পেজ (অপশনাল) ---
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
    # ডেভেলপমেন্টের জন্য debug=True ব্যবহার করতে পারেন, কিন্তু প্রোডাকশনে এটি বন্ধ রাখুন
    # প্রোডাকশনের জন্য Waitress বা Gunicorn এর মতো WSGI সার্ভার ব্যবহার করুন
    # উদাহরণ: waitress-serve --host 0.0.0.0 --port 5000 app:app
    # app.run(host='0.0.0.0', port=5000, debug=True) # Debug মোড বন্ধ রাখুন প্রোডাকশনে
    print("Starting Flask Server...")
    try:
        from waitress import serve
        serve(app, host='0.0.0.0', port=5000)
    except ImportError:
        print("Waitress not found, using Flask's development server (Not recommended for production).")
        app.run(host='0.0.0.0', port=5000)
