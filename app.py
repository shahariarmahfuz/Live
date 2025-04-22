import os
import requests
import json
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
                        source_url = info
                        try:
                            base_url = urljoin(source_url, '.')
                            updated_data[normalized_id] = {'source_url': source_url, 'base_url': base_url}
                            made_changes = True # পরিবর্তন হয়েছে
                        except ValueError:
                            print(f"Warning: Invalid source URL '{source_url}' for channel '{channel_id}'. Skipping.")
                            continue # অবৈধ হলে বাদ দিন
                    elif isinstance(info, dict) and 'source_url' in info:
                        # নতুন ফরম্যাট, শুধু নিশ্চিত করুন base_url আছে এবং আইডি লোয়ারকেস
                        source_url = info['source_url']
                        if 'base_url' not in info or not info['base_url']:
                             try:
                                 info['base_url'] = urljoin(source_url, '.')
                                 made_changes = True
                             except ValueError:
                                 print(f"Warning: Invalid source URL '{source_url}' for channel '{channel_id}'. Skipping base_url update.")
                                 # Base URL ছাড়াও সোর্স URL সহ ডেটা রাখা যায়
                                 info['base_url'] = None # বা খালি রাখুন

                        if channel_id != normalized_id: # যদি আইডি কেস ভিন্ন থাকে
                            made_changes = True

                        updated_data[normalized_id] = info # লোয়ারকেস আইডি দিয়ে সংরক্ষণ করুন

                    else:
                         print(f"Warning: Invalid data format for channel '{channel_id}'. Skipping.")

                if made_changes:
                    print("Updating channel data format or IDs to lowercase...")
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
    return render_template('admin_multi.html', channels=channels_data)

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
    source_url = request.form.get('source_url', '').strip()
    action = request.form.get('action', 'add') # অ্যাকশন (add বা update)

    # বেসিক ভ্যালিডেশন
    if not channel_id or not source_url:
        flash('Channel ID and Source URL cannot be empty.', 'error')
        return redirect(url_for('admin_panel'))

    if not channel_id.isalnum() or ' ' in channel_id: # আইডি তে শুধু অক্ষর ও সংখ্যা রাখুন
         flash('Channel ID should only contain letters and numbers (no spaces).', 'error')
         return redirect(url_for('admin_panel'))

    is_update = channel_id in channels_data

    try:
        parsed_url = urlparse(source_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise ValueError("Invalid URL format (Requires scheme and netloc)")
        base_url = urljoin(source_url, '.') # বেস ইউআরএল ক্যালকুলেট করুন
    except ValueError as e:
        flash(f'Invalid Source URL format: {e}. Please provide a full URL (e.g., http://...).', 'error')
        return redirect(url_for('admin_panel'))

    # ডেটা আপডেট করুন
    channels_data[channel_id] = {'source_url': source_url, 'base_url': base_url}
    if save_channels(channels_data):
        if is_update:
             flash(f'Channel "{channel_id}" updated successfully.', 'success')
        else:
             flash(f'Channel "{channel_id}" added successfully.', 'success')
    else:
        # ডেটা রোলব্যাক করুন যদি সেভ না হয়
        if not is_update: # শুধু নতুন অ্যাড এর ক্ষেত্রে রোলব্যাক করুন
            channels_data.pop(channel_id, None)
        # আপডেট ফেইল করলে আগের ডেটা রাখা যেতে পারে, অথবা এখানেও রোলব্যাক করা যায়
        flash('Failed to save channel data. Check server logs.', 'error')

    return redirect(url_for('admin_panel'))

@app.route('/admin/channels/delete/<channel_id>', methods=['POST'])
def delete_channel(channel_id):
    """নির্দিষ্ট চ্যানেল মুছে ফেলে।"""
    global channels_data
    if not session.get('logged_in'):
        abort(403)

    channel_id = channel_id.lower() # নিশ্চিত করুন আইডি লোয়ারকেস আছে
    if channel_id in channels_data:
        removed_data = channels_data.pop(channel_id) # ডিকশনারি থেকে রিমুভ করুন
        if save_channels(channels_data):
            flash(f'Channel "{channel_id}" deleted successfully.', 'success')
        else:
            # রোলব্যাক: যদি সেভ ফেইল করে, ডেটা ফেরত আনুন
            channels_data[channel_id] = removed_data
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
    if channel_id not in channels_data:
        abort(404, description=f"Channel '{channel_id}' not found.")

    channel_info = channels_data[channel_id]
    source_url = channel_info.get('source_url')
    base_url = channel_info.get('base_url') # এখন base_url ও দরকার হতে পারে (Key URI এর জন্য)

    if not source_url:
         abort(500, description=f"Configuration error: Source URL missing for channel '{channel_id}'.")

    print(f"Fetching M3U8 for channel '{channel_id}' from: {source_url}")

    try:
        # User-Agent সেট করা অনেক সময় সাহায্য করে ব্লক এড়াতে
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(source_url, timeout=10, headers=headers)
        response.raise_for_status() # HTTP error থাকলে exception raise করবে

        original_m3u8_content = response.text
        modified_lines = []
        # সেগমেন্ট পাথ এর জন্য চ্যানেলের আইডি সহ নতুন প্রিফিক্স
        segment_proxy_prefix = url_for('serve_segment', channel_id=channel_id, segment='placeholder', _external=False)
        segment_proxy_prefix = segment_proxy_prefix.rsplit('/', 1)[0] # '/stream/<channel_id>' অংশটি নিন

        for line in original_m3u8_content.splitlines():
            line = line.strip()
            if not line:
                continue

            if line.startswith('#EXT-X-KEY'):
                # কী URI প্রক্সি করার প্রয়োজন হতে পারে
                if 'URI="' in line:
                    parts = line.split('URI="')
                    uri_part = parts[1].split('"')[0]
                    # যদি URI আপেক্ষিক হয়, base_url ব্যবহার করে পূর্ণ URL তৈরি করুন
                    # তারপর সেই URL কে প্রক্সি রুটের মাধ্যমে পরিবর্তন করুন (এখানে সরল রাখা হয়েছে)
                    # আপাতত শুধু মূল URL রেখে দিচ্ছি, জটিলতা এড়াতে
                    # full_key_uri = urljoin(base_url, uri_part) if base_url and not urlparse(uri_part).scheme else uri_part
                    # proxy_key_uri = ... (যদি কী প্রক্সি করতে চান, নতুন রুট লাগবে)
                    # modified_lines.append(f'{parts[0]}URI="{proxy_key_uri}"{parts[1].split("\"", 1)[1]}')
                    modified_lines.append(line) # আপাতত অপরিবর্তিত রাখছি
                else:
                    modified_lines.append(line)

            elif not line.startswith('#'):
                # এটি একটি সেগমেন্ট বা প্লেলিস্ট URI হতে পারে
                # সম্পূর্ণ পাথ সহ URL তৈরি করার চেষ্টা করুন, যদি প্রয়োজন হয়
                # segment_url = urljoin(base_url, line) if base_url and not urlparse(line).scheme else line

                # শুধু ফাইলের নাম বা শেষ অংশ দিয়ে প্রক্সি URL তৈরি করুন
                segment_name = line.split('/')[-1] # শুধু শেষ অংশ বা ফাইলের নাম নিন
                # নিশ্চিত করুন সেগমেন্ট নামে Query প্যারামিটার থাকলে সেটাও থাকে
                if '?' in segment_name:
                    segment_name = segment_name.split('?', 1)[0] + '?' + segment_name.split('?', 1)[1]


                # .m3u8 দিয়ে শেষ হলে, এটি অন্য প্লেলিস্ট, এটিকেও প্রক্সি করতে হবে
                if line.lower().endswith('.m3u8'):
                     # অন্য প্লেলিস্টের আইডি কী হবে? এখানে একটি সমস্যা আছে।
                     # সহজ করার জন্য, ধরে নিচ্ছি সাব-প্লেলিস্ট নেই বা সেগুলো অ্যাবসোলিউট URL
                     # অথবা একটি সাধারণ প্রক্সি URL তৈরি করতে হবে যা বেস ইউআরএল ব্যবহার করে রিডাইরেক্ট করে
                     # আপাতত, অপরিবর্তিত রাখছি জটিলতা এড়াতে
                     modified_lines.append(line)
                else:
                    # এটি সম্ভবত একটি সেগমেন্ট (.ts, .aac, ইত্যাদি)
                    modified_line = f"{segment_proxy_prefix}/{segment_name}"
                    modified_lines.append(modified_line)

            else:
                # অন্য সব # ট্যাগ লাইন அப்படியே রাখুন
                modified_lines.append(line)

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
    if channel_id not in channels_data:
        abort(404, description=f"Channel '{channel_id}' not found for segment request.")

    channel_info = channels_data[channel_id]
    # এখন চ্যানেলের নিজস্ব বেস ইউআরএল ব্যবহার করুন
    base_url = channel_info.get('base_url') # লোড করার সময় ক্যালকুলেট করা হয়েছে

    if not base_url:
         # যদি কোনো কারণে বেস ইউআরএল না থাকে (যেমন অবৈধ সোর্স ইউআরএল)
         print(f"Error: Base URL not found for channel '{channel_id}'")
         abort(500, description=f"Configuration error: Base URL missing for channel '{channel_id}'.")


    # urljoin ব্যবহার করে আপেক্ষিক পাথ সঠিকভাবে হ্যান্ডেল করুন
    original_segment_url = urljoin(base_url, segment)
    print(f"Channel '{channel_id}': Fetching segment '{segment}' from {original_segment_url}") # Debug log

    try:
        # এখানেও User-Agent যোগ করা ভালো
        headers = {'User-Agent': 'Mozilla/5.0'}
        # stream=True ব্যবহার করুন বড় ফাইলের জন্য মেমরি বাঁচাতে
        response = requests.get(original_segment_url, stream=True, timeout=10, headers=headers)
        response.raise_for_status() # 4xx বা 5xx স্ট্যাটাস কোডের জন্য exception raise করবে

        # Content-Type সঠিকভাবে সেট করুন
        content_type = response.headers.get('Content-Type', 'video/MP2T')

        # ডেটা স্ট্রিম করুন ক্লায়েন্টের কাছে
        return Response(response.iter_content(chunk_size=1024*1024), # 1MB চাঙ্ক সাইজ
                        content_type=content_type,
                        status=response.status_code) # মূল স্ট্যাটাস কোড পাস করুন (যেমন 200 OK)

    except requests.exceptions.Timeout:
        print(f"Timeout fetching segment {segment} for channel '{channel_id}' from {original_segment_url}")
        abort(504, description=f"Timeout fetching segment: {segment}")
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 502
        print(f"Error fetching segment '{segment}' for channel '{channel_id}' from {original_segment_url}. Status: {status_code}. Error: {e}")
        # ক্লায়েন্টকে মূল সার্ভারের দেওয়া এরর কোড দেখানো ভালো (যেমন 404)
        abort(status_code if status_code in [403, 404] else 502, description=f"Failed to fetch segment '{segment}' for channel '{channel_id}'. Origin error: {status_code}")
    except Exception as e:
        print(f"Unexpected error serving segment {segment} for channel '{channel_id}': {e}")
        abort(500, description="Internal server error fetching segment.")


# --- রুট পেজ (অপশনাল) ---
@app.route('/')
def index():
    """হোমপেজে উপলব্ধ চ্যানেলগুলির একটি তালিকা দেখায়।"""
    # শুধুমাত্র চ্যানেল আইডি এবং প্রক্সি URL দেখাবে
    available_channels = {
        cid: url_for('serve_m3u8', channel_id=cid, _external=True)
        for cid in channels_data.keys()
    }
    return render_template('index.html', available_channels=available_channels)


if __name__ == '__main__':
    # ডেভেলপমেন্টের জন্য debug=True ব্যবহার করতে পারেন, কিন্তু প্রোডাকশনে এটি বন্ধ রাখুন
    # প্রোডাকশনের জন্য Waitress বা Gunicorn এর মতো WSGI সার্ভার ব্যবহার করুন
    # app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000)
