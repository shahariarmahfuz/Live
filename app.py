import os
import requests
import json
from flask import Flask, Response, abort, request, render_template, redirect, url_for, flash, session
from urllib.parse import urljoin, urlparse

app = Flask(__name__)

# সিক্রেট কী (অ্যাডমিন পাসওয়ার্ডের মতই এনভায়রনমেন্ট ভেরিয়েবল থেকে নেওয়া ভালো)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_key_for_multichannel')

# অ্যাডমিন পাসওয়ার্ড (এনভায়রনমেন্ট ভেরিয়েবল ব্যবহার করুন!)
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'password123') # এটি পরিবর্তন করুন!

# যে JSON ফাইলে চ্যানেলগুলোর তথ্য (ID -> URL) সংরক্ষণ করা হবে
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
                # বেস ইউআরএল ক্যালকুলেট করে নিন যদি না থাকে (পুরাতন ফরম্যাট থেকে আপগ্রেড)
                updated_data = {}
                for channel_id, info in data.items():
                    if isinstance(info, str): # পুরাতন ফরম্যাট (শুধু URL স্ট্রিং)
                         source_url = info
                         try:
                             base_url = urljoin(source_url, '.')
                             updated_data[channel_id] = {'source_url': source_url, 'base_url': base_url}
                         except ValueError:
                             print(f"Warning: Invalid source URL '{source_url}' for channel '{channel_id}'. Skipping.")
                             continue # অবৈধ হলে বাদ দিন
                    elif isinstance(info, dict) and 'source_url' in info:
                        # নতুন ফরম্যাট, শুধু নিশ্চিত করুন base_url আছে
                        if 'base_url' not in info:
                             try:
                                 info['base_url'] = urljoin(info['source_url'], '.')
                             except ValueError:
                                 print(f"Warning: Invalid source URL '{info['source_url']}' for channel '{channel_id}'. Skipping base_url.")
                        updated_data[channel_id] = info
                    else:
                         print(f"Warning: Invalid data format for channel '{channel_id}'. Skipping.")

                if updated_data != data:
                    save_channels(updated_data) # যদি কোনো পরিবর্তন হয়ে থাকে তবে সেভ করুন
                return updated_data
            else:
                print("Warning: channels.json is not a valid dictionary. Starting fresh.")
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
        with open(CHANNELS_STORAGE_FILE, 'w') as f:
            json.dump(channels_data, f, indent=4) # সুন্দর ফরম্যাটিং এর জন্য indent ব্যবহার করুন
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
    """অ্যাডমিন পেজ রেন্ডার করে।"""
    if not session.get('logged_in'):
        return render_template('admin_multi.html') # নতুন টেমপ্লেট ব্যবহার করুন
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
def add_channel():
    """নতুন চ্যানেল যোগ করে।"""
    global channels_data
    if not session.get('logged_in'):
        abort(403) # Forbidden

    channel_id = request.form.get('channel_id', '').strip().lower() # আইডি কে লোয়ারকেস করুন
    source_url = request.form.get('source_url', '').strip()

    # বেসিক ভ্যালিডেশন
    if not channel_id or not source_url:
        flash('Channel ID and Source URL cannot be empty.', 'error')
        return redirect(url_for('admin_panel'))

    if not channel_id.isalnum() or ' ' in channel_id: # আইডি তে শুধু অক্ষর ও সংখ্যা রাখুন
         flash('Channel ID should only contain letters and numbers (no spaces).', 'error')
         return redirect(url_for('admin_panel'))

    if channel_id in channels_data:
        flash(f'Channel ID "{channel_id}" already exists.', 'error')
        return redirect(url_for('admin_panel'))

    try:
        parsed_url = urlparse(source_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise ValueError("Invalid URL format")
        base_url = urljoin(source_url, '.') # বেস ইউআরএল ক্যালকুলেট করুন
    except ValueError as e:
        flash(f'Invalid Source URL format: {e}.', 'error')
        return redirect(url_for('admin_panel'))

    # ডেটা আপডেট করুন
    channels_data[channel_id] = {'source_url': source_url, 'base_url': base_url}
    if save_channels(channels_data):
        flash(f'Channel "{channel_id}" added successfully.', 'success')
    else:
        # ডেটা রোলব্যাক করুন যদি সেভ না হয়
        channels_data.pop(channel_id, None)
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
    source_url = channel_info['source_url']
    # Base URL লাগবে না এখানে, কিন্তু সেগমেন্ট রুটে লাগবে

    try:
        response = requests.get(source_url, timeout=10)
        response.raise_for_status()

        original_m3u8_content = response.text
        modified_lines = []
        # সেগমেন্ট পাথ এর জন্য চ্যানেলের আইডি সহ নতুন প্রিফিক্স
        segment_proxy_prefix = f"/stream/{channel_id}"

        for line in original_m3u8_content.splitlines():
            line = line.strip()
            if not line:
                continue
            if not line.startswith('#'):
                # URI লাইন: এটিকে আমাদের প্রক্সি সেগমেন্ট হ্যান্ডলারের দিকে পয়েন্ট করান
                # এখানেও ধরে নিচ্ছি সেগমেন্ট পাথ relativo বা filename
                segment_name = line.split('/')[-1] # শুধু শেষ অংশ বা ফাইলের নাম নিন
                modified_line = f"{segment_proxy_prefix}/{segment_name}"
                modified_lines.append(modified_line)
            else:
                # কমেন্ট বা ট্যাগ লাইন அப்படியே রাখুন
                modified_lines.append(line)

        modified_m3u8_content = "\n".join(modified_lines)
        return Response(modified_m3u8_content, mimetype='application/vnd.apple.mpegurl')

    except requests.exceptions.Timeout:
        abort(504, description=f"Timeout fetching manifest for channel '{channel_id}' from {source_url}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching original M3U8 for channel '{channel_id}' ({source_url}): {e}")
        abort(502, description=f"Could not fetch manifest for channel '{channel_id}'. Error: {e}")
    except Exception as e:
        print(f"Unexpected error processing manifest for channel '{channel_id}': {e}")
        abort(500, description="Internal error processing manifest.")

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


    # Use urljoin to correctly handle relative paths in segment names if any
    original_segment_url = urljoin(base_url, segment)
    print(f"Channel '{channel_id}': Fetching segment {segment} from {original_segment_url}") # Debug log

    try:
        response = requests.get(original_segment_url, stream=True, timeout=10)
        response.raise_for_status()
        return Response(response.iter_content(chunk_size=1024*1024),
                        content_type=response.headers.get('Content-Type', 'video/MP2T'))

    except requests.exceptions.Timeout:
        print(f"Timeout fetching segment {segment} for channel '{channel_id}' from {original_segment_url}")
        abort(504, description=f"Timeout fetching segment: {segment}")
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 502
        print(f"Error fetching segment {segment} for channel '{channel_id}' from {original_segment_url}. Status: {status_code}. Error: {e}")
        abort(status_code if status_code == 404 else 502, description=f"Failed to fetch segment '{segment}' for channel '{channel_id}'. Origin error: {status_code}")
    except Exception as e:
        print(f"Unexpected error serving segment {segment} for channel '{channel_id}': {e}")
        abort(500, description="Internal error fetching segment.")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
