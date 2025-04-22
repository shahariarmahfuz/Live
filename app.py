import os
import requests
import json
import html  # <-- HTML এনটিটি ডিকোড করার জন্য ইম্পোর্ট করা হয়েছে
from flask import Flask, Response, abort, request, render_template, redirect, url_for, flash, session
from urllib.parse import urljoin, urlparse

app = Flask(__name__)

# সিক্রেট কী (পরিবেশ ভেরিয়েবল থেকে নেওয়া ভালো)
# বাংলাদেশ সময় অনুযায়ী একটি ডিফল্ট কী দেওয়া হলো, পরিবর্তন করতে পারেন
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dhaka_secret_key_multichannel_2025')

# অ্যাডমিন পাসওয়ার্ড (পরিবেশ ভেরিয়েবল ব্যবহার করুন!)
# একটি ডিফল্ট পাসওয়ার্ড দেওয়া হলো, অবশ্যই এটি পরিবর্তন করুন!
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'adminpassBD123')

# যে JSON ফাইলে চ্যানেলগুলোর তথ্য (ID -> URL Info) সংরক্ষণ করা হবে
CHANNELS_STORAGE_FILE = "channels.json"

# --- Helper Functions ---

def load_channels():
    """JSON ফাইল থেকে চ্যানেল ডেটা লোড করে।"""
    if not os.path.exists(CHANNELS_STORAGE_FILE):
        return {}
    try:
        with open(CHANNELS_STORAGE_FILE, 'r', encoding='utf-8') as f: # এনকোডিং উল্লেখ করা ভালো
            data = json.load(f)
            if isinstance(data, dict):
                updated_data = {}
                made_changes = False
                for channel_id, info in data.items():
                    normalized_id = channel_id.lower()
                    current_source_url = None
                    current_base_url = None

                    if isinstance(info, str): # পুরাতন ফরম্যাট (শুধু URL স্ট্রিং)
                        current_source_url = info
                        made_changes = True # Format needs update
                    elif isinstance(info, dict) and 'source_url' in info:
                         current_source_url = info.get('source_url')
                         current_base_url = info.get('base_url')
                    else:
                         print(f"Warning: Invalid data format for channel '{channel_id}'. Skipping.")
                         continue

                    # সোর্স URL ঠিক আছে কিনা এবং HTML এনকোডিং আছে কিনা দেখুন
                    if current_source_url:
                         try:
                             # আগে সেভ করা URL এ &amp; থাকলে ঠিক করুন
                             corrected_source_url = html.unescape(current_source_url)
                             if corrected_source_url != current_source_url:
                                 print(f"Correcting stored URL for channel '{normalized_id}'")
                                 current_source_url = corrected_source_url
                                 made_changes = True # URL was corrected

                             # Base URL গণনা করুন বা আপডেট করুন যদি না থাকে বা ভুল মনে হয়
                             parsed_src = urlparse(current_source_url)
                             if not all([parsed_src.scheme, parsed_src.netloc]):
                                 raise ValueError("Invalid source URL format stored")

                             calculated_base_url = urljoin(current_source_url, '.')
                             if current_base_url != calculated_base_url:
                                  current_base_url = calculated_base_url
                                  made_changes = True # Base URL updated or added

                         except ValueError as e:
                             print(f"Warning: Invalid source URL '{current_source_url}' for channel '{channel_id}' during load. Skipping. Error: {e}")
                             continue # সমস্যাযুক্ত চ্যানেল বাদ দিন

                    else: # সোর্স URL না থাকলে বাদ দিন
                         print(f"Warning: Missing source_url for channel '{channel_id}'. Skipping.")
                         continue

                    # আইডি লোয়ারকেস করা হয়েছে কিনা দেখুন
                    if channel_id != normalized_id:
                         made_changes = True

                    updated_data[normalized_id] = {'source_url': current_source_url, 'base_url': current_base_url}

                if made_changes:
                    print("Updating channel data format, correcting URLs, or normalizing IDs...")
                    # যদি কোনো পরিবর্তন হয়ে থাকে তবে সেভ করুন
                    # save_channels এখানে কল না করে, শুরুতে লোড করার পর একবার সেভ করা যেতে পারে
                    # অথবা শুধু updated_data রিটার্ন করুন, কারণ save_channels কল হবে অ্যাড/আপডেট/ডিলিটে
                    pass # শুধু মেমরিতে আপডেট রাখুন, ফাইল সেভ হবে অপারেশনে

                return updated_data
            else:
                print(f"Warning: {CHANNELS_STORAGE_FILE} does not contain a valid dictionary. Starting fresh.")
                return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {CHANNELS_STORAGE_FILE}. Starting fresh.")
        return {}
    except Exception as e:
        print(f"Error loading channels from file: {e}")
        return {}

def save_channels(channels_data):
    """চ্যানেল ডেটা JSON ফাইলে সংরক্ষণ করে।"""
    try:
        normalized_data = {k.lower(): v for k, v in channels_data.items()}
        with open(CHANNELS_STORAGE_FILE, 'w', encoding='utf-8') as f:
            json.dump(normalized_data, f, indent=4, ensure_ascii=False) # ensure_ascii=False বাংলা অক্ষরের জন্য ভালো
        print(f"Channel data saved to {CHANNELS_STORAGE_FILE}")
        return True
    except Exception as e:
        print(f"Error saving channels to file: {e}")
        return False

# --- Global State ---
channels_data = load_channels()
# যদি load_channels এ কোনো পরিবর্তন করে থাকে, একবার সেভ করা যেতে পারে
# save_channels(channels_data) # তবে এটি প্রতিবার স্টার্টআপে ফাইল লিখবে

# --- Admin Routes ---
@app.route('/admin', methods=['GET'])
def admin_panel():
    if not session.get('logged_in'):
        return render_template('admin_multi.html')
    return render_template('admin_multi.html', channels=channels_data)

@app.route('/admin/login', methods=['POST'])
def admin_login():
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
    session.pop('logged_in', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('admin_panel'))

@app.route('/admin/channels', methods=['POST'])
def add_or_update_channel():
    """নতুন চ্যানেল যোগ করে বা বিদ্যমান চ্যানেল আপডেট করে এবং URL ঠিক করে।"""
    global channels_data
    if not session.get('logged_in'):
        abort(403)

    channel_id = request.form.get('channel_id', '').strip().lower()
    source_url = request.form.get('source_url', '').strip()

    # --- URL ঠিক করার কোড ---
    if source_url:
        original_url = source_url
        # HTML এনটিটি ডিকোড করুন (যেমন &amp; -> &)
        source_url = html.unescape(source_url)
        if original_url != source_url:
             print(f"Corrected submitted URL: '{original_url}' -> '{source_url}'")
    # --- পরিবর্তন শেষ ---

    # বেসিক ভ্যালিডেশন (এখন ঠিক করা URL ব্যবহার করবে)
    if not channel_id or not source_url:
        flash('Channel ID and Source URL cannot be empty.', 'error')
        return redirect(url_for('admin_panel'))

    if not channel_id.isalnum() or ' ' in channel_id:
         flash('Channel ID should only contain letters and numbers (no spaces).', 'error')
         return redirect(url_for('admin_panel'))

    is_update = channel_id in channels_data

    try:
        # এখন ঠিক করা source_url ব্যবহার হচ্ছে
        parsed_url = urlparse(source_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            # স্কিম (http/https) এবং নেটওয়ার্ক লোকেশন (domain) আবশ্যক
            raise ValueError("Invalid URL format (Requires scheme like 'http://' and domain name)")
        # ঠিক করা source_url থেকে base_url গণনা করুন
        base_url = urljoin(source_url, '.')
    except ValueError as e:
        flash(f'Invalid Source URL format: {e}. Please provide a full URL (e.g., http://example.com/stream).', 'error')
        return redirect(url_for('admin_panel'))

    # ডেটা আপডেট করুন (ঠিক করা URL সহ)
    channels_data[channel_id] = {'source_url': source_url, 'base_url': base_url}
    if save_channels(channels_data):
        if is_update:
             flash(f'Channel "{channel_id}" updated successfully.', 'success')
        else:
             flash(f'Channel "{channel_id}" added successfully.', 'success')
    else:
        # এখানে রোলব্যাক লজিক যোগ করা যেতে পারে যদি প্রয়োজন হয়
        flash('Failed to save channel data. Check server logs.', 'error')
        # যদি সেভ ফেইল করে, মেমরিতে থাকা ডেটা রোলব্যাক করা উচিত
        # কিন্তু যেহেতু channels_data গ্লোবাল, এটি জটিল হতে পারে।
        # আপাতত শুধু এরর দেখানো হচ্ছে।

    return redirect(url_for('admin_panel'))

@app.route('/admin/channels/delete/<channel_id>', methods=['POST'])
def delete_channel(channel_id):
    """নির্দিষ্ট চ্যানেল মুছে ফেলে।"""
    global channels_data
    if not session.get('logged_in'):
        abort(403)

    channel_id = channel_id.lower()
    if channel_id in channels_data:
        removed_data = channels_data.pop(channel_id)
        if save_channels(channels_data):
            flash(f'Channel "{channel_id}" deleted successfully.', 'success')
        else:
            # রোলব্যাক
            channels_data[channel_id] = removed_data
            flash('Failed to save changes after deletion. Check server logs.', 'error')
    else:
        flash(f'Channel ID "{channel_id}" not found.', 'error')

    return redirect(url_for('admin_panel'))

# --- Stream Proxy Routes (Dynamic) ---

@app.route('/live/<channel_id>.m3u8')
def serve_m3u8(channel_id):
    """নির্দিষ্ট চ্যানেলের জন্য পরিবর্তিত M3U8 ফাইল পরিবেশন করে।"""
    channel_id = channel_id.lower()
    if channel_id not in channels_data:
        abort(404, description=f"Channel '{channel_id}' not found.")

    channel_info = channels_data[channel_id]
    source_url = channel_info.get('source_url')
    base_url = channel_info.get('base_url') # কী এবং সেগমেন্ট URL সমাধানের জন্য

    if not source_url or not base_url:
         abort(500, description=f"Configuration error: Source or Base URL missing for channel '{channel_id}'.")

    print(f"Fetching M3U8 for channel '{channel_id}' from: {source_url}")

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'}
        response = requests.get(source_url, timeout=15, headers=headers, allow_redirects=True) # Timeout বাড়ানো হলো, redirect ফলো করা হলো
        response.raise_for_status()

        # যদি কন্টেন্ট টাইপ M3U8 না হয়, তবে সমস্যা হতে পারে
        content_type = response.headers.get('Content-Type', '').lower()
        if 'mpegurl' not in content_type and 'x-mpegurl' not in content_type:
             print(f"Warning: Unexpected Content-Type '{content_type}' for M3U8 request from {source_url}")
             # আপনি এখানে abort করতে পারেন অথবা চেষ্টা চালিয়ে যেতে পারেন

        original_m3u8_content = response.text
        modified_lines = []
        # সেগমেন্ট পাথ এর জন্য চ্যানেলের আইডি সহ নতুন প্রিফিক্স
        # url_for ব্যবহার করলে এটি হোস্টনেম সহ পূর্ণ URL তৈরি করতে পারে, যা প্লেয়ার সবসময় পছন্দ নাও করতে পারে।
        # আপেক্ষিক পাথ ব্যবহার করা ভালো: /stream/<channel_id>/segment_name
        segment_proxy_prefix = f"/stream/{channel_id}"

        for line in original_m3u8_content.splitlines():
            line = line.strip()
            if not line:
                continue

            if line.startswith('#EXT-X-KEY'):
                # কী URI প্রক্সি করার চেষ্টা করুন (যদি আপেক্ষিক হয়)
                if 'URI="' in line:
                    try:
                        parts = line.split('URI="')
                        uri_part = parts[1].split('"')[0]
                        # কী URI কি আপেক্ষিক?
                        parsed_key_uri = urlparse(uri_part)
                        if not parsed_key_uri.scheme and not parsed_key_uri.netloc:
                             # আপেক্ষিক হলে, আমাদের প্রক্সি দিয়ে নিয়ে আসতে হবে
                             # একটি ডেডিকেটেড কী প্রক্সি রুট তৈরি করা ভালো, কিন্তু সরলতার জন্য সেগমেন্ট রুট ব্যবহার করা যেতে পারে
                             # ধরে নিচ্ছি কী ফাইলের নাম সেগমেন্টের মতই ইউনিক
                             proxy_key_uri = f"{segment_proxy_prefix}/{uri_part.lstrip('/')}"
                             modified_line = f'{parts[0]}URI="{proxy_key_uri}"{parts[1].split("\"", 1)[1]}'
                             modified_lines.append(modified_line)
                             print(f"Proxied KEY URI: {uri_part} -> {proxy_key_uri}")
                        else:
                             # অ্যাবসোলিউট URI হলে அப்படியே রাখুন
                             modified_lines.append(line)
                    except Exception as key_e:
                         print(f"Error processing EXT-X-KEY line: {line} - Error: {key_e}")
                         modified_lines.append(line) # সমস্যা হলে আসল লাইন রাখুন
                else:
                    modified_lines.append(line)

            elif not line.startswith('#'):
                # এটি একটি সেগমেন্ট বা প্লেলিস্ট URI
                parsed_segment_uri = urlparse(line)
                if not parsed_segment_uri.scheme and not parsed_segment_uri.netloc:
                     # আপেক্ষিক URI, এটিকে প্রক্সি করুন
                     # পাথ থেকে শুধুমাত্র শেষ অংশটি নিন (ফাইলনাম বা শেষ ডিরেক্টরি/ফাইলনাম)
                     # Query প্যারামিটার সহ নামটি সঠিকভাবে পেতে হবে
                     segment_full_name = line.split('/')[-1]
                     # আপেক্ষিক পাথ হ্যান্ডেল করার জন্য base_url এর সাথে join করা উচিত নয় এখানে,
                     # কারণ আমরা শুধু শেষ অংশটুকু চাইছি প্রক্সি রুটে পাঠাতে।
                     # কিন্তু মূল সার্ভার থেকে ফ্লেচ করার সময় base_url লাগবে।
                     # নিশ্চিত করুন URI থেকে লিডিং স্ল্যাশ বাদ দেওয়া হয়েছে প্রক্সি রুটে যোগ করার আগে
                     modified_line = f"{segment_proxy_prefix}/{line.lstrip('/')}"
                     modified_lines.append(modified_line)
                else:
                     # অ্যাবসোলিউট URI হলে அப்படியே রাখুন (অন্য সার্ভারের দিকে নির্দেশ করছে)
                     # অথবা যদি নিজের ডোমেইন হয় তাহলে প্রক্সি করা যেতে পারে (জটিলতা বাড়াবে)
                     modified_lines.append(line)

            else:
                # অন্য সব # ট্যাগ লাইন (যেমন #EXTINF, #EXT-X-VERSION ইত্যাদি) அப்படியே রাখুন
                modified_lines.append(line)

        modified_m3u8_content = "\n".join(modified_lines)
        # ক্লায়েন্টকে বলুন এটি UTF-8 এনকোডেড
        return Response(modified_m3u8_content, mimetype='application/vnd.apple.mpegurl; charset=utf-8')

    except requests.exceptions.Timeout:
        print(f"Timeout fetching manifest for channel '{channel_id}' from {source_url}")
        abort(504, description=f"Timeout fetching manifest for channel '{channel_id}'")
    except requests.exceptions.RequestException as e:
        status = e.response.status_code if e.response is not None else 502
        print(f"Error fetching M3U8 for channel '{channel_id}' ({source_url}). Status: {status}. Error: {e}")
        abort(status if status in [403, 404] else 502, description=f"Could not fetch manifest for channel '{channel_id}'. Origin error: {status}")
    except Exception as e:
        print(f"Unexpected error processing manifest for channel '{channel_id}': {e}")
        import traceback
        traceback.print_exc() # ডিবাগিং এর জন্য বিস্তারিত ট্রেসব্যাক প্রিন্ট করুন
        abort(500, description="Internal server error processing manifest.")


@app.route('/stream/<channel_id>/<path:segment>')
def serve_segment(channel_id, segment):
    """নির্দিষ্ট চ্যানেলের জন্য সেগমেন্ট ফাইল বা কী ফাইল রিলে করে।"""
    channel_id = channel_id.lower()
    if channel_id not in channels_data:
        abort(404, description=f"Channel '{channel_id}' not found for segment/key request.")

    channel_info = channels_data[channel_id]
    base_url = channel_info.get('base_url')

    if not base_url:
         print(f"Error: Base URL not found for channel '{channel_id}'")
         abort(500, description=f"Configuration error: Base URL missing for channel '{channel_id}'.")

    # urljoin ব্যবহার করে আপেক্ষিক পাথ সঠিকভাবে হ্যান্ডেল করুন
    # segment ভেরিয়েবলে সম্পূর্ণ আপেক্ষিক পাথ থাকে (যেমন dir/file.ts?query=...)
    original_resource_url = urljoin(base_url, segment)
    print(f"Channel '{channel_id}': Fetching segment/key '{segment}' from {original_resource_url}")

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'}
        response = requests.get(original_resource_url, stream=True, timeout=15, headers=headers, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', 'application/octet-stream') # ডিফল্ট বাইনারি স্ট্রিম
        # সাধারণত সেগমেন্টের জন্য 'video/MP2T' বা কী ফাইলের জন্য ভিন্ন কিছু হতে পারে

        # ডেটা স্ট্রিম করুন
        return Response(response.iter_content(chunk_size=1024*1024), # 1MB চাঙ্ক
                        content_type=content_type,
                        status=response.status_code)

    except requests.exceptions.Timeout:
        print(f"Timeout fetching {segment} for channel '{channel_id}' from {original_resource_url}")
        abort(504, description=f"Timeout fetching resource: {segment}")
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 502
        print(f"Error fetching {segment} for channel '{channel_id}' from {original_resource_url}. Status: {status_code}. Error: {e}")
        abort(status_code if status_code in [403, 404] else 502, description=f"Failed to fetch resource '{segment}' for channel '{channel_id}'. Origin error: {status_code}")
    except Exception as e:
        print(f"Unexpected error serving {segment} for channel '{channel_id}': {e}")
        import traceback
        traceback.print_exc()
        abort(500, description="Internal server error fetching resource.")


# --- রুট পেজ (অপশনাল) ---
@app.route('/')
def index():
    """হোমপেজে উপলব্ধ চ্যানেলগুলির একটি তালিকা দেখায়।"""
    # আইডি অনুযায়ী সর্ট করে দেখানো ভালো
    sorted_channel_ids = sorted(channels_data.keys())
    available_channels = {
        cid: url_for('serve_m3u8', channel_id=cid, _external=True)
        for cid in sorted_channel_ids
    }
    return render_template('index.html', available_channels=available_channels)


if __name__ == '__main__':
    # প্রোডাকশনের জন্য Waitress বা Gunicorn ব্যবহার করুন:
    # waitress-serve --host=0.0.0.0 --port=5000 app:app
    # gunicorn -w 4 -b 0.0.0.0:5000 app:app
    # ডেভেলপমেন্টের জন্য:
    # debug=True দিলে কোড পরিবর্তন করলে অটো রিলোড হবে, কিন্তু প্রোডাকশনে বন্ধ রাখুন
    app.run(host='0.0.0.0', port=5000, debug=False)
