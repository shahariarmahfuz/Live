import os
import requests # API কলের জন্য
import json
import html
from flask import (
    Flask, Response, abort, request, render_template,
    redirect, url_for, flash, session
)
from urllib.parse import urljoin, urlparse
from collections import defaultdict # ক্যাটাগরি অনুযায়ী গ্রুপ করার জন্য

# --- Flask অ্যাপ ইনিশিয়ালাইজেশন ---
app = Flask(__name__)

# --- কনফিগারেশন ---
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_key_for_multichannel_proxy')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'password123') # আপনার এডমিন পাসওয়ার্ড এখানে দিন বা এনভায়রনমেন্ট ভেরিয়েবল সেট করুন

# গুরুত্বপূর্ণ: ডাটাবেস সার্ভারের সঠিক URL এখানে দিন বা এনভায়রনমেন্ট ভেরিয়েবল সেট করুন
DATABASE_API_URL = os.environ.get('DATABASE_API_URL', 'https://itachi321.pythonanywhere.com/api/channels')

# --- ক্যাটাগরি লিস্ট ---
# এই লিস্টটি অ্যাডমিন প্যানেলে ড্রপডাউন তৈরি করতে ব্যবহৃত হয়
CATEGORIES = [
    "News", "Entertainment", "Movies", "Sports", "Music", "Kids",
    "Documentary", "Lifestyle", "Religious", "Educational", "Regional (BD)",
    "Regional (IN)", "International", "Drama", "Comedy", "Action", "Sci-Fi",
    "Cooking", "Travel", "Music Video", "Web Series", "General", "Other"
]
CATEGORIES.sort() # বর্ণানুক্রমে সাজানো
VALID_CATEGORIES_SET = set(CATEGORIES) # দ্রুত Validation এর জন্য Set

# --- রিমোট ডাটাবেস সার্ভারের সাথে যোগাযোগের হেল্পার ফাংশন ---

def get_remote_channels():
    """ডাটাবেস সার্ভার থেকে সব চ্যানেল ডেটা (নাম, ক্যাটাগরি সহ) আনে"""
    try:
        response = requests.get(DATABASE_API_URL, timeout=10)
        response.raise_for_status() # HTTP Error থাকলে Exception raise করবে
        channels_list = response.json()
        # API থেকে পাওয়া লিস্টকে ডিকশনারিতে রূপান্তর করুন যা টেমপ্লেট আশা করে
        # Key: channel_id, Value: channel_info dictionary
        channels_dict = {ch['channel_id']: ch for ch in channels_list}
        return channels_dict
    except requests.exceptions.Timeout:
        error_msg = "Error: Timeout connecting to the database server."
        print(error_msg)
        flash(error_msg, "error")
        return {}
    except requests.exceptions.RequestException as e:
        error_msg = f"Error connecting to database server: {e}"
        print(error_msg)
        flash(error_msg, "error")
        return {} # ব্যর্থ হলে খালি ডিকশনারি ফেরত দিন
    except json.JSONDecodeError:
        error_msg = "Error: Received invalid data format from database server."
        print(error_msg)
        flash(error_msg, "error")
        return {}

def get_remote_channel(channel_id):
    """ডাটাবেস সার্ভার থেকে নির্দিষ্ট চ্যানেল ডেটা (নাম, ক্যাটাগরি সহ) আনে"""
    channel_id_lower = channel_id.lower() # সর্বদা ছোট হাতের আইডি ব্যবহার করুন
    try:
        url = f"{DATABASE_API_URL}/{channel_id_lower}"
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            # চ্যানেল পাওয়া না গেলে None ফেরত দিন, যা রুট হ্যান্ডলার চেক করবে
            return None
        response.raise_for_status() # অন্যান্য HTTP Error হ্যান্ডেল করুন
        return response.json() # চ্যানেল ইনফো ডিকশনারি রিটার্ন করবে
    except requests.exceptions.Timeout:
        print(f"Error: Timeout fetching channel '{channel_id_lower}' from remote API.")
        # এখানে flash মেসেজ দেখানো উচিত নয় কারণ এটি প্রায়শই ব্যাকগ্রাউন্ডে কল হতে পারে
        return None # ব্যর্থ হলে None ফেরত দিন
    except requests.exceptions.RequestException as e:
        print(f"Error fetching channel '{channel_id_lower}' from remote API: {e}")
        return None
    except json.JSONDecodeError:
         print(f"Error decoding JSON for channel '{channel_id_lower}'.")
         return None

def save_channel_remote(channel_id, channel_name, category, source_url):
    """ডাটাবেস সার্ভারে চ্যানেল যোগ বা আপডেট করে (নাম ও ক্যাটাগরি সহ)"""
    channel_id_lower = channel_id.strip().lower()
    channel_name_cleaned = channel_name.strip()
    source_url_unescaped = html.unescape(source_url.strip())

    # Payload তৈরির আগে ভ্যালিডেশন
    if not all([channel_id_lower, channel_name_cleaned, category, source_url_unescaped]):
         return False, "Missing required channel information."
    if category not in VALID_CATEGORIES_SET:
         return False, f"Invalid category selected: {category}"

    payload = {
        "channel_id": channel_id_lower,
        "channel_name": channel_name_cleaned,
        "category": category,
        "source_url": source_url_unescaped
    }

    try:
        response = requests.post(DATABASE_API_URL, json=payload, timeout=15) # টাইমআউট কিছুটা বাড়ানো যেতে পারে
        response.raise_for_status() # HTTP 4xx/5xx এর জন্য Exception raise করবে

        # সফল স্ট্যাটাস কোড (200 OK বা 201 Created)
        if response.status_code in [200, 201]:
            success_msg = response.json().get("message", f"Channel '{channel_id_lower}' saved successfully.")
            print(success_msg)
            return True, success_msg
        else:
            # অপ্রত্যাশিত সফল স্ট্যাটাস কোড (যদিও raise_for_status এটি হ্যান্ডেল করার কথা)
            error_msg = f"Unexpected success status code {response.status_code} from API during save."
            print(error_msg)
            return False, error_msg

    except requests.exceptions.Timeout:
         error_msg = f"Error: Timeout saving channel '{channel_id_lower}' via API."
         print(error_msg)
         return False, error_msg
    except requests.exceptions.RequestException as e:
        # API থেকে আসা এরর মেসেজ দেখানোর চেষ্টা করুন
        api_error = str(e)
        try:
            # চেষ্টা করুন response থেকে JSON error message পেতে
            if e.response is not None:
                api_error = e.response.json().get("error", e.response.text)
        except (json.JSONDecodeError, AttributeError):
             # যদি JSON না থাকে বা অন্য সমস্যা হয়, আগের error ব্যবহার করুন
             pass
        error_message = f"Error saving channel '{channel_id_lower}' via API. Status: {e.response.status_code if e.response is not None else 'N/A'}. Message: {api_error}"
        print(error_message)
        return False, error_message
    except Exception as e: # অন্যান্য অপ্রত্যাশিত সমস্যা
         error_msg = f"An unexpected error occurred while saving channel '{channel_id_lower}': {e}"
         print(error_msg)
         return False, error_msg


def delete_channel_remote(channel_id):
    """ডাটাবেস সার্ভার থেকে চ্যানেল মুছে ফেলে"""
    channel_id_lower = channel_id.lower()
    try:
        url = f"{DATABASE_API_URL}/{channel_id_lower}"
        response = requests.delete(url, timeout=10)

        if response.status_code == 404:
            # সার্ভারে চ্যানেল পাওয়া যায়নি
            error_msg = f"Channel '{channel_id_lower}' not found on database server."
            print(error_msg)
            return False, error_msg

        response.raise_for_status() # অন্যান্য HTTP Error (4xx, 5xx) চেক করুন

        # সফলভাবে ডিলিট হলে (সাধারণত 200 OK বা 204 No Content)
        success_msg = response.json().get("message", f"Channel '{channel_id_lower}' deleted successfully.")
        print(success_msg)
        return True, success_msg

    except requests.exceptions.Timeout:
        error_msg = f"Error: Timeout deleting channel '{channel_id_lower}' via API."
        print(error_msg)
        return False, error_msg
    except requests.exceptions.RequestException as e:
        # API থেকে আসা এরর মেসেজ দেখানোর চেষ্টা করুন
        api_error = str(e)
        try:
             if e.response is not None:
                api_error = e.response.json().get("error", e.response.text)
        except (json.JSONDecodeError, AttributeError):
             pass
        error_message = f"Error deleting channel '{channel_id_lower}' via API. Status: {e.response.status_code if e.response is not None else 'N/A'}. Message: {api_error}"
        print(error_message)
        return False, error_message
    except Exception as e:
         error_msg = f"An unexpected error occurred while deleting channel '{channel_id_lower}': {e}"
         print(error_msg)
         return False, error_msg

# --- Flask Routes ---

@app.route('/admin', methods=['GET'])
def admin_panel():
    """অ্যাডমিন প্যানেল প্রদর্শন করে"""
    if not session.get('logged_in'):
        # লগইন করা না থাকলে শুধু লগইন ফর্ম দেখান
        return render_template('admin_multi.html', channels=None, categories=None)

    # লগইন করা থাকলে চ্যানেল ও ক্যাটাগরি লোড করুন
    current_channels = get_remote_channels() # API থেকে চ্যানেল ডেটা পান
    # টেমপ্লেটে ক্যাটাগরি লিস্ট পাঠান ড্রপডাউনের জন্য
    return render_template('admin_multi.html', channels=current_channels, categories=CATEGORIES)

@app.route('/admin/login', methods=['POST'])
def admin_login():
    """অ্যাডমিন লগইন হ্যান্ডেল করে"""
    password_attempt = request.form.get('password')
    if password_attempt == ADMIN_PASSWORD:
        session['logged_in'] = True
        flash('সফলভাবে লগইন করেছেন!', 'success') # বাংলা বার্তা
    else:
        session.pop('logged_in', None) # ভুল পাসওয়ার্ড দিলে সেশন সরিয়ে ফেলুন
        flash('ভুল পাসওয়ার্ড।', 'error') # বাংলা বার্তা
    return redirect(url_for('admin_panel'))

@app.route('/admin/logout', methods=['POST'])
def admin_logout():
    """অ্যাডমিন লগআউট হ্যান্ডেল করে"""
    session.pop('logged_in', None)
    flash('আপনি সফলভাবে লগআউট হয়েছেন।', 'success') # বাংলা বার্তা
    return redirect(url_for('admin_panel'))

@app.route('/admin/channels', methods=['POST'])
def add_or_update_channel():
    """অ্যাডমিন প্যানেল থেকে চ্যানেল যোগ বা আপডেট করার অনুরোধ হ্যান্ডেল করে"""
    if not session.get('logged_in'):
        abort(403) # Forbidden access

    # ফর্ম থেকে ডেটা সংগ্রহ করুন
    channel_id = request.form.get('channel_id', '').strip()
    channel_name = request.form.get('channel_name', '').strip()
    category = request.form.get('category', '') # Select থেকে আসবে, strip দরকার নেই
    source_url_raw = request.form.get('source_url', '').strip()

    # --- ইনপুট ভ্যালিডেশন ---
    errors = []
    if not channel_id: errors.append("চ্যানেল আইডি আবশ্যক।")
    if not channel_name: errors.append("চ্যানেলের নাম আবশ্যক।")
    if not category: errors.append("ক্যাটাগরি নির্বাচন আবশ্যক।")
    if not source_url_raw: errors.append("সোর্স M3U8 URL আবশ্যক।")

    if channel_id and (not channel_id.isalnum() or ' ' in channel_id):
        errors.append("চ্যানেল আইডিতে শুধু অক্ষর এবং সংখ্যা ব্যবহার করা যাবে (স্পেস ছাড়া)।")

    if category and category not in VALID_CATEGORIES_SET:
        errors.append(f'অবৈধ ক্যাটাগরি "{category}" নির্বাচন করা হয়েছে।')

    source_url_unescaped = html.unescape(source_url_raw)
    try:
        parsed_url = urlparse(source_url_unescaped)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            # scheme (http/https) এবং netloc (domain/ip) না থাকলে URL অবৈধ
            if source_url_raw: # শুধু URL দেওয়া থাকলেই এই error দেখান
                 errors.append('সঠিক URL ফরম্যাট প্রয়োজন (যেমন: http://example.com/stream)।')
    except ValueError:
        if source_url_raw:
             errors.append('অবৈধ সোর্স URL ফরম্যাট।')

    # যদি কোনো error থাকে, ফর্ম আবার দেখান
    if errors:
        for error in errors:
            flash(error, 'error')
        # ভ্যালিডেশন ফেইল করলে আগের ইনপুটগুলো সহ ফর্ম দেখানোর জন্য ডেটা পাস করা যেতে পারে,
        # তবে সরলতার জন্য শুধু redirect করা হচ্ছে।
        return redirect(url_for('admin_panel'))

    # --- রিমোট API তে সেভ করার চেষ্টা করুন ---
    success, message = save_channel_remote(channel_id, channel_name, category, source_url_unescaped)

    if success:
        flash(f'চ্যানেল "{channel_name}" ({channel_id}) সফলভাবে সংরক্ষিত হয়েছে।', 'success') # বাংলা বার্তা
    else:
        flash(f'চ্যানেল "{channel_name}" ({channel_id}) সংরক্ষণ ব্যর্থ হয়েছে। ত্রুটি: {message}', 'error') # বাংলা বার্তা

    return redirect(url_for('admin_panel'))

@app.route('/admin/channels/delete/<channel_id>', methods=['POST'])
def delete_channel(channel_id):
    """অ্যাডমিন প্যানেল থেকে চ্যানেল মুছে ফেলার অনুরোধ হ্যান্ডেল করে"""
    if not session.get('logged_in'):
        abort(403)

    channel_id_lower = channel_id.lower() # ছোট হাতের আইডি ব্যবহার করুন

    # রিমোট API ব্যবহার করে ডিলিট করুন
    success, message = delete_channel_remote(channel_id_lower)

    if success:
        flash(f'চ্যানেল আইডি "{channel_id_lower}" সফলভাবে মুছে ফেলা হয়েছে।', 'success') # বাংলা বার্তা
    else:
        flash(f'চ্যানেল আইডি "{channel_id_lower}" মুছতে ব্যর্থ হয়েছে। ত্রুটি: {message}', 'error') # বাংলা বার্তা

    return redirect(url_for('admin_panel'))


@app.route('/live/<channel_id>.m3u8')
def serve_m3u8(channel_id):
    """নির্দিষ্ট চ্যানেলের জন্য প্রক্সি M3U8 প্লেলিস্ট তৈরি এবং পরিবেশন করে"""
    channel_id_lower = channel_id.lower()
    # API থেকে নির্দিষ্ট চ্যানেল ডেটা আনুন
    channel_info = get_remote_channel(channel_id_lower)

    # চ্যানেল পাওয়া না গেলে বা API কানেকশনে সমস্যা হলে 404 দিন
    if channel_info is None:
        print(f"Channel info not found or API error for ID: {channel_id_lower}")
        abort(404, description=f"Channel '{channel_id_lower}' not found or database unavailable.")

    # প্রয়োজনীয় URL গুলো পান
    source_url = channel_info.get('source_url')
    base_url = channel_info.get('base_url') # API থেকে base_url আসবে

    # URL না থাকলে কনফিগারেশন ত্রুটি
    if not source_url or not base_url:
         error_msg = f"Configuration error from API: Source or Base URL missing for channel '{channel_id_lower}'."
         print(error_msg)
         abort(500, description=error_msg)

    channel_name = channel_info.get('channel_name', 'N/A') # লগিংয়ের জন্য নাম ব্যবহার
    print(f"Fetching M3U8 for channel '{channel_id_lower}' ({channel_name}) from: {source_url}")

    try:
        # হেডার সেট করুন (User-Agent গুরুত্বপূর্ণ হতে পারে)
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36'}
        response = requests.get(source_url, timeout=15, headers=headers, allow_redirects=True)
        response.raise_for_status() # 4xx বা 5xx স্ট্যাটাস কোডের জন্য exception

        # কন্টেন্ট টাইপ চেক (ঐচ্ছিক কিন্তু ভালো)
        content_type = response.headers.get('Content-Type', '').lower()
        if 'mpegurl' not in content_type and 'x-mpegurl' not in content_type:
             print(f"Warning: Content-Type for channel '{channel_id_lower}' is '{content_type}', not M3U8. Processing anyway.")

        original_m3u8_content = response.text
        modified_lines = []

        # base_url ব্যবহার করে আপেক্ষিক URL গুলোকে পূর্ণ URL এ রূপান্তর করুন
        # response.url ব্যবহার করা ভালো যদি রিডাইরেক্ট হয়ে থাকে, কিন্তু API থেকে base_url পাওয়াই যথেষ্ট
        effective_base_url = base_url

        # M3U8 কন্টেন্ট লাইন বাই লাইন প্রসেস করুন
        for line in original_m3u8_content.splitlines():
            line = line.strip()
            if not line:
                continue # খালি লাইন বাদ দিন

            # --- কী (Key) URL পুনঃলিখন ---
            if line.startswith('#EXT-X-KEY'):
                if 'URI="' in line:
                    try:
                        parts = line.split('URI="')
                        uri_part_and_rest = parts[1].split('"', 1)
                        uri_part = uri_part_and_rest[0]
                        rest_of_line = uri_part_and_rest[1] if len(uri_part_and_rest) > 1 else ""

                        # যদি URI আপেক্ষিক হয় এবং base_url থাকে
                        if effective_base_url and not urlparse(uri_part).scheme:
                            full_key_uri = urljoin(effective_base_url, uri_part)
                            new_key_line = f'{parts[0]}URI="{full_key_uri}"{rest_of_line}'
                            modified_lines.append(new_key_line)
                            # print(f"Rewritten KEY URI: {uri_part} -> {full_key_uri}")
                        else:
                             # URI যদি পূর্ণ হয় বা base_url না থাকে, তাহলে অপরিবর্তিত রাখুন
                             modified_lines.append(line)
                    except IndexError:
                         # URI ফরম্যাট অপ্রত্যাশিত হলে লাইন অপরিবর্তিত রাখুন
                         modified_lines.append(line)
                         print(f"Warning: Could not parse KEY URI line: {line}")
                else:
                     # URI attribute না থাকলে লাইন অপরিবর্তিত রাখুন
                     modified_lines.append(line)

            # --- সেগমেন্ট বা সাব-প্লেলিস্ট URL পুনঃলিখন ---
            elif not line.startswith('#'):
                segment_uri_part = line
                parsed_segment_uri = urlparse(segment_uri_part)

                # যদি URL টি পূর্ণ হয় (scheme আছে), তাহলে অপরিবর্তিত রাখুন
                if parsed_segment_uri.scheme:
                    modified_lines.append(line)

                # যদি এটি একটি সাব-প্লেলিস্ট (.m3u8) হয়
                elif segment_uri_part.lower().endswith('.m3u8'):
                    if effective_base_url:
                        # আপেক্ষিক হলে পূর্ণ URL তৈরি করুন
                        absolute_sublist_url = urljoin(effective_base_url, segment_uri_part)
                        modified_lines.append(absolute_sublist_url)
                        # print(f"Rewritten nested playlist URI: {segment_uri_part} -> {absolute_sublist_url}")
                    else:
                        # base_url না থাকলে আপেক্ষিক রাখুন (সমস্যা হতে পারে)
                       modified_lines.append(line)
                       print(f"Warning: Keeping relative nested playlist URI (no base_url): {line}")

                # যদি এটি একটি মিডিয়া সেগমেন্ট (.ts, .aac, ইত্যাদি) হয়
                else:
                    # সেগমেন্টটিকে এই প্রক্সি সার্ভারের মাধ্যমে পরিবেশন করুন
                    # segment অংশে '/' থাকতে পারে, তাই path কনভার্টার দরকার হবে রুটে
                    proxy_segment_url = url_for('serve_segment', channel_id=channel_id_lower, segment=segment_uri_part, _external=False)
                    modified_lines.append(proxy_segment_url)

            # --- অন্যান্য M3U8 ট্যাগ ---
            else:
                # অন্যান্য সব লাইন (#EXTM3U, #EXT-X-VERSION, ইত্যাদি) অপরিবর্তিত রাখুন
                modified_lines.append(line)

        # পরিবর্তিত M3U8 কন্টেন্ট তৈরি করুন
        modified_m3u8_content = "\n".join(modified_lines)
        # সঠিক MIME টাইপ সহ Response ফেরত দিন
        return Response(modified_m3u8_content, mimetype='application/vnd.apple.mpegurl')

    # --- Error Handling ---
    except requests.exceptions.Timeout:
        error_msg = f"Timeout fetching manifest for channel '{channel_id_lower}' from {source_url}"
        print(error_msg)
        abort(504, description=error_msg) # 504 Gateway Timeout
    except requests.exceptions.RequestException as e:
        status = e.response.status_code if e.response is not None else 502 # 502 Bad Gateway default
        error_msg = f"Error fetching original M3U8 for channel '{channel_id_lower}' ({source_url}). Status: {status}. Error: {e}"
        print(error_msg)
        # যদি মূল সার্ভার 403 বা 404 দেয়, সেই স্ট্যাটাসই ফেরত দিন, নইলে 502 দিন
        abort(status if status in [403, 404] else 502, description=f"Could not fetch manifest for channel '{channel_id_lower}'. Origin error: {status}")
    except Exception as e:
        # অন্যান্য অপ্রত্যাশিত ত্রুটি
        error_msg = f"Unexpected error processing manifest for channel '{channel_id_lower}': {e}"
        print(error_msg)
        abort(500, description="Internal server error processing manifest.")


# --- মিডিয়া সেগমেন্ট পরিবেশন করার রুট ---
# segment অংশে '/' থাকতে পারে, তাই path কনভার্টার ব্যবহার করুন
@app.route('/stream/<channel_id>/<path:segment>')
def serve_segment(channel_id, segment):
    """মূল উৎস থেকে মিডিয়া সেগমেন্ট এনে ক্লায়েন্টকে স্ট্রিম করে"""
    channel_id_lower = channel_id.lower()

    # সেগমেন্ট URL তৈরি করার জন্য base_url দরকার, তাই চ্যানেল ইনফো আবার আনুন
    # এখানে ক্যাশিং ব্যবহার করা যেতে পারে পারফরম্যান্স বাড়ানোর জন্য
    channel_info = get_remote_channel(channel_id_lower)

    if channel_info is None:
        abort(404, description=f"Channel '{channel_id_lower}' not found for segment request or DB unavailable.")

    base_url = channel_info.get('base_url')
    if not base_url:
        error_msg = f"Configuration error from API: Base URL missing for channel '{channel_id_lower}' when serving segment."
        print(error_msg)
        abort(500, description=error_msg)

    try:
        # আপেক্ষিক সেগমেন্ট পাথ এবং base_url ব্যবহার করে পূর্ণ URL তৈরি করুন
        original_segment_url = urljoin(base_url, segment)
    except ValueError as e:
        error_msg = f"Error creating segment URL for channel '{channel_id_lower}': base='{base_url}', segment='{segment}'. Error: {e}"
        print(error_msg)
        abort(500, description="Internal error creating segment URL.")

    # print(f"Proxying segment: {original_segment_url}") # লগিং (বেশি বিস্তারিত হতে পারে)

    try:
        # মূল উৎস থেকে সেগমেন্ট আনার জন্য রিকোয়েস্ট পাঠান (stream=True গুরুত্বপূর্ণ)
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36'}
        # Referer যোগ করা অনেক সময় প্রয়োজন হয়
        headers['Referer'] = base_url
        seg_response = requests.get(
            original_segment_url,
            stream=True, # ডেটা মেমরিতে লোড না করে স্ট্রিম করুন
            timeout=20,   # সেগমেন্টের জন্য টাইমআউট কিছুটা বেশি হতে পারে
            headers=headers,
            allow_redirects=True
        )
        seg_response.raise_for_status() # 4xx/5xx স্ট্যাটাসের জন্য exception

        # মূল কন্টেন্ট টাইপ ব্যবহার করুন, না পেলে ডিফল্ট দিন
        content_type = seg_response.headers.get('Content-Type', 'video/MP2T')

        # স্ট্রিম জেনারেটর ফাংশন
        def generate_stream():
            try:
                # chunk_size=8192 বা 1MB (1024*1024) ব্যবহার করা যেতে পারে
                for chunk in seg_response.iter_content(chunk_size=1024*1024):
                    if chunk: # ফিল্টার আউট keep-alive নতুনチャンক
                        yield chunk
            except requests.exceptions.ChunkedEncodingError:
                 print(f"ChunkedEncodingError while streaming segment for channel {channel_id_lower}")
            except Exception as stream_err:
                 # স্ট্রিমিং এর সময় কোনো সমস্যা হলে লগ করুন
                 print(f"Error during segment streaming for {channel_id_lower}: {stream_err}")
            finally:
                # নিশ্চিত করুন যে মূল response অবজেক্টটি বন্ধ হয়েছে
                seg_response.close()

        # মূল response থেকে কিছু হেডার ফরোয়ার্ড করা যেতে পারে (যেমন Content-Length)
        resp_headers = {}
        if 'Content-Length' in seg_response.headers:
            resp_headers['Content-Length'] = seg_response.headers['Content-Length']
        # resp_headers['Cache-Control'] = 'no-cache' # ক্লায়েন্ট ক্যাশিং বন্ধ করতে চাইলে

        # জেনারেটর ব্যবহার করে Response তৈরি করুন
        return Response(generate_stream(),
                        content_type=content_type,
                        status=seg_response.status_code, # মূল স্ট্যাটাস কোড ব্যবহার করুন
                        headers=resp_headers)

    # --- Error Handling ---
    except requests.exceptions.Timeout:
        error_msg = f"Timeout fetching segment '{segment}' for channel '{channel_id_lower}' from {original_segment_url}"
        print(error_msg)
        abort(504, description=error_msg) # 504 Gateway Timeout
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 502
        error_msg = f"Error fetching segment '{segment}' for channel '{channel_id_lower}' from {original_segment_url}. Status: {status_code}. Error: {e}"
        print(error_msg)
        # মূল সার্ভারের error স্ট্যাটাস (403, 404) ফরোয়ার্ড করুন, নইলে 502 দিন
        abort(status_code if status_code in [403, 404] else 502, description=f"Failed to fetch segment '{segment}'. Origin error: {status_code}")
    except Exception as e:
        # অন্যান্য অপ্রত্যাশিত ত্রুটি
        error_msg = f"Unexpected error serving segment '{segment}' for channel '{channel_id_lower}': {e}"
        print(error_msg)
        abort(500, description="Internal server error fetching segment.")


# --- প্রধান ইনডেক্স পেজ ---
@app.route('/')
def index():
    """প্রধান পৃষ্ঠা, যেখানে চ্যানেলগুলো ক্যাটাগরি অনুযায়ী গ্রুপ করে দেখানো হয়"""
    all_channels_dict = get_remote_channels() # সব চ্যানেল ডেটা পান

    # ক্যাটাগরি অনুযায়ী চ্যানেল গ্রুপ করার জন্য defaultdict ব্যবহার করুন
    grouped_channels = defaultdict(list)
    for channel_id, info in all_channels_dict.items():
        # যদি কোনো চ্যানেলের ক্যাটাগরি না থাকে (ডাটাবেস ত্রুটি বা পুরনো ডেটা), তাহলে 'Other' ক্যাটাগরিতে দিন
        category = info.get('category', 'Other')
        # প্রতিটি চ্যানেলের জন্য প্রয়োজনীয় তথ্য (আইডি, নাম, প্রক্সি URL) যোগ করুন
        grouped_channels[category].append({
            'id': channel_id,
            # যদি নাম না থাকে, আইডি ব্যবহার করুন
            'name': info.get('channel_name', channel_id),
            'proxy_url': url_for('serve_m3u8', channel_id=channel_id, _external=True) # পূর্ণ URL তৈরি করুন
        })

    # ক্যাটাগরিগুলোকে বর্ণানুক্রমে সাজান (A-Z)
    sorted_grouped_channels = dict(sorted(grouped_channels.items()))

    # প্রতিটি ক্যাটাগরির ভিতরের চ্যানেলগুলোকে নামের ভিত্তিতে বর্ণানুক্রমে সাজান
    for category in sorted_grouped_channels:
        sorted_grouped_channels[category].sort(key=lambda x: x['name'].lower()) # ছোট হাতের অক্ষরে রূপান্তর করে সাজানো ভালো

    # সাজানো ও গ্রুপ করা ডেটা টেমপ্লেটে পাঠান
    return render_template('index.html', grouped_channels=sorted_grouped_channels)


# --- সার্ভার চালু করার কোড ---
if __name__ == '__main__':
    print("Starting IPTV Proxy Server (with Remote DB and Categories)...")
    print(f"Admin Password: {'Set (Hidden)' if ADMIN_PASSWORD else 'Not Set (Using Default)'}")
    print(f"Database API URL: {DATABASE_API_URL}")

    # প্রোডাকশনের জন্য Waitress ব্যবহার করা ভালো
    try:
        from waitress import serve
        print(f"Running with Waitress WSGI server on http://0.0.0.0:5000")
        # থ্রেডের সংখ্যা আপনার সার্ভারের কোর অনুযায়ী সেট করতে পারেন
        serve(app, host='0.0.0.0', port=5000, threads=8)
    except ImportError:
        # Waitress ইনস্টল করা না থাকলে Flask এর ডেভেলপমেন্ট সার্ভার ব্যবহার করুন
        # এটি প্রোডাকশনের জন্য উপযুক্ত নয়!
        print("Waitress not found, using Flask's development server (NOT recommended for production).")
        # debug=False রাখুন প্রোডাকশন বা শেয়ার করার সময়
        app.run(host='0.0.0.0', port=5000, debug=False)
