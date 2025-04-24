# proxy_server.py (পরিবর্তিত অংশ)
import os
import requests # requests লাইব্রেরি যোগ করুন
import json
import html
from flask import Flask, Response, abort, request, render_template, redirect, url_for, flash, session
from urllib.parse import urljoin, urlparse

app = Flask(__name__)

app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_key_for_multichannel')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'password123')

# ডাটাবেস সার্ভারের URL কনফিগার করুন (এনভায়রনমেন্ট ভেরিয়েবল থেকে নেওয়া ভালো)
DATABASE_API_URL = os.environ.get('DATABASE_API_URL', 'http://localhost:5001/api/channels') # ডাটাবেস সার্ভারের সঠিক IP/ডোমেইন ও পোর্ট দিন

# --- রিমোট ডাটাবেস সার্ভারের সাথে যোগাযোগের ফাংশন ---

def get_remote_channels():
    """ডাটাবেস সার্ভার থেকে সব চ্যানেল ডেটা আনে"""
    try:
        response = requests.get(DATABASE_API_URL, timeout=10)
        response.raise_for_status() # HTTP Error থাকলে Exception raise করবে
        channels_list = response.json()
        # API থেকে পাওয়া লিস্টকে ডিকশনারিতে রূপান্তর করুন যা টেমপ্লেট আশা করে
        channels_dict = {ch['channel_id']: ch for ch in channels_list}
        return channels_dict
    except requests.exceptions.RequestException as e:
        print(f"Error fetching channels from remote API: {e}")
        flash(f"Error connecting to database server: {e}", "error")
        return {} # ব্যর্থ হলে খালি ডিকশনারি ফেরত দিন
    except json.JSONDecodeError:
        print("Error decoding JSON response from database server.")
        flash("Received invalid data from database server.", "error")
        return {}

def get_remote_channel(channel_id):
    """ডাটাবেস সার্ভার থেকে নির্দিষ্ট চ্যানেল ডেটা আনে"""
    try:
        url = f"{DATABASE_API_URL}/{channel_id.lower()}"
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            return None # পাওয়া না গেলে None ফেরত দিন
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching channel '{channel_id}' from remote API: {e}")
        # এখানে flash মেসেজ দেখানো থেকে বিরত থাকুন কারণ এটি প্রায়শই ব্যাকগ্রাউন্ডে কল হতে পারে
        return None # ব্যর্থ হলে None ফেরত দিন
    except json.JSONDecodeError:
         print(f"Error decoding JSON for channel '{channel_id}'.")
         return None

def save_channel_remote(channel_id, source_url):
    """ডাটাবেস সার্ভারে চ্যানেল যোগ বা আপডেট করে"""
    payload = {
        "channel_id": channel_id.strip().lower(),
        "source_url": html.unescape(source_url.strip())
    }
    try:
        response = requests.post(DATABASE_API_URL, json=payload, timeout=10)
        response.raise_for_status()
        # 201 বা 200 স্ট্যাটাস কোড সফল নির্দেশ করে
        if response.status_code in [200, 201]:
            print(f"Channel '{payload['channel_id']}' saved successfully via API.")
            return True, response.json().get("message", "Success")
        else:
            # অপ্রত্যাশিত সফল স্ট্যাটাস কোড
            print(f"Unexpected success status code {response.status_code} from API during save.")
            return False, f"Unexpected API response: {response.status_code}"
    except requests.exceptions.RequestException as e:
        error_message = f"Error saving channel '{payload['channel_id']}' via API: {e}"
        # API থেকে আসা এরর মেসেজ দেখানোর চেষ্টা করুন
        try:
            api_error = e.response.json().get("error", str(e))
            error_message += f" (API Error: {api_error})"
        except:
            pass # যদি JSON না থাকে বা অন্য সমস্যা হয়
        print(error_message)
        return False, error_message
    except json.JSONDecodeError:
         print("Error decoding JSON response during save.")
         return False, "Received invalid data from database server during save."

def delete_channel_remote(channel_id):
    """ডাটাবেস সার্ভার থেকে চ্যানেল মুছে ফেলে"""
    channel_id = channel_id.lower()
    try:
        url = f"{DATABASE_API_URL}/{channel_id}"
        response = requests.delete(url, timeout=10)
        if response.status_code == 404:
            print(f"Channel '{channel_id}' not found on remote server for deletion.")
            return False, "Channel not found on database server."
        response.raise_for_status() # অন্যান্য HTTP Error চেক করুন
        print(f"Channel '{channel_id}' deleted successfully via API.")
        return True, response.json().get("message", "Deleted successfully")
    except requests.exceptions.RequestException as e:
        error_message = f"Error deleting channel '{channel_id}' via API: {e}"
        try:
             api_error = e.response.json().get("error", str(e))
             error_message += f" (API Error: {api_error})"
        except:
            pass
        print(error_message)
        return False, error_message
    except json.JSONDecodeError:
         print("Error decoding JSON response during delete.")
         return False, "Received invalid data from database server during delete."


# --- Flask রুটগুলির পরিবর্তন ---

# load_channels() এবং save_channels() ফাংশন দুটি মুছে ফেলুন
# global channels_data ভেরিয়েবলটি আর প্রয়োজন নেই

@app.route('/admin', methods=['GET'])
def admin_panel():
    if not session.get('logged_in'):
        return render_template('admin_multi.html', channels=None)
    # এখন API থেকে চ্যানেল লোড করুন
    current_channels = get_remote_channels()
    return render_template('admin_multi.html', channels=current_channels) # টেমপ্লেট একই থাকবে

# login/logout অপরিবর্তিত থাকবে...
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
    # global channels_data সরিয়ে ফেলুন
    if not session.get('logged_in'):
        abort(403)

    channel_id = request.form.get('channel_id', '').strip().lower()
    source_url_raw = request.form.get('source_url', '').strip()
    # action = request.form.get('action', 'add') # এই অ্যাকশনটি আর খুব জরুরি নয় কারণ API upsert করে

    if not channel_id or not source_url_raw:
        flash('Channel ID and Source URL cannot be empty.', 'error')
        return redirect(url_for('admin_panel'))

    if not channel_id.isalnum() or ' ' in channel_id:
         flash('Channel ID should only contain letters and numbers (no spaces).', 'error')
         return redirect(url_for('admin_panel'))

    # URL validation এখন API সার্ভারে করা হচ্ছে, তবে এখানে বেসিক চেক রাখা যেতে পারে
    try:
        parsed_url = urlparse(html.unescape(source_url_raw))
        if not all([parsed_url.scheme, parsed_url.netloc]):
             raise ValueError("Invalid URL format")
    except ValueError:
         flash('Invalid Source URL format. Please provide a full URL (e.g., http://...).', 'error')
         return redirect(url_for('admin_panel'))

    # রিমোট API তে সেভ করার চেষ্টা করুন
    success, message = save_channel_remote(channel_id, source_url_raw)

    if success:
        flash(f'Channel "{channel_id}" processed successfully. API Message: {message}', 'success')
    else:
        flash(f'Failed to save channel "{channel_id}". Error: {message}', 'error')

    return redirect(url_for('admin_panel'))


@app.route('/admin/channels/delete/<channel_id>', methods=['POST'])
def delete_channel(channel_id):
    # global channels_data সরিয়ে ফেলুন
    if not session.get('logged_in'):
        abort(403)

    channel_id_lower = channel_id.lower() # নিশ্চিত করুন আইডি ছোট হাতের

    # রিমোট API ব্যবহার করে ডিলিট করুন
    success, message = delete_channel_remote(channel_id_lower)

    if success:
        flash(f'Channel "{channel_id_lower}" delete request sent. API Message: {message}', 'success')
    else:
        flash(f'Failed to delete channel "{channel_id_lower}". Error: {message}', 'error')

    return redirect(url_for('admin_panel'))


@app.route('/live/<channel_id>.m3u8')
def serve_m3u8(channel_id):
    channel_id = channel_id.lower()
    # API থেকে নির্দিষ্ট চ্যানেল ডেটা আনুন
    channel_info = get_remote_channel(channel_id)

    if channel_info is None:
        # API কল ব্যর্থ হলে বা চ্যানেল না থাকলে 404 দিন
        # get_remote_channel ব্যর্থ হলে None ফেরত দেয়
        abort(404, description=f"Channel '{channel_id}' not found or database connection failed.")

    source_url = channel_info.get('source_url')
    base_url = channel_info.get('base_url') # API থেকে base_url আসবে

    if not source_url or not base_url: # base_url ও এখন জরুরি
         print(f"Configuration error from API: Source or Base URL missing for channel '{channel_id}'.")
         abort(500, description=f"Configuration error for channel '{channel_id}'.")

    print(f"Fetching M3U8 for channel '{channel_id}' from: {source_url}")

    # বাকি অংশ অপরিবর্তিত থাকবে... (requests.get, কন্টেন্ট প্রসেসিং ইত্যাদি)
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(source_url, timeout=15, headers=headers, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '').lower()
        if 'mpegurl' not in content_type and 'x-mpegurl' not in content_type:
             print(f"Warning: Content-Type for channel '{channel_id}' is '{content_type}', not M3U8. Trying to process anyway.")

        original_m3u8_content = response.text
        modified_lines = []

        # Use the base_url provided by the API
        effective_base_url = base_url

        for line in original_m3u8_content.splitlines():
            line = line.strip()
            if not line:
                continue

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
                    else:
                         modified_lines.append(line)
                else:
                     modified_lines.append(line)

            elif not line.startswith('#'):
                segment_uri_part = line
                parsed_segment_uri = urlparse(segment_uri_part)

                if parsed_segment_uri.scheme:
                    modified_lines.append(line) # Absolute URL, keep as is
                elif segment_uri_part.lower().endswith('.m3u8'):
                     if effective_base_url:
                         absolute_sublist_url = urljoin(effective_base_url, segment_uri_part)
                         modified_lines.append(absolute_sublist_url)
                         print(f"Rewriting nested playlist URI: '{segment_uri_part}' -> '{absolute_sublist_url}' (Absolute)")
                     else:
                        # This case should ideally not happen if API always provides base_url
                        modified_lines.append(line)
                        print(f"Warning: Keeping relative nested playlist URI (no effective base_url): {line}")

                else: # Segment (.ts file or similar)
                    # Proxy the segment through this server
                    # Use request.host_url as the base for the proxy URL
                    proxy_segment_url = url_for('serve_segment', channel_id=channel_id, segment=segment_uri_part, _external=False)
                    modified_lines.append(proxy_segment_url)

            else: # Other M3U8 tags
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


@app.route('/stream/<channel_id>/<path:segment>') # path কনভার্টার ব্যবহার করুন
def serve_segment(channel_id, segment):
    channel_id = channel_id.lower()
    # API থেকে চ্যানেল ডেটা আনুন শুধু base_url পাওয়ার জন্য
    channel_info = get_remote_channel(channel_id)

    if channel_info is None:
        abort(404, description=f"Channel '{channel_id}' not found for segment request or DB connection failed.")

    base_url = channel_info.get('base_url')

    if not base_url:
        print(f"Error: Base URL not found in API response for channel '{channel_id}'")
        abort(500, description=f"Configuration error from API: Base URL missing for channel '{channel_id}'.")

    try:
        # API থেকে পাওয়া base_url ব্যবহার করে সেগমেন্ট URL তৈরি করুন
        original_segment_url = urljoin(base_url, segment)
    except ValueError as e:
        print(f"Error creating segment URL for channel '{channel_id}': base='{base_url}', segment='{segment}'. Error: {e}")
        abort(500, description="Internal error creating segment URL.")

    # বাকি অংশ অপরিবর্তিত থাকবে... (requests.get stream, generate_stream ইত্যাদি)
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        # Referer যোগ করা অনেক সময় সাহায্য করে
        headers['Referer'] = base_url
        response = requests.get(original_segment_url, stream=True, timeout=20, headers=headers, allow_redirects=True) # টাইমআউট কিছুটা বাড়ানো যেতে পারে
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', 'video/MP2T') # ডিফল্ট কন্টেন্ট টাইপ

        def generate_stream():
            try:
                for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunk size
                    if chunk: # ফিল্টার আউট keep-alive নতুনチャンক
                        yield chunk
            except requests.exceptions.ChunkedEncodingError:
                 print(f"ChunkedEncodingError while streaming segment for channel {channel_id}")
            except Exception as stream_err:
                 print(f"Error during segment streaming for {channel_id}: {stream_err}")
            finally:
                response.close() # Response object বন্ধ করা নিশ্চিত করুন

        # Headers ফরওয়ার্ড করা (ঐচ্ছিক কিন্তু সহায়ক হতে পারে)
        resp_headers = {}
        if 'Content-Length' in response.headers:
            resp_headers['Content-Length'] = response.headers['Content-Length']
        # resp_headers['Cache-Control'] = 'no-cache' # ক্যাশিং বন্ধ করতে চাইলে

        return Response(generate_stream(),
                        content_type=content_type,
                        status=response.status_code,
                        headers=resp_headers)

    except requests.exceptions.Timeout:
        print(f"Timeout fetching segment '{segment}' for channel '{channel_id}' from {original_segment_url}")
        abort(504, description=f"Timeout fetching segment: {segment}")
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 502
        print(f"Error fetching segment '{segment}' for channel '{channel_id}' from {original_segment_url}. Status: {status_code}. Error: {e}")
        abort(status_code if status_code in [403, 404] else 502, description=f"Failed to fetch segment '{segment}' for channel '{channel_id}'. Origin error: {status_code}")
    except Exception as e:
        print(f"Unexpected error serving segment '{segment}' for channel '{channel_id}': {e}")
        abort(500, description="Internal server error fetching segment.")


@app.route('/')
def index():
    # API থেকে চ্যানেল লোড করুন
    current_channels_data = get_remote_channels() # এটি এখন ডিকশনারি ফরম্যাটে ডেটা দেবে
    available_channels = {
        cid: url_for('serve_m3u8', channel_id=cid, _external=True)
        for cid in current_channels_data.keys()
    }
    # টেমপ্লেট আগের মতই থাকবে
    return render_template('index.html', available_channels=available_channels)


if __name__ == '__main__':
    print("Starting IPTV Proxy Server (using remote database API)...")
    # Waitress ব্যবহার বা Flask ডেভেলপমেন্ট সার্ভার অপরিবর্তিত থাকবে
    try:
        from waitress import serve
        print(f"Running with Waitress WSGI server on http://0.0.0.0:5000")
        print(f"Connecting to Database API at: {DATABASE_API_URL}")
        serve(app, host='0.0.0.0', port=5000, threads=8)
    except ImportError:
        print("Waitress not found, using Flask's development server (NOT recommended for production).")
        print(f"Connecting to Database API at: {DATABASE_API_URL}")
        app.run(host='0.0.0.0', port=5000, debug=False)                              
