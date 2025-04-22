import os
import requests
from flask import Flask, Response, abort, request, render_template, redirect, url_for, flash, session
from urllib.parse import urljoin, urlparse

app = Flask(__name__)

# একটি সিক্রেট কী সেট করুন সেশন ম্যানেজমেন্টের জন্য (ফ্ল্যাশ মেসেজ)
# প্রোডাকশনের জন্য এটি একটি শক্তিশালী এবং গোপন কী হওয়া উচিত, এনভায়রনমেন্ট ভেরিয়েবল থেকে লোড করা ভালো
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_very_secret_key_for_development_only')

# অ্যাডমিন পাসওয়ার্ড (প্রোডাকশনের জন্য এনভায়রনমেন্ট ভেরিয়েবল বা কনফিগ ফাইল ব্যবহার করুন)
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'password123') # এটি পরিবর্তন করুন!

# যে ফাইলে স্ট্রিমের URL সংরক্ষণ করা হবে
URL_STORAGE_FILE = "stream_url.txt"

# --- Helper Functions ---
def load_stream_url():
    """ফাইল থেকে স্ট্রিমের URL লোড করে।"""
    try:
        if os.path.exists(URL_STORAGE_FILE):
            with open(URL_STORAGE_FILE, 'r') as f:
                url = f.read().strip()
                return url if url else None
    except Exception as e:
        print(f"Error loading URL from file: {e}")
    return None

def save_stream_url(url):
    """স্ট্রিমের URL ফাইলে সংরক্ষণ করে।"""
    try:
        with open(URL_STORAGE_FILE, 'w') as f:
            f.write(url)
        return True
    except Exception as e:
        print(f"Error saving URL to file: {e}")
        return False

# --- Global State ---
# অ্যাপ্লিকেশন শুরু হওয়ার সময় URL লোড করুন
current_stream_url = load_stream_url()
current_base_url = None
if current_stream_url:
    try:
        current_base_url = urljoin(current_stream_url, '.') # বেস ইউআরএল ক্যালকুলেট করুন
    except ValueError:
        print(f"Warning: Loaded URL '{current_stream_url}' seems invalid.")
        current_stream_url = None # অবৈধ হলে রিসেট করুন

# --- Admin Routes ---
@app.route('/admin')
def admin_panel():
    """অ্যাডমিন পেজ রেন্ডার করে।"""
    global current_stream_url
    # লগইন করা না থাকলে লগইন পেজ দেখান
    if not session.get('logged_in'):
        return render_template('admin.html')

    # লগইন করা থাকলে URL আপডেট ফর্ম এবং বর্তমান URL দেখান
    return render_template('admin.html', current_url=current_stream_url)

@app.route('/admin/login', methods=['POST'])
def admin_login():
    """অ্যাডমিন লগইন হ্যান্ডেল করে।"""
    password_attempt = request.form.get('password')
    if password_attempt == ADMIN_PASSWORD:
        session['logged_in'] = True
        flash('Login successful!', 'success')
        return redirect(url_for('admin_panel'))
    else:
        flash('Incorrect password.', 'error')
        return redirect(url_for('admin_panel'))

@app.route('/admin/logout', methods=['POST'])
def admin_logout():
    """অ্যাডমিন লগআউট হ্যান্ডেল করে।"""
    session.pop('logged_in', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('admin_panel'))

@app.route('/admin/update', methods=['POST'])
def admin_update():
    """অ্যাডমিন দ্বারা স্ট্রিম URL আপডেট হ্যান্ডেল করে।"""
    global current_stream_url, current_base_url

    if not session.get('logged_in'):
        flash('You must be logged in to perform this action.', 'error')
        return redirect(url_for('admin_panel'))

    new_url = request.form.get('stream_url', '').strip()

    if not new_url:
        flash('Stream URL cannot be empty.', 'error')
        return redirect(url_for('admin_panel'))

    # সাধারণ URL ভ্যালিডেশন (খুব বেসিক)
    try:
        parsed_url = urlparse(new_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise ValueError("Invalid URL format")
        # নতুন বেস ইউআরএল ক্যালকুলেট করুন
        new_base_url = urljoin(new_url, '.')
    except ValueError as e:
        flash(f'Invalid URL format: {e}. Please enter a valid M3U8 URL (e.g., http://.../playlist.m3u8)', 'error')
        return redirect(url_for('admin_panel'))

    if save_stream_url(new_url):
        current_stream_url = new_url
        current_base_url = new_base_url
        flash(f'Stream URL updated successfully to: {new_url}', 'success')
    else:
        flash('Failed to save the new stream URL. Check server logs.', 'error')

    return redirect(url_for('admin_panel'))


# --- Stream Proxy Routes ---
PROXY_PREFIX = "/stream" # সেগমেন্ট ফাইলের জন্য পাথ প্রিফিক্স

@app.route('/jamuna.m3u8')
def serve_m3u8():
    """ক্লায়েন্টের কাছে পরিবর্তিত M3U8 ফাইল পরিবেশন করে।"""
    global current_stream_url, current_base_url

    if not current_stream_url:
        # return "Stream URL is not configured by the admin yet.", 503 # 503 Service Unavailable
        abort(503, description="Stream URL is not configured by the admin yet.")


    try:
        # Use a reasonable timeout
        response = requests.get(current_stream_url, timeout=10)
        response.raise_for_status() # Check for HTTP errors (like 404, 500)

        original_m3u8_content = response.text
        modified_lines = []

        for line in original_m3u8_content.splitlines():
            line = line.strip()
            if not line:
                continue
            # If the line is a URI (not starting with #)
            if not line.startswith('#'):
                # Make the URL relative to our proxy segment handler
                # Handle both relative paths and absolute URLs in the original M3U8
                if line.startswith('http://') or line.startswith('https://'):
                    # If it's an absolute URL, extract the path part maybe?
                    # Or maybe just proxy it as is? Proxying seems safer.
                    # For simplicity, let's assume segments are relative or just filenames
                    # If absolute URLs for segments are needed, this needs more logic.
                    # Here we assume relative path or filename
                    segment_path = line.split('/')[-1] # Get filename/last part
                    modified_line = f"{PROXY_PREFIX}/{segment_path}"
                else:
                     # Assuming it's a relative path or just filename
                    modified_line = f"{PROXY_PREFIX}/{line}"
                modified_lines.append(modified_line)
            else:
                # Keep comment or tag lines as they are
                modified_lines.append(line)

        modified_m3u8_content = "\n".join(modified_lines)

        # Send the response with the correct content type
        return Response(modified_m3u8_content, mimetype='application/vnd.apple.mpegurl')

    except requests.exceptions.Timeout:
        abort(504, description=f"Timeout while fetching the original stream manifest from {current_stream_url}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching original M3U8 ({current_stream_url}): {e}")
        # Maybe notify admin or retry? For now, return error.
        abort(502, description=f"Could not fetch the original stream manifest. Error: {e}") # 502 Bad Gateway
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        abort(500, description="An internal error occurred while processing the manifest.")


@app.route(f'{PROXY_PREFIX}/<path:segment>')
def serve_segment(segment):
    """আসল সার্ভার থেকে সেগমেন্ট ফাইল এনে ক্লায়েন্টকে পরিবেশন করে।"""
    global current_base_url

    if not current_base_url:
        abort(503, description="Stream URL is not configured, cannot determine base URL for segments.")

    # Construct the original segment URL using the dynamically determined base URL
    original_segment_url = urljoin(current_base_url, segment)
    print(f"Attempting to fetch segment: {original_segment_url}") # Logging for debug

    try:
        # Use stream=True to stream the data efficiently
        response = requests.get(original_segment_url, stream=True, timeout=10) # Timeout for segment fetch
        response.raise_for_status() # Check for errors (404 etc.)

        # Stream the content back to the client
        # iter_content() allows sending data in chunks
        return Response(response.iter_content(chunk_size=1024*1024), # 1MB chunks
                        content_type=response.headers.get('Content-Type', 'video/MP2T')) # Use original content type or default

    except requests.exceptions.Timeout:
         print(f"Timeout fetching segment {segment} from {original_segment_url}")
         abort(504, description=f"Timeout while fetching segment: {segment}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching segment {segment} from {original_segment_url}: {e}")
        if e.response is not None:
             print(f"Original server response status: {e.response.status_code}")
             # Pass through the original error code if possible?
             if e.response.status_code == 404:
                 abort(404, description=f"Segment not found on original server: {segment}")
             else:
                 abort(502, description=f"Failed to fetch segment {segment} from origin. Status: {e.response.status_code}")
        else:
            abort(502, description=f"Failed to fetch segment {segment} from origin. Error: {e}") # 502 Bad Gateway
    except Exception as e:
        print(f"An unexpected error occurred while serving segment {segment}: {e}")
        abort(500, description="An internal error occurred while fetching the segment.")

if __name__ == '__main__':
    # Run on 0.0.0.0 to be accessible within Docker, specify a port
    # Use port 5000 or change as needed
    app.run(host='0.0.0.0', port=5000)                                          
