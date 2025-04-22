import requests
from flask import Flask, Response, abort, request
from urllib.parse import urljoin, urlparse

app = Flask(__name__)

# আপনার আসল বা গোপন M3U8 লিঙ্ক
ORIGINAL_M3U8_URL = "https://tvs1.aynaott.com/somoytv/tracks-v1/index.fmp4.m3u8?amp%3Bremote=no_check_ip&token=4894a5d455ddfc1cd0c740ebde3156faa50ce967-3f128356c076d2fdaef8614f7d112e2b-1745293867-1745292967&amp;remote=no_check_ip"
# আসল লিঙ্কের বেস ইউআরএল (URL এর শেষ অংশ বাদে)
BASE_URL = ORIGINAL_M3U8_URL.rsplit('/', 1)[0] + '/'

# এই সার্ভারের ঠিকানা (ডকার এ চালালে এটি ইন্টারনাল হবে, বাইরে থেকে অন্য ডোমেইন/আইপি দিয়ে অ্যাক্সেস হবে)
# আমরা রিলেটিভ পাথ ব্যবহার করবো, তাই এটি খুব জরুরি নয় যদি M3U8 ফাইলের মধ্যে সম্পূর্ণ URL না থাকে।
# তবে ক্লায়েন্ট সাইড থেকে পাথ ঠিক রাখার জন্য এটি জরুরি।
PROXY_PREFIX = "/stream" # সেগমেন্ট ফাইলের জন্য পাথ প্রিফিক্স

@app.route('/jamuna.m3u8')
def serve_m3u8():
    try:
        response = requests.get(ORIGINAL_M3U8_URL, timeout=10)
        response.raise_for_status() # HTTP एरर চেক করার জন্য

        original_m3u8_content = response.text
        modified_lines = []

        for line in original_m3u8_content.splitlines():
            line = line.strip()
            if not line:
                continue
            # যদি লাইনটি একটি URI হয় ( # দিয়ে শুরু না হয়)
            if not line.startswith('#'):
                 # যদি এটি একটি முழுமையான URL হয়, তবে এটিকে প্রক্সি URL দিয়ে প্রতিস্থাপন করুন
                 # এই ক্ষেত্রে, আমরা ধরে নিচ্ছি এটি একটি আপেক্ষিক পাথ বা ফাইলের নাম
                 # এটিকে আমাদের প্রক্সি পাথে পরিবর্তন করতে হবে
                 # উদাহরণ: mono.ts.ts -> /stream/mono.ts.ts
                modified_line = f"{PROXY_PREFIX}/{line}"
                modified_lines.append(modified_line)
            else:
                # কমেন্ট বা ট্যাগ লাইন அப்படியே রাখুন
                modified_lines.append(line)

        modified_m3u8_content = "\n".join(modified_lines)

        # সঠিক কন্টেন্ট টাইপ সহ রেসপন্স পাঠান
        return Response(modified_m3u8_content, mimetype='application/vnd.apple.mpegurl')

    except requests.exceptions.RequestException as e:
        print(f"Error fetching original M3U8: {e}")
        abort(500, description="Could not fetch the original stream manifest.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        abort(500, description="An internal error occurred.")


@app.route(f'{PROXY_PREFIX}/<path:segment>')
def serve_segment(segment):
    # আসল সেগমেন্ট ফাইলের URL তৈরি করুন
    original_segment_url = urljoin(BASE_URL, segment)

    try:
        # stream=True ব্যবহার করে ডেটা স্ট্রিম করুন মেমোরি बचाने के लिए
        response = requests.get(original_segment_url, stream=True, timeout=10)
        response.raise_for_status()

        # ক্লায়েন্টের কাছে সেগমেন্ট ডেটা স্ট্রিম করুন
        # iter_content ব্যবহার করে ডেটা চাঙ্ক আকারে পাঠানো হয়
        return Response(response.iter_content(chunk_size=1024*1024),
                        content_type=response.headers.get('Content-Type', 'video/MP2T')) # সঠিক MIME টাইপ দিন

    except requests.exceptions.RequestException as e:
        print(f"Error fetching segment {segment}: {e}")
        abort(404, description="Segment not found or could not be fetched.")
    except Exception as e:
        print(f"An unexpected error occurred while serving segment {segment}: {e}")
        abort(500, description="An internal error occurred while fetching the segment.")

if __name__ == '__main__':
    # ডকার এর জন্য 0.0.0.0 তে রান করুন এবং একটি পোর্ট নির্দিষ্ট করুন
    # আপনি প্রয়োজন অনুযায়ী পোর্ট পরিবর্তন করতে পারেন (যেমন 80 বা অন্যকিছু)
    app.run(host='0.0.0.0', port=5000)
