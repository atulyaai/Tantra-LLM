import requests
import hashlib
import re
import datetime
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

BOT_TOKEN = "8578737502:AAEIuTYpjYDbBsxVOSjxYr-ReeBB5KuN5nY"
DVR_IP = "169.254.27.249"
USERNAME = "admin"
PASSWORD = "prince1989"
CHAT_ID = 1484854122

def capture_dvr_snapshot(channel=1):
    path = f"/ISAPI/Streaming/channels/{channel}01/picture"
    url = f"http://{DVR_IP}{path}"
    
    try:
        r1 = requests.get(url, timeout=4)
        auth_header = r1.headers.get("WWW-Authenticate", "")
        realm_match = re.search(r'realm="([^"]+)"', auth_header)
        nonce_match = re.search(r'nonce="([^"]+)"', auth_header)
        qop_match = re.search(r'qop="([^"]+)"', auth_header)

        if not (realm_match and nonce_match):
            return None

        realm = realm_match.group(1)
        nonce = nonce_match.group(1)
        qop = qop_match.group(1) if qop_match else None
        
        nc = "00000001"
        cnonce = "0a4f113b"
        ha1 = hashlib.md5(f"{USERNAME}:{realm}:{PASSWORD}".encode('utf-8')).hexdigest()
        ha2 = hashlib.md5(f"GET:{path}".encode('utf-8')).hexdigest()
        
        if qop == "auth":
            resp_str = f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}"
            response_hash = hashlib.md5(resp_str.encode('utf-8')).hexdigest()
            digest_header = f'Digest username="{USERNAME}", realm="{realm}", nonce="{nonce}", uri="{path}", response="{response_hash}", qop={qop}, nc={nc}, cnonce="{cnonce}"'
        else:
            resp_str = f"{ha1}:{nonce}:{ha2}"
            response_hash = hashlib.md5(resp_str.encode('utf-8')).hexdigest()
            digest_header = f'Digest username="{USERNAME}", realm="{realm}", nonce="{nonce}", uri="{path}", response="{response_hash}"'
            
        headers = {"Authorization": digest_header}
        r2 = requests.get(url, headers=headers, timeout=4)
        if r2.status_code == 200:
            return r2.content
    except:
        pass
    return None

def send_photo(img_bytes, caption):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    files = {"photo": ("motion.jpg", img_bytes, "image/jpeg")}
    data = {"chat_id": CHAT_ID, "caption": caption, "parse_mode": "HTML"}
    try:
        r = requests.post(url, files=files, data=data, timeout=10)
        print(f"[TELEGRAM] Photo sent: {r.status_code}")
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")

def listen_alert_stream():
    path = "/ISAPI/Event/notification/alertStream"
    url = f"http://{DVR_IP}{path}"
    
    print(f"[*] Connecting to DVR Real-Time Alert Stream at {url}...")
    
    # Challenge
    r1 = requests.get(url, timeout=5)
    auth_header = r1.headers.get("WWW-Authenticate", "")
    realm_match = re.search(r'realm="([^"]+)"', auth_header)
    nonce_match = re.search(r'nonce="([^"]+)"', auth_header)
    qop_match = re.search(r'qop="([^"]+)"', auth_header)

    if not (realm_match and nonce_match):
        print("[!] Failed to get auth challenge from alert stream.")
        return

    realm = realm_match.group(1)
    nonce = nonce_match.group(1)
    qop = qop_match.group(1) if qop_match else None
    
    nc = "00000001"
    cnonce = "0a4f113b"
    ha1 = hashlib.md5(f"{USERNAME}:{realm}:{PASSWORD}".encode('utf-8')).hexdigest()
    ha2 = hashlib.md5(f"GET:{path}".encode('utf-8')).hexdigest()
    
    if qop == "auth":
        resp_str = f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}"
        response_hash = hashlib.md5(resp_str.encode('utf-8')).hexdigest()
        digest_header = f'Digest username="{USERNAME}", realm="{realm}", nonce="{nonce}", uri="{path}", response="{response_hash}", qop={qop}, nc={nc}, cnonce="{cnonce}"'
    else:
        resp_str = f"{ha1}:{nonce}:{ha2}"
        response_hash = hashlib.md5(resp_str.encode('utf-8')).hexdigest()
        digest_header = f'Digest username="{USERNAME}", realm="{realm}", nonce="{nonce}", uri="{path}", response="{response_hash}"'
        
    headers = {"Authorization": digest_header}
    
    # Stream alert events in real-time
    r2 = requests.get(url, headers=headers, stream=True, timeout=120)
    print(f"[*] Alert Stream Connected! (Status: {r2.status_code})")
    print("[*] Listening for LIVE Motion Detection Events across all cameras...")
    
    last_trigger_time = {}
    buffer = ""
    
    for chunk in r2.iter_content(chunk_size=1024):
        if chunk:
            text = chunk.decode('utf-8', errors='ignore')
            buffer += text
            
            # Check for Motion event (VMD = Video Motion Detection)
            if "<eventType>VMD</eventType>" in buffer or "<eventType>shelteralarm</eventType>" in buffer:
                # Find channel
                ch_match = re.search(r'<dynChannelID>(\d+)</dynChannelID>', buffer) or re.search(r'<channelID>(\d+)</channelID>', buffer)
                ch = int(ch_match.group(1)) if ch_match else 1
                
                now = time.time()
                # Cooldown: only send 1 alert per camera every 10 seconds to avoid spamming
                if now - last_trigger_time.get(ch, 0) > 10:
                    last_trigger_time[ch] = now
                    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n🚨 [MOTION DETECTED] Camera {ch} at {now_str}! Capturing snapshot...")
                    
                    img = capture_dvr_snapshot(ch)
                    if img:
                        caption = f"🚨 <b>MOTION DETECTED!</b>\n📹 <b>Camera {ch:02d}</b>\n⏰ <code>{now_str}</code>\n📍 Automatic Cloud Event Snapshot"
                        send_photo(img, caption)
                
                # Reset buffer
                buffer = ""

if __name__ == "__main__":
    while True:
        try:
            listen_alert_stream()
        except Exception as e:
            print(f"[!] Stream reconnecting in 5s: {e}")
            time.sleep(5)
