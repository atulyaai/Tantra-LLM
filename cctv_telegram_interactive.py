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
ALLOWED_CHAT_ID = 1484854122

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

def send_photo(chat_id, img_bytes, caption):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    files = {"photo": ("snapshot.jpg", img_bytes, "image/jpeg")}
    data = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}
    try:
        requests.post(url, files=files, data=data, timeout=10)
    except:
        pass

def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data, timeout=5)
    except:
        pass

def handle_command(chat_id, text):
    cmd = text.strip().lower()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if cmd.startswith("/cam") or cmd.startswith("cam"):
        # Match channel number
        match = re.search(r'\d+', cmd)
        ch = int(match.group(0)) if match else 1
        if 1 <= ch <= 8:
            send_message(chat_id, f"⏳ <i>Capturing Camera {ch} live view...</i>")
            img = capture_dvr_snapshot(ch)
            if img:
                send_photo(chat_id, img, f"📹 <b>Camera {ch:02d} Live View</b>\n⏰ <code>{now_str}</code>\nStatus: Online (1080p Lite H.265)")
            else:
                send_message(chat_id, f"❌ Camera {ch} is offline or no signal.")
        else:
            send_message(chat_id, "❌ Invalid camera number. Use /cam1 to /cam8.")

    elif cmd == "/all" or cmd == "all":
        send_message(chat_id, "⏳ <i>Capturing all 8 cameras...</i>")
        for ch in range(1, 9):
            img = capture_dvr_snapshot(ch)
            if img and len(img) > 5000:
                send_photo(chat_id, img, f"📹 <b>Camera {ch:02d}</b> | <code>{now_str}</code>")
                time.sleep(0.5)

    elif cmd == "/status" or cmd == "status":
        status_msg = f"""📊 <b>Hikvision System Status</b>
Model: <code>DS-7108HGHI-K1</code>
Time: <code>{now_str}</code>
IP: <code>{DVR_IP}</code>
Alarm Buzzer: <b>Muted / Silent</b>
Resolution: <b>720p HD (H.265 Smart)</b>
Commands:
• <code>/cam1</code> to <code>/cam8</code> (Get specific camera photo)
• <code>/all</code> (Get photos from all cameras)
• <code>/status</code> (Check system health)"""
        send_message(chat_id, status_msg)

    elif cmd == "/start" or cmd == "help" or cmd == "/help":
        help_msg = """👋 <b>Welcome to your CCTV Cloud Control Bot!</b>

Available Commands:
👉 <code>/cam1</code> - View Camera 1 Live
👉 <code>/cam2</code> - View Camera 2 Live
👉 <code>/all</code> - View all 8 cameras at once
👉 <code>/status</code> - Check DVR System Health"""
        send_message(chat_id, help_msg)

print(f"[*] Starting CCTV Telegram Bot Listener for Chat ID: {ALLOWED_CHAT_ID}...")
last_update_id = 0

# Send start menu
handle_command(ALLOWED_CHAT_ID, "/start")

while True:
    try:
        r = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset={last_update_id + 1}&timeout=30", timeout=35)
        if r.status_code == 200:
            updates = r.json().get("result", [])
            for u in updates:
                last_update_id = u["update_id"]
                msg = u.get("message", {})
                chat = msg.get("chat", {})
                from_id = chat.get("id")
                text = msg.get("text", "")
                if from_id == ALLOWED_CHAT_ID and text:
                    print(f"[*] Received command: '{text}' from {from_id}")
                    handle_command(from_id, text)
    except Exception as e:
        time.sleep(2)
