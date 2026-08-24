import requests
import hashlib
import re
import datetime
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

DVR_IP = "169.254.27.249"
USERNAME = "admin"
PASSWORD = "prince1989"

# Telegram Config (Replace with your actual token and chat_id)
BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
CHAT_ID = "YOUR_CHAT_ID_HERE"

def capture_dvr_snapshot(channel=1):
    path = f"/ISAPI/Streaming/channels/{channel}01/picture"
    url = f"http://{DVR_IP}{path}"
    
    r1 = requests.get(url, timeout=5)
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
    r2 = requests.get(url, headers=headers, timeout=5)
    if r2.status_code == 200:
        return r2.content
    return None

def send_telegram_alert(channel=1, custom_caption=None):
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or CHAT_ID == "YOUR_CHAT_ID_HERE":
        print("[ERROR] Please set your actual BOT_TOKEN and CHAT_ID in the script!")
        return False

    print(f"[*] Capturing live snapshot from Camera {channel}...")
    img_data = capture_dvr_snapshot(channel)
    if not img_data:
        print("[!] Failed to capture snapshot from DVR.")
        return False

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    caption = custom_caption or f"🚨 <b>CCTV Alert: Camera {channel}</b>\n⏰ Time: <code>{now_str}</code>\n📍 Model: DS-7108HGHI-K1"

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    files = {"photo": ("snapshot.jpg", img_data, "image/jpeg")}
    data = {"chat_id": CHAT_ID, "caption": caption, "parse_mode": "HTML"}

    print("[*] Uploading snapshot to Telegram Cloud...")
    try:
        r = requests.post(url, files=files, data=data, timeout=10)
        if r.status_code == 200:
            print("✅ Alert photo sent successfully to Telegram!")
            return True
        else:
            print(f"[!] Telegram API error: {r.status_code} - {r.text}")
            return False
    except Exception as e:
        print(f"[!] Network error sending to Telegram: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        BOT_TOKEN = sys.argv[1]
        CHAT_ID = sys.argv[2]
        ch = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        send_telegram_alert(ch)
    else:
        print("Usage: python telegram_cctv_bot.py <BOT_TOKEN> <CHAT_ID> [CHANNEL_NUM]")
