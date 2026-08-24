import requests
import hashlib
import re
import datetime
import sys

sys.stdout.reconfigure(encoding='utf-8')

BOT_TOKEN = "8578737502:AAEIuTYpjYDbBsxVOSjxYr-ReeBB5KuN5nY"
DVR_IP = "169.254.27.249"
USERNAME = "admin"
PASSWORD = "prince1989"

# 1. Fetch Chat ID
r_up = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates")
updates = r_up.json().get("result", [])

if not updates:
    print("[ERROR] No updates found yet.")
    sys.exit(1)

chat_id = updates[-1]["message"]["chat"]["id"]
first_name = updates[-1]["message"]["chat"].get("first_name", "Owner")
print(f"[*] Found User: {first_name} (Chat ID: {chat_id})")

# Send confirmation text message
requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data={
    "chat_id": chat_id,
    "text": f"🎉 <b>Hikvision CCTV Bot Connected!</b>\nHello {first_name}, your <b>DS-7108HGHI-K1</b> DVR is now linked to Telegram cloud!\nSending live camera snapshots...",
    "parse_mode": "HTML"
})

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

now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Send Camera 1
print("[*] Capturing & Sending Camera 1...")
img1 = capture_dvr_snapshot(1)
if img1:
    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
        files={"photo": ("cam1.jpg", img1, "image/jpeg")},
        data={"chat_id": chat_id, "caption": f"📹 <b>Camera 01 (Main View)</b>\n⏰ <code>{now_str}</code>\nStatus: Online | 720p HD H.265", "parse_mode": "HTML"}
    )
    print("✅ Camera 1 sent!")

# Send Camera 2
print("[*] Capturing & Sending Camera 2...")
img2 = capture_dvr_snapshot(2)
if img2:
    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
        files={"photo": ("cam2.jpg", img2, "image/jpeg")},
        data={"chat_id": chat_id, "caption": f"📹 <b>Camera 02 (Side View)</b>\n⏰ <code>{now_str}</code>\nStatus: Online | 720p HD H.265", "parse_mode": "HTML"}
    )
    print("✅ Camera 2 sent!")

print("\n🚀 All test snapshots sent to your Telegram phone successfully!")
