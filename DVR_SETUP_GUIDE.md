# 📹 Hikvision DVR DS-7108HGHI-K1 — Complete Configuration & Handover Guide

## 1. Device Summary & Credentials
* **Model:** Hikvision Turbo HD `DS-7108HGHI-K1` (8-Channel)
* **Firmware:** `V4.30.203` (Build 220222)
* **Current Direct IP:** `http://169.254.27.249` (or `http://192.168.1.8` when on Wi-Fi router)
* **Username:** `admin`
* **Password:** `prince1989`

---

## 2. Completed Automated Optimizations (Executed via API)

### A. Image Quality & Day/Night Clarity (All 8 Channels)
* **Brightness:** `128` (Washed-out 255 white glare completely removed).
* **Contrast:** `135` (High dynamic contrast for clear person/vehicle identification).
* **Sharpness:** `15` (Maximum sharpness applied across all channels).
* **Digital Noise Reduction (DNR):** `Level 5` (Maximum night-time grain reduction).
* **Image Profile:** `Outdoor` mode enabled.

### B. Video Stream Compression (3x Recording Duration)
* **Main Stream Codec:** Upgraded to **`H.265`** with **`SmartCodec`** enabled.
* **Bitrate Type:** Switched to **`VBR`** (Variable Bitrate) with dynamic cap (`832 kbps`).
* **Frame Rate:** Optimized `15–25 fps` PAL.

### C. System & Motion Detection
* **Motion Detection:** Activated on all 8 channels with full-screen grid coverage.
* **Time Synchronization:** Configured with `time.windows.com` (`pool.ntp.org`) on `GMT+05:30`.
* **Alarm Buzzer:** Silenced for all storage/missing drive exceptions.

---

## 3. Local Camera Live Dashboard
A lightweight, fast dashboard has been created in your project workspace:
* **Path:** `d:\Atulya Tantra\Tantra-LLM\cctv_dashboard.html`
* Simply double-click `cctv_dashboard.html` to view all camera feeds in your browser without any slow Hikvision plugins or cloud software.

---

## 4. Pending Hardware & Network Checklist (When Back From Dinner)

### 1. Physical Hard Drive (HDD) Connection:
* **The Problem:** The DVR storage controller returned an empty list (`<hddList></hddList>`).
* **Action:** 
  1. Open the 4 screws of the DVR metal box.
  2. Re-seat both the red SATA data cable and power cable.
  3. Ensure your power adapter is **12V 2A, 3A, or 5A** (12V 1.5A is too weak to spin mechanical drives).
  4. Once spinning, go to **Storage $\rightarrow$ HDD Management** and click **Init** (Format).

### 2. Camera 2 Going Black Intermittently:
* **Cause 1 (Voltage Drop):** Night vision IR LEDs draw extra power; if voltage drops below 10.5V, camera reboots.
* **Cause 2 (Loose BNC):** Tighten the BNC twist connector and power jack at Camera 2.
* **Cause 3 (IR Cut Filter Switch):** Day/Night threshold switch delay has been lengthened to prevent flickering.

### 3. Remote Internet Access (DDNS + Port Forwarding):
1. Plug the LAN cable from the DVR into your **Wi-Fi Router**.
2. Log into your router (`http://192.168.1.1`).
3. Forward Ports:
   * **Web Port:** `8088` (or `80`) $\rightarrow$ DVR IP
   * **Server Port:** `8000` $\rightarrow$ DVR IP
   * **RTSP Video Port:** `554` $\rightarrow$ DVR IP
4. Set up free DDNS (`yourname.duckdns.org`) under **Configuration $\rightarrow$ Network $\rightarrow$ DDNS**.
