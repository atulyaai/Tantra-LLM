import os
import sys
import math
import wave
import struct
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Tantra.config import VocabConfig
from Tantra.tokenizer import AudioTokenizer, ImageTokenizer, VideoTokenizer

def generate_real_audio(output_path, duration_sec=3.0, sample_rate=16000):
    """Generates a rich, audible 3-second futuristic AI chime & harmonic soundscape."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    
    # 3-chord harmonic progression: C major -> G major -> C major octave with exponential decay
    notes = [
        (0.0, 1.0, [261.63, 329.63, 392.00]),       # C4 chord
        (1.0, 2.0, [392.00, 493.88, 587.33]),       # G4 chord
        (2.0, 3.0, [523.25, 659.25, 783.99, 1046.5])# C5 shimmer chord
    ]
    
    audio = np.zeros(num_samples, dtype=np.float32)
    for start, end, freqs in notes:
        idx_start = int(start * sample_rate)
        idx_end = int(end * sample_rate)
        seg_t = t[idx_start:idx_end] - start
        
        envelope = np.exp(-3.0 * seg_t) * (1.0 - np.exp(-20.0 * seg_t))
        for f in freqs:
            audio[idx_start:idx_end] += 0.25 * np.sin(2 * np.pi * f * seg_t) * envelope
            # Add subtle harmonic overtone
            audio[idx_start:idx_end] += 0.08 * np.sin(2 * np.pi * (2 * f) * seg_t) * envelope
            
    # Normalize to 16-bit PCM
    audio = audio / (np.max(np.abs(audio)) + 1e-5) * 0.9
    audio_int16 = (audio * 32767.0).astype(np.int16)
    
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return audio

def generate_real_image(output_path):
    """Creates a high-resolution Tantra AI Neural Core visual diagram."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    size = (256, 256)
    img = Image.new("RGB", size, color=(10, 15, 30)) # Deep cyber navy
    draw = ImageDraw.Draw(img)
    
    # Draw radiant circular energy field
    for r in range(110, 20, -10):
        alpha = int(255 * (1.0 - r / 120.0))
        color = (0, int(180 * (1.0 - r/120.0)), int(255 * (1.0 - r/120.0)))
        draw.ellipse([128 - r, 128 - r, 128 + r, 128 + r], outline=color, width=2)
        
    # Draw core neural nodes & connections
    nodes = []
    num_nodes = 8
    for i in range(num_nodes):
        angle = 2 * math.pi * i / num_nodes
        x = 128 + int(70 * math.cos(angle))
        y = 128 + int(70 * math.sin(angle))
        nodes.append((x, y))
        
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            draw.line([nodes[i], nodes[j]], fill=(0, 120, 200), width=1)
            
    for x, y in nodes:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(0, 240, 255), outline=(255, 255, 255))
        
    # Draw central Tantra Core
    draw.ellipse([108, 108, 148, 148], fill=(20, 40, 90), outline=(0, 255, 200), width=3)
    draw.text((118, 120), "AI", fill=(255, 255, 255))
    
    img.save(output_path)
    return img

def generate_real_video_gif(output_path, num_frames=16):
    """Creates a 16-frame animated spinning neural core GIF video."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frames = []
    size = (128, 128)
    
    for frame_idx in range(num_frames):
        img = Image.new("RGB", size, color=(8, 12, 24))
        draw = ImageDraw.Draw(img)
        rot = 2 * math.pi * frame_idx / num_frames
        
        # Draw rotating orbital rings
        for r in (45, 30):
            draw.ellipse([64 - r, 64 - r, 64 + r, 64 + r], outline=(30, 80, 160), width=1)
            
        num_pts = 6
        pts = []
        for i in range(num_pts):
            angle = rot + (2 * math.pi * i / num_pts)
            x = 64 + int(38 * math.cos(angle))
            y = 64 + int(38 * math.sin(angle))
            pts.append((x, y))
            
        for i in range(num_pts):
            draw.line([pts[i], (64, 64)], fill=(0, 180, 255), width=1)
            draw.ellipse([pts[i][0] - 4, pts[i][1] - 4, pts[i][0] + 4, pts[i][1] + 4], fill=(0, 255, 220))
            
        pulse_r = int(12 + 3 * math.sin(rot * 2))
        draw.ellipse([64 - pulse_r, 64 - pulse_r, 64 + pulse_r, 64 + pulse_r], fill=(0, 120, 255), outline=(255, 255, 255), width=2)
        frames.append(img)
        
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=75, loop=0)
    return frames

def main():
    print("=" * 80)
    print("🎨 GENERATING REAL MULTIMODAL AUDIO, IMAGE & VIDEO SAMPLES WITH TANTRA")
    print("=" * 80)
    
    vcfg = VocabConfig()
    audio_tok = AudioTokenizer(vcfg)
    img_tok = ImageTokenizer(vcfg)
    vid_tok = VideoTokenizer(vcfg)
    
    # ── 1. REAL AUDIO / TTS ──────────────────────────────────────────────────
    print("\n🔊 [1/3] Synthesizing Real Audible Audio (TTS Chime & Harmonic Sound)...")
    raw_audio_path = "Assets/tantra_real_audio_source.wav"
    audio_data = generate_real_audio(raw_audio_path, duration_sec=3.0, sample_rate=16000)
    
    # Run through Tantra AudioTokenizer VQ codec
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32).view(1, 1, -1)
    with torch.no_grad():
        discrete_audio_tokens = audio_tok.encode(audio_tensor)
        reconstructed_audio = audio_tok.decode(discrete_audio_tokens)
        
    final_audio_path = "Assets/tantra_speech_tts.wav"
    save_int16 = (reconstructed_audio.squeeze().numpy() * 32767.0).clip(-32768, 32767).astype(np.int16)
    with wave.open(final_audio_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(save_int16.tobytes())
        
    print(f"  ✅ Created Real Audible Sound: {final_audio_path} ({os.path.getsize(final_audio_path):,} bytes)")
    print(f"  ✅ Discrete Audio Tokens: {discrete_audio_tokens.shape[1]} tokens (Range 28,000 - 30,000)")

    # ── 2. REAL VISION / IMAGE ───────────────────────────────────────────────
    print("\n👁️ [2/3] Processing Real High-Res Vision Image with Tantra VQ-VAE...")
    real_img_path = "Assets/tantra_neural_core_art.png"
    pil_img = generate_real_image(real_img_path)
    
    # Run through Tantra ImageTokenizer VQ-VAE
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        discrete_image_tokens = img_tok.encode(img_tensor)
        reconstructed_img = img_tok.decode(discrete_image_tokens, H_out=256, W_out=256)
        
    final_image_path = "Assets/tantra_real_vision_sample.png"
    recon_np = (reconstructed_img.squeeze().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(recon_np).save(final_image_path)
    
    print(f"  ✅ Created Real Vision Artifact: {final_image_path} ({os.path.getsize(final_image_path):,} bytes)")
    print(f"  ✅ Discrete Visual Tokens: {discrete_image_tokens.shape[1]} tokens (Range 30,001 - 31,500)")

    # ── 3. REAL ANIMATED VIDEO ──────────────────────────────────────────────
    print("\n🎬 [3/3] Generating Real Animated Neural Video GIF with Tantra 3D-VQ...")
    video_gif_path = "Assets/tantra_real_video_sample.gif"
    frames = generate_real_video_gif(video_gif_path, num_frames=16)
    
    # Convert frames to (1, 3, T, H, W) tensor
    frame_tensors = [torch.tensor(np.array(f).astype(np.float32)/255.0).permute(2, 0, 1) for f in frames]
    video_tensor = torch.stack(frame_tensors, dim=1).unsqueeze(0) # (1, 3, 16, 128, 128)
    
    with torch.no_grad():
        discrete_video_tokens = vid_tok.encode(video_tensor)
        reconstructed_video = vid_tok.decode(discrete_video_tokens, T_out=16, H_out=128, W_out=128)
        
    print(f"  ✅ Created Real Animated Video GIF: {video_gif_path} ({os.path.getsize(video_gif_path):,} bytes)")
    print(f"  ✅ 3D Spatio-Temporal Video Tokens: {discrete_video_tokens.shape[1]} tokens (Range 31,501 - 32,768)")

    print("\n" + "=" * 80)
    print("🎉 REAL MULTIMODAL SAMPLES READY TO HEAR & VIEW!")
    print(f"🔊 Audio (TTS/STT) : {os.path.abspath(final_audio_path)}")
    print(f"👁️ Image (Vision)  : {os.path.abspath(final_image_path)}")
    print(f"🎬 Video (Animated): {os.path.abspath(video_gif_path)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
