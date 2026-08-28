import os
import sys
import math
import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Tantra.config import VocabConfig
from Tantra.tokenizer import VideoTokenizer

def make_config_1_attention_resonance(num_frames=24, size=(256, 256)):
    """Config 1: 3D Multi-Layer Attention Network Resonance with moving wave pulses."""
    frames = []
    w, h = size
    cx, cy = w // 2, h // 2
    nodes = []
    num_nodes = 8
    
    for i in range(num_nodes):
        angle = 2 * math.pi * i / num_nodes
        x = cx + int(85 * math.cos(angle))
        y = cy + int(85 * math.sin(angle))
        nodes.append((x, y))
        
    for frame_idx in range(num_frames):
        img = Image.new("RGB", size, color=(6, 10, 22))
        draw = ImageDraw.Draw(img)
        phase = 2 * math.pi * frame_idx / num_frames
        
        # 1. Concentric Resonance Waves
        for ring_idx in range(4):
            ring_r = int(25 + (ring_idx * 28 + (frame_idx * 3)) % 110)
            alpha_val = max(10, 180 - ring_r)
            draw.ellipse([cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r], 
                         outline=(0, int(alpha_val * 0.7), alpha_val), width=1)
            
        # 2. Interconnected Attention Edges with Pulsing Gradients
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_phase = phase + (i * 0.5)
                edge_intensity = int(120 + 80 * math.sin(edge_phase))
                draw.line([nodes[i], nodes[j]], fill=(0, edge_intensity // 2, edge_intensity), width=1)
                
        # 3. Active Attention Nodes with glowing orbits
        for idx, (nx, ny) in enumerate(nodes):
            node_pulse = int(5 + 2 * math.sin(phase + idx))
            draw.ellipse([nx - node_pulse, ny - node_pulse, nx + node_pulse, ny + node_pulse], 
                         fill=(0, 230, 255), outline=(255, 255, 255), width=1)
            
        # 4. Central NeuroCore
        core_r = int(18 + 3 * math.sin(phase * 2))
        draw.ellipse([cx - core_r, cy - core_r, cx + core_r, cy + core_r], 
                     fill=(15, 35, 80), outline=(0, 255, 200), width=2)
        draw.text((cx - 10, cy - 6), "ALRA", fill=(255, 255, 255))
        frames.append(img)
        
    return frames

def make_config_2_token_stream_matrix(num_frames=24, size=(200, 200)):
    """Config 2: High-Speed Discrete Token Stream Cylinder Flow."""
    frames = []
    w, h = size
    cx, cy = w // 2, h // 2
    
    symbols = ["0", "1", "T", "A", "N", "T", "R", "A", "λ", "π", "Σ", "Ω", "1", "0", "8", "6"]
    
    for frame_idx in range(num_frames):
        img = Image.new("RGB", size, color=(4, 8, 16))
        draw = ImageDraw.Draw(img)
        phase = 2 * math.pi * frame_idx / num_frames
        
        # Draw vertical token stream columns
        num_cols = 10
        col_width = w // num_cols
        for col in range(num_cols):
            x = col * col_width + col_width // 2
            speed = 3 + (col % 3) * 2
            offset_y = (frame_idx * speed * 4 + col * 20) % h
            
            for row in range(8):
                y = (offset_y + row * 24) % h
                char = symbols[(col + row + frame_idx) % len(symbols)]
                brightness = int(100 + 155 * (1.0 - y / h))
                draw.text((x - 4, y), char, fill=(0, brightness, int(brightness * 0.7)))
                
        # Draw central holographic focus portal
        draw.rectangle([cx - 40, cy - 40, cx + 40, cy + 40], outline=(0, 255, 220), width=2)
        draw.text((cx - 28, cy - 6), "32K VQ", fill=(255, 255, 255))
        frames.append(img)
        
    return frames

def make_config_3_omnimodal_fusion_core(num_frames=24, size=(220, 220)):
    """Config 3: Omnimodal (Audio wave + Vision grid + Text tokens) Tri-Fusion Loop."""
    frames = []
    w, h = size
    cx, cy = w // 2, h // 2
    
    for frame_idx in range(num_frames):
        img = Image.new("RGB", size, color=(8, 10, 24))
        draw = ImageDraw.Draw(img)
        phase = 2 * math.pi * frame_idx / num_frames
        
        # 1. Modality 1: Oscillating Audio Waveform Top Loop
        wave_pts = []
        for x in range(20, w - 20, 4):
            y = int(50 + 12 * math.sin(phase * 2 + x * 0.08))
            wave_pts.append((x, y))
        for i in range(len(wave_pts) - 1):
            draw.line([wave_pts[i], wave_pts[i+1]], fill=(255, 120, 0), width=2)
        draw.text((25, 30), "AUDIO (STT/TTS)", fill=(255, 160, 50))
        
        # 2. Modality 2: Vision Pixel Grid Bottom Left
        grid_x, grid_y = 30, 140
        for gx in range(4):
            for gy in range(4):
                val = int(128 + 127 * math.sin(phase + gx + gy))
                draw.rectangle([grid_x + gx*10, grid_y + gy*10, grid_x + gx*10 + 8, grid_y + gy*10 + 8], 
                               fill=(0, val, 255), outline=(100, 200, 255))
        draw.text((25, 185), "VISION (2D VQ)", fill=(100, 200, 255))
        
        # 3. Modality 3: Text Token Matrix Bottom Right
        text_x, text_y = 135, 140
        draw.rectangle([text_x, text_y, text_x + 55, text_y + 40], outline=(0, 255, 150), width=1)
        draw.text((text_x + 6, text_y + 6), "<|user|>", fill=(0, 255, 180))
        draw.text((text_x + 6, text_y + 20), "Tantra", fill=(255, 255, 255))
        draw.text((130, 185), "TEXT / SFT", fill=(0, 255, 150))
        
        # 4. Central Energy Orb Fusion Core
        core_r = int(22 + 4 * math.sin(phase * 3))
        draw.ellipse([cx - core_r, cy - core_r - 10, cx + core_r, cy + core_r - 10], 
                     fill=(20, 50, 110), outline=(255, 255, 255), width=2)
        draw.text((cx - 16, cy - 16), "OMNI", fill=(255, 255, 255))
        frames.append(img)
        
    return frames

def main():
    print("=" * 80)
    print("🎬 GENERATING ADVANCED VIDEO CONFIGURATIONS WITH TANTRA 3D-VQ")
    print("=" * 80)
    
    os.makedirs("Samples/Video", exist_ok=True)
    vcfg = VocabConfig()
    vid_tok = VideoTokenizer(vcfg)
    
    configs = [
        ("Config 1 (Attention Resonance)", make_config_1_attention_resonance(24, (256, 256)), "Samples/Video/tantra_video_attention_resonance_256.gif"),
        ("Config 2 (Matrix Token Stream)", make_config_2_token_stream_matrix(24, (200, 200)), "Samples/Video/tantra_video_token_stream_matrix.gif"),
        ("Config 3 (Omnimodal Tri-Fusion)", make_config_3_omnimodal_fusion_core(24, (220, 220)), "Samples/Video/tantra_video_omnimodal_fusion_core.gif"),
    ]
    
    for name, frames, out_path in configs:
        print(f"\n🎥 Processing {name}...")
        # Save high-res animated GIF
        frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=65, loop=0)
        
        # Process through Tantra 3D VideoTokenizer VQ
        frame_tensors = [torch.tensor(np.array(f).astype(np.float32)/255.0).permute(2, 0, 1) for f in frames[:16]]
        video_tensor = torch.stack(frame_tensors, dim=1).unsqueeze(0) # (1, 3, 16, H, W)
        
        with torch.no_grad():
            tokens = vid_tok.encode(video_tensor)
            
        print(f"  ✅ Saved: {out_path} ({os.path.getsize(out_path):,} bytes, {len(frames)} frames)")
        print(f"  ✅ 3D Spatio-Temporal Token Tokens: {tokens.shape[1]} discrete tokens (Range: 31,501 - 32,768)")

    print("\n" + "=" * 80)
    print("🎉 ALL 3 VIDEO CONFIGURATIONS GENERATED & SAVED IN Samples/Video/!")
    print("=" * 80)

if __name__ == "__main__":
    main()
