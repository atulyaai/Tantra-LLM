import sys
import os
import asyncio

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.local.rwkv_adapter import RWKVAdapter
from core.vision import VisionOrgan
from core.voice import VoiceOrgan
from core.sentiment import SentimentCore
from atulya_core.schema.models import TantraRequest, Message

async def test_rwkv_adapter():
    print("\n[TEST] Testing RWKV Adapter...")
    adapter = RWKVAdapter(model_path="mock_model_path.pth")
    if not adapter.health_check():
        print("[WARN] RWKV Adapter health check failed (Expected if lib/model missing).")
    else:
        print("[PASS] RWKV Adapter health check passed.")

    req = TantraRequest(messages=[Message(role="user", content="Hello RWKV!")])
    resp = await adapter.generate(req)
    print(f"[RESULT] Generate output: {resp.content}")

async def test_vision():
    print("\n[TEST] Testing Vision Organ...")
    vision = VisionOrgan()
    # Mock capture
    res = await vision.analyze_environment()
    print(f"[RESULT] Vision Analysis: {res}")

async def test_voice():
    print("\n[TEST] Testing Voice Organ...")
    voice = VoiceOrgan()
    res = await voice.listen("mock_audio.wav")
    print(f"[RESULT] Voice Listen: {res}")

async def test_sentiment():
    print("\n[TEST] Testing Sentiment Core...")
    rwkv_mock = RWKVAdapter(model_path="mock.pth")
    sentiment = SentimentCore(adapter=rwkv_mock)
    res = await sentiment.analyze_vibe("I am very happy!")
    print(f"[RESULT] Sentiment Analysis: {res}")

async def main():
    print("=== Tantra-LLM RWKV Integration Verification ===")
    await test_rwkv_adapter()
    await test_vision()
    await test_voice()
    await test_sentiment()
    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    asyncio.run(main())
