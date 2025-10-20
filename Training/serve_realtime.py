import asyncio
import json
from typing import Dict, Any

from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect

from Training.serve_api import build_agent
try:
    import whisper  # optional
except Exception:
    whisper = None
try:
    import pyttsx3  # lightweight TTS
except Exception:
    pyttsx3 = None


app = FastAPI()
agent = build_agent()


@app.websocket("/voice")
async def voice(ws: WebSocket):
    await ws.accept()
    try:
        # For portability: accept text frames as ASR transcripts; TTS is optional
        while True:
            text = await ws.receive_text()
            _, final_text = agent.run(text)
            await ws.send_text(json.dumps({"type": "partial", "data": final_text[:40]}))
            await asyncio.sleep(0.05)
            await ws.send_text(json.dumps({"type": "final", "data": final_text}))
            if pyttsx3 is not None:
                try:
                    engine = pyttsx3.init()
                    engine.say(final_text)
                    engine.runAndWait()
                except Exception:
                    pass
    except WebSocketDisconnect:
        return


