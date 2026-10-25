"""
Full WebSocket ASR test with qwen-audio-3.0-realtime-plus (Manual mode).
"""
import asyncio
import base64
import json
import math
import os
import struct
import sys

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bot.config import config

ws_id = config.ai_base_url.split("://")[1].split("/")[0].split(".")[0]
MODEL = "qwen-audio-3.0-realtime-plus"
URL = f"wss://{ws_id}.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime?model={MODEL}"
API_KEY = config.ai_api_key


def make_wav_pcm(dur=2.0, freq=440.0):
    """WAV with tone, return raw PCM16 bytes."""
    sr = 16000
    n = int(sr * dur)
    samples = []
    for i in range(n):
        t = i / sr
        val = int(16000 * 0.3 * math.sin(2 * math.pi * freq * t))
        samples.append(struct.pack("<h", max(-32768, min(32767, val))))
    return b"".join(samples)


async def recv_event(ws, timeout=10):
    try:
        msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
        if msg.type == aiohttp.WSMsgType.TEXT:
            return json.loads(msg.data)
        return {"_raw": str(msg.type)}
    except TimeoutError:
        return {"_timeout": True}


async def main():
    h = {"Authorization": f"Bearer {API_KEY}"}
    print(f"Model: {MODEL}")
    print(f"URL: {URL}")
    print()

    try:
        async with aiohttp.ClientSession() as s:  # noqa: SIM117
            async with s.ws_connect(URL, headers=h,
                                     timeout=aiohttp.ClientTimeout(total=60, sock_read=15)) as ws:
                # 1. session.update (Manual mode)
                await ws.send_json({
                    "type": "session.update",
                    "session": {
                        "turn_detection": None,
                        "input_audio_format": "pcm16",
                        "input_audio_sample_rate": 16000,
                    },
                })
                evt = await recv_event(ws, 10)
                print(f"1. After update: {evt.get('type', evt)}")

                # 2. Send PCM audio (1 sec tone 440Hz + 0.5 sec silence)
                pcm = make_wav_pcm(1.0, 440.0) + b"\x00\x00" * 8000
                b64 = base64.b64encode(pcm).decode("ascii")
                await ws.send_json({
                    "type": "input_audio_buffer.append",
                    "audio": b64,
                })
                print(f"2. Sent audio: {len(pcm)} bytes PCM")

                # 3. Commit
                await ws.send_json({"type": "input_audio_buffer.commit"})
                print("3. Sent commit")

                # 4. Finish
                await ws.send_json({"type": "session.finish"})
                print("4. Sent finish")

                # 5. Read responses
                print("5. Events:")
                transcription = None
                for i in range(20):
                    evt = await recv_event(ws, 15)
                    t = evt.get("type", str(evt)[:80])
                    print(f"   <- {t}")

                    if "_timeout" in evt:
                        print("   (timeout)")
                        break
                    if t == "conversation.item.input_audio_transcription.completed":
                        transcription = evt.get("transcript", "")
                        content = evt.get("content", [])
                        if isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and "transcript" in part:
                                    transcription = part["transcript"]
                                    break
                        print(f"   >>> TRANSCRIPTION: {transcription!r}")
                        print(f"   >>> Full event: {json.dumps(evt, ensure_ascii=False)[:500]}")
                    if t == "error":
                        print(f"   ERROR: {json.dumps(evt, ensure_ascii=False)[:500]}")
                        break
                    if t == "session.finished":
                        print("   [DONE]")
                        break
                    if t == "session.created":
                        print(f"   Session id: {evt.get('session', {}).get('id', '?')}")

                if transcription:
                    print("\n[SUCCESS] ASR via WebSocket works!")
                    print(f"Transcription: {transcription!r}")
                elif transcription is not None:
                    print("\n[OK] WebSocket flow works, transcription empty (tone)")
                else:
                    print("\n[PARTIAL] Flow worked but no transcription received")

    except aiohttp.ClientError as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001 - ручной скрипт, печатаем любую ошибку
        print(f"[FAIL] {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(main())