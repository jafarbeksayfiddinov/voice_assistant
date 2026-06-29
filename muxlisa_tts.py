"""Text-to-speech via the Muxlisa AI API -- speaks the LLM's answer back to the user.

Uses the same MUXLISA_API_KEY as muxlisa_stt.py (same .env file, same dashboard key).
"""

import os
import json
import base64
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

MUXLISA_TTS_URL = "https://service.muxlisa.uz/api/v2/tts"
DEFAULT_SPEAKER = 1

API_KEY = os.environ.get("MUXLISA_API_KEY")


def synthesize(text: str, speaker: int = DEFAULT_SPEAKER) -> tuple[bytes, str]:
    """Returns (audio_bytes, mime_type)."""
    if not API_KEY:
        raise RuntimeError("MUXLISA_API_KEY is not set. Put it in the .env file.")

    headers = {"Content-Type": "application/json", "x-api-key": API_KEY}
    payload = json.dumps({"text": text, "speaker": speaker})
    resp = requests.post(MUXLISA_TTS_URL, headers=headers, data=payload, timeout=60)

    if not resp.ok:
        raise RuntimeError(f"Muxlisa TTS {resp.status_code}: {resp.text[:500]}")

    content_type = resp.headers.get("Content-Type", "")

    if "audio" in content_type or "octet-stream" in content_type:
        return resp.content, content_type or "audio/mpeg"

    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(
            f"Unexpected TTS response (content-type={content_type}, "
            f"{len(resp.content)} bytes) -- not audio, not JSON."
        )

    print("[muxlisa_tts] raw JSON response:", data)

    for path in (("audio",), ("audio_base64",), ("data", "audio"), ("result", "audio")):
        node = data
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, str) and node:
            return base64.b64decode(node), "audio/mpeg"

    for path in (("url",), ("audio_url",), ("data", "url")):
        node = data
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, str) and node:
            audio_resp = requests.get(node, timeout=60)
            audio_resp.raise_for_status()
            return audio_resp.content, audio_resp.headers.get("Content-Type", "audio/mpeg")

    raise RuntimeError(
        f"Couldn't find audio in TTS response: {data}. "
        "Check the printed raw JSON above and tell me the correct key path."
    )

