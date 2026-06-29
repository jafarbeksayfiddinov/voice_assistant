

import os
import subprocess
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

MUXLISA_STT_URL = "https://service.muxlisa.uz/api/v2/stt"
AUDIO_FIELD_NAME = "audio"

API_KEY = os.environ.get("MUXLISA_API_KEY")


def _convert_to_wav(input_path: str) -> str:
    """Muxlisa's documented example uploads .wav/audio/wav. Browsers record
    webm/opus, so we convert before sending."""
    output_path = input_path + ".converted.wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg is not installed. Run: brew install ffmpeg (macOS)."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg conversion failed: {e.stderr.decode(errors='ignore')[:500]}")
    return output_path


def transcribe(audio_path: str) -> str:
    if not API_KEY:
        raise RuntimeError(
            "MUXLISA_API_KEY is not set. Put it in a .env file next to muxlisa_stt.py."
        )

    wav_path = _convert_to_wav(audio_path)
    try:
        headers = {"x-api-key": API_KEY}
        with open(wav_path, "rb") as f:
            files = {AUDIO_FIELD_NAME: ("audio.wav", f, "audio/wav")}
            resp = requests.post(MUXLISA_STT_URL, headers=headers, files=files, timeout=60)
    finally:
        os.remove(wav_path)

    if not resp.ok:
        # Surface the actual error body instead of a blind raise_for_status().
        raise RuntimeError(f"Muxlisa STT {resp.status_code}: {resp.text[:500]}")

    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(f"Muxlisa STT returned non-JSON response: {resp.text[:500]}")

    print("[muxlisa_stt] raw response:", data)

    for path in (("text",), ("transcript",), ("result", "text"), ("data", "text")):
        node = data
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                node = None
                break
        if isinstance(node, str) and node:
            return node.strip()

    raise RuntimeError(
        f"Couldn't find transcript text in response: {data}. "
        "Check the printed raw response above and tell me the correct key path."
    )