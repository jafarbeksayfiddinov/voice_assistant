import tempfile
import os
import base64
import time

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

import pipeline   # noqa: F401  (import triggers model + corpus loading at startup)
import muxlisa_stt as stt
import muxlisa_tts as tts

app = FastAPI(title="SQB Ovozli Yordamchi")


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.post("/api/ask")
async def ask(audio: UploadFile = File(...)):
    t_start = time.perf_counter()

    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        transcript = stt.transcribe(tmp_path)
    finally:
        os.remove(tmp_path)
    t_stt = time.perf_counter()
    print(f"[timing] STT: {t_stt - t_start:.2f}s")

    if not transcript:
        return JSONResponse(
            {"transcript": "", "answer": "Ovoz aniqlanmadi. Iltimos, qaytadan urinib ko'ring.", "audio_base64": None}
        )

    answer = pipeline.get_answer(transcript)  # logs retrieve + ollama timing internally
    t_answer = time.perf_counter()

    audio_base64 = None
    try:
        audio_bytes, mime_type = tts.synthesize(answer)
        audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
    except Exception as e:
        print(f"[app] TTS failed, continuing with text-only answer: {e}")
    t_tts = time.perf_counter()
    print(f"[timing] TTS: {t_tts - t_answer:.2f}s")
    print(f"[timing] TOTAL: {t_tts - t_start:.2f}s")

    return JSONResponse({"transcript": transcript, "answer": answer, "audio_base64": audio_base64})


# Run with: uvicorn app:app --port 8000