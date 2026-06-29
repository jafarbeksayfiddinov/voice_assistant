"""Local, offline speech-to-text (mirrors the local-Ollama philosophy of the
rest of this project — audio never leaves the machine).

Model size tradeoff:
  - "small"   : fast, but Uzbek is low-resource for Whisper — easily bleeds
                into Turkish/Russian vocabulary on unclear audio.
  - "medium"  : noticeably better Uzbek accuracy, ~2-3x slower on CPU. Default below.
  - "large-v3": best accuracy, much slower on CPU — worth it if you have a GPU.
"""

from faster_whisper import WhisperModel

WHISPER_MODEL_SIZE = "medium"

# Biases the decoder's internal language model toward Uzbek banking vocabulary
# instead of letting it drift into Turkish when it's unsure. Edit this to match
# your actual domain terms.
INITIAL_PROMPT = (
    "Quyidagi nutq o'zbek tilida, bank krediti haqida savol. "
    "Mikroqarz, kredit, foiz stavkasi, mahalla tadbirkori, avtokredit, "
    "o'zini o'zi band qilgan shaxs."
)

print(f"[stt] loading faster-whisper '{WHISPER_MODEL_SIZE}' model...")
_whisper = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cpu",          # change to "cuda" if you have a GPU available
    compute_type="int8",   # change to "float16" if device="cuda"
)
print("[stt] ready.")


def transcribe(audio_path: str) -> str:
    """Transcribes an audio file (any format ffmpeg/PyAV can decode,
    including the webm/opus blobs MediaRecorder produces in the browser)."""
    segments, info = _whisper.transcribe(
        audio_path,
        language="uz",                  # forces output script/vocab to Uzbek
        initial_prompt=INITIAL_PROMPT,  # biases decoding away from Turkish
        condition_on_previous_text=False,  # stop one bad guess from cascading
        vad_filter=True,    # trims the leading/trailing silence around your click
        beam_size=5,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text