import re
import time
import warnings
from google import genai
from google.genai import types
from dotenv import load_dotenv
import statics

load_dotenv()
warnings.filterwarnings("ignore")

client = genai.Client()


def ask_gemini_flash(question, context):
    # 1. Use max_output_tokens=250 so the model never gets cut off mid-number
    config = types.GenerateContentConfig(
        system_instruction=(
            "Siz bank xizmatlari bo'yicha virtual yordamchisiz. "
            "Berilgan kontekstdan foydalanib, foydalanuvchining savoliga faqat aniq fakt "
            "yoki raqam bilan juda qisqa javob bering. Hech qanday kirish so'zlari qo'shmang."
        ),
        temperature=0.0,
        max_output_tokens=250,
    )

    structured_prompt = (
        f"<KONTEKST>\n{context}\n</KONTEKST>\n\n"
        f"SAVOL: {question}\n\n"
        f"JAVOB:"
    )

    full_content = ""
    first_token_time = None
    request_start = time.perf_counter()

    primary_model = "gemini-2.5-flash"
    fallback_model = "gemini-2.5-flash-lite"

    try:
        response_stream = client.models.generate_content_stream(
            model=primary_model,
            contents=structured_prompt,
            config=config
        )

        for chunk in response_stream:
            if chunk.text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                # Append chunks directly rather than printing raw partial strings
                # that look like errors in your system logs
                full_content += chunk.text

    except Exception as e:
        # If primary model hits high-demand 503, fallback immediately
        if "503" in str(e) or "UNAVAILABLE" in str(e):
            print(f"\n[System] Primary model busy. Retrying with {fallback_model}...")
            try:
                full_content = ""  # Reset buffer
                response_stream = client.models.generate_content_stream(
                    model=fallback_model,
                    contents=structured_prompt,
                    config=config
                )
                for chunk in response_stream:
                    if chunk.text:
                        full_content += chunk.text
            except Exception as fallback_err:
                return "Kechirasiz, bank tizimi hozirda band. Birozdan so'ng qayta urinib ko'ring."
        else:
            return "Tizimda xatolik yuz berdi."

    # Process completion metrics cleanly
    done_time = time.perf_counter()
    if first_token_time:
        print(f"\n[Gemini Output Received]: {full_content.strip()}")
        print(f"[timing] Gemini Generation Total: {done_time - request_start:.2f}s")

    # Clean text using strict word boundaries to prevent digit cropping
    cleaned = full_content.strip()
    cleaned = re.sub(
        r"^\b(Faqat javobni beram|Mana javob|Javob shu)\b\s*:?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Final guardrail for Muxlisa TTS pipeline
    if not cleaned or "bilmayman" in cleaned.lower():
        return "Kechirasiz, ushbu ma'lumot topilmadi."

    return cleaned