# SQB Ovozli Yordamchi — ishga tushirish

## Fayllar
- `static/index.html` — mikrofon tugmasi bo'lgan sahifa (frontend)
- `app.py` — FastAPI server: sahifani beradi + `/api/ask` endpoint
- `stt.py` — lokal Whisper (faster-whisper) orqali speech-to-text
- `pipeline.py` — sizning oldingi retrieval + Ollama mantiqingiz (chunking_fixed.py asosida)
- `chunking_fixed.py`, `data.py`, `ollama.py` — avvalgi tuzatilgan fayllar, o'zgarishsiz ishlatiladi

## O'rnatish
```bash
pip install -r requirements.txt
```
Birinchi marta ishga tushirganda ikkita model avtomatik yuklab olinadi (internet kerak bo'ladi bir martalik):
- `intfloat/multilingual-e5-base` (Hugging Face)
- Whisper `small` modeli (faster-whisper orqali)

Bundan keyin Ollama allaqachon ishlab turishi kerak (`mistral:7b` bilan), chunking.py'dagidek.

## Ishga tushirish
```bash
uvicorn app:app --reload --port 8000
```
Brauzerda: `http://localhost:8000`

## Diqqat
- Mikrofonga ruxsat so'raladi — `localhost` HTTPS talab qilmaydi, shuning uchun muammosiz ishlaydi.
- Birinchi audio so'rov sekin bo'lishi mumkin (model "isinish" vaqti). Keyingilari tezroq.
- Agar Uzbek tilida transkripsiya sifati past bo'lsa, `stt.py` ichidagi `WHISPER_MODEL_SIZE = "small"` ni `"medium"` ga o'zgartiring (sekinroq, lekin aniqroq).
- `pipeline.py` ichidagi `CONFIDENCE_THRESHOLD = 0.78` — agar javoblar noto'g'ri kredit turini tanlasa, buni oshiring; agar haddan tashqari ko'p "Bilmayman" chiqsa, tushiring.
