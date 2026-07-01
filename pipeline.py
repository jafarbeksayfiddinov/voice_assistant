"""Shared RAG pipeline: load once at process startup, reuse for every request."""

from sentence_transformers import SentenceTransformer, util

from chunking_fixed import get_records, normalize_text
import statics
from gemini import ask_gemini_flash
from ollama import ask_ollama

print("[pipeline] loading embedding model...")
_model = SentenceTransformer("intfloat/multilingual-e5-base")

print("[pipeline] building credit-record corpus...")
_records = get_records(statics.PATH)                       # [{full, summary, turi}, ...]
_summaries = [normalize_text(r["summary"]) for r in _records]
_doc_embeddings = _model.encode(
    [f"passage: {s}" for s in _summaries],
    normalize_embeddings=True,
)
print(f"[pipeline] ready. {len(_records)} unique credit records loaded.")

CONFIDENCE_THRESHOLD = 0.78  # tune against your own eval set


import time


def retrieve(question: str, k: int = 3, debug: bool = False):
    """Returns the best-matching record's full text, or None if no record
    clears the confidence threshold."""
    q_norm = normalize_text(question)
    query_emb = _model.encode([f"query: {q_norm}"], normalize_embeddings=True)

    scores = util.cos_sim(query_emb, _doc_embeddings)[0]
    order = scores.argsort(descending=True)[:k]

    if debug:
        print(f"[pipeline] query: {question}")
        for idx in order:
            idx = int(idx)
            print(f"   score={scores[idx]:.3f}  turi={_records[idx]['turi']!r}")

    best_idx = int(order[0])
    if float(scores[best_idx]) < CONFIDENCE_THRESHOLD:
        return None

    return _records[best_idx]["full"]


def get_answer(question: str) -> str:
    """Full text-in, text-out pipeline: retrieve context, ask the LLM."""
    t0 = time.perf_counter()
    context = retrieve(question, debug=True)
    t1 = time.perf_counter()
    print(f"[timing] retrieve: {t1 - t0:.2f}s")

    if context is None:
        return "Bilmayman."

    try:
        answer = ask_gemini_flash(question, context)
        t2 = time.perf_counter()
        print(f"[timing] ollama generation: {t2 - t1:.2f}s")
        return answer
    except Exception as e:
        return f"Xatolik yuz berdi: {e}"