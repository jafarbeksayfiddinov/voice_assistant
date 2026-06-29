import numpy as np
from sentence_transformers import SentenceTransformer, util
from chunking_fixed import get_records, normalize_text
import statics
from ollama import ask_ollama

model = SentenceTransformer("intfloat/multilingual-e5-base")

# --- Build the corpus -------------------------------------------------
records = get_records(data.PATH)                       # [{full, summary, turi}, ...]
summaries = [normalize_text(r["summary"]) for r in records]

# Embed the SHORT denoised summary, not the 2000+ char raw chunk.
doc_embeddings = model.encode(
    [f"passage: {s}" for s in summaries],
    normalize_embeddings=True,
)

CONFIDENCE_THRESHOLD = 0.78   # tune this against your own eval set


def retrieve(question, k=2, debug=False):
    q_norm = normalize_text(question)
    query_emb = model.encode([f"query: {q_norm}"], normalize_embeddings=True)

    scores = util.cos_sim(query_emb, doc_embeddings)[0]
    order = scores.argsort(descending=True)[:k]

    if debug:
        print(f"\n[debug] query: {question}")
        for idx in order:
            idx = int(idx)
            print(f"   score={scores[idx]:.3f}  turi={records[idx]['turi']!r}")

    best_idx = int(order[0])
    if float(scores[best_idx]) < CONFIDENCE_THRESHOLD:
        return None  # no confident match -> let the caller say "Bilmayman"

    return records[best_idx]["full"]


if __name__ == "__main__":
    for question in data.test_questions_tiny:
        print("-" * 50)
        print(question)

        context = retrieve(question, debug=True)
        if context is None:
            print("(past threshold) -> Bilmayman.")
            continue

        try:
            ask_ollama(question, context)
        except Exception as e:
            print(f"Ollama xato: {e}")
