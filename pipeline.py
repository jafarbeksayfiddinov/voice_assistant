
from sentence_transformers import SentenceTransformer, util

from chunking_fixed import get_records, normalize_text
import statics
from ollama import ask_ollama

print("[pipeline] loading embedding model...")
_model = SentenceTransformer("intfloat/multilingual-e5-base")

print("[pipeline] building credit-record corpus...")
_records = get_records(data.PATH)                       # [{full, summary, turi}, ...]
_summaries = [normalize_text(r["summary"]) for r in _records]
_doc_embeddings = _model.encode(
    [f"passage: {s}" for s in _summaries],
    normalize_embeddings=True,
)
print(f"[pipeline] ready. {len(_records)} unique credit records loaded.")

CONFIDENCE_THRESHOLD = 0.78  # tune against your own eval set


def retrieve(question: str, k: int = 3, debug: bool = False):

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
    context = retrieve(question, debug=True)
    if context is None:
        return "Bilmayman."

    try:
        return ask_ollama(question, context)
    except Exception as e:
        return f"Xatolik yuz berdi: {e}"
