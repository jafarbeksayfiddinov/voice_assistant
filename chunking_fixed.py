from pypdf import PdfReader
import re
import unicodedata


def normalize_text(s: str) -> str:
    """Normalize apostrophe/diacritic variants so 'o'zim', 'oʻzim', 'o‘zim'
    all match. Run this on BOTH the corpus and the incoming user query."""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("ʻ", "'").replace("‘", "'").replace("’", "'")
    return s


def get_chunks(pdf_name) -> list[str]:
    """Original record-based chunking, unchanged: one chunk per credit record."""
    reader = PdfReader(pdf_name)

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    chunks = [
        chunk.strip()
        for chunk in re.split(r"(?=1\.\s*Kreditning turi)", text)
        if chunk.strip().startswith("1. Kreditning turi")
    ]
    return chunks


def extract_fields(chunk: str) -> dict:
    """Pull out only the fields that actually differ between records.
    Everything else in the raw chunk (legal boilerplate, field labels,
    the disclaimer footer) is identical across ALL records and only
    adds noise to an embedding."""

    def grab(pattern):
        m = re.search(pattern, chunk, re.S)
        return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""

    return {
        "turi": grab(r"1\.\s*Kreditning turi\s*(.*?)\s*2\.\s*Kreditning maqsadi"),
        "miqdor": grab(r"3\.\s*Kreditning miqdori\s*(.*?)\s*4\.\s*Kreditdan"),
        "muddat": grab(r"\(oyda\)\s*(.*?)\s*5\.\s*Kreditdan"),
        "foiz": grab(r"summa\s*\n?\s*(\d+(?:[.,]\d+)?%)"),
    }


def get_records(pdf_name) -> list[dict]:
    """Returns one dict per credit record:
    - full: the raw chunk (use this as LLM context, it has every field)
    - summary: a short, denoised string (use THIS for embedding/retrieval)
    - turi: the credit type name alone (useful for keyword routing / debugging)
    """
    records = []
    seen_summaries = set()
    for chunk in get_chunks(pdf_name):
        fields = extract_fields(chunk)
        summary = (
            f"Kredit turi: {fields['turi']}. "
            f"Miqdori: {fields['miqdor']}. "
            f"Muddati: {fields['muddat']}. "
            f"Foiz stavkasi: {fields['foiz']}."
        )
        key = normalize_text(summary).lower()
        if key in seen_summaries:
            continue  # drop exact-duplicate records (e.g. the two identical AutoLoan entries)
        seen_summaries.add(key)
        records.append({
            "full": chunk,
            "summary": summary,
            "turi": fields["turi"],
        })
    return records


if __name__ == "__main__":
    recs = get_records("info_list-2-merged.pdf")
    print(f"Total unique records after dedup: {len(recs)}\n")
    for r in recs:
        print(r["summary"])
