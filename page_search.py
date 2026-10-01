"""Page search without PageIndex: one Choice whose options are the document's pages.

    python page_search.py handbook.pdf "How many vacation days do new employees get?"

With no arguments it runs the benchmark baseline instead (the whole document in `state`, one Choice
over its page ids). Both run only on documents that fit in one request."""
import ast
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from pageindex import PageIndexClient
from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient

load_dotenv()

HERE = Path(__file__).parent
MAX_STATE_TOKENS = 30000  # Jev: state plus the longest question within 32k
MAX_OPTIONS = 255         # Jev: options per Choice

typesafe = TypeSafeClient(model="jev-1.13.0", retry=RetryPolicy(max_retries=5, backoff_initial=1.0, backoff_max=20.0))
enc = tiktoken.get_encoding("o200k_base")


def page_search(pages, question):
    """One Choice over the pages, each option described by its text; returns the most likely page number."""
    if len(pages) > MAX_OPTIONS:
        raise ValueError(f"{len(pages)} pages, but a Choice takes at most {MAX_OPTIONS} options; use tree_search.py")
    tokens = len(enc.encode(question)) + sum(len(enc.encode(text)) for text in pages)
    if tokens > MAX_STATE_TOKENS:
        raise ValueError(f"about {tokens} tokens, but one request fits {MAX_STATE_TOKENS}; use tree_search.py")
    r = typesafe.system_one(state={"question": question}, questions={"page": Choice(
        instructions="Which page contains the answer to the question?",
        criteria={f"p{i}": text for i, text in enumerate(pages, 1)})})
    return int(r.answers["page"].choice[1:])


def main():
    client = PageIndexClient(api_key=os.environ["PAGEINDEX_API_KEY"])
    ids = json.loads((HERE / "doc_ids.json").read_text())

    def ask(row):
        doc_id = ids[row["doc_id"]]
        n = client.get_document(doc_id)["pageNum"]
        text = {int(c["page_index"]): c["markdown"] for c in client.get_page_content(doc_id, f"1-{n}")}
        document = "\n".join(f"p{p}| {text[p]}" for p in sorted(text))
        if len(enc.encode(document)) > MAX_STATE_TOKENS:
            return None
        r = typesafe.system_one(state={"document": document}, questions={"where": Choice(
            instructions=f'Which page of `document` contains the answer to: "{row["question"]}"?',
            criteria={f"p{p}": None for p in sorted(text)})})
        probs = r.answers["where"].probabilities
        return {"doc_id": row["doc_id"], "question": row["question"], "pages": n,
                "evidence": ast.literal_eval(row["evidence_pages"]),
                "ranked": [int(k[1:]) for k in sorted(probs, key=probs.get, reverse=True)][:5],
                "top_probs": sorted(probs.values(), reverse=True)[:3], "jev_tokens": r.usage.input_tokens}

    rows = [r for r in json.load(open(HERE / "bench" / "questions.json")) if ids.get(r["doc_id"])]
    with ThreadPoolExecutor(4) as ex:
        out = [r for r in ex.map(ask, rows) if r]
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "flat_choice.json").write_text(json.dumps(out, indent=2))
    top1 = sum(bool(set(r["evidence"]) & set(r["ranked"][:1])) for r in out)
    top3 = sum(bool(set(r["evidence"]) & set(r["ranked"][:3])) for r in out)
    print(f"{len(out)} questions fit in one request | answer page ranked 1st: {top1} | in top 3: {top3}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        from pypdf import PdfReader

        pdf, question = sys.argv[1], sys.argv[2]
        pages = [page.extract_text() or "" for page in PdfReader(pdf).pages]
        best = page_search(pages, question)
        print(f"p{best}\n\n{pages[best - 1][:1500]}")
    else:
        main()
