"""Baseline without PageIndex, as in TypeSafe's line-by-line search: the whole document in `state`,
one Choice over its page ids. Runs only on documents that fit in one request."""
import ast
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tiktoken
from pageindex import PageIndexClient
from typesafe_sdk import Choice

import locate

HERE = Path(__file__).parent
MAX_STATE_TOKENS = 30000  # Jev: state plus the longest question within 32k

enc = tiktoken.get_encoding("o200k_base")


def main():
    client = PageIndexClient(storage_path=str(HERE / "store"))
    ids = json.loads((HERE / "doc_ids.json").read_text())

    def ask(row):
        doc_id = ids[row["doc_id"]]
        n = client.get_document(doc_id)["pageNum"]
        text = {int(c["page_index"]): c["markdown"] for c in client.get_page_content(doc_id, f"1-{n}")}
        document = "\n".join(f"p{p}| {text[p]}" for p in sorted(text))
        if len(enc.encode(document)) > MAX_STATE_TOKENS:
            return None
        r = locate.jev.system_one(state={"document": document}, questions={"where": Choice(
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
    main()
