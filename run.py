"""Answer the benchmark questions with one method, timed and priced.

    python run.py A --out results/A.json          # PageIndex agent
    python run.py B --out results/B.choice.json   # Jev locate + one LLM call
    JEV_NAV=noul python run.py B --out results/B.noul.json
"""
import argparse
import ast
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pageindex import PageIndexClient

import jev

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "bench"))
import pi_bench  # noqa: E402  (pricing and envelope helpers from PageIndex-OSS-Benchmark)

JEV_PRICE = 0.042 / 1e6  # USD per input token; output is free

ANSWER_PROMPT = """Answer the question using only the document pages below. Be concise.
If the pages do not contain the answer, say so.

Question: {question}

{pages}"""


def pages_read(items):
    """Pages the agent opened with get_page_content."""
    pages = set()
    for i in items:
        if i.get("type") == "function_call" and i.get("name") == "get_page_content":
            for part in json.loads(i.get("arguments") or "{}").get("pages", "").replace(" ", "").split(","):
                if "-" in part:
                    a, b = part.split("-")
                    pages.update(range(int(a), int(b) + 1))
                elif part:
                    pages.add(int(part))
    return pages


def arm_a(client, doc_id, row, model, effort):
    t = time.perf_counter()
    env = client.chat(row["question"], doc_id=doc_id, protocol="responses", model=model, reasoning_effort=effort)
    return {"response": pi_bench.answer_text(env), "latency_s": time.perf_counter() - t,
            "llm_cost": pi_bench.price(model, env["usage"]), "jev_cost": 0.0,
            "tool_calls": pi_bench.tool_calls(env), "pages": sorted(pages_read(env["items"]))}


def arm_b(client, doc_id, row, model, effort):
    import openai

    t = time.perf_counter()
    found = jev.locate(client, doc_id, row["question"])
    pages = "\n\n".join(f"--- Page {p} ---\n{found['text'][p]}" for p in found["pages"])
    r = openai.OpenAI().responses.create(model=model, reasoning={"effort": effort},
                                         input=ANSWER_PROMPT.format(question=row["question"], pages=pages))
    usage = {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens,
             "input_tokens_details": {"cached_tokens": r.usage.input_tokens_details.cached_tokens or 0}}
    return {"response": r.output_text, "latency_s": time.perf_counter() - t,
            "llm_cost": pi_bench.price(model, usage), "jev_cost": found["jev_tokens"] * JEV_PRICE,
            "jev_tokens": found["jev_tokens"], "nav_s": found["nav_s"], "verify_s": found["verify_s"],
            "candidates": found["candidates"], "pages": found["pages"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("arm", choices=["A", "B"])
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--effort", default="none")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids = json.loads((HERE / "doc_ids.json").read_text())
    rows = [r for r in json.load(open(HERE / "bench" / "questions.json")) if ids.get(r["doc_id"])]
    client = PageIndexClient(storage_path=str(HERE / ".pageindex"))
    fn = {"A": arm_a, "B": arm_b}[args.arm]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["doc_id"], r["question"]): r for r in json.loads(out.read_text())
            if not r.get("error")} if out.exists() else {}

    def one(row):
        if (row["doc_id"], row["question"]) in done:
            return done[(row["doc_id"], row["question"])]
        try:
            res = fn(client, ids[row["doc_id"]], row, args.model, args.effort)
        except Exception as e:
            res = {"response": "", "error": f"{type(e).__name__}: {e}"}
        res["page_hit"] = bool(set(ast.literal_eval(row["evidence_pages"])) & set(res.get("pages", [])))
        print(f"{args.arm} {res.get('latency_s', 0):5.1f}s hit={res['page_hit']!s:<5} {row['question'][:60]}", flush=True)
        return {**row, **res, "arm": args.arm, "nav": jev.NAV, "chat_model": args.model, "effort": args.effort}

    # B runs one question at a time so each latency is measured alone
    with ThreadPoolExecutor(4 if args.arm == "A" else 1) as ex:
        preds = list(ex.map(one, rows))
    out.write_text(json.dumps(preds, indent=2, ensure_ascii=False))
    print(f"wrote {out} ({len(preds)} rows)")


if __name__ == "__main__":
    main()
