"""Find the pages that answer a question by searching a PageIndex tree with Jev.

    python jev.py report.pdf "What was NVIDIA's total revenue for fiscal year 2026?"

The search runs top to bottom, in three stages:

1. Sections. From the root, one Choice per open node, over its children and its own opening pages
   (the pages before its first child, which no child covers). Beam search keeps the BEAM best paths,
   scored by the geometric mean of their step probabilities, until every path ends in a section
   with no subsections. The tree can be any depth.
2. Pages. One Choice over the pages of each of those sections, each option described by the page's
   text, as in flat page search. A section too long for one request is split into windows that fit.
3. Check. One Noul per top candidate page, with its full text. Pages at VERIFY_MIN or above are kept;
   if none is, the best 2.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import tiktoken
from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, RetryPolicy, TypeSafeClient

load_dotenv()

MODEL = "jev-1.13.0"
BEAM = 3                 # paths kept at each level, as in TypeSafe's hierarchical classification cookbook
PAGES_PER_WINDOW = 3     # best pages taken from each page-level Choice
MAX_CANDIDATES = 16      # pages checked with Noul
VERIFY_MIN = 0.5         # a page is kept when its Noul clears this
MAX_OPTIONS = 255        # Jev: options per Choice
MAX_TOKENS = 30_000      # Jev: one request fits 32k tokens; leave room for the instructions
MAX_PAGE_CHARS = 60_000  # a single page is cut to this before Jev reads it

typesafe = TypeSafeClient(model=MODEL, retry=RetryPolicy(max_retries=5, backoff_initial=1.0, backoff_max=20.0))
enc = tiktoken.get_encoding("o200k_base")


def ask_choice(question, instructions, options):
    """One Choice over {key: description}; returns ({key: probability}, input tokens)."""
    if len(options) == 1:
        return {key: 1.0 for key in options}, 0
    r = typesafe.system_one(state={"question": question}, questions={"pick": Choice(
        instructions=instructions, criteria=options)})
    return dict(r.answers["pick"].probabilities), r.usage.input_tokens or 0


# ---------- 1. sections ----------

def section_options(node, children):
    """What Jev sees at a node: each child as "title. summary", and the node's own opening pages if any.
    Returns ({key: description}, {key: (child or None, label, start, end)})."""
    options, targets = {}, {}
    for child in children:
        options[child["node_id"]] = f"{child['title']}. {child.get('summary', '')}".strip()
        targets[child["node_id"]] = (child, child["title"], child["start_index"], child["end_index"])
    if node is not None and children[0]["start_index"] > node["start_index"]:
        key = f"{node['node_id']}:opening"
        options[key] = f"The opening pages of \"{node['title']}\", before its subsections. {node.get('summary', '')}"
        targets[key] = (None, "(opening pages)", node["start_index"], children[0]["start_index"] - 1)
    return options, targets


def search_sections(question, tree, beam=BEAM):
    """Beam search down the tree. Returns the final sections as dicts with score, path, start, end."""
    paths = [{"score": 1.0, "prod": 1.0, "steps": 0, "path": [], "node": None, "children": tree, "range": None}]
    trace, tokens = [], 0
    while any(p["range"] is None for p in paths):
        open_paths = [p for p in paths if p["range"] is None]
        menus = [section_options(p["node"], p["children"]) for p in open_paths]
        with ThreadPoolExecutor(len(open_paths)) as ex:
            answers = list(ex.map(lambda m: ask_choice(
                question, "Which part of the document most likely contains the answer to the question?", m[0]), menus))
        grown = [p for p in paths if p["range"] is not None]
        for p, (options, targets), (probs, used) in zip(open_paths, menus, answers):
            tokens += used
            trace.append({"path": p["path"], "probs": probs})
            decided = len(probs) > 1
            for key, prob in probs.items():
                child, label, start, end = targets[key]
                prod = p["prod"] * (max(prob, 1e-9) if decided else 1.0)
                steps = p["steps"] + decided
                kids = (child or {}).get("nodes") or []
                grown.append({"score": prod ** (1 / steps) if steps else 1.0, "prod": prod, "steps": steps,
                              "path": p["path"] + [label],
                              "node": child, "children": kids, "range": None if kids else (start, end)})
        paths = sorted(grown, key=lambda p: -p["score"])[:beam]
    sections = [{"score": p["score"], "path": p["path"], "start": p["range"][0], "end": p["range"][1]} for p in paths]
    return sections, trace, tokens


# ---------- 2. pages ----------

def windows(pages, text):
    """Split pages into runs that fit one request: at most MAX_OPTIONS pages and MAX_TOKENS tokens."""
    out, run, used = [], [], 0
    for page in pages:
        cost = len(enc.encode(text[page][:MAX_PAGE_CHARS]))
        if run and (len(run) == MAX_OPTIONS or used + cost > MAX_TOKENS):
            out.append(run)
            run, used = [], 0
        run.append(page)
        used += cost
    return out + [run] if run else out


def search_pages(question, sections, text):
    """One Choice per window of each section's pages; returns candidates ranked by section score x page probability."""
    jobs = [(s, run) for s in sections for run in windows(range(s["start"], s["end"] + 1), text)]
    if not jobs:
        return [], 0

    def ask(job):
        section, run = job
        options = {f"p{page}": text[page][:MAX_PAGE_CHARS] for page in run}
        probs, used = ask_choice(question, "Which page contains the answer to the question?", options)
        best = sorted(probs, key=probs.get, reverse=True)[:PAGES_PER_WINDOW]
        return [(section["score"] * probs[key], int(key[1:])) for key in best], used

    with ThreadPoolExecutor(min(8, len(jobs))) as ex:
        answers = list(ex.map(ask, jobs))
    ranked, seen = [], set()
    for score, page in sorted((hit for hits, _ in answers for hit in hits), reverse=True):
        if page not in seen:
            seen.add(page)
            ranked.append(page)
    return ranked[:MAX_CANDIDATES], sum(used for _, used in answers)


# ---------- 3. check ----------

def check_pages(question, pages, text):
    """One Noul per page: does this page state the answer? Returns kept pages, scores, input tokens."""
    if not pages:
        return [], {}, 0

    def ask(page):
        r = typesafe.system_one(
            state={"question": question, "page": text[page][:MAX_PAGE_CHARS]},
            questions={"answers": Noul(instructions="Does this page state information that answers the question?")})
        return page, r.answers["answers"].noul, r.usage.input_tokens or 0

    with ThreadPoolExecutor(len(pages)) as ex:
        scored = sorted(ex.map(ask, pages), key=lambda s: -s[1])
    kept = [p for p, s, _ in scored if s >= VERIFY_MIN] or [p for p, _, _ in scored[:2]]
    return kept, {p: s for p, s, _ in scored}, sum(t for _, _, t in scored)


def locate(client, doc_id, question, beam=BEAM):
    """Sections, then pages, then a Noul check. Returns the kept pages and how the search got there."""
    t0 = time.perf_counter()
    tree = client.get_document_structure(doc_id)  # pageindex>=0.2.21: start_index, end_index, summary per node
    if not tree:
        raise ValueError(f"document {doc_id} has no tree")
    sections, trace, section_tokens = search_sections(question, tree, beam)
    wanted = sorted({p for s in sections for p in range(s["start"], s["end"] + 1)})
    text = {int(c["page_index"]): c["markdown"] or "" for c in client.get_page_content(doc_id, ",".join(map(str, wanted)))}
    candidates, page_tokens = search_pages(question, sections, text)
    t1 = time.perf_counter()
    kept, page_scores, check_tokens = check_pages(question, candidates, text)
    return {"pages": sorted(kept), "candidates": candidates, "page_scores": page_scores, "sections": sections,
            "trace": trace, "text": text, "jev_tokens": section_tokens + page_tokens + check_tokens,
            "nav_s": t1 - t0, "verify_s": time.perf_counter() - t1}


if __name__ == "__main__":
    import sys

    from pageindex import PageIndexClient

    pdf, question = sys.argv[1], sys.argv[2]
    client = PageIndexClient(api_key=os.environ["PAGEINDEX_API_KEY"])
    doc_id = client.submit_document(pdf, wait=True)["doc_id"]
    found = locate(client, doc_id, question)
    for s in found["sections"]:
        print(f"{s['score']:.2f}  p{s['start']}-{s['end']}  {' > '.join(s['path'])}")
    print("\npages:", found["pages"])
    for page in found["pages"]:
        print(f"\n--- p{page} ---\n{found['text'][page][:1500]}")
