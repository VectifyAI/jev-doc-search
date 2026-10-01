"""Find the pages that answer a question: Jev walks a PageIndex tree, then checks each candidate page."""
import os
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, RetryPolicy, TypeSafeClient

load_dotenv()

MODEL = "jev-1.13.0"
NAV = os.environ.get("JEV_NAV", "choice")  # "choice": beam search as in TypeSafe's cookbook; "noul": per-section threshold
BEAM_WIDTH = 3     # TypeSafe's hierarchical classification cookbook uses 3
NAV_MIN = 0.2      # noul navigation only
VERIFY_MIN = 0.5   # a candidate page is kept when its noul clears this
MAX_VERIFY = 16    # candidate pages Jev reads

typesafe = TypeSafeClient(model=MODEL, retry=RetryPolicy(max_retries=5, backoff_initial=1.0, backoff_max=20.0))


def page_spans(tree):
    """node_id -> (first page, last page, last page of the text before its first child)."""
    spans = {}

    def walk(nodes):
        for n in nodes:
            kids = n.get("nodes") or []
            # a child may start mid-page, so the parent's own text can run onto that page
            own_end = kids[0]["start_index"] if kids else n["end_index"]
            spans[n["node_id"]] = (n["start_index"], n["end_index"], max(n["start_index"], own_end))
            walk(kids)

    walk(tree)
    return spans


def _nodes_by_id(tree):
    by_id = {}

    def walk(nodes):
        for n in nodes:
            by_id[n["node_id"]] = n
            walk(n.get("nodes") or [])

    walk(tree)
    return by_id


def _options(parent, children):
    """What Jev sees per child: "title. summary". A parent also offers the pages before its first child."""
    options = {c["node_id"]: f"{c['title']}. {c.get('summary') or ''}".strip() for c in children}
    if parent is not None and parent.get("summary"):
        options[f"{parent['node_id']}:intro"] = (
            f"The opening part of \"{parent['title']}\", before its subsections. {parent['summary']}")
    return options


def ask_choice(question, parent, children):
    """One request, one Choice over this menu; the distribution sums to 1."""
    options = _options(parent, children)
    if len(options) == 1:
        return {key: 1.0 for key in options}, 0
    r = typesafe.system_one(state={"question": question}, questions={"part": Choice(
        instructions="Which part of the document most likely contains the answer to the question?",
        criteria=options)})
    return dict(r.answers["part"].probabilities), r.usage.input_tokens


def navigate_choice(question, tree):
    """Beam search as in TypeSafe's hierarchical classification cookbook: expand every open path,
    keep the BEAM_WIDTH best by geometric-mean edge probability; finished paths compete too."""
    spans, by_id = page_spans(tree), _nodes_by_id(tree)
    trace, tokens = [], 0
    beam = [{"path": (), "node": None, "children": tree, "prod": 1.0, "k": 0, "score": 1.0, "span": None}]
    for _ in range(12):
        open_ = [b for b in beam if b["span"] is None]
        if not open_:
            break
        with ThreadPoolExecutor(BEAM_WIDTH) as ex:
            answers = list(ex.map(lambda b: ask_choice(question, b["node"], b["children"]), open_))
        expanded = []
        for b, (probs, used) in zip(open_, answers):
            tokens += used
            trace.append({"path": b["path"], "probs": probs})
            decision = len(probs) > 1
            for key, p in probs.items():
                prod = b["prod"] * (max(p, 1e-9) if decision else 1.0)
                k = b["k"] + decision
                step = {"path": b["path"] + (key,), "prod": prod, "k": k,
                        "score": prod ** (1 / k) if k else 1.0, "node": None, "children": None, "span": None}
                if key.endswith(":intro"):
                    start, _, own_end = spans[b["node"]["node_id"]]
                    step["span"] = (start, own_end)
                elif by_id[key].get("nodes"):
                    step["node"], step["children"] = by_id[key], by_id[key]["nodes"]
                else:
                    step["span"] = spans[key][:2]
                expanded.append(step)
        beam = sorted([b for b in beam if b["span"] is not None] + expanded, key=lambda b: -b["score"])[:BEAM_WIDTH]
    hits = [(b["score"], *b["span"], b["path"]) for b in beam if b["span"] is not None]
    return hits, trace, tokens


def ask_nouls(question, parent, children):
    """One request, one Noul per option; each option is judged on its own."""
    questions = {
        key: Noul(instructions=f"Would this part of the document contain the answer to the question? Part: {text}")
        for key, text in _options(parent, children).items()
    }
    r = typesafe.system_one(state={"question": question}, questions=questions)
    return {key: a.noul for key, a in r.answers.items()}, r.usage.input_tokens


def navigate_noul(question, tree):
    """Level by level; open every option over NAV_MIN (the best one if none clears it)."""
    spans, by_id = page_spans(tree), _nodes_by_id(tree)
    trace, tokens, hits = [], 0, []
    frontier = [(None, tree, (), 1.0, 0)]  # parent, children, path, prob product, decisions
    while frontier:
        with ThreadPoolExecutor(8) as ex:
            answers = list(ex.map(lambda f: ask_nouls(question, f[0], f[1]), frontier))
        nxt = []
        for (parent, _, path, prod, k), (probs, used) in zip(frontier, answers):
            tokens += used
            trace.append({"path": path, "probs": probs})
            for key in [key for key, p in probs.items() if p >= NAV_MIN] or [max(probs, key=probs.get)]:
                score_prod = prod * max(probs[key], 1e-9)
                score = score_prod ** (1 / (k + 1))
                if key.endswith(":intro"):
                    start, _, own_end = spans[parent["node_id"]]
                    hits.append((score, start, own_end, path + (key,)))
                elif by_id[key].get("nodes"):
                    nxt.append((by_id[key], by_id[key]["nodes"], path + (key,), score_prod, k + 1))
                else:
                    hits.append((score, *spans[key][:2], path + (key,)))
        frontier = nxt
    hits.sort(key=lambda h: -h[0])
    return hits, trace, tokens


def candidate_pages(hits, limit=MAX_VERIFY):
    """Round-robin over sections in score order, so one long section cannot fill the budget."""
    queues = [list(range(start, end + 1)) for _, start, end, _ in hits]
    pages = []
    while any(queues) and len(pages) < limit:
        for q in queues:
            if q and len(pages) < limit:
                p = q.pop(0)
                if p not in pages:
                    pages.append(p)
    return pages


def verify(question, pages_text):
    """One request per page: does this page state the answer?"""
    def ask(item):
        page, text = item
        r = typesafe.system_one(
            state={"question": question, "page": text[:60000]},
            questions={"answers": Noul(instructions="Does this page state information that answers the question?")},
        )
        return page, r.answers["answers"].noul, r.usage.input_tokens

    with ThreadPoolExecutor(MAX_VERIFY) as ex:
        scored = sorted(ex.map(ask, pages_text.items()), key=lambda s: -s[1])
    kept = [p for p, s, _ in scored if s >= VERIFY_MIN] or [p for p, _, _ in scored[:2]]
    return kept, {p: s for p, s, _ in scored}, sum(t for _, _, t in scored)


def tree_with_ranges(client, doc_id):
    """The public structure gives only start pages; a node ends where its next sibling starts."""
    def walk(nodes, end):
        out = []
        for k, n in enumerate(nodes):
            stop = nodes[k + 1]["page_index"] if k + 1 < len(nodes) else end
            out.append({"node_id": n["node_id"], "title": n["title"],
                        "summary": n.get("summary") or n.get("prefix_summary"),
                        "start_index": n["page_index"], "end_index": max(n["page_index"], stop),
                        "nodes": walk(n.get("nodes") or [], stop)})
        return out

    return walk(client.get_document_structure(doc_id), client.get_document(doc_id)["pageNum"])


def locate(client, doc_id, question):
    """Navigate the tree, then verify the candidate pages. Returns the kept pages and the route."""
    t0 = time.perf_counter()
    # local mode keeps exact page ranges behind a private call; the cloud has only the public structure
    raw = getattr(client._api, "raw_tree", None)
    tree = raw(doc_id) if raw else tree_with_ranges(client, doc_id)
    hits, trace, nav_tokens = (navigate_choice if NAV == "choice" else navigate_noul)(question, tree)
    t1 = time.perf_counter()
    pages = candidate_pages(hits)
    text = {int(c["page_index"]): c["markdown"]
            for c in client.get_page_content(doc_id, ",".join(map(str, pages)))}
    kept, page_scores, verify_tokens = verify(question, text)
    return {"pages": sorted(kept), "candidates": pages, "page_scores": page_scores, "hits": hits,
            "trace": trace, "text": text, "jev_tokens": nav_tokens + verify_tokens,
            "nav_s": t1 - t0, "verify_s": time.perf_counter() - t1}


if __name__ == "__main__":
    import sys

    from pageindex import PageIndexClient

    pdf, question = sys.argv[1], sys.argv[2]
    client = PageIndexClient(api_key=os.environ["PAGEINDEX_API_KEY"])
    doc_id = client.submit_document(pdf, wait=True)["doc_id"]
    found = locate(client, doc_id, question)
    print("pages:", found["pages"])
    for page in found["pages"]:
        print(f"\n--- p{page} ---\n{found['text'][page][:1500]}")
