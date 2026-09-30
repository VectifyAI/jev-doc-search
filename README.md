# jev-pageindex

PageIndex as the index structure for [Jev](https://docs.typesafe.ai): long-document retrieval as a chain of small classifications.

## Why Jev needs an index

With Jev, retrieval is a classification problem: put the document in `state`, ask one `Choice` over its page ids ("which page answers this question?"), and read the ranking. TypeSafe's [line-by-line search](https://docs.typesafe.ai/cookbooks/semantic_find) cookbook does exactly this over lines. It stops working when the document is long:

| Case | Example (from PageIndex-OSS-Benchmark) | Why flat classification fails |
| --- | --- | --- |
| More than 255 pages | MMDetection 2.18.0 docs: 468 pages, ≈249k tokens | A `Choice` takes at most 255 options, and the text is 8× one request |
| Fewer than 255 pages, but too long for one request | Best Buy FY2023 10-K: 75 pages, ≈66k tokens | `state` plus the longest question must fit in 32k tokens, which is about 40 pages |
| Fits, but the flat choice misses | Xiaomi Mi phone user guide: 37 pages, ≈7k tokens | "How many steps are needed for editing an SMS?" The answer is on page 22. Flat `Choice` ranks page 23 first (0.29) and page 22 fifth; pages 19 to 23 all discuss SMS |

In the benchmark, 23 of 34 documents fit in one request (the largest is 44 pages). None of the six annual reports (72 to 198 pages) do. The third case is rare: on the 39 questions whose document fits, flat `Choice` puts the answer page first 29 times and in the top 3 35 times.

TypeSafe's own docs point the same way. Past 255 lines, line-by-line search runs "in two passes: one Choice question picks a window of lines, and a second ranks the lines inside it". And from the [Jev 1.13 failure modes](https://docs.typesafe.ai/model-jaggedness/jev-1.13): "Accuracy falls as the state grows with content unrelated to the decision… retrieve and filter in code first."

A document that does not fit in `state` needs a short description per window before Jev can pick one. That is an index. PageIndex builds it from the document's own structure: a tree of sections, each with a title, a summary, and a page range.

## How it works

1. **Tree.** PageIndex indexes the PDF locally into a section tree.
2. **Navigate.** Each node's children become one `Choice`; every option reads `"title. summary"`, and a parent also offers "the opening part, before its subsections". Beam search keeps the 3 best paths by geometric-mean edge probability, the method of TypeSafe's [hierarchical classification](https://docs.typesafe.ai/cookbooks/hierarchical_classification) cookbook.
3. **Candidates.** Pages from the 3 winning sections, taken round-robin, at most 16.
4. **Verify.** One `Noul` per candidate page with its full text: "Does this page state information that answers the question?" Pages at 0.5 or above are kept; if none is, the top 2.

For the Xiaomi example, Jev picks the section "Editing An SMS" at 0.95 among 24 top-level sections, which lands on page 22 in one step, with about 6k Jev tokens.

```python
from pageindex import PageIndexClient
import locate

client = PageIndexClient(storage_path="store")  # local mode; OPENAI_API_KEY builds the tree
doc_id = client.submit_document("BESTBUY_2023_10K.pdf", wait=True)["doc_id"]
found = locate.locate(client, doc_id, "What goodwill does Best Buy have for the fiscal year ending January 28, 2023?")
print(found["pages"], found["jev_tokens"])      # pages to hand to your LLM
```

`locate.py` holds the whole method.

## Results

[PageIndex-OSS-Benchmark](https://github.com/VectifyAI/PageIndex-OSS-Benchmark): 62 questions over 34 PDFs (1,945 pages). Trees built locally with flash. Every method answers with `gpt-5.6-luna` at reasoning effort `none`, and [MMLongBench-Doc-V2](https://github.com/VectifyAI/MMLongBench-Doc-V2)'s judge scores the answers. Costs include Jev (`jev-1.13.0`, $0.042 per million input tokens).

| Method | Correct (run 1 / run 2) | Median latency | Cost per question |
| --- | --- | --- | --- |
| **A.** PageIndex agent (`client.chat`), no Jev | 57 / 56 (91%) | 6.4 / 6.0 s | $0.0038 / $0.0033 |
| **B.** Jev locates, one LLM call answers | 51 (82%, one run) | 3.8 s | $0.0009 |
| **C.** PageIndex agent with a Jev `locate_pages` tool that returns the pages' text | 55 / 56 (90%) | 5.6 / 5.1 s | $0.0015 |

- B is the pure "PageIndex + Jev" retrieval: no LLM touches retrieval, and it cannot search again after a miss.
- C keeps the agent's tools. In over 80% of questions it answers straight from the pages Jev returns; otherwise it reads more pages itself. It matches A within run-to-run noise at under half the cost.
- Navigating with one `Noul` per section instead of a `Choice` was worse: B 43, C 56 / 53. TypeSafe's failure-mode notes explain why: a `Choice` is relative, while each `Noul` is absolute "and can be low for all of them".
- On the 39 questions whose document fits in one request, the answer page is in flat `Choice`'s top 3 for 35 and in the tree's candidates (about 6 pages) for 37.

Every row's output is in `results/`.

## Limitations

- Two runs per method at most; the same setup moves by 2 to 3 questions between runs.
- The benchmark leans short: 39 of 62 questions have a document that fits in one request, and only 2 are on a document over 255 pages.
- Some benchmark page labels are off. In the NYU housing guide, three answers sit 2 pages after their label, which undercounts page hits for every method but not answer accuracy.
- Local flash trees have defects on some annual reports, such as pages 1 to 99 of the Activision Blizzard 10-K as one undivided section.
- The verification threshold was picked on the first few questions.
- `locate.py` reads page ranges through a private SDK call, because the public `get_document_structure()` returns only start pages in local mode.

## Reproduce

```bash
git clone https://github.com/VectifyAI/PageIndex-OSS-Benchmark bench
git clone https://github.com/VectifyAI/MMLongBench-Doc-V2 mmlb
pip install -r requirements.txt
cp .env.example .env              # OPENAI_API_KEY, TYPESAFE_API_KEY

python index_docs.py              # 34 local trees, about $1.6 of gpt-5.6-luna
python run.py A --out results/A.run1.json
python run.py B --out results/B.choice.json
python run.py C --out results/C.choice.run1.json
JEV_NAV=noul python run.py B --out results/B.noul.json
python flat.py                    # flat Choice baseline
python summarize.py               # judge and tabulate
```
