# jev-pageindex

[PageIndex](https://github.com/VectifyAI/PageIndex) as the index structure for [Jev](https://docs.typesafe.ai): long-document retrieval as a chain of small classifications.

Jev walks down a PageIndex tree with `Choice` questions, then checks each candidate page with a `Noul`. On [PageIndex-OSS-Benchmark](https://github.com/VectifyAI/PageIndex-OSS-Benchmark), one LLM call answering from the pages Jev keeps gets 51 of 62 questions right (82%), at $0.0009 per question and a median 3.8 s. The PageIndex agent gets 57 (92%) at $0.0038 and 6.4 s.

## Why Jev needs an index

Jev can treat retrieval as classification: put the document in `state`, ask one `Choice` over its page ids ("which page answers this question?"), and read the ranking. TypeSafe's [line-by-line search](https://docs.typesafe.ai/cookbooks/semantic_find) cookbook does exactly this over lines. On a long document, it breaks down:

| Case | Example from the benchmark | Why one flat `Choice` fails |
| --- | --- | --- |
| Over 255 pages | MMDetection 2.18.0 docs: 468 pages, ≈249k tokens | A `Choice` takes at most 255 options, and the text is about 8 times what fits in one request |
| Under 255 pages, but too long for one request | Best Buy FY2023 10-K: 75 pages, ≈66k tokens | `state` plus the longest question must fit in 32k tokens, about 40 pages |
| Fits, but the answer page is not ranked first | Xiaomi Mi phone user guide: 37 pages, ≈7k tokens | Pages 19 to 23 all discuss SMS, so the probability spreads across them |

In the Xiaomi case, the question is "How many steps are needed for editing an SMS?" and the answer is on page 22. Over two runs, flat `Choice` ranked a wrong page first (page 23 at 0.29, then page 19 at 0.32) and page 22 fifth, then third.

In the benchmark, 23 of 34 documents fit in one request (the largest is 44 pages); none of the six annual reports (72 to 198 pages) do. The third case is uncommon: on the 39 questions whose document fits, flat `Choice` ranks the answer page first for 29 and in its top 3 for 35.

TypeSafe's own docs point the same way. Past 255 lines, line-by-line search runs "in two passes: one Choice question picks a window of lines, and a second ranks the lines inside it". And the [Jev 1.13 failure modes](https://docs.typesafe.ai/model-jaggedness/jev-1.13) warn: "Accuracy falls as the state grows with content unrelated to the decision… retrieve and filter in code first."

To pick a window of a document that does not fit in `state`, Jev needs a short description of each window. That is an index. PageIndex builds one from the document's own structure: a tree of sections, each with a title, a summary, and a page range.

## How it works

1. **Index.** PageIndex turns the PDF into a section tree, locally.
2. **Navigate.** Jev descends the tree with one `Choice` per section it opens: which part of the document most likely contains the answer? Each option reads `"title. summary"`, and the section's opening text, before its first subsection, is an option too. Beam search keeps the 3 best paths, scored by the geometric mean of their step probabilities, as in TypeSafe's [hierarchical classification](https://docs.typesafe.ai/cookbooks/hierarchical_classification) cookbook.
3. **Candidates.** The pages of the 3 winning sections, taken from each in turn so that one long section cannot fill the list, up to 16.
4. **Verify.** One `Noul` per candidate page, with the page's full text: "Does this page state information that answers the question?" Pages at 0.5 or above are kept; if none is, the top 2.

On the Xiaomi question, Jev picks "Editing An SMS" at 0.95 out of 24 top-level sections. That lands on page 22 in one step, for about 6k Jev tokens.

```python
from pageindex import PageIndexClient
import jev

client = PageIndexClient()  # local mode: trees are saved in ./.pageindex and built with OPENAI_API_KEY
doc_id = client.submit_document("BESTBUY_2023_10K.pdf", wait=True)["doc_id"]
found = jev.locate(client, doc_id, "What goodwill does Best Buy have for the fiscal year ending January 28, 2023?")
print(found["pages"], found["jev_tokens"])  # the pages to hand to your LLM, and Jev's input tokens
```

`jev.py` holds the whole method.

## Results

[PageIndex-OSS-Benchmark](https://github.com/VectifyAI/PageIndex-OSS-Benchmark): 62 lookup questions over 34 PDFs (1,945 pages). Trees are built locally with [PageIndex Flash](https://pageindex.ai/blog/pageindex-flash). Every method answers with `gpt-5.6-luna` at reasoning effort `none`, and [MMLongBench-Doc-V2](https://github.com/VectifyAI/MMLongBench-Doc-V2)'s judge scores the answers. Costs include Jev (`jev-1.13.0`, $0.042 per million input tokens).

| Method | Correct | Median latency | Cost per question |
| --- | --- | --- | --- |
| **A.** PageIndex agent (`client.chat`), no Jev | 57 (92%) | 6.4 s | $0.0038 |
| **B.** Jev locates, one LLM call answers | 51 (82%) | 3.8 s | $0.0009 |

- B is pure PageIndex + Jev retrieval: no LLM helps find the pages, and nothing searches again after a miss.
- Navigating with one `Noul` per section instead of a `Choice` over them was worse: B then answers 43. TypeSafe's failure-mode notes explain why: a `Choice` is relative, while each `Noul` is absolute "and can be low for all of them".
- Where the document fits in one request, the tree finds the answer page as often as flat `Choice`: on those 39 questions, the page is among the tree's candidates (median 6 pages) for 37, and in flat `Choice`'s top 5 for 37.

Per-question outputs are in `results/`.

## Limitations

- Each number comes from one run; the same setup moves by 2 to 3 questions between runs.
- The benchmark leans short: 39 of 62 questions have a document that fits in one request, and only 2 are on a document over 255 pages.
- Some benchmark page labels are off. In the NYU housing guide, three answers sit 2 pages after their labeled page. This undercounts page hits for every method, but not answer accuracy.
- Flash trees have defects on some annual reports, such as pages 1 to 99 of the Activision Blizzard 10-K in one undivided section.
- The verification threshold (0.5) was picked on the first few questions.
- `jev.py` reads page ranges through a private SDK call, because the public `get_document_structure()` returns only start pages in local mode.

## Reproduce

```bash
git clone https://github.com/VectifyAI/PageIndex-OSS-Benchmark bench
git clone https://github.com/VectifyAI/MMLongBench-Doc-V2 mmlb
pip install -r requirements.txt
cp .env.example .env              # OPENAI_API_KEY, TYPESAFE_API_KEY

python index_docs.py              # 34 local trees, about $1.6 of gpt-5.6-luna
python run.py A --out results/A.json
python run.py B --out results/B.choice.json
JEV_NAV=noul python run.py B --out results/B.noul.json
python flat.py                    # flat Choice baseline
python summarize.py               # judge and tabulate
```
