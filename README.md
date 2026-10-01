# Long-document search with Jev and PageIndex

**Find the page that answers a question in a 300-page report with two Jev `Choice` calls. No embeddings needed!**

[Jev](https://docs.typesafe.ai) answers multiple-choice questions: give it the options, and it returns a probability for each. "Which page answers this question?" is one of them, and it works well, until the document outgrows what Jev can read at once. [PageIndex](https://github.com/VectifyAI/PageIndex) removes that limit by turning the document into a hierarchical tree representation. Jev picks a node, then one of its children, and so on down the tree, choosing among a handful of options each time, however long the document.

## Page search with Jev's `Choice`

Page search can be framed as a multiple-choice question. The question is "which page answers this?", and the options are the pages themselves: each page is one option, described by its own text. Jev reads every option and returns a probability for each page; the most likely page is the answer.

<img src="assets/page-search.gif" width="900" alt="Each page is one option of a Choice; Jev returns a probability for every page">

To try it on a PDF of your own, run [flat.py](flat.py):

```bash
pip install -r requirements.txt
export TYPESAFE_API_KEY="..."
python flat.py handbook.pdf "How many vacation days do new employees get?"
```

It prints the most likely page and its text. The core of it is a single call:

```python
from pypdf import PdfReader
from typesafe_sdk import Choice, TypeSafeClient

typesafe = TypeSafeClient(model="jev-1.13.0")

pages = [page.extract_text() for page in PdfReader("handbook.pdf").pages]
question = "How many vacation days do new employees get?"

r = typesafe.system_one(
    state={"question": question},
    questions={"page": Choice(
        instructions="Which page contains the answer to the question?",
        criteria={f"p{i}": text for i, text in enumerate(pages, 1)},
    )},
)
print(r.answers["page"].choice)  # 'p7', the most likely page
```

In the code:

- **`state`** is what Jev reads before deciding. Here it is just the question.
- **`Choice`** is the decision. Its options go in `criteria`: one key per page (`p1`, `p2`, …), each described by that page's text.
- **The answer** names the most likely page in `choice`, with every page's probability in `probabilities`, summing to 1. The chosen page goes to your LLM.

## Two challenges of scaling to long documents

Flat page search hits two limits, one after the other:

1. **Too many tokens.** Every page's text goes into the request as an option, and the whole request must fit in 32k tokens. That runs out after a few dozen pages. NVIDIA's [10-K for fiscal 2026](https://d18rn0p25nwr6d.cloudfront.net/CIK-0001045810/e361e58a-7483-44f5-bc62-a9080ae6ec72.pdf) has only 93 pages, but its text is about 76k tokens, more than twice what fits.
2. **Too many options.** A `Choice` takes at most 255 options, so one option per page stops at 255 pages. Some annual reports are longer than that: Citigroup's [10-K for 2025](https://www.citigroup.com/rcs/citigpa/storage/public/citi-2025-10-k-2-20-26.pdf) has 318 pages.

[PageIndex](https://github.com/VectifyAI/PageIndex) solves both at once. It turns the flat choice over pages into a tree: the document splits into sections, each with a title and a short summary, and each section into its pages. Jev then makes one small choice per level instead of one huge one, so every `Choice` has only a handful of options and fewer tokens.

## Scaling Jev to long documents with PageIndex

It takes two steps: PageIndex builds the tree, and Jev searches it instead of the pages.

### 1. Generate a tree representation of the document with PageIndex

PageIndex builds the tree from the document's own structure. The root is the whole document, its children are sections, and each section has a title, a summary, and a page range, down to the pages.

<img src="assets/tree-index.gif" width="990" alt="PageIndex turns a 300-page document into a three-level tree: the document, its sections, their pages">

```python
from pageindex import PageIndexClient

pageindex = PageIndexClient(api_key="YOUR_PAGEINDEX_API_KEY")

doc_id = pageindex.submit_document("NVIDIA_2026_10K.pdf", wait=True)["doc_id"]
tree = pageindex.get_document_structure(doc_id)
```

`tree` is a list of nodes, each with its `title`, its page range (`start_index` to `end_index`), a `summary` of those pages, and its children in `nodes`. For the NVIDIA 10-K it looks like this (abridged; built with PageIndex Flash):

```jsonc
[
  {
    "title": "Preface",
    "node_id": "0000",
    "start_index": 1,
    "end_index": 48,
    "summary": "The Preface introduces NVIDIA’s fiscal 2026 Form 10-K…",
    "nodes": [
      {
        "title": "Our Company",
        "node_id": "0004",
        "start_index": 4,
        "end_index": 5,
        "summary": "This section of Part I, Item 1 describes…"
      }
      // … 70 more
    ]
  }
  // … 17 more
]
```

### 2. Tree search with Jev's `Choice`

Jev starts at the root and asks one `Choice` per level: which of these children most likely contains the answer? Each option reads `"title. summary"`. The chosen section's children become the next menu, until the search reaches pages.

<img src="assets/tree-search.gif" width="990" alt="Jev searches the tree with one Choice per level: a section, then a page">

Each `Choice` sees a handful of titles and summaries instead of the whole document, so neither limit applies: the 318-page Citigroup 10-K becomes a few short menus, one per level.

Continuing from the tree above, the search takes two steps.

**First, pick a section.** One `Choice` over the top-level sections, each described by its title and summary.

```python
from pageindex.utils import get_node
from typesafe_sdk import Choice, TypeSafeClient

typesafe = TypeSafeClient(model="jev-1.13.0")
question = "What was NVIDIA's total revenue for fiscal year 2026?"

sections = {n["node_id"]: f"{n['title']}. {n.get('summary', '')}" for n in tree}

r = typesafe.system_one(
    state={"question": question},
    questions={"section": Choice(
        instructions="Which section most likely contains the answer to the question?",
        criteria=sections,
    )},
)
section = get_node(tree, r.answers["section"].choice)  # choice is the picked node_id, e.g. "0003"
```

**Then, pick a page inside it.** One `Choice` over the pages of that section, `start_index` to `end_index`, as in flat page search.

```python
pages = pageindex.get_page_content(doc_id, f"{section['start_index']}-{section['end_index']}")
r = typesafe.system_one(
    state={"question": question},
    questions={"page": Choice(
        instructions="Which page contains the answer to the question?",
        criteria={f"p{p['page_index']}": p["markdown"] for p in pages},
    )},
)
print(r.answers["page"].choice)  # the page to hand to your LLM
```

## Going further: deeper trees, top-K search, and checking with `Noul`

The two-step search above is the simplest version. Three changes make it general and sturdier.

**Deeper trees.** A real tree has more than two levels: sections have subsections, which can have their own. The search is the same step repeated: one `Choice` over the children of the section just picked, until a section has no subsections, then one over its pages. Each level keeps the menu short, however long the document. The tree can go one level further, below pages: once a page is picked, one more `Choice` over its lines finds the exact line, as in TypeSafe's [line-by-line search](https://docs.typesafe.ai/cookbooks/semantic_find) cookbook.

**Top-K search.** Taking only the most likely option at each step is brittle: if the right section or page ranks second, it is lost. Instead, keep the top K: the K most likely keys of `probabilities`, not just `choice`. In the tree this is beam search: keep the K best paths at each level, scored by the geometric mean of their step probabilities, as in TypeSafe's [hierarchical classification](https://docs.typesafe.ai/cookbooks/hierarchical_classification) cookbook.

**Checking with `Noul`.** A `Choice` is relative: its probabilities sum to 1, so it always names a winner, even when no option answers the question. A `Noul` is a yes/no question with its own probability, so each candidate page can be judged on its own, with its full text in `state`, and kept or dropped by a threshold:

```python
from typesafe_sdk import Noul

candidates = [32, 33, 41]  # page numbers from the top K, e.g. the pages of the beam's sections
kept = []
for page in pageindex.get_page_content(doc_id, ",".join(map(str, candidates))):
    r = typesafe.system_one(
        state={"question": question, "page": page["markdown"]},
        questions={"answers": Noul(
            instructions="Does this page state information that answers the question?",
        )},
    )
    if r.answers["answers"].noul >= 0.5:
        kept.append(page["page_index"])  # pages to hand to your LLM
```

[jev.py](jev.py) puts all three together. A beam of 3 goes down a tree of any depth, with each section's opening pages, before its first subsection, as an option too. A `Choice` over the pages of each of the 3 sections it ends in then picks the candidates; a section too long for one request is split into windows that fit. Finally, one `Noul` checks each of up to 16 candidates: pages at 0.5 or above are kept, or the best 2 if none is. Run it on a PDF and a question:

```bash
pip install -r requirements.txt
export TYPESAFE_API_KEY="..."
export PAGEINDEX_API_KEY="..."
python jev.py NVIDIA_2026_10K.pdf "What was NVIDIA's total revenue for fiscal year 2026?"
```

It prints the sections the search ended in, then the pages it kept and their text.
