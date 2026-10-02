"""Page search without PageIndex: one Choice whose options are the document's pages.

    python page_search.py handbook.pdf "How many vacation days do new employees get?"

It runs only on documents that fit in one request."""
import tiktoken
from dotenv import load_dotenv
from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient

load_dotenv()

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


if __name__ == "__main__":
    import sys

    from pypdf import PdfReader

    pdf, question = sys.argv[1], sys.argv[2]
    pages = [page.extract_text() or "" for page in PdfReader(pdf).pages]
    best = page_search(pages, question)
    print(f"p{best}\n\n{pages[best - 1][:1500]}")
