"""Build local PageIndex trees (flash) for the benchmark PDFs; doc ids go to doc_ids.json."""
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from pageindex import PageIndexClient

load_dotenv()
HERE = Path(__file__).parent


def main():
    client = PageIndexClient(storage_path=str(HERE / ".pageindex"))
    cache_path = HERE / "doc_ids.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for name in sorted({q["doc_id"] for q in json.load(open(HERE / "bench" / "questions.json"))}):
        if name in cache:
            continue
        t = time.perf_counter()
        try:
            cache[name] = client.submit_document(str(HERE / "bench" / "documents" / name), wait=True)["doc_id"]
            print(f"ok   {name} {time.perf_counter() - t:.0f}s", flush=True)
        except Exception as e:
            cache[name] = None
            print(f"FAIL {name}: {e}", flush=True)
        cache_path.write_text(json.dumps(cache, indent=2))


if __name__ == "__main__":
    main()
