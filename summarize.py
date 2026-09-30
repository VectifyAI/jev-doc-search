"""Judge each results/*.json in place with MMLongBench-Doc-V2's judge, then print one row per file."""
import json
import statistics
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # the judge subprocess inherits OPENAI_API_KEY
HERE = Path(__file__).parent

print(f"{'file':<22}{'n':>4}{'correct':>13}{'page hit':>11}{'median s':>10}{'$/q':>9}{'Jev $/q':>10}")
for f in sorted((HERE / "results").glob("*.json")):
    preds = json.loads(f.read_text())
    if not preds or "response" not in preds[0]:
        continue  # flat_choice.json: page ranks only, no answers
    subprocess.run([sys.executable, "-m", "eval.judge", str(f.resolve()), "--out", str(f.resolve())],
                   cwd=HERE / "mmlb", check=True, capture_output=True)
    preds = json.loads(f.read_text())
    ok = [p for p in preds if not p.get("error")]
    n, correct = len(preds), sum(p["llm_judge"]["equivalent"] for p in preds)
    print(f"{f.stem:<22}{n:>4}{correct:>6} {correct / n:>6.1%}{sum(p['page_hit'] for p in preds):>11}"
          f"{statistics.median(p['latency_s'] for p in ok):>10.1f}"
          f"{statistics.mean(p['llm_cost'] + p['jev_cost'] for p in ok):>9.4f}"
          f"{statistics.mean(p['jev_cost'] for p in ok):>10.5f}")
