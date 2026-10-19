"""Retrieval quality, measured without any API key.

  hit@1 / hit@3     answerable questions whose correct source document is the top / in the top 3 results
  false-context     off-topic questions that still receive retrieved text (they should receive none)
  avg chars         average context injected per question (a proxy for wasted tokens)

    python evals/run_retrieval_eval.py [--scorer tfidf_legacy|bm25]

The corpus and questions are small and written by the project author. Read the numbers as a regression
test and a direction, not as a benchmark of retrieval in general.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from core.rag import RAGEngine  # noqa: E402


def evaluate(scorer=None, top_k=3):
    kwargs = {"docs_dir": str(ROOT / "corpus"), "top_k": top_k}
    if scorer:
        kwargs["scorer"] = scorer
    rag = RAGEngine(**kwargs)
    rag.load_documents()
    cases = json.loads((ROOT / "retrieval_cases.json").read_text())
    out = {}
    for kind in ("exact", "paraphrase"):
        rows = [c for c in cases["answerable"] if c["kind"] == kind]
        top1 = top3 = 0
        for c in rows:
            sources = [h["source"] for h in rag.retrieve_chunks(c["query"])]
            top1 += bool(sources) and sources[0] == c["source"]
            top3 += c["source"] in sources
        out[kind] = {"n": len(rows), "hit@1": top1, "hit@3": top3}
    off = cases["off_topic"]
    out["off_topic"] = {"n": len(off), "false_context": sum(bool(rag.retrieve(q)) for q in off)}
    allq = [c["query"] for c in cases["answerable"]] + off
    out["avg_chars"] = round(sum(len(rag.retrieve(q)) for q in allq) / len(allq))
    return out


def render(name, o):
    pc = lambda a, b: f"{a}/{b} = {100 * a / b:.0f}%"  # noqa: E731
    return "\n".join([
        f"retrieval [{name}]",
        f"  exact-wording questions   hit@1 {pc(o['exact']['hit@1'], o['exact']['n'])}   hit@3 {pc(o['exact']['hit@3'], o['exact']['n'])}",
        f"  paraphrased questions     hit@1 {pc(o['paraphrase']['hit@1'], o['paraphrase']['n'])}   hit@3 {pc(o['paraphrase']['hit@3'], o['paraphrase']['n'])}",
        f"  off-topic given context   {pc(o['off_topic']['false_context'], o['off_topic']['n'])}   (lower is better; 0 is the goal)",
        f"  avg context per question  {o['avg_chars']} characters",
    ])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scorer", default=None)
    a = p.parse_args()
    print(render(a.scorer or "current default", evaluate(a.scorer)))
