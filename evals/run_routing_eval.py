"""Router accuracy against hand-labelled questions, no API key needed.

  python evals/run_routing_eval.py --split dev|test|both [--router keyword_legacy|weighted]

'dev' was used while improving the router; 'test' was written first and run once at the end.
Both sets were labelled by the project author, so they share the author's blind spots.
Also reports the average number of specialists activated (each one is an extra paid model call).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from core.router import QueryRouter  # noqa: E402

DOMAINS = ("math", "code", "science")


def evaluate(split, router_name=None):
    router = QueryRouter(mode=router_name) if router_name else QueryRouter()
    cases = json.loads((ROOT / f"routing_{split}.json").read_text())
    tp = {d: 0 for d in DOMAINS}; fp = dict(tp); fn = dict(tp)  # noqa: E702
    exact, activated, misses = 0, 0, []
    for c in cases:
        want = set(c["specialists"])
        got = set(router.route(c["query"])) - {"general"}
        activated += len(got)
        exact += got == want
        for d in DOMAINS:
            tp[d] += d in got and d in want
            fp[d] += d in got and d not in want
            fn[d] += d not in got and d in want
        if got != want:
            misses.append((c["query"], sorted(want), sorted(got)))
    T, F, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
    prec = T / (T + F) if T + F else 1.0
    rec = T / (T + N) if T + N else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n": len(cases), "exact": exact, "precision": prec, "recall": rec, "f1": f1,
            "avg_specialists": activated / len(cases), "misses": misses}


def render(split, name, o, show_misses):
    lines = [f"routing [{name}] split={split}  n={o['n']}",
             f"  exact-set accuracy {o['exact']}/{o['n']} = {100 * o['exact'] / o['n']:.0f}%   "
             f"precision {o['precision']:.2f}  recall {o['recall']:.2f}  F1 {o['f1']:.2f}   "
             f"avg specialists/query {o['avg_specialists']:.2f}"]
    if show_misses:
        for q, want, got in o["misses"]:
            lines.append(f"    MISS want={want} got={got} | {q}")
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="both", choices=["dev", "test", "both"])
    p.add_argument("--router", default=None)
    p.add_argument("--misses", action="store_true")
    a = p.parse_args()
    for s in (["dev", "test"] if a.split == "both" else [a.split]):
        print(render(s, a.router or "current default", evaluate(s, a.router), a.misses))
