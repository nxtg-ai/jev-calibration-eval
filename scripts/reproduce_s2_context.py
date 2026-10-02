#!/usr/bin/env python3
"""Recompute every Part 2 number in docs/results2-v1.md from the released per-item file.

The S2 items are private. The released file, results/jevctx-p2-20261002T000132Z-S2/per-item.csv,
holds one row per (run, arm, item): an opaque item id, the class, the number of options,
whether the answer was correct (0/1) and the top-1 probability. It holds no text and no labels.
That is enough to recompute accuracy, ECE (the shipped scorer's equal-mass, 15 bins) and every
paired-bootstrap interval and verdict word (prereg2-v1.md section 4, A1 and A2).

    python3 scripts/reproduce_s2_context.py            # tables
    python3 scripts/reproduce_s2_context.py --json x   # every number as JSON
"""
import argparse, csv, json, os, sys
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evals", "jev-calibration"))
from jevcal import scorer  # noqa: E402

CSV = os.path.join(ROOT, "results", "jevctx-p2-20261002T000132Z-S2", "per-item.csv")
SEED, B = 20260929, 2000
CLASSES = ("S2a", "S2b", "S2c")
ARMS = ("JEV+CTX", "JEV", "FRONTIER+CTX", "FRONTIER", "RETRIEVAL-ONLY", "LOCAL-P+CTX")

rows = list(csv.DictReader(open(CSV)))
D = {(r["run"], r["arm"], r["item_id"]): r for r in rows}
assert len(D) == len(rows) == 1296, len(rows)


def items(cls):
    return sorted(i for (run, arm, i) in D if run == "part2" and arm == "RETRIEVAL-ONLY" and i.startswith(cls))


def vec(run, arm, ids, field="correct"):
    return [float(D[(run, arm, i)][field]) for i in ids]   # KeyError = a missing item: refuse


def pair(a, b, ids):
    r = scorer.paired_bootstrap(vec(*a, ids), vec(*b, ids), SEED, ids, ids, n_resamples=B)
    return {"diff": r["mean_diff"], "ci95": [r["lo"], r["hi"]], "n_items": r["n_items"]}


def h5_word(p):   # prereg section 4: CI entirely above 0 AND gain >= 10 points; "entirely" read strictly
    lo, hi = p["ci95"]
    if lo > 0 and p["diff"] >= 0.10: return "context helps"
    if hi < 0: return "context hurts"
    return "no clear effect"


def h6_word(p):   # amendment A2.2
    lo, hi = p["ci95"]
    if lo > 0: return "beats"
    if hi < 0: return "trails"
    return "matches" if abs(Fraction(p["diff"]).limit_denominator(1000)) <= Fraction(3, 100) else "no clear difference"


out = {}
for cls in CLASSES:
    ids = items(cls)
    c = out[cls] = {"n_items": len(ids), "chance": sum(1 / float(D[("part2", "JEV", i)]["n_options"]) for i in ids) / len(ids)}
    c["arms"] = {a: {"accuracy": sum(vec("part2", a, ids)) / len(ids),
                     "correct": int(sum(vec("part2", a, ids))),
                     "ece": scorer.ece(vec("part2", a, ids, "top1_confidence"), vec("part2", a, ids))} for a in ARMS}
    for m in ("JEV", "FRONTIER"):
        p = pair(("part2", m + "+CTX"), ("part2", m), ids); p["word"] = h5_word(p); c[f"H5 {m}"] = p
        p = pair(("part2", m + "+CTX"), ("part1-draw1", m), ids); p["word"] = h5_word(p); c[f"H5' {m} vs Part 1"] = p
    p = pair(("part2", "LOCAL-P+CTX"), ("part1-draw1", "LOCAL-P"), ids); p["word"] = h5_word(p); c["H5' LOCAL-P vs Part 1 (secondary only)"] = p
    best = max(("JEV", "FRONTIER"), key=lambda m: sum(vec("part2", m, ids)))            # A2.1
    p = pair(("part2", "RETRIEVAL-ONLY"), ("part2", best), ids); p["word"] = h6_word(p); p["comparator"] = best; c["H6"] = p
    best1 = max(("JEV", "FRONTIER", "LOCAL-P"), key=lambda m: sum(vec("part1-draw1", m, ids)))
    p = pair(("part2", "RETRIEVAL-ONLY"), ("part1-draw1", best1), ids); p["word"] = h6_word(p); p["comparator"] = "Part 1 " + best1; c["H6 vs Part 1"] = p
    c["H7 (descriptive)"] = pair(("part2", "FRONTIER+CTX"), ("part2", "JEV+CTX"), ids)

pooled = {a: sum(sum(vec("part2", a, items(cls))) for cls in CLASSES) / sum(len(items(cls)) for cls in CLASSES) for a in ARMS}

ap = argparse.ArgumentParser(); ap.add_argument("--json"); args = ap.parse_args()
if args.json:
    json.dump({"by_class": out, "pooled_accuracy": pooled}, open(args.json, "w"), indent=1)
f = lambda p: f"{p['diff']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]"
for cls, c in out.items():
    print(f"== {cls}  n={c['n_items']}  chance={c['chance']:.3f}")
    for a, m in c["arms"].items():
        print(f"  {a:15s} accuracy {m['accuracy']:.3f} ({m['correct']}/{c['n_items']})  ECE {m['ece']:.3f}")
    for k, p in c.items():
        if isinstance(p, dict) and "diff" in p:
            extra = f"  (comparator {p['comparator']})" if "comparator" in p else ""
            print(f"  {k:40s} {f(p)}  {p.get('word', '')}{extra}")
print("== pooled accuracy, 144 items:", ", ".join(f"{a} {v:.3f}" for a, v in pooled.items()))
