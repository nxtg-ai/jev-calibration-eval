#!/usr/bin/env python3
"""Reproduce every S1 number of the Jev calibration study, v1, from the released raw records.

Inputs (all in this repo):
  governance/evals/jev-calibration/items-v1.jsonl             the 160 S1 items (CUAD-derived), pinned
  governance/evals/jev-calibration/frontier-subset-v1-s1.txt  the 47 S1 members of the FRONTIER subset
  results/jevcal-d2b-3arm-20261001T043407Z-S1/raw-s1.jsonl    the run's 1,214 S1 per-call records

Every number is computed by the SHIPPED harness code (jevcal.scorer, jevcal.run.h2_read,
jevcal.run.record_vector), never by a re-implementation. Population and draw conventions are
those of the pre-registration (docs/prereg-v1.md): test split, draw 1 primary, k = 3 draws,
bootstrap seed 20260929.

Usage:
  python3 scripts/reproduce_s1.py                 # prints the tables
  python3 scripts/reproduce_s1.py --json out.json # also writes every number as JSON

What this can NOT reproduce from public data (it needs the private S2 slices): pooled numbers,
the S2 rows, H4 latency/cost over all 340 items, the pooled temperature-scaling read, and
jitter over all 340 items. The S1-only versions of pairwise Brier, latency, cost and jitter are
printed and labelled S1-only; they are not the figures in the results document.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evals", "jev-calibration"))

from jevcal import run as R  # noqa: E402
from jevcal import scorer as S  # noqa: E402
from jevcal.items import load_items  # noqa: E402

RUN = "jevcal-d2b-3arm-20261001T043407Z"
DATA = os.path.join(ROOT, "governance", "evals", "jev-calibration")
ITEMS = os.path.join(DATA, "items-v1.jsonl")              # the S1 rows of the frozen set, pinned
SUBSET = os.path.join(DATA, "frontier-subset-v1-s1.txt")
RAW = os.path.join(ROOT, "results", f"{RUN}-S1", "raw-s1.jsonl")
SEED = 20260929
ARMS = ("LOCAL-P", "JEV", "FRONTIER")


def J(x):
    return json.loads(json.dumps(x))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="write every reproduced number to this JSON file")
    a = ap.parse_args(argv)

    # "certifying" = load only the pinned bytes (items-v1.sha256); "fixture" refuses them by design.
    items = load_items(ITEMS, "certifying")
    idx = {it.item_id: it for it in items}
    by = collections.defaultdict(lambda: collections.defaultdict(dict))   # arm -> item -> draw -> rec
    n_rec = 0
    for line in open(RAW, encoding="utf-8"):
        d = json.loads(line)
        by[d["arm"]][d["item_id"]][int(d["draw"])] = d
        n_rec += 1
    subset = {x.strip() for x in open(SUBSET) if x.strip()}
    test = [it for it in items if it.split == "test"]
    slices = sorted({it.slice for it in test})
    out = {"run_id": RUN, "records_read": n_rec, "items": len(items), "test_items": len(test),
           "frontier_subset_s1": len(subset), "seed": SEED}
    print(f"run {RUN}: {n_rec} S1 records, {len(items)} items ({len(test)} test), "
          f"FRONTIER subset (S1 part) {len(subset)}")

    def vecs(arm, its, d):
        rs = [by[arm][it.item_id][d] for it in its]
        return ([R.record_vector(r, it.options) for r, it in zip(rs, its)],
                [it.gold_index for it in its], [it.item_id for it in its],
                [list(it.options) for it in its], [bool(r["abstained"]) for r in rs])

    def drawrun(arm, its, d):
        return S.DrawRun(d, *vecs(arm, its, d))

    def point(arm, its, d):
        p, g, _ids, k, ab = vecs(arm, its, d)
        return S.point_metrics(p, g, k, ab)

    # ---- H1: JEV, per S1 slice, per draw -------------------------------------------------
    print("\nH1 (JEV, test split, raw probabilities)")
    print(f"{'slice':10} {'n':>3}  d  ece_mass  ece_width  bins  verdict")
    out["h1"] = {}
    for sl in slices:
        its = [it for it in test if it.slice == sl]
        rows, words = [], []
        for d in (1, 2, 3):
            v = S.h1_for_draw(drawrun("JEV", its, d))
            bins = point("JEV", its, d)["ece_bins_realized"]
            words.append(v["h1"])
            rows.append({"draw": d, "n": len(its), "ece_bins_realized": bins, **J(v)})
            print(f"{sl:10} {len(its):3}  {d}  {v['ece_equal_mass_15']:.4f}    {v['ece_equal_width_15']:.4f}"
                  f"    {bins:2}   {v['h1']}")
        claim = S.claim_across_draws(words)
        out["h1"][sl] = {"per_draw": rows, "claim": claim}
        print(f"{'':10} claim across draws: {claim}")

    # ---- H2: LOCAL-P vs best non-local arm, Read C (all test items) and Read A (subset) ----
    arms_ns = [types.SimpleNamespace(arm_name=n) for n in ARMS]

    def h2(its, draw_of):
        ids = [it.item_id for it in its]
        recs = {n: {i: by[n][i][draw_of[n]] for i in ids} for n in draw_of}
        return R.h2_read(arms_ns, its, recs, ids, [list(it.options) for it in its],
                         [it.gold_index for it in its], SEED)

    print("\nH2 (does LOCAL-P match on this class?)  C = all test items, FRONTIER at draw 1; "
          "A = FRONTIER subset, every arm at draw d")
    print(f"{'slice':10} read d  n   acc_L  acc_J  acc_F  best      gap_L-best  bUCI_L  bUCI_J  verdict")
    out["h2"] = collections.defaultdict(list)
    sub_items = [it for it in items if it.item_id in subset]
    for label, its_all, fr in (("C", items, lambda d: 1), ("A", sub_items, lambda d: d)):
        for d in (1, 2, 3):
            h = h2(its_all, {"LOCAL-P": d, "JEV": d, "FRONTIER": fr(d)})
            for sl, v in h["by_slice"].items():
                acc = {k: x["point"] for k, x in v["accuracy"].items()}
                row = {"read": label, "draw": d, "n": v["n_items"], "verdict": v["verdict"],
                       "best_non_local": v["best_non_local"],
                       "accuracy_gap_local_minus_best": v["accuracy_gap_local_minus_best"],
                       "brier_upper_ci": v["brier_upper_ci"], "accuracy": acc}
                out["h2"][sl].append(J(row))
                print(f"{sl:10} {label}    {d} {v['n_items']:3}   {acc['LOCAL-P']:.3f}  {acc['JEV']:.3f}  "
                      f"{acc['FRONTIER']:.3f}  {v['best_non_local']:9} {v['accuracy_gap_local_minus_best']:+.4f}"
                      f"     {v['brier_upper_ci']['LOCAL-P']:.4f}  {v['brier_upper_ci']['JEV']:.4f}  {v['verdict']}")
    out["h2"] = dict(out["h2"])
    for sl in slices:
        rows = out["h2"][sl]
        vc = {r["verdict"] for r in rows if r["read"] == "C"}
        va = {r["verdict"] for r in rows if r["read"] == "A"}
        claim = (sorted(vc)[0] if len(vc) == 1 else "draw-sensitive")
        out["h2"][sl + ":claim"] = {"read_C_governs": claim, "read_C": sorted(vc), "read_A": sorted(va)}
        print(f"{sl:10} claim (Read C governs): {claim}   Read C {sorted(vc)}  Read A {sorted(va)}")

    # ---- per-arm point metrics (test, draw 1) incl. H3 coverage at 95% precision ----------
    print("\nPer arm, test split, draw 1 (H3 = coverage at >=95% precision)")
    print(f"{'arm':9} {'slice':10} {'n':>3}  acc    brier  ece_mass ece_width cov@95  macroF1")
    out["point"] = {}
    for arm in ARMS:
        out["point"][arm] = {}
        for sl in slices:
            m = point(arm, [it for it in test if it.slice == sl], 1)
            out["point"][arm][sl] = J({k: m[k] for k in (
                "n", "abstentions", "accuracy", "brier", "ece_equal_mass_15", "ece_equal_width_15", "nll",
                "coverage_at_95_precision", "aurc", "macro_f1", "ece_bins_realized")})
            print(f"{arm:9} {sl:10} {m['n']:3}  {m['accuracy']:.3f}  {m['brier']:.3f}  {m['ece_equal_mass_15']:.3f}"
                  f"    {m['ece_equal_width_15']:.3f}     {m['coverage_at_95_precision']:.2f}    {m['macro_f1']:.3f}")

    # ---- sub-slices by tag (every tagged item is S1) ----------------------------------------
    print("\nSub-slices by tag (test, draw 1, accuracy)")
    out["subslice"] = {}
    for t in sorted({t for it in test for t in it.tags}):
        its = [it for it in test if t in it.tags]
        row = {"n": len(its), **{arm: point(arm, its, 1)["accuracy"] for arm in ARMS}}
        out["subslice"][t] = row
        print(f"  {t:18} n={row['n']:3}  " + "  ".join(f"{arm} {row[arm]:.3f}" for arm in ARMS))

    # ---- S1-only extras (labelled; the results document pools S1 and S2 for these) ---------
    print("\nS1-only figures (NOT the figures in the results document, which pool S1 and S2):")
    out["s1_only"] = {"pairwise_brier_test_draw1": {}, "latency_cost_draw1_all_s1": {}, "jitter": {}}
    ids = sorted(it.item_id for it in test)

    def brier_items(arm):
        return [float(S.brier_per_item([R.record_vector(by[arm][i][1], idx[i].options)],
                                       [idx[i].gold_index])[0]) for i in ids]
    for x, y in (("LOCAL-P", "JEV"), ("JEV", "FRONTIER"), ("LOCAL-P", "FRONTIER")):
        pb = S.paired_bootstrap(brier_items(x), brier_items(y), SEED, ids, ids)
        out["s1_only"]["pairwise_brier_test_draw1"][f"{x} - {y}"] = J(pb)
        print(f"  paired Brier {x} - {y} (test, n={pb['n_items']}): {pb['mean_diff']:+.3f} "
              f"[{pb['lo']:+.3f}, {pb['hi']:+.3f}]")
    for arm in ARMS:
        rs = [by[arm][it.item_id][1] for it in items]
        lat = [r["latency_s"] for r in rs]
        cost = sum(r["cost_usd_list"] or 0.0 for r in rs)
        row = {"n": len(rs), "p50_s": R.percentile(lat, 50), "p95_s": R.percentile(lat, 95),
               "cost_usd_list_total": cost, "cost_usd_list_per_decision": cost / len(rs)}
        out["s1_only"]["latency_cost_draw1_all_s1"][arm] = row
        print(f"  {arm:9} draw 1, all {len(rs)} S1 items: p50 {row['p50_s']:.3f} s, p95 {row['p95_s']:.3f} s, "
              f"list cost per decision ${row['cost_usd_list_per_decision']:.7f}")
    for arm in ARMS:
        sub = [it for it in items if len(by[arm][it.item_id]) == 3]
        j = S.jitter([[R.record_vector(by[arm][it.item_id][d], it.options) for d in (1, 2, 3)] for it in sub],
                     [[bool(by[arm][it.item_id][d]["abstained"]) for d in (1, 2, 3)] for it in sub])
        out["s1_only"]["jitter"][arm] = J(j)
        print(f"  jitter {arm:9} items {j['items_with_draws']:3}  mean |dp| {j['mean_abs_dp']:.4f}  "
              f"max |dp| {j['max_abs_dp']:.2f}  answer flips {j['answer_flip_share']:.3f}")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
