#!/usr/bin/env python3
"""Mutation proof for prereg A7 (draws): break each rule in the SHIPPED code and require
the arm named for it to go RED. Engine: tests/mutation_engine.py.
Report path: $JEVCAL_DRAWS_MUTATION_REPORT.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mutation_engine import run_suite  # noqa: E402

ARMS = ["ArmDrawJitter", "ArmDrawSensitivity", "ArmKAveragedIsolation", "ArmDrawPrimary",
        "ArmRunMode", "ArmTestPopulation"]
S = "jevcal/scorer.py"
R = "jevcal/run.py"
MUTATIONS = [
    ("D1-jitter-never-counts-flips", "ArmDrawJitter", S,
     "        if len(set(answers)) > 1:",
     "        if False:"),
    ("D2-jitter-max-is-min", "ArmDrawJitter", S,
     '"max_abs_dp": float(max(diffs)) if diffs else None,',
     '"max_abs_dp": float(min(diffs)) if diffs else None,'),
    ("D3-verdict-from-draw-1-only", "ArmDrawSensitivity", S,
     'return vs[0] if all(v == vs[0] for v in vs) else "draw-sensitive"',
     "return vs[0]"),
    ("D4-h-verdict-accepts-k-averaged", "ArmKAveragedIsolation", S,
     "    if not isinstance(run, DrawRun):",
     "    if False:"),
    ("D5-primary-overwritten-by-latest-draw", "ArmDrawPrimary", R,
     "                    draws[arm.arm_name][it.item_id].append(rec)",
     "                    draws[arm.arm_name][it.item_id].append(rec); records[arm.arm_name][it.item_id] = rec"),
    # CODEX re-grade P0-2: a verdict re-materialized under the k-averaged block
    ("D6-k-averaged-carries-a-verdict", "ArmKAveragedIsolation", S,
     '        out.update({"descriptive_only": True, "draws": self.draws})',
     '        out.update({"descriptive_only": True, "draws": self.draws, '
     '"h1": _combine_h1_bands(out["ece_equal_mass_15"], out["ece_equal_width_15"])["h1"]})'),
    # CODEX re-grade P1-3: each certifying-mode check reverted, and the fixture ledger guard
    ("R1-certifying-accepts-any-k", "ArmRunMode", R,
     "    if a.draws != CERT_K:",
     "    if False:"),
    ("R2-certifying-accepts-partial-redraw", "ArmRunMode", R,
     "    if a.draw_items:\n        raise SystemExit",
     "    if False:\n        raise SystemExit"),
    ("R3-subset-size-unchecked", "ArmRunMode", R,
     "    if len(ids) != CERT_FRONTIER_SUBSET_N or len(set(ids)) != len(ids):",
     "    if False:"),
    ("R4-subset-need-not-be-committed", "ArmRunMode", R,
     '    if git("ls-files", "--", rel) != rel or git("status", "--porcelain", "--", rel):',
     "    if False:"),
    ("R5-fixture-run-written-to-ledger", "ArmRunMode", R,
     "    if not certifying:\n        # CODEX re-grade P1-3",
     "    if False:\n        # CODEX re-grade P1-3"),
    ("R6-frontier-redraw-set-unchecked", "ArmRunMode", R,
     "    if a.frontier_draw_items is not None and set(",
     "    if False and set("),
    # Advisor review P1: revert each use of the test population
    ("T1-ledger-headline-pooled", "ArmTestPopulation", R,
     '**(headline(v["test"]) or {"population": "test", "n": 0}),',
     '**(headline(v["raw"]) or {"population": "test", "n": 0}),'),
    ("T2-win-gap-pooled", "ArmTestPopulation", R,
     "pb = S.paired_bootstrap(brier_items[A][test_ix], brier_items[B][test_ix], seed=a.seed,\n"
     "                                ids_a=test_ids, ids_b=test_ids)",
     "pb = S.paired_bootstrap(brier_items[A], brier_items[B], seed=a.seed,\n"
     "                                ids_a=all_ids, ids_b=all_ids)"),
    ("T3-parity-pooled", "ArmTestPopulation", R,
     "pa = S.paired_bootstrap(acc_a, acc_b, seed=a.seed + 1, ids_a=test_ids, ids_b=test_ids)",
     "pa = S.paired_bootstrap(np.array([t['per_model_correct'][arms[0].model_id] for t in tasks], dtype=float), "
     "np.array([t['per_model_correct'][arms[1].model_id] for t in tasks], dtype=float), "
     "seed=a.seed + 1, ids_a=all_ids, ids_b=all_ids)"),
]

if __name__ == "__main__":
    sys.exit(run_suite("tests.test_draws", ARMS, MUTATIONS, "JEVCAL_DRAWS_MUTATION_REPORT"))
