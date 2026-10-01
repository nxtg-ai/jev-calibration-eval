#!/usr/bin/env python3
"""Mutation proof for Jev eval step D1: revert each new guard in the SHIPPED file, reproducing the
defect's mechanism, and require the arm named for it to go RED. Engine: tests/mutation_engine.py.
Report path: $JEVCAL_D1_MUTATION_REPORT.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mutation_engine import run_suite  # noqa: E402

ARMS = ["ArmThreeArmFixtureRun", "ArmPairLabel", "ArmH2BestByAccuracy", "ArmH2Runner", "ArmH2Withheld",
        "ArmRefuseSingleArm", "ArmRefuseLocalWithoutBoth", "ArmCompareChoices",
        "ArmLocalMaxTokens"]
R = "jevcal/run.py"
S = "jevcal/scorer.py"
C = "jevcal/compare_choices.py"
LOC = "jevcal/arms/local.py"
MUTATIONS = [
    # 1: the pre-D1 hard-coded parity label comes back (false in a run that has LOCAL-P in the pair)
    ("D1-1-parity-label-hard-coded", "ArmPairLabel", R,
     "f\"±{PARITY_POINTS:.2f} ({pl['text']}). A pairwise GATE E read, \"",
     "f\"±{PARITY_POINTS:.2f} (LOCAL arm absent in this run). A pairwise GATE E read, \""),
    # 2a: best non-local arm chosen by NAME, not by accuracy (every arm counts as tied)
    ("D1-2a-best-by-name", "ArmH2BestByAccuracy", S,
     "    tied = [n for n in names if acc[n] == top]",
     "    tied = names"),
    # 2b: h2 computed over every split instead of the TEST split
    ("D1-2b-h2-all-splits", "ArmH2Runner", R,
     '    by_slice = {}\n    test_ix = [i for i, it in enumerate(items) if it.split == "test"]',
     "    by_slice = {}\n    test_ix = list(range(len(items)))"),
    # 2c: a verdict is printed although FRONTIER did not run
    ("D1-2c-verdict-not-withheld", "ArmH2Withheld", R,
     '        v["verdict"] = word if not missing else None',
     '        v["verdict"] = word'),
    # 3a: the single-arm refusal removed, so the run proceeds and calls the arm
    ("D1-3a-single-arm-proceeds", "ArmRefuseSingleArm", R,
     "    if len(arm_names) < 2:",
     "    if False:"),
    # 3b: the LOCAL-without-both refusal removed, so LOCAL-P is built (VRAM probe + arm) and runs
    ("D1-3b-local-without-both-proceeds", "ArmRefuseLocalWithoutBoth", R,
     "    if local and missing:",
     "    if False:"),
    # 4: every draw read, the last one wins (the draw-1 filter and the duplicate refusal gone)
    ("D1-4-last-draw-wins", "ArmCompareChoices", C,
     '            if rec.get("draw", 1) != draw:\n                continue\n'
     '            iid = rec["item_id"]\n            if iid in labels:',
     '            iid = rec["item_id"]\n            if False:'),
    # 5: the LOCAL arm stops reporting its output cap (the GATE H defect of run
    # jevcal-d2-3arm-20260930T230651Z: max_tokens null although num_predict was sent on every call)
    ("D1-5-local-max-tokens-dropped", "ArmLocalMaxTokens", LOC,
     '        rail["max_tokens"] = cap if isinstance(cap, int) and not isinstance(cap, bool) else None\n',
     '        pass  # mutant D1-5: no max_tokens key (sampling_config().get("max_tokens") -> None)\n'),
]

if __name__ == "__main__":
    sys.exit(run_suite("tests.test_d1", ARMS, MUTATIONS, "JEVCAL_D1_MUTATION_REPORT"))
