#!/usr/bin/env python3
"""Mutation proof on the SHIPPED isolation checks (prereg A5.2): revert each check,
one at a time, and require the arm named for it to go RED, then restore byte-for-byte.
Engine and discipline: tests/mutation_engine.py. Report path: $JEVCAL_ISOLATION_MUTATION_REPORT.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mutation_engine import run_suite  # noqa: E402

ARMS = ["ArmIsolationEnv", "ArmIsolationOutsideAsif", "ArmIsolationAncestors", "ArmIsolationEnforced"]
I = "jevcal/isolation.py"
F = "jevcal/arms/frontier.py"
MUTATIONS = [
    ("I1-env-check-reverted", "ArmIsolationEnv", I,
     "    if EFFORT_KEY in env:\n        return False",
     "    if False:\n        return False"),
    ("I2-child-env-keeps-inherited-pin", "ArmIsolationEnv", F,
     "env = {k: v for k, v in os.environ.items() if k != isolation.EFFORT_KEY}",
     "env = dict(os.environ)"),
    ("I3-outside-asif-check-reverted", "ArmIsolationOutsideAsif", I,
     "    if inside:\n",
     "    if False:\n"),
    ("I4-ancestor-settings-check-reverted", "ArmIsolationAncestors", I,
     "if isinstance(env, dict) and EFFORT_KEY in env:",
     "if False:"),
    ("I5-ancestor-walk-cwd-only", "ArmIsolationAncestors", I,
     "for d in ancestors(cwd):",
     "for d in ancestors(cwd)[:1]:"),
    ("I6-unreadable-settings-fail-open", "ArmIsolationAncestors", I,
     'bad.append(f"{p} unreadable ({type(e).__name__}); cannot show it does not pin {EFFORT_KEY}")',
     "pass"),
    ("I7-arm-skips-per-call-check", "ArmIsolationEnforced", F,
     "        if not iso[\"ok\"]:\n            raise ArmHalt(",
     "        if False:\n            raise ArmHalt("),
]

if __name__ == "__main__":
    sys.exit(run_suite("tests.test_isolation", ARMS, MUTATIONS, "JEVCAL_ISOLATION_MUTATION_REPORT"))
