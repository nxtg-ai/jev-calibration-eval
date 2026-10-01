#!/usr/bin/env python3
"""Mutation proof for CODEX PR 72 findings 2-5: revert each fix in the SHIPPED file and
require the arm named for it to go RED. Engine: tests/mutation_engine.py.
Report path: $JEVCAL_HARNESS_MUTATION_REPORT.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mutation_engine import run_suite  # noqa: E402

ARMS = ["ArmEffortPin", "ArmPromptParity", "ArmLangfuseFailClosed", "ArmRubricConfig",
        "ArmE4FixtureRefusesFrozen", "ArmE4HashMismatch", "ArmE4FixtureRefusesRealStore",
        "ArmE4MissingPin", "ArmE4CertifyingRequiresPinnedContent", "ArmE4FixtureRefusesSymlink",
        "ArmE4FixtureRefusesCopy", "ArmE4ProvenanceUnknownKeyRefused", "ArmE4NoLeakInPrompt",
        "ArmE4CertIgnoresAsifRoot", "ArmE4FixtureRefusesPartialFrozenById",
        "ArmE4FixtureRefusesPartialFrozenByState", "ArmE4FixturesStillLoad"]
F = "jevcal/arms/frontier.py"
J = "jevcal/arms/jev.py"
R = "jevcal/run.py"
I = "jevcal/items.py"
MUTATIONS = [
    ("C2-effort-pin-loosened", "ArmEffortPin", F,
     "        if effort != PINNED_EFFORT:",
     "        if not effort:"),
    ("C3a-noul-wire-drops-keys", "ArmPromptParity", J,
     '"criteria": {"true": yes_text, "false": no_text}}',
     '"criteria": {"true": r.options[0][2], "false": r.options[1][2]}}'),
    ("C3b-score-wire-drops-keys", "ArmPromptParity", J,
     '"criteria": list(r.option_texts)}',
     '"criteria": [desc or key for _, key, desc in r.options]}'),
    ("C4a-no-early-abort-on-helper-error", "ArmLangfuseFailClosed", R,
     '    if lfs["mode"] == "error":\n',
     "    if False:\n"),
    ("C4b-partial-tracing-accepted", "ArmLangfuseFailClosed", R,
     'if lfs["mode"] == "trace" and (lf_errors or not trace_ids):',
     "if False:"),
    ("C5-missing-endorsement-defaults-true", "ArmRubricConfig", R,
     'endorsed = cfg.get("operator_endorsed", False)',
     'endorsed = cfg.get("operator_endorsed", True)'),
    # E.4 go-live guards: each revert reproduces the defect's real mechanism.
    ("E4a-fixture-reads-frozen-path", "ArmE4FixtureRefusesFrozen", I,
     "    if looks_frozen:",
     "    if False:"),
    ("E4c-cert-hash-comparison-dropped", "ArmE4HashMismatch", I,
     "        if actual != pinned:",
     "        if False:"),
    ("E4f-cert-missing-pin-treated-as-ok", "ArmE4MissingPin", I,
     "        if pinned is None:",
     "        if False:"),
    ("E4d-fixture-uses-real-store", "ArmE4FixtureRefusesRealStore", R,
     '    if mode == "certifying":',
     "    if True:"),
    # F1+F2 (reviewer): deleting the content gate restores the path-only guard; certifying then
    # accepts an unpinned file (F1) and fixture accepts the pinned bytes at a non-frozen path (F2).
    ("E4-content-gate-deleted", "ArmE4CertifyingRequiresPinnedContent", I,
     "    if _content_refuse(path, mode, pinned, actual):",
     "    if False:"),
    ("E4h-any-unknown-key-accepted", "ArmE4ProvenanceUnknownKeyRefused", I,
     "    if extra:",
     "    if False:"),
    ("E4i-leak-strings-surfaced-into-render", "ArmE4NoLeakInPrompt", I,
     '    state = _need_str(row, "state", idx)',
     '    state = _need_str(row, "state", idx) + " " + " ".join(row.get("leak_strings", []))'),
    # N1 (reviewer PR #89 review): restore the ASIF_ROOT pin redirect so certifying honours it again.
    ("N1-cert-honours-asif-root", "ArmE4CertIgnoresAsifRoot", I,
     '    root = _repo_root() if mode == "certifying" else (os.environ.get("ASIF_ROOT") or _repo_root())',
     '    root = os.environ.get("ASIF_ROOT") or _repo_root()'),
    # N2 (reviewer PR #89 review): drop the per-row frozen-overlap refusal -> a partial/renamed copy
    # is accepted. One drop turns both N2 arms red; each tuple names one so both are proven KILLED.
    ("N2a-per-row-check-dropped", "ArmE4FixtureRefusesPartialFrozenById", I,
     "    if overlap is not None:",
     "    if False:"),
    ("N2b-per-row-check-dropped", "ArmE4FixtureRefusesPartialFrozenByState", I,
     "    if overlap is not None:",
     "    if False:"),
]

if __name__ == "__main__":
    sys.exit(run_suite("tests.test_codex_fixes", ARMS, MUTATIONS, "JEVCAL_HARNESS_MUTATION_REPORT"))
