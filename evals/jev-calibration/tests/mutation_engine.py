"""Mutation engine shared by mutate_scorer.py and mutate_isolation.py.

Discipline (canon: mutate the shipped code, not a copy):
  - each mutation edits a SHIPPED file IN PLACE (the file the runner imports);
  - it refuses to start unless every target file is committed and clean, so a
    crash mid-run is recoverable with `git checkout -- <file>` and nothing is lost;
  - every mutation asserts its anchor occurs EXACTLY once and that the file's
    bytes changed; a mutation that applies nothing is a failure, never a pass;
  - after each mutation the original bytes are restored and their SHA-256 is
    re-verified before the next one;
  - every arm runs against every mutation in a FRESH interpreter, and the report
    is the OBSERVED red/green matrix, including collateral reds;
  - `isolating_mutations` lists, per arm, the mutations that turn ONLY that arm red.

Exit 0 only if: baseline all green, every mutation applied, every named arm RED
on its own mutation, and every file restored.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import signal
import subprocess
import sys
from typing import List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # evals/jev-calibration

RUN_ARMS = r"""
import json, sys, unittest
sys.path.insert(0, {root!r})
arms = {arms!r}
out = {{}}
for a in arms:
    suite = unittest.defaultTestLoader.loadTestsFromName({module!r} + '.' + a)
    r = unittest.TextTestRunner(stream=open('/dev/null', 'w'), verbosity=0).run(suite)
    out[a] = 'GREEN' if (r.wasSuccessful() and r.testsRun > 0) else 'RED'
print(json.dumps(out))
"""

Mutation = Tuple[str, str, str, str, str]  # (id, named arm, target file rel to ROOT, anchor, replacement)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run_arms(module: str, arms: List[str]):
    code = RUN_ARMS.format(root=ROOT, arms=arms, module=module)
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       timeout=900)
    if p.returncode != 0:
        raise RuntimeError(f"arm runner crashed: {p.stderr[-800:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def run_suite(module: str, arms: List[str], mutations: List[Mutation], report_env: str) -> int:
    targets = sorted({m[2] for m in mutations})
    for rel in targets:
        tracked = subprocess.run(["git", "-C", ROOT, "ls-files", "--error-unmatch", rel],
                                 capture_output=True).returncode == 0
        clean = subprocess.run(["git", "-C", ROOT, "diff", "--quiet", "HEAD", "--", rel]).returncode == 0
        if not (tracked and clean):
            print(f"REFUSED: {rel} must be committed and clean before mutating it "
                  f"(tracked={tracked} clean={clean})")
            return 2
    originals = {}
    for rel in targets:
        with open(os.path.join(ROOT, rel), "rb") as fh:
            originals[rel] = fh.read()

    def restore(*_a):
        for rel, data in originals.items():
            with open(os.path.join(ROOT, rel), "wb") as fh:
                fh.write(data)
    atexit.register(restore)
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *a: (restore(), sys.exit(130)))

    def restored() -> bool:
        ok = True
        for rel, data in originals.items():
            with open(os.path.join(ROOT, rel), "rb") as fh:
                ok = ok and sha(fh.read()) == sha(data)
        return ok

    report = {"module": module, "targets": {rel: sha(d) for rel, d in originals.items()},
              "arms": arms, "mutations": []}
    baseline = run_arms(module, arms)
    report["baseline"] = baseline
    ok = all(v == "GREEN" for v in baseline.values())

    for mid, named, rel, anchor, repl in mutations:
        text = originals[rel].decode("utf-8")
        count = text.count(anchor)
        row = {"id": mid, "named_arm": named, "target": rel, "anchor_count": count}
        if count != 1:
            row["status"] = "ANCHOR-MISMATCH"
            ok = False
            report["mutations"].append(row)
            continue
        path = os.path.join(ROOT, rel)
        with open(path, "wb") as fh:
            fh.write(text.replace(anchor, repl).encode("utf-8"))
        with open(path, "rb") as fh:
            row["file_changed"] = sha(fh.read()) != sha(originals[rel])
        try:
            res = run_arms(module, arms) if row["file_changed"] else {}
        finally:
            restore()
        row["restored"] = restored()
        row["result"] = res
        row["red_arms"] = sorted(a for a, v in res.items() if v == "RED")
        row["named_arm_red"] = res.get(named) == "RED"
        row["status"] = ("KILLED" if (row["file_changed"] and row["named_arm_red"] and row["restored"])
                         else "SURVIVED")
        ok = ok and row["status"] == "KILLED"
        report["mutations"].append(row)

    report["isolating_mutations"] = {
        a: [m["id"] for m in report["mutations"] if m.get("red_arms") == [a]] for a in arms}
    report["arms_without_isolating_mutation"] = sorted(
        a for a, v in report["isolating_mutations"].items() if not v)
    report["final_restored"] = restored()
    ok = ok and report["final_restored"]
    report["verdict"] = "PASS" if ok else "FAIL"

    print(f"baseline: {sum(v == 'GREEN' for v in baseline.values())}/{len(baseline)} arms GREEN")
    print(f"{'mutation':<40} {'named arm':<26} {'named':<5} red arms (observed)")
    for m in report["mutations"]:
        print(f"{m['id']:<40} {m['named_arm']:<26} "
              f"{'RED' if m.get('named_arm_red') else m['status']:<5} {','.join(m.get('red_arms', []))}")
    killed = sum(1 for m in report["mutations"] if m["status"] == "KILLED")
    print(f"killed {killed}/{len(mutations)}; files restored: {report['final_restored']}; "
          f"arms with no isolating mutation: {report['arms_without_isolating_mutation'] or 'none'}; "
          f"verdict {report['verdict']}")
    out = os.environ.get(report_env)
    if out:
        with open(out, "w") as fh:
            json.dump(report, fh, indent=2)
    return 0 if ok else 1
