#!/usr/bin/env python3
"""Deterministic FRONTIER 100-item subset for the Jev calibration study, prereg A8.

FRONTIER runs through the subscription CLI, so its k=3 redraws (A7.2) and its
A4.1 repeat samples cover 100 items, not all 340. The harness's certifying mode
refuses any FRONTIER redraw set that is not exactly this list.

Rule (fixed before any eval-data call):
  * population = the TEST split only (verdicts and draw-sensitivity are read there);
  * strata = slice, plus gold within S1-noul (keeps its yes/no balance);
  * quota = largest-remainder proportional allocation of 100 over stratum sizes;
  * order within a stratum = sha256("frontier-subset-v1:" + item_id) ascending,
    take the first `quota`. No RNG library, nothing version-dependent.

Output: one item_id per line, sorted, plus a sha256 file beside it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

SALT = "frontier-subset-v1:"
N = 100


def stratum(item: dict) -> str:
    if item["slice"] == "S1-noul":
        return f"S1-noul/{item['gold']}"
    return item["slice"]


def allocate(sizes: dict, n: int) -> dict:
    total = sum(sizes.values())
    exact = {k: n * v / total for k, v in sizes.items()}
    quota = {k: int(x) for k, x in exact.items()}
    left = n - sum(quota.values())
    for k in sorted(exact, key=lambda k: (-(exact[k] - quota[k]), k))[:left]:
        quota[k] += 1
    return quota


def select(items: list, n: int = N) -> list:
    groups = defaultdict(list)
    for it in items:
        if it["split"] == "test":
            groups[stratum(it)].append(it["item_id"])
    quota = allocate({k: len(v) for k, v in groups.items()}, n)
    chosen = []
    for k, ids in groups.items():
        ids.sort(key=lambda i: hashlib.sha256((SALT + i).encode()).hexdigest())
        chosen.extend(ids[: quota[k]])
    return sorted(chosen)


def main() -> int:
    d = Path(__file__).resolve().parents[1] / "governance/evals/jev-calibration"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default=str(d / "items-v1.jsonl"))
    ap.add_argument("--out", default=str(d / "frontier-subset-v1.txt"))
    ap.add_argument("--check", action="store_true", help="exit 1 if --out differs from a fresh selection")
    a = ap.parse_args()
    items = [json.loads(line) for line in open(a.items) if line.strip()]
    ids = select(items)
    body = "".join(i + "\n" for i in ids)
    if a.check:
        # compare BYTES (a CRLF copy must not match) and verify the sha256 sidecar too
        want = hashlib.sha256(body.encode()).hexdigest()
        side = Path(a.out + ".sha256")
        same = Path(a.out).read_bytes() == body.encode() and side.exists() and side.read_text().split()[0] == want
        print(f"frontier-subset check: {'MATCH' if same else 'MISMATCH'} ({len(ids)} ids)")
        return 0 if same else 1
    Path(a.out).write_text(body)
    digest = hashlib.sha256(body.encode()).hexdigest()
    Path(a.out + ".sha256").write_text(f"{digest}  {Path(a.out).name}\n")
    idx = {it["item_id"]: it for it in items}
    per = defaultdict(int)
    for i in ids:
        per[stratum(idx[i])] += 1
    print(json.dumps({"n": len(ids), "sha256": digest, "per_stratum": dict(sorted(per.items()))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
