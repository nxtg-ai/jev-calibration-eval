"""Post-hoc: per-item argmax-label agreement of ONE arm across two raw.jsonl run logs.

    python3 -m jevcal.compare_choices A.raw.jsonl B.raw.jsonl --arm LOCAL-P [--draw 1] [--json]

Written for the prereg §9 determinism check (the reported three-arm run versus the disclosed
single-arm LOCAL-P run 193843Z), but generic across arms and runs. It reads the runner's raw
records as written: halt rows are skipped and counted; the label is the record's `answer` (the
argmax option, ties to the lowest option index) or `<abstain>`. Only rows of the requested draw
are compared (default: draw 1, the primary); a row with no `draw` field is a pre-A7 row and is
counted as draw 1. A duplicate (item_id, draw) for the arm is refused, never silently resolved.
Every count is printed with each file's row counts beside it.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

ABSTAIN = "<abstain>"


def load_labels(path: str, arm: str, draw: int = 1) -> Dict[str, Any]:
    rows = halts = other_arm = arm_rows = no_draw_field = 0
    labels: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rows += 1
            rec = json.loads(line)
            if rec.get("halt"):
                halts += 1
                continue
            if rec.get("arm") != arm:
                other_arm += 1
                continue
            arm_rows += 1
            if "draw" not in rec:
                no_draw_field += 1
            if rec.get("draw", 1) != draw:
                continue
            iid = rec["item_id"]
            if iid in labels:
                raise SystemExit(f"REFUSED: {path} line {n}: a second {arm} row for "
                                 f"(item_id={iid}, draw={draw}); which one is primary is ambiguous")
            labels[iid] = ABSTAIN if rec.get("abstained") else rec.get("answer")
    return {"path": path, "rows": rows, "halt_rows": halts, "other_arm_rows": other_arm,
            "arm_rows_all_draws": arm_rows, "rows_without_draw_field": no_draw_field,
            "arm_rows_selected_draw": len(labels), "labels": labels}


def compare(path_a: str, path_b: str, arm: str, draw: int = 1) -> Dict[str, Any]:
    A, B = load_labels(path_a, arm, draw), load_labels(path_b, arm, draw)
    la, lb = A["labels"], B["labels"]
    shared = sorted(set(la) & set(lb))
    mismatches = [{"item_id": i, "a": la[i], "b": lb[i]} for i in shared if la[i] != lb[i]]
    return {"arm": arm, "draw": draw,
            "files": {"a": {k: v for k, v in A.items() if k != "labels"},
                      "b": {k: v for k, v in B.items() if k != "labels"}},
            "shared_items": len(shared),
            "only_in_a": sorted(set(la) - set(lb)), "only_in_b": sorted(set(lb) - set(la)),
            "matches": len(shared) - len(mismatches),
            "match_rate": ((len(shared) - len(mismatches)) / len(shared)) if shared else None,
            "mismatches": mismatches}


def render(res: Dict[str, Any]) -> str:
    fa, fb = res["files"]["a"], res["files"]["b"]
    rows = f"(A rows={fa['rows']}, B rows={fb['rows']})"
    out = [f"arm {res['arm']}, draw {res['draw']}"]
    for tag, f in (("A", fa), ("B", fb)):
        out.append(f"file {tag}: {f['path']}: rows={f['rows']}; halt rows={f['halt_rows']} of {f['rows']}; "
                   f"other-arm rows={f['other_arm_rows']} of {f['rows']}; {res['arm']} rows (all draws)="
                   f"{f['arm_rows_all_draws']} of {f['rows']}; {res['arm']} draw-{res['draw']} items="
                   f"{f['arm_rows_selected_draw']} of {f['rows']}; rows without a draw field="
                   f"{f['rows_without_draw_field']} of {f['rows']}")
    out.append(f"shared items: {res['shared_items']} {rows}")
    out.append(f"only in A: {len(res['only_in_a'])} {rows} {res['only_in_a']}")
    out.append(f"only in B: {len(res['only_in_b'])} {rows} {res['only_in_b']}")
    rate = "n/a (no shared items)" if res["match_rate"] is None else f"{res['match_rate']:.4f}"
    out.append(f"argmax-label matches: {res['matches']} of {res['shared_items']} shared {rows}; "
               f"match rate {rate}")
    out.append(f"mismatches: {len(res['mismatches'])} {rows}")
    for m in res["mismatches"]:
        out.append(f"  {m['item_id']}: A={m['a']} B={m['b']}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("raw_a")
    ap.add_argument("raw_b")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--draw", type=int, default=1)
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    a = ap.parse_args(argv)
    res = compare(a.raw_a, a.raw_b, a.arm, a.draw)
    print(json.dumps(res, indent=2) if a.json else render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
