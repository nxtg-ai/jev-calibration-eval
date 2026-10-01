"""ONE runner: items JSONL in -> per-item raw JSONL + an eval-rail run record out.

    python3 -m jevcal.run --items fixtures/items-fixture.jsonl --arms JEV,FRONTIER \
        --evals-dir <scratch evals dir> --run-id <id> --purpose "<registered purpose>"

Writes, all under --evals-dir (never the real governance/evals; refused):
  runs/<run-id>.raw.jsonl   one record per (item, arm) call, streamed as it returns
  runs/<run-id>.json        the run record (prereg §6 fields + eval-rail gate fields)
  metric-history.jsonl      one run_id-stamped cert-ledger row (appended)
  sampling-registry.json    a byte copy of the repo registry, if absent (GATE F reads it)

HALT (exit 3, no run record): any ArmHalt (non-200, pinned-model mismatch, billing /
quota / spend signal). The partial raw log is kept and ends with a halt record.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List

import numpy as np

from . import scorer as S
from .arms import ARMS, LOCAL_ARMS, vram_preflight
from .arms.base import ArmHalt, canonical_json, install_terminate_handlers, sha256_text
from .items import file_sha256, load_items
from .render import render

HARNESS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # evals/jev-calibration
REPO = os.path.dirname(os.path.dirname(HARNESS_DIR))
SCORER_FILE = os.path.join(HARNESS_DIR, "jevcal", "scorer.py")
RENDER_FILE = os.path.join(HARNESS_DIR, "jevcal", "render.py")
PREREG = "docs/prereg-v1.md"
JUDGE_FAMILY = "deterministic-scorer"
PARITY_POINTS = 0.03  # H2's accuracy band (prereg §5)

RUBRIC_FILE = os.path.join(HARNESS_DIR, "config", "abstention-rubric.json")


def load_abstention_rubric(path: str = RUBRIC_FILE) -> Dict[str, Any]:
    """GATE M input, READ-ONLY (CODEX finding 5, prereg A6.4). operator_endorsed comes
    from the git-tracked config and defaults to false when absent; the harness never
    writes it. A non-bool value is refused rather than coerced."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("rubric"), str) or not cfg["rubric"].strip():
        raise SystemExit(f"REFUSED: {path} must be an object with a non-empty 'rubric'")
    endorsed = cfg.get("operator_endorsed", False)
    if not isinstance(endorsed, bool):
        raise SystemExit(f"REFUSED: {path} operator_endorsed must be a JSON bool, got {endorsed!r}")
    src = cfg.get("source_ref")
    return {"definition": cfg["rubric"], "operator_endorsed": endorsed,
            "source_ref": src if isinstance(src, str) else "",
            "endorsed_by": cfg.get("endorsed_by"), "endorsement_ref": cfg.get("endorsement_ref"),
            "config_path": os.path.relpath(path, REPO), "config_sha256": file_sha256(path)}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", HARNESS_DIR, *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def refuse_real_evals_dir(evals_dir: str, mode: str = "fixture") -> None:
    """The real eval store (governance/evals) is allowed ONLY in --mode certifying, so the run
    record and cert-ledger row land where the internal eval-rail certifier reads them.
    --mode fixture refuses it, so an uncertified run writes only to a scratch evals dir."""
    if mode == "certifying":
        return
    real = {os.path.realpath(os.path.join(REPO, "governance", "evals"))}
    asif_root = os.environ.get("ASIF_ROOT")
    if asif_root:
        real.add(os.path.realpath(os.path.join(asif_root, "governance", "evals")))
    if os.path.realpath(evals_dir) in real:
        raise SystemExit(f"REFUSED: {evals_dir} is the real eval store; --mode fixture writes "
                         "fixture runs to a scratch evals dir only (use --mode certifying)")


def langfuse_setup(run_id: str) -> Dict[str, Any]:
    """Probe once. Unreachable -> a recorded proof and no tracing (fail-open).
    Reachable -> per-call tracing through deploy/langfuse/eval_rail_trace.py."""
    sys.path.insert(0, os.path.join(REPO, "deploy", "langfuse"))
    try:
        import eval_rail_trace as lf  # noqa: WPS433
    except Exception as e:  # SDK or helper missing: neither traced nor a proven outage
        return {"mode": "error", "error": f"helper import failed: {type(e).__name__}: {e}"}
    env_file = os.environ.get("JEVCAL_LANGFUSE_ENV_FILE")
    try:
        _pub, _sec, base = (lf.load_keypair(env_file=__import__("pathlib").Path(env_file))
                            if env_file else lf.load_keypair())
    except Exception as e:
        return {"mode": "error", "error": f"keypair unresolved: {e}"}
    checked_at = lf._now_iso()
    used, endpoint, result, reachable = lf.health_probe(base, attempts=3)
    if not reachable:
        return {"mode": "unreachable", "proof": {"checked_at": checked_at, "endpoint": endpoint,
                                                  "probe_result": result, "attempts": used}}
    return {"mode": "trace", "lf": lf, "env_file": env_file}


EXIT_INVALID = 4


def write_invalid(art_path: str, run_id: str, reason: str, art: Dict[str, Any] = None) -> int:
    """Write a run record marked INVALID with the reason, write NO cert-ledger row,
    and return the non-zero exit code. Used when observability fails closed."""
    rec = dict(art or {"run_id": run_id})
    rec.update({"valid": False, "invalid_reason": reason})
    with open(art_path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"INVALID: {reason}", file=sys.stderr)
    return EXIT_INVALID


def registry_path() -> str:
    return os.path.join(REPO, "governance", "evals", "sampling-registry.json")


def preflight_registry(arms, path: str = None) -> None:
    """Prereg A4.2, generic across arms: every arm's model_id must be a VERBATIM key
    of the sampling registry, checked before the first call of any arm."""
    path = path or registry_path()
    with open(path, "r", encoding="utf-8") as fh:
        reg = json.load(fh)
    keys = set((reg.get("models") or {}).keys()) if isinstance(reg, dict) else set()
    bad = [f"{arm.arm_name}:{arm.model_id!r}" for arm in arms if arm.model_id not in keys]
    if bad:
        raise SystemExit(f"PREFLIGHT FAILED: model_id not a verbatim key of {path}: {bad} "
                         "(prereg A4.2; no arm was called)")


def preflight_isolation(arms) -> Dict[str, Any]:
    """Prereg A5.2: every arm that launches a child process (FRONTIER) must show
    (a) no CLAUDE_CODE_EFFORT_LEVEL in the child env and (b) a child cwd outside
    ~/ASIF with no effort-pinning .claude/settings*.json up its ancestor chain,
    evaluated on the EXACT env and cwd the child will get. Fails before any call."""
    reports = {}
    for arm in arms:
        if hasattr(arm, "isolation_report"):
            rep = arm.isolation_report()
            reports[arm.arm_name] = rep
            if not rep["ok"]:
                raise SystemExit(f"PREFLIGHT FAILED ({arm.arm_name} isolation, prereg A5.2; no arm "
                                 f"was called): " + " | ".join(rep["details"]))
    return reports


def record_vector(rec: Dict[str, Any], options) -> List[float]:
    """A record's probability vector in option order; an abstention is the uniform vector."""
    if rec["abstained"]:
        return [1.0 / len(options)] * len(options)
    return [rec["probabilities"][k] for k in options]


def draw_analysis(arm, items, draws, gold_idx, keys, k, draw_subset) -> Dict[str, Any]:
    """A7.3-A7.5 for one arm, over the items that were re-drawn: jitter, the H1 verdict per
    single-draw run on the TEST split with draw-sensitivity, and the k-averaged variant
    (descriptive only; built as KAveragedRun, which h1_for_draw refuses)."""
    sub = [i for i, it in enumerate(items) if len(draws[arm.arm_name][it.item_id]) == k and k >= 2]
    out = {"k": k, "primary": "draw 1 (first call per item; never best-of-k)",
           "subset_item_ids": sorted(draw_subset[arm.arm_name]) if k >= 2 else [],
           "items_with_k_draws": len(sub)}
    if not sub:
        out["note"] = "k=1 or no item re-drawn: jitter and draw-sensitivity not assessed"
        return out

    def run_for(d, ix):
        return S.DrawRun(d, [record_vector(draws[arm.arm_name][items[i].item_id][d - 1], items[i].options)
                             for i in ix], [gold_idx[i] for i in ix], [items[i].item_id for i in ix],
                         [keys[i] for i in ix],
                         [bool(draws[arm.arm_name][items[i].item_id][d - 1]["abstained"]) for i in ix])
    out["jitter"] = S.jitter(
        [[record_vector(rec, items[i].options) for rec in draws[arm.arm_name][items[i].item_id]] for i in sub],
        [[bool(rec["abstained"]) for rec in draws[arm.arm_name][items[i].item_id]] for i in sub])
    test_sub = [i for i in sub if items[i].split == "test"]
    out["h1_draw_sensitivity"] = (S.h1_across_draws([run_for(d, test_sub) for d in range(1, k + 1)])
                                  if test_sub else None)
    out["h2_draw_sensitivity"] = ("not computed per draw by this runner: reads.h2 reads draw 1 only. "
                                  "Per-draw H2 = S.h2_for_draw per single draw + S.claim_across_draws")
    out["k_averaged_descriptive"] = S.KAveragedRun([run_for(d, sub) for d in range(1, k + 1)]).metrics()
    return out


CERT_K = 3                   # prereg A7.2
CERT_FRONTIER_SUBSET_N = 100  # prereg A7.2 / §2: the fixed 100-item FRONTIER subset
# Prereg A8: the fixed FRONTIER subset is THIS committed file (read by repo path, never
# copied or edited here); its sha256 is recorded in every certifying run record.
FRONTIER_SUBSET_PATH = os.path.join(REPO, "governance", "evals", "jev-calibration",
                                    "frontier-subset-v1.txt")


def certifying_preflight(a, arm_names, all_ids) -> Any:
    """Certifying mode (default). Refuses, before any call, anything that is not the A7
    design: k must be exactly 3; JEV/LOCAL redraw ALL items; FRONTIER redraws exactly the
    fixed 100-item subset in the committed FRONTIER_SUBSET_PATH (A8), whose hash is recorded."""
    if a.draws != CERT_K:
        raise SystemExit(f"REFUSED (certifying): --draws must be exactly {CERT_K} (prereg A7.2); "
                         f"got {a.draws}. Use --mode fixture for anything else.")
    if a.draw_items:
        raise SystemExit("REFUSED (certifying): JEV/LOCAL redraw ALL items (prereg A7.2); "
                         "--draw-items is fixture-only")
    if "FRONTIER" not in arm_names:
        return None
    path = FRONTIER_SUBSET_PATH
    if not os.path.isfile(path):
        raise SystemExit(f"REFUSED (certifying): the committed FRONTIER subset {path} is missing (A8)")
    rel = os.path.relpath(os.path.abspath(path), HARNESS_DIR)
    if git("ls-files", "--", rel) != rel or git("status", "--porcelain", "--", rel):
        raise SystemExit(f"REFUSED (certifying): {path} must be committed and clean (its hash is "
                         "the subset's identity)")
    with open(path, "r", encoding="utf-8") as fh:
        ids = [l.strip() for l in fh if l.strip() and not l.lstrip().startswith("#")]
    if len(ids) != CERT_FRONTIER_SUBSET_N or len(set(ids)) != len(ids):
        raise SystemExit(f"REFUSED (certifying): the FRONTIER subset must be exactly "
                         f"{CERT_FRONTIER_SUBSET_N} distinct item_ids; {path} has {len(ids)} "
                         f"({len(set(ids))} distinct)")
    missing = [x for x in ids if x not in set(all_ids)]
    if missing:
        raise SystemExit(f"REFUSED (certifying): subset ids not in the items file: {missing[:5]}")
    if a.frontier_draw_items is not None and set(
            x.strip() for x in a.frontier_draw_items.split(",") if x.strip()) != set(ids):
        raise SystemExit("REFUSED (certifying): FRONTIER redraws exactly the fixed subset; "
                         "--frontier-draw-items differs")
    return {"path": os.path.relpath(os.path.abspath(path), REPO), "sha256": file_sha256(path),
            "n": len(ids), "ids": ids}


H2_REQUIRED_NON_LOCAL = ("JEV", "FRONTIER")   # prereg §5 H2 + §9 declared order LOCAL-P,JEV,FRONTIER


def certifying_arm_set(arm_names: List[str]) -> None:
    """D1, certifying mode only, checked BEFORE any arm is built or called:
    (a) fewer than 2 arms -> refused: GATE E needs reads.win_gap and reads.parity, which the
        runner computes only from two arms, so such a run can never certify;
    (b) any LOCAL arm without BOTH JEV and FRONTIER -> refused: H2 (prereg §5) compares LOCAL-P
        with the best non-local arm per slice, which needs both non-local arms present."""
    if len(arm_names) < 2:
        raise SystemExit(f"REFUSED (certifying): --arms names {len(arm_names)} arm(s) {arm_names}; "
                         "GATE E needs two arms (reads.win_gap and reads.parity), so this run could "
                         "never certify. No arm was built or called. Use --mode fixture for a "
                         "single-arm run.")
    local = [n for n in arm_names if n in LOCAL_ARMS]
    missing = [n for n in H2_REQUIRED_NON_LOCAL if n not in arm_names]
    if local and missing:
        raise SystemExit(f"REFUSED (certifying): LOCAL arm(s) {local} without both JEV and FRONTIER "
                         f"(missing: {missing}); H2 (prereg §5) compares LOCAL-P with the best "
                         "non-local arm per slice and cannot be answered. Declared order: "
                         "LOCAL-P,JEV,FRONTIER (prereg §9). No arm was built or called.")


def pair_label(pair: List[str], arm_names: List[str]) -> Dict[str, Any]:
    """D1: what the two-arm reads compare, derived from the actual pair (never hard-coded)."""
    in_pair = [n for n in pair if n in LOCAL_ARMS]
    in_run = [n for n in arm_names if n in LOCAL_ARMS]
    text = ((f"LOCAL arm in this pair: {', '.join(in_pair)}" if in_pair else "no LOCAL arm in this pair")
            + "; " + (f"LOCAL arm(s) in this run: {', '.join(in_run)}" if in_run
                      else "no LOCAL arm in this run"))
    return {"pair": list(pair), "local_arms_in_pair": in_pair, "local_arms_in_run": in_run,
            "text": text}


def h2_read(arms, items, records, all_ids, keys, gold_idx, seed: int):
    """D1: reads["h2"], prereg §5 H2 per slice on the TEST split, draw 1 (the primary, A7.1).
    Computed only when LOCAL-P and at least one non-local arm ran; the numbers and the verdict
    come from ONE S.h2_for_draw call per slice. A verdict is withheld (null, with the reason)
    unless both JEV and FRONTIER ran, because §5's comparator is the best of BOTH."""
    names = [arm.arm_name for arm in arms]
    local = "LOCAL-P"
    non_local = [n for n in names if n not in LOCAL_ARMS]
    if local not in names or not non_local:
        return None
    missing = [n for n in H2_REQUIRED_NON_LOCAL if n not in names]
    out = {"hypothesis": "H2 (prereg §5): per slice, LOCAL-P test accuracy within 3 points of the "
                         "best non-local arm AND LOCAL-P's Brier upper-CI <= JEV's Brier upper-CI",
           "local_arm": local, "non_local_arms_considered": non_local, "population": "test",
           "draw": 1, "seed": seed,
           "draw_sensitivity": "not assessed here: this block reads draw 1 only. Per A7.4 a verdict is "
                               "claimed only if it is the same on every single-draw run.",
           "complete": not missing}
    if missing:
        out["verdict_withheld_reason"] = (f"{missing} did not run: H2's comparator is the best of JEV "
                                          "and FRONTIER per slice, and its Brier condition reads JEV")
    if "JEV" not in names:
        out["by_slice"] = None
        return out

    def run_for(name, ix):
        recs = [records[name][all_ids[i]] for i in ix]
        return S.DrawRun(1, [record_vector(r, items[i].options) for r, i in zip(recs, ix)],
                         [gold_idx[i] for i in ix], [all_ids[i] for i in ix], [keys[i] for i in ix],
                         [bool(r["abstained"]) for r in recs])
    by_slice = {}
    test_ix = [i for i, it in enumerate(items) if it.split == "test"]
    for sl in sorted({items[i].slice for i in test_ix}):
        ix = [i for i in test_ix if items[i].slice == sl]
        v = S.h2_for_draw(run_for(local, ix), {n: run_for(n, ix) for n in non_local},
                          seed=seed, local_arm=local)
        v["item_ids"] = [all_ids[i] for i in ix]
        word = v.pop("h2")
        v["verdict"] = word if not missing else None
        by_slice[sl] = v
    out["by_slice"] = by_slice
    return out


def percentile(xs: List[float], q: float):
    return float(np.percentile(np.asarray(xs, dtype=float), q)) if xs else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--items", required=True)
    ap.add_argument("--arms", default="JEV,FRONTIER")
    ap.add_argument("--evals-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--purpose", required=True)
    ap.add_argument("--eval-name", default="jev-calibration")
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--mode", choices=("certifying", "fixture"), default="certifying",
                    help="certifying (default): prereg A7 enforced (k=3, JEV/LOCAL redraw all items, "
                         "FRONTIER redraws exactly the committed 100-item subset), refused otherwise "
                         "before any call. fixture: any k/subset, record says certifying=false, "
                         "NO cert-ledger row is written.")
    ap.add_argument("--frontier-effort", default=None,
                    help="REQUIRED when FRONTIER runs; no default (prereg A4.1)")
    ap.add_argument("--frontier-secondary-items", default="",
                    help="comma-separated item_ids for the A4.1 secondary estimator "
                         "(N independent single-letter samples; descriptive only)")
    ap.add_argument("--frontier-secondary-n", type=int, default=5)
    ap.add_argument("--draws", type=int, default=1,
                    help="k draws per arm (prereg A7.2; E.4 uses 3). Draw 1 is the primary.")
    ap.add_argument("--draw-items", default="",
                    help="item_ids re-drawn for non-FRONTIER arms (default: all items)")
    ap.add_argument("--frontier-draw-items", default=None,
                    help="item_ids re-drawn for FRONTIER (default: the A4.1 secondary subset)")
    ap.add_argument("--local-precheck", action="append", default=[], metavar="NAME=PATH",
                    help="bind a LOCAL arm's A1/A4.3 precheck receipt (repeatable). Alternatively set "
                         "JEVCAL_LOCAL_PRECHECK_<ARM> (hyphens->underscores). Required in certifying mode.")
    a = ap.parse_args(argv)

    # A SIGTERM/SIGINT must unwind through the try/finally that unloads LOCAL arms (keep_alive:0), not
    # kill the process with a model left resident (E.4 first live LOCAL-P run, 2026-09-30).
    install_terminate_handlers()

    refuse_real_evals_dir(a.evals_dir, a.mode)        # real store allowed only in certifying
    items = load_items(a.items, a.mode)               # frozen items: certifying + pinned hash only
    arm_names = [x.strip() for x in a.arms.split(",") if x.strip()]
    unknown = [n for n in arm_names if n not in ARMS]
    if unknown or len(arm_names) < 1:
        raise SystemExit(f"unknown arm(s) {unknown}; registered: {sorted(ARMS)}")
    certifying = a.mode == "certifying"
    if certifying:
        certifying_arm_set(arm_names)                 # D1: before any arm is built or called
    # E.4b: bind LOCAL arm precheck receipts by --local-precheck NAME=PATH or JEVCAL_LOCAL_PRECHECK_<ARM>.
    local_precheck_paths: Dict[str, str] = {}
    for spec in a.local_precheck:
        if "=" not in spec:
            raise SystemExit(f"--local-precheck expects NAME=PATH, got {spec!r}")
        n, p = spec.split("=", 1)
        local_precheck_paths[n.strip()] = p.strip()
    local_preflight: Dict[str, Any] = {}

    def build_arm(n: str):
        if n == "FRONTIER":
            return ARMS[n](effort=a.frontier_effort)
        if n in LOCAL_ARMS:
            model_id = LOCAL_ARMS[n]
            # Prereg §7: the free-VRAM precondition is checked BEFORE the arm is constructed.
            vram = vram_preflight(model_id)
            pc = (local_precheck_paths.get(n)
                  or os.environ.get(f"JEVCAL_LOCAL_PRECHECK_{n.replace('-', '_')}"))
            arm = ARMS[n](precheck_path=pc, require_precheck=certifying, vram_report=vram)
            local_preflight[n] = {"model_id": model_id, "vram": vram, "precheck": arm.precheck}
            return arm
        return ARMS[n]()

    try:
        arms = [build_arm(n) for n in arm_names]
    except ValueError as e:
        raise SystemExit(f"REFUSED: {e}")
    except ArmHalt as e:                              # LOCAL VRAM/precheck/digest refusal (before any call)
        raise SystemExit(f"REFUSED: {e}")
    preflight_registry(arms)                          # before the first call of any arm
    isolation_reports = preflight_isolation(arms)     # A5.2, on the exact child env + cwd
    secondary_ids = [x.strip() for x in a.frontier_secondary_items.split(",") if x.strip()]
    if a.draws < 1:
        raise SystemExit("--draws must be >= 1")
    all_ids = [it.item_id for it in items]
    certifying = a.mode == "certifying"
    frontier_subset = None
    if certifying:   # CODEX re-grade P1-3: A7 is enforced, not advisory; refused before any call
        frontier_subset = certifying_preflight(a, arm_names, all_ids)
        if frontier_subset is not None:
            subset_ids = frontier_subset["ids"]
            if secondary_ids and set(secondary_ids) != set(subset_ids):
                raise SystemExit("REFUSED (certifying): the A4.1 secondary runs on the fixed FRONTIER "
                                 "subset; --frontier-secondary-items differs from it")
            secondary_ids = list(subset_ids)

    def id_list(spec, default):
        ids = [x.strip() for x in spec.split(",") if x.strip()] if spec else list(default)
        bad = [x for x in ids if x not in all_ids]
        if bad:
            raise SystemExit(f"draw item_ids not in the items file: {bad}")
        return set(ids)
    # A7.2: JEV/LOCAL re-draw all items by default; FRONTIER only its fixed subset
    # (the A4.1 subset), because subscription latency makes k draws over all items infeasible.
    draw_subset = {n: (id_list(a.frontier_draw_items if a.frontier_draw_items is not None else "",
                               frontier_subset["ids"] if frontier_subset else secondary_ids)
                       if n == "FRONTIER" else id_list(a.draw_items, all_ids))
                   for n in arm_names}
    if secondary_ids:
        if "FRONTIER" not in arm_names:
            raise SystemExit("--frontier-secondary-items needs the FRONTIER arm")
        missing = [x for x in secondary_ids if x not in {it.item_id for it in items}]
        if missing:
            raise SystemExit(f"secondary item_ids not in the items file: {missing}")

    runner_sha = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", "jevcal", "fixtures", "config")
    if dirty:
        raise SystemExit("REFUSED: harness code or fixtures are uncommitted; runner_git_sha "
                         "must name the exact code that ran:\n" + dirty)

    runs_dir = os.path.join(a.evals_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    art_path = os.path.join(runs_dir, f"{a.run_id}.json")
    raw_path = os.path.join(runs_dir, f"{a.run_id}.raw.jsonl")
    if os.path.exists(art_path) or os.path.exists(raw_path):
        raise SystemExit(f"REFUSED: run {a.run_id} already has artifacts in {runs_dir}")

    rubric = load_abstention_rubric()
    families = [arm.family for arm in arms]
    if len(set(families)) != len(families):
        raise SystemExit(f"two arms share a family {families}; pins.contestants is keyed by family")
    arm_cfg = {arm.arm_name: {"model": arm.model_id, "family": arm.family, "config": arm.config(),
                              "config_sha256": arm.config_sha256(),
                              "sampling_config": arm.sampling_config()} for arm in arms}
    arm_config_sha = sha256_text(canonical_json({k: v["config_sha256"] for k, v in arm_cfg.items()}))
    scorer_sha = file_sha256(SCORER_FILE)
    judge_id = f"deterministic-scorer@{runner_sha[:12]}"
    renderer_id = f"prompt-renderer@{runner_sha[:12]}"
    lfs = langfuse_setup(a.run_id)
    trace_ids: List[str] = []
    lf_errors: List[str] = []
    if lfs["mode"] == "error":
        # Neither traceable nor a proven outage: GATE O could only be met by a fake proof.
        # Abort BEFORE any arm is called (CODEX PR 72 finding 4).
        return write_invalid(art_path, a.run_id, "langfuse neither traceable nor provably "
                             f"unreachable before the first call: {lfs.get('error')}; no arm was called")

    render_calls = {"n": 0}

    def counted_render(it):
        render_calls["n"] += 1
        return render(it)

    # One render per item, shared by every arm (arms never render); the grader
    # re-renders once more per item below. Liveness reports the COUNTED total.
    rendered = {it.item_id: counted_render(it) for it in items}

    # E.4b §2: LOCAL arms pin num_ctx, so BEFORE any call refuse the run if any item's estimated
    # prompt would exceed the pinned window (never truncate silently). Pure local estimate, no calls.
    for _arm in arms:
        _numctx_pf = getattr(_arm, "numctx_preflight", None)
        if _numctx_pf is not None:
            try:
                _numctx_rep = _numctx_pf(list(rendered.values()))
            except ArmHalt as _e:
                raise SystemExit(f"REFUSED: {_e}")
            if _arm.arm_name in local_preflight:
                local_preflight[_arm.arm_name]["numctx"] = _numctx_rep

    records: Dict[str, Dict[str, Any]] = {n: {} for n in arm_names}
    started = datetime.datetime.now(datetime.timezone.utc)

    def do_call(arm, it, r, draw, raw):
        """One arm call on one item as draw `draw`; returns the record, or None on HALT."""
        def model_call():
            return arm.call(r)
        try:
            if lfs["mode"] == "trace":
                box = {}

                def mc():
                    box["res"] = arm.call(r)
                    return box["res"].to_record(r)["probabilities"]

                def jc(_out):
                    res = box["res"]
                    return {"answer_index": res.chosen, "gold_index": it.gold_index,
                            "correct": (not res.abstained) and res.chosen == it.gold_index}
                from pathlib import Path
                kw = {"env_file": Path(lfs["env_file"])} if lfs.get("env_file") else {}
                try:
                    tr = lfs["lf"].trace_eval_run(
                        run_id=f"{a.run_id}:{arm.arm_name}:{it.item_id}:d{draw}",
                        model_id=arm.model_id, task_input=r.canonical_text,
                        model_call=mc, judge_call=jc, judge_id=judge_id, **kw)
                    trace_ids.extend(tr.get("langfuse_trace_ids", []))
                    if "langfuse_unreachable_proof" in tr:
                        lf_errors.append("mid-run outage: " + json.dumps(tr["langfuse_unreachable_proof"]))
                    res = box.get("res") or model_call()
                except ArmHalt:
                    raise
                except Exception as e:  # reachable but tracing failed: recorded, never faked
                    lf_errors.append(f"{type(e).__name__}: {e}")
                    res = box.get("res") or model_call()
            else:
                res = model_call()
        except ArmHalt as e:
            raw.write(json.dumps({"halt": True, "arm": arm.arm_name, "item_id": it.item_id,
                                  "draw": draw, "reason": str(e)}) + "\n")
            raw.flush()
            print(f"HALT: {e}", file=sys.stderr)
            return None
        rec = res.to_record(r)
        rec.update({"draw": draw, "slice": it.slice, "split": it.split, "gold": it.gold,
                    "question_type": it.question_type, "tags": list(it.tags)})
        raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
        raw.flush()
        return rec

    # draws[arm][item_id] = [record of draw 1, draw 2, ...]; draw 1 is the PRIMARY (A7.1)
    draws: Dict[str, Dict[str, List[Dict[str, Any]]]] = {n: {} for n in arm_names}

    def _close_arms():
        # E.4b: unload any arm that holds a resident model (LOCAL close() -> keep_alive:0) on EVERY
        # exit path, including an ArmHalt (return 3) or the identity-invalidation SystemExit. Best
        # effort: a close() failure is logged, never raised into this finally.
        for _arm in arms:
            _close = getattr(_arm, "close", None)
            if callable(_close):
                try:
                    _close()
                except Exception as _e:  # noqa: BLE001
                    print(f"close() warning ({_arm.arm_name}): {_e}", file=sys.stderr)

    try:
        with open(raw_path, "a", encoding="utf-8") as raw:
            for it in items:
                r = rendered[it.item_id]
                for arm in arms:
                    rec = do_call(arm, it, r, 1, raw)
                    if rec is None:
                        return 3
                    records[arm.arm_name][it.item_id] = rec          # the primary = draw 1
                    draws[arm.arm_name][it.item_id] = [rec]
            # A7.2: re-draws 2..k, only to measure jitter / draw-sensitivity; never the primary
            for d in range(2, a.draws + 1):
                for it in items:
                    r = rendered[it.item_id]
                    for arm in arms:
                        if it.item_id not in draw_subset[arm.arm_name]:
                            continue
                        rec = do_call(arm, it, r, d, raw)
                        if rec is None:
                            return 3
                        draws[arm.arm_name][it.item_id].append(rec)

        # ---------------- A4.1 secondary estimator (descriptive only; feeds no H-test) ----------------
        secondary: Dict[str, Any] = {}
        if secondary_ids:
            fr = next(arm for arm in arms if arm.arm_name == "FRONTIER")
            sec_path = os.path.join(runs_dir, f"{a.run_id}.secondary.jsonl")
            with open(sec_path, "a", encoding="utf-8") as sec:
                for iid in secondary_ids:
                    r = rendered[iid]
                    samples = []
                    for k in range(a.frontier_secondary_n):
                        try:
                            s = fr.sample_answer(r, k)
                        except ArmHalt as e:
                            sec.write(json.dumps({"halt": True, "item_id": iid, "sample_index": k,
                                                  "reason": str(e)}) + "\n")
                            print(f"HALT (secondary): {e}", file=sys.stderr)
                            return 3
                        sec.write(json.dumps(s, ensure_ascii=False) + "\n")
                        sec.flush()
                        samples.append(s)
                    valid = [s["answer"] for s in samples if s["answer"] is not None]
                    secondary[iid] = {
                        "estimator": "5 independent claude -p samples at provider-default sampling "
                                     "(prereg A4.1); empirical answer frequency",
                        "descriptive_only": True, "n_samples": len(samples), "n_valid": len(valid),
                        "frequencies": {key: (valid.count(key) / len(valid) if valid else None)
                                        for key in r.keys},
                        "effort": fr.effort, "temperature": samples[0]["temperature"],
                        "gold": next(it.gold for it in items if it.item_id == iid)}
            secondary_log = {"path": f"runs/{a.run_id}.secondary.jsonl", "sha256": file_sha256(sec_path)}

        # ---------------- arm identity must not move mid-run (prereg §2, A4.2, A5.1) ----------------
        for arm in arms:
            recs = [rec for it in items for rec in draws[arm.arm_name][it.item_id]]   # every draw
            digests = {x["model_digest"] for x in recs}
            efforts = {(x.get("sampling_config") or {}).get("effort") for x in recs}
            if len(digests) != 1 or len(efforts) != 1:
                raise SystemExit(f"INVALIDATED: {arm.arm_name} identity changed mid-run "
                                 f"(model_digest {sorted(map(str, digests))}, effort {sorted(map(str, efforts))}); "
                                 "a version change invalidates the arm and it re-runs (prereg §2)")
    finally:
        _close_arms()

    # ---------------- grading (deterministic, against gold; no arm grades an arm) ----------------
    tasks, per_arm = [], {}
    for it in items:
        r = counted_render(it)  # recomputed from the items file: the grader's own view of user-seen text
        legs, pmc, saw = {}, {}, True
        for arm in arms:
            rec = records[arm.arm_name][it.item_id]
            saw = saw and rec["msg_sha256"] == r.msg_sha256
            ot = rec.get("output_tokens")
            legs[arm.model_id] = {"msg_sha256": rec["msg_sha256"], "wire_sha256": rec["wire_sha256"],
                                  **({"compl_tok": ot} if isinstance(ot, int) else {})}
            pmc[arm.model_id] = (not rec["abstained"]) and rec["answer"] == it.gold
        tasks.append({"task_id": it.item_id, "slice": it.slice, "split": it.split, "gold": it.gold,
                      "graded_by": "exact_match", "per_model_correct": pmc, "legs": legs,
                      "user_seen_full": r.canonical_text, "truncated": False,
                      "judge_saw_user_seen": saw})

    gold_idx = [it.gold_index for it in items]
    keys = [list(it.options) for it in items]
    splits = [it.split for it in items]
    brier_items = {}
    for arm in arms:
        recs = [records[arm.arm_name][it.item_id] for it in items]
        ab = [bool(x["abstained"]) for x in recs]
        probs = [([x["probabilities"][k] for k in it.options] if not x["abstained"]
                  else [1.0 / len(it.options)] * len(it.options)) for x, it in zip(recs, items)]
        overall = S.score_arm(probs, gold_idx, all_ids, keys, seed=a.seed, abstain=ab)
        overall["population"] = "all_splits"
        test_ix = [i for i, it in enumerate(items) if it.split == "test"]
        # Advisor review P1: the headline population is the TEST split (prereg §5); the
        # pooled block above stays, labelled all_splits.
        test_block = (S.score_arm([probs[i] for i in test_ix], [gold_idx[i] for i in test_ix],
                                  [all_ids[i] for i in test_ix], [keys[i] for i in test_ix],
                                  seed=a.seed, abstain=[ab[i] for i in test_ix]) if test_ix else None)
        if test_block is not None:
            test_block["population"] = "test"
        by_slice = {}
        for sl in sorted({it.slice for it in items}):
            ix = [i for i, it in enumerate(items) if it.slice == sl]
            by_slice[sl] = S.point_metrics([probs[i] for i in ix], [gold_idx[i] for i in ix],
                                           [keys[i] for i in ix], [ab[i] for i in ix])

        def split_metrics(ix):
            if not ix:
                return None
            m = S.point_metrics([probs[i] for i in ix], [gold_idx[i] for i in ix],
                                [keys[i] for i in ix], [ab[i] for i in ix])
            # the verdict comes ONLY from h1_for_draw on the primary single-draw run (draw 1)
            m["h1_verdict"] = S.h1_for_draw(S.DrawRun(1, [probs[i] for i in ix], [gold_idx[i] for i in ix],
                                                      [all_ids[i] for i in ix], [keys[i] for i in ix],
                                                      [ab[i] for i in ix]))
            return m
        # H1 is judged on the TEST split (prereg §5), pooled and per slice; each block
        # carries both ECEs, realized bins and the binning-sensitive verdict (A6.1/A6.2).
        h1_test = {"pooled": split_metrics(test_ix),
                   "by_slice": {sl: split_metrics([i for i in test_ix if items[i].slice == sl])
                                for sl in sorted({items[i].slice for i in test_ix})}}
        lat = [x["latency_s"] for x in recs]
        per_arm[arm.arm_name] = {
            "model": arm.model_id, "probability_source": arm.config().get("probability_source"),
            "raw": overall, "test": test_block, "by_slice": by_slice, "test_split_h1": h1_test,
            "temperature_scaled_secondary": {
                **S.temperature_scaled(probs, gold_idx, splits, ab),
                "population": "test (T fitted on the calibration split)",
                "compare_with": "metrics.<arm>.test (raw, same test population)"},
            "latency_s": {"p50": percentile(lat, 50), "p95": percentile(lat, 95),
                          "clock": "client wall-clock"},
            "tokens": {"input_mean": float(np.mean([x["input_tokens"] or 0 for x in recs])),
                       "output_mean": float(np.mean([x["output_tokens"] or 0 for x in recs]))},
            "cost_usd_list_total": float(sum(x["cost_usd_list"] or 0.0 for x in recs)),
            "returned_models": sorted({x["returned_model"] for it in items
                                       for x in draws[arm.arm_name][it.item_id]}),
            "executed_calls": sum(len(draws[arm.arm_name][it.item_id]) for it in items),
            "draws": draw_analysis(arm, items, draws, gold_idx, keys, a.draws, draw_subset),
        }
        brier_items[arm.arm_name] = S.brier_per_item(probs, gold_idx)    # draw 1 (A7.1)

    reads = {}
    test_ix = [i for i, it in enumerate(items) if it.split == "test"]
    test_ids = [all_ids[i] for i in test_ix]
    if len(arms) >= 2 and test_ix:
        # Advisor review P1: both reads on the TEST population, arms joined by the test ids.
        A, B = arms[0].arm_name, arms[1].arm_name
        pb = S.paired_bootstrap(brier_items[A][test_ix], brier_items[B][test_ix], seed=a.seed,
                                ids_a=test_ids, ids_b=test_ids)
        pl = pair_label([A, B], arm_names)
        reads["win_gap"] = {"metric": f"paired bootstrap of per-item Brier, {A} minus {B} "
                                      f"(negative = first arm better calibrated+accurate; {pl['text']})",
                            "pair": pl, "population": "test", "item_ids": test_ids, **pb}
        acc_a = np.array([tasks[i]["per_model_correct"][arms[0].model_id] for i in test_ix], dtype=float)
        acc_b = np.array([tasks[i]["per_model_correct"][arms[1].model_id] for i in test_ix], dtype=float)
        pa = S.paired_bootstrap(acc_a, acc_b, seed=a.seed + 1, ids_a=test_ids, ids_b=test_ids)
        reads["parity"] = {"metric": f"accuracy difference {A} minus {B}, H2-shaped band "
                                     f"±{PARITY_POINTS:.2f} ({pl['text']}). A pairwise GATE E read, "
                                     "not the H2 verdict (H2 is reads.h2, present only when LOCAL-P "
                                     "and a non-local arm ran)",
                           "pair": pl, "population": "test", "item_ids": test_ids, "n_items": pa["n_items"],
                           "band": PARITY_POINTS, "diff": pa["mean_diff"],
                           "ci95": [pa["lo"], pa["hi"]],
                           "within_band_point": bool(abs(pa["mean_diff"]) <= PARITY_POINTS),
                           "within_band_ci": bool(-PARITY_POINTS <= pa["lo"] and pa["hi"] <= PARITY_POINTS)}

    h2 = h2_read(arms, items, records, all_ids, keys, gold_idx, a.seed) if test_ix else None
    if h2 is not None:
        reads["h2"] = h2

    # ---------------- run record ----------------
    reg_src = registry_path()
    reg_dst = os.path.join(a.evals_dir, "sampling-registry.json")
    if not os.path.exists(reg_dst):
        shutil.copyfile(reg_src, reg_dst)
    liveness = {
        arm.model_id: {"executed_calls": per_arm[arm.arm_name]["executed_calls"],
                       "evidence": f"runs/{a.run_id}.raw.jsonl: {per_arm[arm.arm_name]['executed_calls']} "
                                   f"records, returned_model {per_arm[arm.arm_name]['returned_models']}"}
        for arm in arms}
    liveness[judge_id] = {"executed_calls": len(tasks),
                          "evidence": f"{len(tasks)} tasks graded exact_match against gold; "
                                      f"scorer file sha256 {scorer_sha}"}
    liveness[renderer_id] = {"executed_calls": render_calls["n"],
                             "evidence": f"{render_calls['n']} counted render() calls: one per item shared "
                                         f"by all {len(arms)} arms + one grader re-render per item; every "
                                         f"leg's msg_sha256 matches the grader's re-render: "
                                         f"{all(t['judge_saw_user_seen'] for t in tasks)}"}
    date = started.strftime("%Y-%m-%d")
    art = {
        "run_id": a.run_id, "date": date, "eval": a.eval_name,
        "config": f"arms={','.join(arm_names)};items_sha256={file_sha256(a.items)[:16]};"
                  f"arm_config_sha256={arm_config_sha[:16]}",
        "mode": a.mode, "certifying": certifying, "a7_compliant": certifying,
        "fixture": not certifying,
        "frontier_subset": ({k: v for k, v in frontier_subset.items() if k != "ids"}
                            if frontier_subset else None),
        "registered_purpose": a.purpose, "preregistration": PREREG,
        "started_at": started.isoformat(),
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runner_git_sha": runner_sha,
        "items_path": os.path.relpath(os.path.abspath(a.items), REPO),
        "items_sha256": file_sha256(a.items), "items_n": len(items),
        "arm_configs": arm_cfg, "arm_config_sha256": arm_config_sha,
        "max_tokens": {arm.model_id: arm.sampling_config().get("max_tokens") for arm in arms},
        "max_tokens_note": {arm.model_id: arm.sampling_config().get("note") for arm in arms},
        "graded_by": judge_id, "scorer_file_sha256": scorer_sha,
        "pins": {
            "judge_id": judge_id, "judge_family": JUDGE_FAMILY,
            "contestant_families": families,
            "contestants": {arm.family: arm.model_id for arm in arms},
            "per_model_gen_params": {arm.model_id: arm.sampling_config() for arm in arms},
            "abstention_rubric": rubric, "seed": a.seed,
        },
        "pinned_components": [{"name": n} for n in [arm.model_id for arm in arms] + [judge_id, renderer_id]],
        "liveness": liveness,
        "judge_context_sighted": all(t["judge_saw_user_seen"] for t in tasks),
        "judge_context_note": "The judge is deterministic code, not a model: it re-renders every item "
                              "from the items file, verifies each leg's msg_sha256 against that render, "
                              "and grades argmax against gold (exact_match).",
        "tasks": tasks,
        "reads": reads,
        "metrics": per_arm,
        "raw_log": {"path": f"runs/{a.run_id}.raw.jsonl", "sha256": file_sha256(raw_path),
                    "records": sum(len(r) for v in draws.values() for r in v.values()),
                    "primary_records": sum(len(v) for v in records.values())},
        "draws_config": {"k": a.draws, "primary": "draw 1",
                         "subset_item_ids": {n: sorted(s) for n, s in draw_subset.items()},
                         "k_averaged": "descriptive only; no H-test reads it (A7.5)"},
        "sampling_registry": {"source": "governance/evals/sampling-registry.json",
                              "sha256": file_sha256(reg_dst),
                              "preflight": "every arm model_id is a verbatim registry key (A4.2)"},
        "model_identity": {arm.model_id: {
            "arm": arm.arm_name, "serving_tag": arm.serving_tag,
            "model_digests_observed": sorted({records[arm.arm_name][it.item_id]["model_digest"]
                                              for it in items})} for arm in arms},
        "frontier_isolation": isolation_reports or None,
        "local_preflight": local_preflight or None,   # E.4b: per-LOCAL-arm VRAM (§7) + precheck (A1/A4.3)
        "frontier_secondary": secondary or None,
        **({"frontier_secondary_log": secondary_log} if secondary else {}),
        "langfuse_status": lfs["mode"] if not lf_errors else f"{lfs['mode']} with errors",
        "langfuse_errors": lf_errors or ([lfs["error"]] if lfs["mode"] == "error" else []),
    }
    if trace_ids:
        art["langfuse_trace_ids"] = trace_ids
    if lfs["mode"] == "unreachable":
        art["langfuse_unreachable_proof"] = lfs["proof"]
    # CODEX PR 72 finding 4: LangFuse is fail-OPEN only for a PROVEN outage. A run that
    # ends with neither trace ids nor a valid unreachable proof, or that traced some
    # calls and silently failed others while reachable, is INVALID: record says why,
    # no cert-ledger row, non-zero exit.
    if lfs["mode"] == "trace" and (lf_errors or not trace_ids):
        return write_invalid(art_path, a.run_id,
                             f"langfuse reachable but tracing incomplete: {len(trace_ids)} trace id(s), "
                             f"{len(lf_errors)} tracing error(s): {lf_errors[:3]}", art)
    if not trace_ids and "langfuse_unreachable_proof" not in art:
        return write_invalid(art_path, a.run_id, "run ended with neither langfuse trace ids nor a "
                             "valid langfuse_unreachable_proof", art)
    art["valid"] = True
    with open(art_path, "w", encoding="utf-8") as fh:
        json.dump(art, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    if not certifying:
        # CODEX re-grade P1-3: a fixture-mode run is NOT A7-compliant and must never enter
        # the cert ledger; without a ledger row GATE A refuses it and doctor ignores it.
        print(f"run record: {art_path} (mode=fixture: certifying=false, NO cert-ledger row written)")
        print(f"raw log:    {raw_path} ({art['raw_log']['records']} records)")
        return 0
    def headline(block):
        if block is None:
            return None
        return {"population": block["population"], "n": block["n"], "accuracy": block["accuracy"],
                "brier": block["brier"], "ece_equal_mass_15": block["ece_equal_mass_15"],
                "brier_ci95": [block["ci95"]["brier"]["lo"], block["ci95"]["brier"]["hi"]]}
    # Advisor review P1: the ledger headline is the TEST split (prereg §5); pooled values
    # are kept only under the labelled all_splits key.
    row = {"run_id": a.run_id, "date": date, "eval": a.eval_name, "config": art["config"],
           "mode": a.mode, "N": len(items), "N_test": len(test_ix), "population": "test",
           "arms": {k: {"model": v["model"], **(headline(v["test"]) or {"population": "test", "n": 0}),
                        "all_splits": headline(v["raw"])}
                    for k, v in per_arm.items()},
           "win_gap": reads.get("win_gap"), "parity": reads.get("parity"),
           "graded_by": judge_id, "runner_git_sha": runner_sha, "items_sha256": art["items_sha256"],
           "key_levers": "certifying run (prereg A7 enforced)"}
    with open(os.path.join(a.evals_dir, "metric-history.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"run record: {art_path}")
    print(f"raw log:    {raw_path} ({art['raw_log']['records']} records)")
    for k, v in per_arm.items():
        h = v["test"] or v["raw"]
        print(f"{k:<9} [{h['population']}] acc={h['accuracy']:.3f} brier={h['brier']:.3f} "
              f"ece15={h['ece_equal_mass_15']:.3f} p50={v['latency_s']['p50']:.2f}s "
              f"abst={h['abstentions']} cost_list=${v['cost_usd_list_total']:.6f}")
    print(f"langfuse: {art['langfuse_status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
