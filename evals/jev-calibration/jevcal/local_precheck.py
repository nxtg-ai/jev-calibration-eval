"""LOCAL arm A1/A4.3 pre-run check (the study author). Uses NO eval data.

    python3 -m jevcal.local_precheck --model-id qwen3-14b --out <receipt.json>

Runs the ONE committed synthetic probe item (fixtures/local-precheck-probe.jsonl, whose passage
repeats the option letters A/B/C so a context repetition penalty has something to act on) through
the same LocalArm request builder, four times:
  (a) registry config, twice   -> same-seed repeatability;
  (b) registry config, temperature = 1.0;
  (c) registry config, penalties neutralised (presence/frequency 0, repeat_penalty 1.0).

Findings (A1/A4.3), each decided from the first generated token's label logprobs:
  temperature_basis   : pre-temperature if (a)==(b) within 1e-6, else post-temperature;
  penalty_basis       : penalties-not-applied-to-logprobs if (a)==(c), else post-sampling-chain;
  repeatable          : (a) call 1 == call 2 within 1e-6;
  label_first_token_feasible : some valid label in the first token's top_logprobs with total
                               label mass >= 0.5.

The receipt records model_id, serving_tag, digest, ollama version, the options sent on each call,
the raw responses and the findings, with a UTC timestamp. In `--mode certifying` the runner REQUIRES
a receipt whose model_id, digest and options match the arm and whose label_first_token_feasible is
true, before the first call (see run.py). This module writes the receipt; it never certifies itself.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict, List, Optional

from .arms import local as L
from .arms.base import ArmHalt, install_terminate_handlers
from .items import load_items
from .render import Rendered, render

HARNESS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROBE = os.path.join(HARNESS_DIR, "fixtures", "local-precheck-probe.jsonl")


def label_logprobs(top: List[Dict[str, Any]], labels: List[str]) -> Dict[str, float]:
    """Per valid label, the highest logprob among tokens whose strip() equals it (absent labels
    omitted). This is the quantity that is invariant under temperature iff logprobs are reported
    pre-temperature, so equality within 1e-6 across calls is the basis test."""
    label_set = set(labels)
    out: Dict[str, float] = {}
    for e in top or []:
        if not isinstance(e, dict):
            continue
        tok, lp = e.get("token"), e.get("logprob")
        if not isinstance(tok, str) or isinstance(lp, bool) or not isinstance(lp, (int, float)):
            continue
        s = tok.strip()
        if s in label_set:
            out[s] = lp if s not in out else max(out[s], float(lp))
    return out


def _eq(d1: Dict[str, float], d2: Dict[str, float], tol: float = 1e-6) -> bool:
    return set(d1) == set(d2) and all(abs(d1[k] - d2[k]) <= tol for k in d1)


UNDETERMINED = "undetermined (cache state differs)"


def _is_warm(call: Dict[str, Any]) -> bool:
    """A call is WARM when its prompt prefix was (almost fully) KV-cached — the state the findings
    compare in. `prompt_eval_cached_count >= prompt_eval_count - 1` (the single generated position is
    never cached, so -1). A COLD call (cached 0) differs from a warm call by cache state, not by
    temperature or penalties: the first live receipt (qwen3:14b 2026-09-30T04:03Z) read a1 COLD
    (cached 0/139, 15 s load) and a2/b/c WARM (138/139), so a1-as-baseline mislabelled the cold/warm
    delta as post-temperature/post-sampling-chain. Comparisons must be warm-vs-warm."""
    pe, cc = call.get("prompt_eval_count"), call.get("prompt_eval_cached_count")
    return (isinstance(pe, int) and not isinstance(pe, bool)
            and isinstance(cc, int) and not isinstance(cc, bool) and cc >= pe - 1)


def _label_probs(top: List[Dict[str, Any]], labels: List[str]) -> Dict[str, float]:
    mass = L.label_masses(top, labels)
    s = sum(mass.values())
    return {l: (mass[l] / s if s > 0 else 0.0) for l in labels}


def compute_findings(warmup, a1, a2, b, c, labels: List[str]) -> Dict[str, Any]:
    """Findings compare WARM calls only (real KV-cache confound, above). Each of warmup/a1/a2/b/c is a
    call dict carrying top_logprobs + prompt_eval_count + prompt_eval_cached_count. a1 is the warm
    baseline; warmup is the discarded cold call, kept only to measure cache_state_effect. If any
    COMPARED call (a1/a2/b/c) is not warm, that basis is UNDETERMINED, never a label — and the
    certifying binding refuses an undetermined basis."""
    ll = lambda call: label_logprobs(call.get("top_logprobs") or [], labels)
    a1l, a2l, bl, cl, wl = ll(a1), ll(a2), ll(b), ll(c), ll(warmup)

    def basis(xl, yl, xcall, ycall, eq_label, neq_label):
        if not (_is_warm(xcall) and _is_warm(ycall)):
            return UNDETERMINED
        return eq_label if _eq(xl, yl) else neq_label

    def max_delta(x, y):
        keys = set(x) & set(y)
        return max((abs(x[k] - y[k]) for k in keys), default=0.0)

    wp = _label_probs(warmup.get("top_logprobs") or [], labels)
    a1p = _label_probs(a1.get("top_logprobs") or [], labels)
    total_mass = sum(L.label_masses(a1.get("top_logprobs") or [], labels).values())
    return {
        "temperature_basis": basis(a1l, bl, a1, b, "pre-temperature", "post-temperature"),
        "penalty_basis": basis(a1l, cl, a1, c, "penalties-not-applied-to-logprobs", "post-sampling-chain"),
        "repeatable": (_eq(a1l, a2l) if (_is_warm(a1) and _is_warm(a2)) else UNDETERMINED),
        "cache_state_effect": {"max_delta_logprob": max_delta(wl, a1l),
                               "max_delta_p": max((abs(wp[l] - a1p[l]) for l in labels), default=0.0)},
        "label_first_token_feasible": bool(a1l) and total_mass >= 0.5,
    }


def run_precheck(arm: "L.LocalArm", r: Rendered) -> Dict[str, Any]:
    """Make the calls through the arm's own request builder and assemble the receipt."""
    def one(options: Dict[str, Any]) -> Dict[str, Any]:
        _lat, resp, _sha = arm._post_chat(arm.chat_body(r, options))
        arm._guard_served_model(resp)
        top = L.first_token_top_logprobs(resp) or []
        return {"options": options, "response": resp, "top_logprobs": top,
                "label_logprobs": label_logprobs(top, r.labels),
                "label_mass_total": sum(L.label_masses(top, r.labels).values()),
                "prompt_eval_count": resp.get("prompt_eval_count"),
                "prompt_eval_cached_count": resp.get("prompt_eval_cached_count")}
    opts = dict(arm.options)
    # Call (b) MUST probe at a temperature that DIFFERS from the registry temperature, or the
    # temperature_basis test is vacuous. gpt-oss:20b's registry temperature is now 1.0 (#80), so a
    # fixed 1.0 probe would read "pre-temperature" by construction. Use 1.0 unless the registry temp
    # is already 1.0, then 0.5 (review note 2). Recorded in the receipt.
    reg_temp = opts.get("temperature")
    b_probe_temp = 1.0 if reg_temp != 1.0 else 0.5
    # a0: a DISCARDED warm-up so a1/a2/b/c all hit a warm KV cache. Without it a1 is the only cold
    # call and its logprobs differ by cache state, confounding every basis (the live 04:03Z receipt).
    warmup = one(dict(opts))
    a1 = one(dict(opts))
    a2 = one(dict(opts))
    b = one({**opts, "temperature": b_probe_temp})
    c = one({**opts, "presence_penalty": 0, "frequency_penalty": 0, "repeat_penalty": 1.0})
    findings = compute_findings(warmup, a1, a2, b, c, r.labels)
    return {
        "model_id": arm.model_id, "serving_tag": arm.serving_tag, "model_digest": arm.pinned_digest,
        "ollama_version": arm.ollama_version, "base_url": arm.base_url, "options": opts,
        "registry_temperature": reg_temp, "b_probe_temperature": b_probe_temp,
        "probe_item_id": r.item_id, "labels": list(r.labels),
        "calls": {"warmup": warmup, "a1": a1, "a2": a2, "b": b, "c": c},
        "findings": findings,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-id", required=True, help="a LOCAL sampling-registry key, e.g. qwen3-14b")
    ap.add_argument("--out", required=True, help="path for the JSON receipt")
    ap.add_argument("--items", default=DEFAULT_PROBE, help="the synthetic probe item (no eval data)")
    ap.add_argument("--base-url", default=None, help="override JEVCAL_OLLAMA_BASE_URL")
    a = ap.parse_args(argv)

    # The precheck loads the model too, so a SIGTERM/SIGINT must unwind through the finally that
    # unloads it (keep_alive:0), never leave it resident (same fix as the runner).
    install_terminate_handlers()

    arm_name = next((n for n, mid in L.LOCAL_ARMS.items() if mid == a.model_id), "LOCAL-PRECHECK")
    # Prereg §7: the free-VRAM precondition must hold BEFORE any local model is loaded. The precheck
    # loads the model (four /api/chat calls), so it enforces the SAME gate as the runner's build_arm
    # path, with the same JEVCAL_VRAM_PROBE env var and the same refuse semantics. An unset probe or a
    # short reading is a clean REFUSED — non-zero exit, no model load, no receipt written.
    try:
        vram = L.vram_preflight(a.model_id, base_url=a.base_url)
    except ArmHalt as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    try:
        arm = L.LocalArm(arm_name=arm_name, model_id=a.model_id, base_url=a.base_url, vram_report=vram)
    except ArmHalt as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    r = render(load_items(a.items)[0])
    try:
        receipt = run_precheck(arm, r)
    finally:
        close = getattr(arm, "close", None)
        if callable(close):
            try:
                close()
            except Exception as e:  # unloading is best-effort; never mask the real result
                print(f"close() warning: {e}", file=sys.stderr)
    receipt["vram_preflight"] = vram   # §7 numbers: free/required MiB, model bytes, probe argv
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    f = receipt["findings"]
    print(f"precheck {a.model_id}: temperature_basis={f['temperature_basis']} "
          f"penalty_basis={f['penalty_basis']} repeatable={f['repeatable']} "
          f"label_first_token_feasible={f['label_first_token_feasible']} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
