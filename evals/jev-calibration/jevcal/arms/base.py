"""Arm adapter interface.

Every arm takes a `Rendered` (the one canonical render) and returns an
`ArmResult` whose `probabilities` are aligned to `rendered.keys` (same order for
every arm). An arm never sees an Item, only its Rendered, which is what makes
prompt parity structural rather than a convention.

LOCAL arm (E.4b, the study author; deliberately NOT built here, see README.md): a
class implementing `Arm` with arm_name "LOCAL-P", family "qwen", that
  - sets model_id to the REGISTRY KEY and serving_tag to the ollama tag (the A4.2
    mapping table in README.md), reads model_digest from /api/tags, and raises
    ArmHalt if the digest changes mid-run (a version change, prereg §2/A4.2);
  - logs the ollama `options` object exactly as sent and the ollama version;
  - reads its ollama base URL from JEVCAL_OLLAMA_BASE_URL (the E.4 harness runs on
    the harness host and reaches the GPU host's ollama through an SSH tunnel; prereg A2), never a
    hard-coded host;
  - pins the registry-recommended sampling config + a fixed seed (prereg A1) and
    returns it from sampling_config();
  - reads top_logprobs over rendered.labels only and renormalises over valid labels;
  - puts ollama's total/load/eval_duration (ns -> s) in ArmResult.server_latency,
    leaving latency_s as client wall-clock;
  - raises ArmHalt if the served model differs from the pin.
The VRAM precondition (prereg §7) is checked before the arm is constructed.
"""
from __future__ import annotations

import hashlib
import json
import math
import signal
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..render import Rendered


class ArmHalt(RuntimeError):
    """Stop the whole run: non-200, pinned-model mismatch, billing/quota/spend signal."""


def install_terminate_handlers() -> None:
    """Convert SIGTERM/SIGINT into a clean exit path so `finally` blocks run.

    A LOCAL arm holds a model resident on the shared GPU during a run; its close() (keep_alive:0)
    lives in the runner's `finally`. Default SIGTERM disposition TERMINATES the process WITHOUT
    running finally, so an aborted run (SIGTERM) left the model resident until it was unloaded by
    hand (E.4 first live LOCAL-P run, 2026-09-30T17:15Z). Raising SystemExit from the handler makes
    the signal unwind through the try/finally instead, so every arm's close() still runs. Both the
    runner (run.py) and the precheck CLI (local_precheck.py) install these before any model loads.
    A no-op when not in the main thread (signal handlers can only be set there)."""
    def _terminate(signum, _frame):
        raise SystemExit(f"terminated by signal {signum}")
    try:
        signal.signal(signal.SIGTERM, _terminate)
        signal.signal(signal.SIGINT, _terminate)
    except ValueError:
        pass  # not the main thread: handlers can only be installed there


@dataclass
class ArmResult:
    arm: str
    item_id: str
    requested_model: str
    returned_model: Optional[str]
    # A4.2 model identity, on EVERY record: model_id is a verbatim sampling-registry
    # key (GATE F looks it up verbatim); serving_tag is what the server is asked for
    # ("n/a" for hosted/CLI arms); model_digest identifies the exact weights served
    # (ollama /api/tags digest; JEV: the returned model string; "n/a" if unobservable).
    model_id: str
    serving_tag: str
    model_digest: str
    probabilities: Optional[List[float]]   # aligned to rendered.keys, renormalised; None = abstention
    raw_probabilities: Any
    raw_sum: Optional[float]
    abstained: bool
    error: Optional[str]
    latency_s: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    wire_request: Any                      # never carries credentials
    wire_response: Any
    wire_sha256: str
    cost_usd_list: Optional[float] = None
    sampling_config: Dict[str, Any] = field(default_factory=dict)
    # Server-reported timing (e.g. ollama total/load/eval_duration, in seconds) kept
    # SEPARATE from latency_s, which is always client wall-clock (prereg A2).
    server_latency: Optional[Dict[str, float]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def chosen(self) -> Optional[int]:
        """Argmax index, ties to the lowest index (same rule as the scorer)."""
        if self.probabilities is None:
            return None
        return max(range(len(self.probabilities)), key=lambda i: (self.probabilities[i], -i))

    def to_record(self, rendered: Rendered) -> Dict[str, Any]:
        """The per-item record every arm emits (the arm interface's output contract)."""
        idx = self.chosen
        return {
            "item_id": self.item_id, "arm": self.arm,
            "model_id": self.model_id, "serving_tag": self.serving_tag,
            "model_digest": self.model_digest,
            "requested_model": self.requested_model, "returned_model": self.returned_model,
            "options": rendered.keys, "labels": rendered.labels,
            "probabilities": (dict(zip(rendered.keys, self.probabilities))
                              if self.probabilities is not None else None),
            "raw_probabilities": self.raw_probabilities, "raw_sum": self.raw_sum,
            "answer": rendered.keys[idx] if idx is not None else None,
            "abstained": self.abstained, "error": self.error,
            "latency_s": self.latency_s, "server_latency": self.server_latency,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "sampling_config": self.sampling_config,
            "msg_sha256": rendered.msg_sha256, "wire_sha256": self.wire_sha256,
            "wire_request": self.wire_request, "wire_response": self.wire_response,
            "cost_usd_list": self.cost_usd_list, "extra": self.extra,
        }


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def normalise(vec: List[Any], tol: float = 1e-3):
    """Validate + renormalise a probability vector.

    Returns (probs, raw_sum, error). error is non-None when the vector is not a
    usable distribution (non-numeric, non-finite, negative, or zero mass); that is
    an ABSTENTION under the rubric, never a silent drop. A sum off 1 by more than
    `tol` is still renormalised but flagged in `error` as a warning prefix 'WARN:'.
    """
    out = []
    for v in vec:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None, None, f"non-numeric probability {v!r}"
        f = float(v)
        if not math.isfinite(f) or f < 0:
            return None, None, f"invalid probability {v!r}"
        out.append(f)
    s = sum(out)
    if s <= 0:
        return None, s, "zero probability mass"
    probs = [x / s for x in out]
    warn = None if abs(s - 1.0) <= tol else f"WARN: raw probabilities summed to {s:.6f}; renormalised"
    return probs, s, warn


class Arm:
    """Contract: call(rendered) -> ArmResult with probabilities aligned to
    rendered.keys (or None = abstention), raising ArmHalt on anything that must
    stop the run. config() returns every knob that can change an answer; its hash
    is the arm config hash in the run record. sampling_config() is the subset
    GATE F reads (log what you cannot pin as a labelled string, never an invented
    number).

    model_id MUST be a verbatim key of governance/evals/sampling-registry.json;
    the runner preflight refuses the run before the first call of any arm if not
    (prereg A4.2)."""
    arm_name: str = ""
    model_id: str = ""        # verbatim sampling-registry key
    serving_tag: str = "n/a"  # e.g. the ollama tag; "n/a" for hosted/CLI arms
    family: str = ""

    def config(self) -> Dict[str, Any]:
        raise NotImplementedError

    def sampling_config(self) -> Dict[str, Any]:
        raise NotImplementedError

    def config_sha256(self) -> str:
        return sha256_text(canonical_json(self.config()))

    def call(self, rendered: Rendered) -> ArmResult:
        raise NotImplementedError
