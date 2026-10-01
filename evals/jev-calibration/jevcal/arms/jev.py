"""JEV arm: TypeSafe System One, pinned to jev-1.13.0 (prereg §2).

Fail-closed rules, each a run HALT (ArmHalt), never a retry:
  - any non-200 status (401/402/422/429/529/...);
  - a returned `model` other than the pinned string;
  - any billing / payment / credit / quota signal in response headers or body keys.
Key handling (portable, prereg A2): the key FILE path comes from the env var
JEV_KEY_FILE (no default, no machine-specific path). The file holds a
`TYPESAFE_API_KEY=...` line; it is read at call time, used only in the
Authorization header, and never logged: wire_request is the JSON body alone.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional, Tuple

from ..render import Rendered
from .base import Arm, ArmHalt, ArmResult, canonical_json, normalise, sha256_text

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
PINNED_MODEL = "jev-1.13.0"
PRICE_PER_M_INPUT_USD = 0.042  # docs.typesafe.ai (capability map §1); output is free
TEMP_LABEL = "n/a (jev-1.13.0 exposes no sampling parameters)"
ACCESS_PATH = "typesafe-systemone-api"  # how the model is reached (GATE F v3 / GATE H); registry-confirmed
SPEND_SIGNAL = re.compile(r"bill|payment|credit|quota|overage|invoice|charge", re.I)

# (url, body_bytes, headers, timeout) -> (status, response_headers, body_bytes)
Transport = Callable[[str, bytes, Dict[str, str], float], Tuple[int, Dict[str, str], bytes]]


def read_key_file(path: str) -> str:
    """Parse KEY=VALUE lines; return TYPESAFE_API_KEY. Never echoes the value."""
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            if line.startswith("TYPESAFE_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise ArmHalt(f"no TYPESAFE_API_KEY line in the file named by JEV_KEY_FILE")


def urllib_transport(url: str, body: bytes, headers: Dict[str, str], timeout: float):
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers.items()) if e.headers else {}, e.read()


def build_question(r: Rendered) -> Dict[str, Any]:
    if r.type == "choice":
        return {"type": "choice", "instructions": r.instruction,
                "criteria": {key: (desc or None) for _, key, desc in r.options}}
    # noul and score wires have no slot for option KEYS, so their criteria carry the
    # renderer's option_texts ("key — description"): the same key and description
    # every other arm sees on its lettered line (prompt parity, prereg §2).
    if r.type == "noul":
        yes_text, no_text = r.option_texts
        return {"type": "noul", "instructions": r.instruction,
                "criteria": {"true": yes_text, "false": no_text}}
    if r.type == "score":
        return {"type": "score", "instructions": r.instruction,
                "criteria": list(r.option_texts)}
    raise ValueError(f"unsupported type {r.type}")


def parse_answer(r: Rendered, ans: Any):
    """Map a Jev answer to a vector aligned with r.keys. Returns (vec, raw, error)."""
    if not isinstance(ans, dict):
        return None, ans, "answer missing or not an object"
    if ans.get("type") != r.type:
        return None, ans, f"answer type {ans.get('type')!r} != question type {r.type!r}"
    if r.type == "noul":
        p = ans.get("noul")
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            return None, ans, "noul value missing"
        return [float(p), 1.0 - float(p)], {"noul": p}, None
    probs = ans.get("probabilities")
    if not isinstance(probs, dict):
        return None, ans, "probabilities missing"
    if r.type == "choice":
        want = list(r.keys)
    else:  # score: level indices as string keys
        want = [str(i) for i in range(len(r.keys))]
    if set(probs) != set(want):
        return None, probs, f"probability keys {sorted(probs)} != options {sorted(want)}"
    return [probs[k] for k in want], probs, None


class JevArm(Arm):
    arm_name = "JEV"
    model_id = PINNED_MODEL
    family = "typesafe"

    def __init__(self, transport: Optional[Transport] = None, timeout: float = 60.0,
                 key_file_env: str = "JEV_KEY_FILE"):
        self.transport = transport or urllib_transport
        self.timeout = timeout
        self.key_file_env = key_file_env

    def sampling_config(self) -> Dict[str, Any]:
        # Jev is non-generative: it returns a distribution and exposes no sampling
        # knobs. Nothing is invented here; GATE F's registry decides (prereg A3).
        # access_path/sampling_exposed/output_cap_exposed are the typed declaration
        # GATE F v3 / GATE H read: exposure is a property of HOW the model is reached
        # (this hosted API), and the registry confirms it per access_path.
        return {"temp": TEMP_LABEL, "top_p": None, "seed": None, "max_tokens": None,
                "access_path": ACCESS_PATH,
                "sampling_exposed": False, "output_cap_exposed": False,
                "note": "jev-1.13.0 exposes no sampling parameters and no output cap"}

    def config(self) -> Dict[str, Any]:
        return {"arm": self.arm_name, "endpoint": ENDPOINT, "model": PINNED_MODEL,
                "question_id": "q", "timeout_s": self.timeout,
                "probability_source": "returned probabilities (choice/score) or noul",
                "generation_cap": None,
                "generation_cap_note": "the API exposes no output-token cap"}

    def call(self, r: Rendered) -> ArmResult:
        kf = os.environ.get(self.key_file_env, "")
        if not kf:
            raise ArmHalt(f"{self.key_file_env} is not set (path to the TypeSafe key file)")
        key = read_key_file(kf)
        if not key:
            raise ArmHalt(f"empty key in the file named by {self.key_file_env}")
        body = {"state": r.state, "model": PINNED_MODEL, "questions": {"q": build_question(r)}}
        body_txt = canonical_json(body)
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        t0 = time.monotonic()
        status, rhead, rbytes = self.transport(ENDPOINT, body_txt.encode("utf-8"), headers, self.timeout)
        latency = time.monotonic() - t0
        del key, headers
        sig = [h for h in rhead if SPEND_SIGNAL.search(h)]
        if status != 200:
            raise ArmHalt(f"JEV non-200: HTTP {status} on item {r.item_id}: "
                          f"{rbytes[:300].decode('utf-8', 'replace')}")
        if sig:
            raise ArmHalt(f"JEV billing/quota signal in response headers {sig} on item {r.item_id}")
        try:
            resp = json.loads(rbytes)
        except json.JSONDecodeError as e:
            raise ArmHalt(f"JEV 200 with unparseable body on item {r.item_id}: {e}")
        if not isinstance(resp, dict):
            raise ArmHalt(f"JEV 200 with non-object body on item {r.item_id}")
        body_sig = [k for k in resp if SPEND_SIGNAL.search(k)]
        if body_sig:
            raise ArmHalt(f"JEV billing/quota signal in response body keys {body_sig}")
        returned = resp.get("model")
        if returned != PINNED_MODEL:
            raise ArmHalt(f"JEV returned model {returned!r} != pinned {PINNED_MODEL!r} "
                          f"(item {r.item_id}); the arm is invalidated (prereg §2)")
        usage = resp.get("usage") or {}
        itok = usage.get("input_tokens") if isinstance(usage, dict) else None
        otok = usage.get("output_tokens") if isinstance(usage, dict) else None
        vec, raw, err = parse_answer(r, (resp.get("answers") or {}).get("q"))
        probs, s, nerr = (None, None, None)
        if vec is not None:
            probs, s, nerr = normalise(vec)
        error = err or nerr
        abstained = probs is None
        cost = (itok * PRICE_PER_M_INPUT_USD / 1e6) if isinstance(itok, int) else None
        return ArmResult(
            arm=self.arm_name, item_id=r.item_id, requested_model=PINNED_MODEL,
            returned_model=returned, model_id=self.model_id, serving_tag=self.serving_tag,
            model_digest=returned, probabilities=probs, raw_probabilities=raw, raw_sum=s,
            abstained=abstained, error=error, latency_s=latency,
            input_tokens=itok, output_tokens=otok, wire_request=body, wire_response=resp,
            wire_sha256=sha256_text(body_txt), cost_usd_list=cost,
            sampling_config=self.sampling_config(),
            extra={"response_header_names": sorted(rhead)})
