"""LOCAL arm (E.4b, the study author): ollama-served open-weight models, logprob probabilities.

Three registered arms, one class parameterised by the sampling-registry KEY (prereg A4.2):
  LOCAL-P        -> qwen3-14b               (family "qwen")
  LOCAL-C-CODER  -> Qwen3-Coder-30B-A3B     (family "qwen")
  LOCAL-C-GPTOSS -> gpt-oss:20b             (family "openai-gpt-oss")

LOCAL-P and LOCAL-C-CODER share the family "qwen", so the runner's family-uniqueness rule
(`pins.contestants` is keyed by family) REFUSES the two in one run. That is intended: they
cannot share the 4090's VRAM either (§7), so a run picks one qwen arm plus, optionally, the
different-family gpt-oss arm. The refusal is documented, not worked around.

Fail-closed rules, each an ArmHalt (never a retry), mirroring the JEV/FRONTIER pattern:
  - /api/tags digest prefix != the A4.2 pin (at construction AND re-checked mid-run; a change
    is a version change and invalidates the arm, prereg §2);
  - the served tag, or the model /api/chat returns, starts with `nxtg-` (a derived in-house model
    that may carry a system prompt or parameters, breaking prompt parity, README/A4.2);
  - the model /api/chat returns is not the serving tag;
  - any non-200 status.

Model identity (A4.2): `model_id` is the registry KEY (what GATE F looks up verbatim);
`serving_tag` is the ollama tag asked for; `model_digest` is the /api/tags digest pin.

Sampling (A1/A4.3): the registry's numeric recommended config is mapped key-by-key onto ollama
`options` (an unmapped key REFUSES), plus a fixed seed and num_predict:1. gpt-oss:20b now carries a
numeric config (temperature 1.0 / top_p 1.0, from openai/gpt-oss's own README §Recommended Sampling
Parameters; landed in the registry by PR #80) and is mapped the same way as the other arms.
`sampling_config()` returns the rail-keyed view (`temp`, not `temperature`) GATE F reads.

Probabilities: the first generated token's `top_logprobs` are read over `rendered.labels` only,
whitespace label variants (`A`, ` A`) summed in probability space, then renormalised over the
valid labels. No valid label -> an ABSTENTION (never a silent drop).

Base URL: JEVCAL_OLLAMA_BASE_URL, no default and no hard-coded host (E.4 tunnels harness host->GPU host,
prereg A2). VRAM precondition (§7) is checked by `vram_preflight()` BEFORE the arm is constructed.
"""
from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..render import Rendered
from .base import Arm, ArmHalt, ArmResult, canonical_json, normalise, sha256_text

# ----- environment seams (no hard-coded host; the probe is injected in tests) -----
BASE_URL_ENV = "JEVCAL_OLLAMA_BASE_URL"       # e.g. http://127.0.0.1:11434 over the A2 SSH tunnel
VRAM_PROBE_ENV = "JEVCAL_VRAM_PROBE"          # e.g. "ssh ... nvidia-smi --query-gpu=memory.free ..."

# ----- fixed, version-controlled knobs (prereg A1) -----
LOCAL_SEED = 20260929                         # the pinned seed, a module constant (A1)
NUM_PREDICT = 1                               # single-token label read (§2)
TOP_LOGPROBS = 20                             # request 20 alternatives on the one generated token
DIGEST_RECHECK_EVERY = 25                     # re-check the /api/tags digest at least this often
VRAM_HEADROOM_MIB = 2048                      # §7: free >= model size + KV + 2 GiB
KEEP_ALIVE_DEFAULT = "5m"                     # hold the model loaded during the window only (§7)
NXTG_PREFIX = "nxtg-"                         # derived in-house models: refused (prompt parity)

# num_ctx: the ollama context window we PIN for every LOCAL arm. ollama's default is the model's
# trained context (qwen3:14b => 40,960), whose KV cache is ~10x this one and was NOT priced by the
# §7 gate, so the first live LOCAL-P run spilled KV into shared system RAM over PCIe and confounded
# LOCAL latency (a pre-registered reported metric). We pin a small window sized to the frozen items:
# the largest rendered LOCAL prompt is 6,736 chars (S1-choice-045, canonical 6,626 + the format
# instruction), ~2,695 tokens at chars/2.5 — well under 4096. num_ctx is in `options`, so it is in
# every record and part of the precheck-binding options equality.
NUM_CTX = 4096                                # pinned ollama context window for LOCAL arms
NUMCTX_TOKEN_MARGIN = 64                      # preflight refuses if worst est prompt > num_ctx - margin
NUMCTX_CHARS_PER_TOKEN = 2.5                  # chars->tokens estimate, conservative at the worst item
KV_BYTES_PER_ELEM_F16 = 2                     # f16 KV cache element size (§7 KV pricing)

# The one instruction appended to the shared canonical text. No system prompt, so the base
# model's own chat template is preserved (prompt parity). It is in config() so it is hashed.
LOCAL_FORMAT_INSTRUCTION = (
    "Answer with ONLY the single letter of the correct option (one of the letters shown above), "
    "and nothing else."
)

# A4.2 identity table (README / prereg A4.2). model_id (registry key) -> (serving_tag, digest pin).
IDENTITY: Dict[str, Tuple[str, str]] = {
    "qwen3-14b": ("qwen3:14b", "bdbd181c33f2"),
    "Qwen3-Coder-30B-A3B": ("hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:UD-Q4_K_XL",
                            "683e1639bbff"),
    "gpt-oss:20b": ("gpt-oss:20b", "17052f91a42e"),
}

# arm_name -> registry model_id (the runner iterates this to VRAM-preflight LOCAL arms).
LOCAL_ARMS: Dict[str, str] = {
    "LOCAL-P": "qwen3-14b",
    "LOCAL-C-CODER": "Qwen3-Coder-30B-A3B",
    "LOCAL-C-GPTOSS": "gpt-oss:20b",
}

# A1 key mapping. A numeric registry sampling key MUST be one of these, else REFUSE (fail-closed):
#   pass-through -> ollama option of the same name; repetition_penalty -> ollama repeat_penalty.
PASS_THROUGH = {"temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty"}
RENAME = {"repetition_penalty": "repeat_penalty"}

# (method, url, body_or_None, headers, timeout) -> (status, resp_headers, resp_bytes)
Transport = Callable[[str, str, Optional[bytes], Dict[str, str], float],
                     Tuple[int, Dict[str, str], bytes]]


def _repo_root() -> str:
    # arms/local.py -> arms -> jevcal -> jev-calibration -> evals -> REPO (5 levels up)
    here = os.path.abspath(__file__)
    for _ in range(5):
        here = os.path.dirname(here)
    return here


def default_registry_path() -> str:
    return os.path.join(_repo_root(), "governance", "evals", "sampling-registry.json")


def urllib_transport(method: str, url: str, body: Optional[bytes],
                     headers: Dict[str, str], timeout: float):
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers.items()) if e.headers else {}, e.read()


def _norm_base(base: str) -> str:
    return base.rstrip("/")


def _norm_digest(entry_digest: Any) -> Optional[str]:
    """The observed digest as hex: strip any 'sha256:' prefix, lower-case. None if not a string."""
    if not isinstance(entry_digest, str):
        return None
    return entry_digest.split(":")[-1].lower()


def _digest_matches(entry_digest: Any, pinned: str) -> bool:
    """ollama /api/tags digest is a full sha256 (optionally 'sha256:'-prefixed); the pin is the
    12-char short id that `ollama list` shows. Compare by prefix, case-insensitively."""
    norm = _norm_digest(entry_digest)
    return norm is not None and norm.startswith(pinned.lower())


def _find_tag(models: List[Any], tag: str) -> Optional[Dict[str, Any]]:
    for m in models or []:
        if isinstance(m, dict) and m.get("name") == tag:
            return m
    return None


def _read_json_get(base: str, path: str, transport: Transport, timeout: float) -> Any:
    status, _hdr, body = transport("GET", _norm_base(base) + path, None, {}, timeout)
    if status != 200:
        raise ArmHalt(f"ollama {path} returned HTTP {status}: {body[:200].decode('utf-8', 'replace')}")
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise ArmHalt(f"ollama {path} returned unparseable JSON: {e}")


def read_tags(base: str, transport: Transport, timeout: float) -> List[Any]:
    data = _read_json_get(base, "/api/tags", transport, timeout)
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        raise ArmHalt("ollama /api/tags did not return a 'models' list")
    return models


def _read_json_post(base: str, path: str, payload: Dict[str, Any], transport: Transport,
                    timeout: float) -> Any:
    body = canonical_json(payload).encode("utf-8")
    status, _hdr, rbytes = transport("POST", _norm_base(base) + path, body,
                                     {"Content-Type": "application/json"}, timeout)
    if status != 200:
        raise ArmHalt(f"ollama {path} returned HTTP {status}: {rbytes[:200].decode('utf-8', 'replace')}")
    try:
        return json.loads(rbytes)
    except json.JSONDecodeError as e:
        raise ArmHalt(f"ollama {path} returned unparseable JSON: {e}")


def _model_info_int(model_info: Dict[str, Any], suffix: str) -> Optional[int]:
    """First integer value in an /api/show model_info dict whose key IS `suffix` or ends with
    `.suffix` (the keys are architecture-prefixed, e.g. `qwen3.attention.head_count_kv`). Bools are
    not ints here. Returns None if absent, so the caller can fail closed."""
    for k, v in model_info.items():
        if (k == suffix or k.endswith("." + suffix)) and isinstance(v, (int, float)) \
                and not isinstance(v, bool):
            return int(v)
    return None


def kv_cache_mib(model_info: Any, num_ctx: int,
                 bytes_per_elem: int = KV_BYTES_PER_ELEM_F16) -> Dict[str, Any]:
    """KV-cache size in MiB at `num_ctx`, from ollama /api/show `model_info` (§7 KV pricing).

        2 x block_count x head_count_kv x head_dim x num_ctx x bytes_per_elem   (K and V => the 2)

    head_dim is `attention.key_length` if present, else `embedding_length / attention.head_count`.
    Any needed field missing => REFUSE (ArmHalt), never guess. bytes_per_elem is f16 (2) unless the
    caller overrides it. Returns the MiB and every formula input for the run record."""
    if not isinstance(model_info, dict) or not model_info:
        raise ArmHalt("ollama /api/show returned no non-empty model_info; cannot price the KV cache "
                      "at num_ctx (§7); refusing rather than guessing")
    block_count = _model_info_int(model_info, "block_count")
    head_count_kv = _model_info_int(model_info, "attention.head_count_kv")
    key_length = _model_info_int(model_info, "attention.key_length")
    if key_length is not None:
        head_dim, head_dim_src = key_length, "attention.key_length"
    else:
        emb = _model_info_int(model_info, "embedding_length")
        head_count = _model_info_int(model_info, "attention.head_count")
        if emb is None or not head_count:
            raise ArmHalt("ollama /api/show model_info lacks attention.key_length and a usable "
                          "embedding_length/attention.head_count fallback; cannot price the KV cache "
                          "(§7); refusing rather than guessing")
        head_dim, head_dim_src = emb // head_count, "embedding_length/attention.head_count"
    if not block_count or not head_count_kv:
        raise ArmHalt("ollama /api/show model_info lacks block_count or attention.head_count_kv; "
                      "cannot price the KV cache (§7); refusing rather than guessing")
    kv_bytes = 2 * block_count * head_count_kv * head_dim * int(num_ctx) * int(bytes_per_elem)
    return {"kv_mib": math.ceil(kv_bytes / (1024 * 1024)), "kv_bytes": kv_bytes,
            "block_count": block_count, "head_count_kv": head_count_kv, "head_dim": head_dim,
            "head_dim_source": head_dim_src, "bytes_per_elem": int(bytes_per_elem),
            "num_ctx": int(num_ctx)}


def build_sampling_options(model_id: str, registry: Dict[str, Any]) -> Dict[str, Any]:
    """Map the registry's numeric recommended config onto ollama `options` (prereg A1/A4.3).

    - uses `recommended.non_thinking`, or the single recommended block if there is only one;
    - a numeric key must be in PASS_THROUGH or RENAME, else REFUSE (an unmapped key never passes
      silently); a non-numeric key other than a descriptive `note` also REFUSES;
    - if no numeric key is present at all (e.g. gpt-oss:20b currently has only a note), REFUSE.
    Adds the fixed seed and num_predict:1. Returns the ollama `options` object, exactly as sent.
    """
    models = registry.get("models", {}) if isinstance(registry, dict) else {}
    entry = models.get(model_id)
    if not isinstance(entry, dict):
        raise ArmHalt(f"model {model_id!r} has no sampling-registry entry (A1); refusing")
    rec = entry.get("recommended")
    if not isinstance(rec, dict):
        raise ArmHalt(f"model {model_id!r} registry entry has no 'recommended' block (A1); refusing")
    block = rec.get("non_thinking") if isinstance(rec.get("non_thinking"), dict) else rec
    mapped: Dict[str, Any] = {}
    for k, v in block.items():
        if k == "note":                        # documented, descriptive; not a sampling knob
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ArmHalt(f"model {model_id!r}: registry sampling value {k!r}={v!r} is not numeric; "
                          "refusing rather than guessing (A1)")
        if k in PASS_THROUGH:
            mapped[k] = v
        elif k in RENAME:
            mapped[RENAME[k]] = v
        else:
            raise ArmHalt(f"model {model_id!r}: unmapped registry sampling key {k!r} (A1); refusing "
                          "rather than passing an unknown knob through silently")
    if not mapped:
        raise ArmHalt(f"model {model_id!r}: registry holds no numeric recommended sampling config "
                      "(only a note); refusing until a cited config lands (A1)")
    mapped["seed"] = LOCAL_SEED
    mapped["num_predict"] = NUM_PREDICT
    mapped["num_ctx"] = NUM_CTX          # pin the context window (KV size + spill control, §7)
    return mapped


def first_token_top_logprobs(resp: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Return the first generated token's top_logprobs list, or None (-> abstention).

    ollama's native /api/chat with logprobs:true returns, per generated token, an entry
    {"token","logprob","top_logprobs":[{"token","logprob"},...]}. The PLACEMENT of that per-token
    list has varied across ollama builds, so we look in the documented native locations and then
    the OpenAI-compat shape, validating structure before use, and ABSTAIN (never guess) if none is
    present. The primary/native shape (top-level "logprobs") is what the tests and README pin; the
    fallbacks exist because this is the one field this arm cannot verify against a live call here
    (see README "logprobs wire shape") and the operator confirms it on the first real E.4 call.
    """
    if not isinstance(resp, dict):
        return None
    msg = resp.get("message") if isinstance(resp.get("message"), dict) else {}
    choices = resp.get("choices") if isinstance(resp.get("choices"), list) else []
    ch0 = choices[0] if (choices and isinstance(choices[0], dict)) else {}
    ch_lp = ch0.get("logprobs") if isinstance(ch0.get("logprobs"), dict) else {}
    for seq in (resp.get("logprobs"), msg.get("logprobs"), ch_lp.get("content")):
        if isinstance(seq, list) and seq and isinstance(seq[0], dict) \
                and isinstance(seq[0].get("top_logprobs"), list):
            return seq[0]["top_logprobs"]
    return None


def label_masses(top_logprobs: List[Dict[str, Any]], labels: List[str]) -> Dict[str, float]:
    """Sum, in probability space, exp(logprob) of every token whose strip() == a valid label.
    Whitespace variants (`A`, ` A`) of one label add together; a lowercase `a` or any non-label
    token is ignored (token.strip() must equal the label exactly)."""
    label_set = set(labels)
    mass = {l: 0.0 for l in labels}
    for e in top_logprobs or []:
        if not isinstance(e, dict):
            continue
        tok = e.get("token")
        lp = e.get("logprob")
        if not isinstance(tok, str) or isinstance(lp, bool) or not isinstance(lp, (int, float)):
            continue
        stripped = tok.strip()
        if stripped in label_set:
            mass[stripped] += math.exp(float(lp))
    return mass


def _first_findings_diff(rederived: Dict[str, Any], stored: Dict[str, Any],
                         float_tol: float = 1e-9) -> Optional[str]:
    """First key (sorted) at which a re-derived findings dict differs from a stored one, or None.

    Strings and bools compare EXACTLY; the nested `cache_state_effect` floats compare within
    `float_tol`. A key present in one dict and not the other is itself a difference (its own name is
    returned). Used by the certifying binding to refuse a receipt whose stored findings do not
    reproduce from its raw calls (the confound fix lives in compute_findings)."""
    def cache_eq(a: Any, b: Any) -> bool:
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return a == b
        if set(a) != set(b):
            return False
        for k in a:
            av, bv = a[k], b[k]
            if (isinstance(av, (int, float)) and not isinstance(av, bool)
                    and isinstance(bv, (int, float)) and not isinstance(bv, bool)):
                if abs(float(av) - float(bv)) > float_tol:
                    return False
            elif av != bv:
                return False
        return True

    for key in sorted(set(rederived) | set(stored)):
        if key not in rederived or key not in stored:
            return key
        rv, sv = rederived[key], stored[key]
        if key == "cache_state_effect":
            if not cache_eq(rv, sv):
                return key
        elif rv != sv:
            return key
    return None


class LocalArm(Arm):
    serving_tag = "n/a"      # set per instance from the A4.2 table

    def __init__(self, *, arm_name: str, model_id: str,
                 transport: Optional[Transport] = None, base_url: Optional[str] = None,
                 keep_alive: str = KEEP_ALIVE_DEFAULT, timeout: float = 120.0,
                 digest_recheck_every: int = DIGEST_RECHECK_EVERY,
                 registry: Optional[Dict[str, Any]] = None, registry_path: Optional[str] = None,
                 precheck_path: Optional[str] = None, require_precheck: bool = False,
                 vram_report: Optional[Dict[str, Any]] = None):
        if model_id not in IDENTITY:
            raise ArmHalt(f"LOCAL arm model_id {model_id!r} has no A4.2 identity entry")
        self.arm_name = arm_name
        self.model_id = model_id
        self.serving_tag, self.pinned_digest = IDENTITY[model_id]
        self.transport = transport or urllib_transport
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.digest_recheck_every = max(1, int(digest_recheck_every))
        self._calls = 0

        self.base_url = base_url or os.environ.get(BASE_URL_ENV)
        if not self.base_url:
            raise ArmHalt(f"{BASE_URL_ENV} is unset; the LOCAL arm has no default host (prereg A2)")

        if registry is not None:
            reg = registry
        else:
            with open(registry_path or default_registry_path(), "r", encoding="utf-8") as _fh:
                reg = json.load(_fh)
        # Build the ollama `options` FIRST: gpt-oss (no numeric config) and any unmapped key
        # REFUSE here, before any /api/tags read, so the refusal is the arm's own (not a network
        # dependency) and its cause is precise.
        self.family = (reg.get("models", {}).get(model_id, {}) or {}).get("family", "")
        self.options = build_sampling_options(model_id, reg)

        # Family must resolve; the runner keys pins.contestants by it.
        if not self.family:
            raise ArmHalt(f"model {model_id!r} has no family in the sampling registry")

        # Model identity at construction: digest pin + nxtg- guard + ollama version.
        models = read_tags(self.base_url, self.transport, self.timeout)
        entry = _find_tag(models, self.serving_tag)
        if entry is None:
            raise ArmHalt(f"serving tag {self.serving_tag!r} absent from ollama /api/tags")
        if self.serving_tag.startswith(NXTG_PREFIX):
            raise ArmHalt(f"refusing derived nxtg- tag {self.serving_tag!r} (prompt parity, A4.2)")
        if not _digest_matches(entry.get("digest"), self.pinned_digest):
            raise ArmHalt(f"{self.serving_tag} digest {entry.get('digest')!r} != pin "
                          f"{self.pinned_digest!r}; a version change invalidates the arm (§2/A4.2)")
        # The digest actually observed on /api/tags (updated at each recheck). Every record carries
        # THIS, not the pin constant, so the runner's cross-record identity check has a real value to
        # compare; the pin is kept separately in extra.pinned_digest (review note 4).
        self.observed_digest = _norm_digest(entry.get("digest"))
        ver = _read_json_get(self.base_url, "/api/version", self.transport, self.timeout)
        self.ollama_version = ver.get("version") if isinstance(ver, dict) else None
        self.vram_report = vram_report

        # Precheck (A1/A4.3): the run-time-recorded finding. In certifying mode a matching receipt
        # with label_first_token_feasible is REQUIRED before the first call; otherwise it is logged
        # when present. Loaded here so every ArmResult can carry the finding identically.
        self.precheck = self._load_precheck(precheck_path, require_precheck)

    # ---------------- precheck binding ----------------
    def _load_precheck(self, path: Optional[str], require: bool) -> Optional[Dict[str, Any]]:
        if not path:
            if require:
                raise ArmHalt(f"certifying mode requires a LOCAL precheck receipt for {self.arm_name} "
                              f"(JEVCAL_LOCAL_PRECHECK_{self.arm_name.replace('-', '_')} or "
                              "--local-precheck); none given (A1/A4.3)")
            return None
        # A specified receipt that is missing, unreadable, or not valid JSON is a clean REFUSED, the
        # same shape as the other refusals — never an uncaught FileNotFoundError/JSONDecodeError. The
        # runner wraps ArmHalt -> SystemExit "REFUSED" (review note 3). json.JSONDecodeError subclasses
        # ValueError, so one except covers both file and parse failures.
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read()
            receipt = json.loads(raw)
        except (OSError, ValueError) as e:
            raise ArmHalt(f"LOCAL precheck receipt {path!r} could not be read: {type(e).__name__}: {e}")
        if not isinstance(receipt, dict):
            raise ArmHalt(f"LOCAL precheck receipt {path!r} is not a JSON object")
        findings = receipt.get("findings", {})
        if not isinstance(findings, dict):
            findings = {}
        info = {"path": os.path.abspath(path), "sha256": sha256_text(raw), "findings": findings,
                "model_id": receipt.get("model_id"), "model_digest": receipt.get("model_digest"),
                "ollama_version": receipt.get("ollama_version")}
        if require:
            if receipt.get("model_id") != self.model_id:
                raise ArmHalt(f"precheck receipt model_id {receipt.get('model_id')!r} != {self.model_id!r}")
            if not _digest_matches(receipt.get("model_digest"), self.pinned_digest):
                raise ArmHalt(f"precheck receipt digest {receipt.get('model_digest')!r} != pin "
                              f"{self.pinned_digest!r} (A4.2)")
            # The findings are properties of the ollama BUILD (its logprob pipeline), not only of the
            # weights, so the receipt must have been produced against the same /api/version the arm
            # reads now (review note 1).
            if receipt.get("ollama_version") != self.ollama_version:
                raise ArmHalt(f"precheck receipt ollama_version {receipt.get('ollama_version')!r} != the "
                              f"arm's /api/version {self.ollama_version!r}; the temperature/penalty "
                              "findings are a property of the ollama build, not only the weights (A1)")
            if receipt.get("options") != self.options:
                raise ArmHalt("precheck receipt options differ from the arm's ollama options; the "
                              "finding does not apply to this configuration (A1)")
            if findings.get("label_first_token_feasible") is not True:
                raise ArmHalt("precheck says label_first_token_feasible is not true; the label-logprob "
                              "read is not established for this model (A1/A4.3); refusing")
            # RE-DERIVE, DON'T TRUST (2026-09-30). A stored `findings` block is the author's CLAIM, not
            # an instrument: the first live receipt (…-CONFOUNDED.json) carried post-cold-baseline
            # findings that this gate accepted. So the receipt must carry the RAW calls the warm-up
            # precheck records, and the binding recomputes the findings itself — PR #85's confound fix
            # lives in compute_findings, so a stale receipt's findings will NOT reproduce.
            calls = receipt.get("calls")
            if not isinstance(calls, dict):
                raise ArmHalt("precheck receipt has no 'calls' object; receipt predates the warm-up "
                              "precheck; re-run it")
            missing_calls = [k for k in ("warmup", "a1", "a2", "b", "c")
                             if not isinstance(calls.get(k), dict)]
            if missing_calls:
                raise ArmHalt(f"precheck receipt calls missing {missing_calls}; receipt predates the "
                              "warm-up precheck; re-run it")
            if not (isinstance(receipt.get("labels"), list) and receipt.get("labels")):
                raise ArmHalt("precheck receipt has no non-empty 'labels' list; cannot re-derive findings")
            from ..local_precheck import compute_findings   # lazy: local_precheck imports THIS module
            rederived = compute_findings(calls["warmup"], calls["a1"], calls["a2"], calls["b"],
                                         calls["c"], receipt["labels"])
            diff_key = _first_findings_diff(rederived, findings)
            if diff_key is not None:
                raise ArmHalt(f"precheck findings do not match a re-derivation from the raw calls "
                              f"(first differing key {diff_key!r}); the stored findings are not "
                              "trustworthy; re-run the precheck (A1/A4.3)")
            # ALLOWLIST for the two bases (replaces the old undetermined denylist): only a DECIDED basis
            # is acceptable. "undetermined (cache state differs)" — a truthy string — is not decided and
            # is refused, as is anything else unexpected.
            tb = findings.get("temperature_basis")
            if tb not in ("pre-temperature", "post-temperature"):
                raise ArmHalt(f"precheck temperature_basis {tb!r} is not a decided basis "
                              "(pre-temperature/post-temperature); refusing (A1/A4.3)")
            pb = findings.get("penalty_basis")
            if pb not in ("penalties-not-applied-to-logprobs", "post-sampling-chain"):
                raise ArmHalt(f"precheck penalty_basis {pb!r} is not a decided basis "
                              "(penalties-not-applied-to-logprobs/post-sampling-chain); refusing (A1/A4.3)")
        return info

    # ---------------- request building ----------------
    def prompt(self, r: Rendered) -> str:
        return r.canonical_text + "\n\n" + LOCAL_FORMAT_INSTRUCTION

    def chat_body(self, r: Rendered, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """The exact /api/chat request body (no credentials). `options` overrides self.options
        (used by the precheck to vary temperature / neutralise penalties)."""
        return {"model": self.serving_tag,
                "messages": [{"role": "user", "content": self.prompt(r)}],
                "stream": False, "think": False,
                "logprobs": True, "top_logprobs": TOP_LOGPROBS,
                "keep_alive": self.keep_alive,
                "options": dict(options if options is not None else self.options)}

    def _post_chat(self, body: Dict[str, Any]) -> Tuple[float, Dict[str, Any], str]:
        txt = canonical_json(body)
        t0 = time.monotonic()
        status, _hdr, rbytes = self.transport(
            "POST", _norm_base(self.base_url) + "/api/chat", txt.encode("utf-8"),
            {"Content-Type": "application/json"}, self.timeout)
        latency = time.monotonic() - t0
        if status != 200:
            raise ArmHalt(f"ollama /api/chat HTTP {status} for {self.serving_tag} on item "
                          f"{body['messages'][0]['content'][:40]!r}: {rbytes[:200].decode('utf-8','replace')}")
        try:
            resp = json.loads(rbytes)
        except json.JSONDecodeError as e:
            raise ArmHalt(f"ollama /api/chat 200 with unparseable body: {e}")
        if not isinstance(resp, dict):
            raise ArmHalt("ollama /api/chat 200 with non-object body")
        return latency, resp, sha256_text(txt)

    def _guard_served_model(self, resp: Dict[str, Any]) -> str:
        served = resp.get("model")
        # BELT-AND-BRACES (review note 5): an nxtg- served model is ALREADY halted by the pinned-tag
        # equality just below (serving_tag is a fixed base tag and never starts with nxtg-). This
        # branch only makes the diagnostic explicit — "derived nxtg- model / prompt parity" rather
        # than a generic tag mismatch — so it is intentionally redundant, not an independent guard.
        # (No mutant targets it precisely because reverting it changes only the message.)
        if isinstance(served, str) and served.startswith(NXTG_PREFIX):
            raise ArmHalt(f"ollama served a derived nxtg- model {served!r}; refusing (prompt parity, A4.2)")
        if served != self.serving_tag:
            raise ArmHalt(f"ollama served model {served!r} != pinned serving tag {self.serving_tag!r} "
                          f"(the arm is invalidated, §2)")
        return served

    def _recheck_digest_if_due(self) -> None:
        """Re-read /api/tags and compare the digest. A change mid-run is a version change (§2)."""
        if self._calls == 1 or self._calls % self.digest_recheck_every == 0:
            models = read_tags(self.base_url, self.transport, self.timeout)
            entry = _find_tag(models, self.serving_tag)
            if entry is None:
                raise ArmHalt(f"{self.serving_tag} disappeared from /api/tags mid-run (§2)")
            if not _digest_matches(entry.get("digest"), self.pinned_digest):
                raise ArmHalt(f"{self.serving_tag} digest changed mid-run to {entry.get('digest')!r} "
                              f"!= pin {self.pinned_digest!r}; a version change invalidates the arm (§2)")
            # keep the observed digest current so each record reflects the latest /api/tags (note 4)
            self.observed_digest = _norm_digest(entry.get("digest"))

    # ---------------- the arm contract ----------------
    def sampling_config(self) -> Dict[str, Any]:
        """The GATE F view: the ollama options rail-keyed (temperature -> `temp`, numeric), plus the
        request-level knobs. GATE F reads `temp`; a numeric temp is required (it is the registry
        temperature, e.g. 0.7, never 0 -> never greedy on a greedy_ok=false model)."""
        rail = {("temp" if k == "temperature" else k): v for k, v in self.options.items()}
        rail.update({"think": False, "logprobs": True, "top_logprobs": TOP_LOGPROBS})
        # GATE H output cap: ollama's num_predict IS this arm's output cap, sent on every /api/chat.
        # Report the value actually in self.options (never a constant); a non-int leaves None so
        # GATE H fails closed rather than certifying a cap nobody sent.
        cap = self.options.get("num_predict")
        rail["max_tokens"] = cap if isinstance(cap, int) and not isinstance(cap, bool) else None
        return rail

    def config(self) -> Dict[str, Any]:
        # Every knob that can change an answer. base_url is deliberately excluded (it is the tunnel
        # endpoint, machine-specific, and does not change the answer); the pinned digest IS included
        # because the served weights are part of the arm's identity.
        return {"arm": self.arm_name, "model_id": self.model_id, "serving_tag": self.serving_tag,
                "model_digest_pin": self.pinned_digest, "endpoint": "/api/chat",
                "probability_source": "top_logprobs over valid labels, renormalised",
                "format_instruction": LOCAL_FORMAT_INSTRUCTION,
                "options": self.options, "think": False, "num_predict": NUM_PREDICT,
                "logprobs": True, "top_logprobs": TOP_LOGPROBS, "keep_alive": self.keep_alive,
                "generation_cap": NUM_PREDICT}

    def call(self, r: Rendered) -> ArmResult:
        self._calls += 1
        self._recheck_digest_if_due()
        body = self.chat_body(r)
        latency, resp, wire_sha = self._post_chat(body)
        self._guard_served_model(resp)

        top = first_token_top_logprobs(resp)
        probs = raw_sum = None
        error = None
        abstained = True
        labels = r.labels
        if top is None:
            error = "ollama response carried no first-token top_logprobs"
            missing = list(labels)
        else:
            mass = label_masses(top, labels)
            missing = [l for l in labels if mass[l] == 0.0]
            vec = [mass[l] for l in labels]                 # aligned to labels == keys order
            if sum(vec) <= 0:
                error = "no valid label token in the first-token top_logprobs (abstention)"
            else:
                probs, raw_sum, error = normalise(vec)
                abstained = probs is None
        if probs is not None:
            abstained = False

        srv = resp.get("server", {}) if isinstance(resp.get("server"), dict) else resp
        server_latency = {}
        for src, dst in (("total_duration", "total_s"), ("load_duration", "load_s"),
                         ("eval_duration", "eval_s")):
            v = srv.get(src)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                server_latency[dst] = float(v) / 1e9      # ns -> s
        itok = resp.get("prompt_eval_count")
        otok = resp.get("eval_count")

        # Runtime truncation belt (independent review 2026-09-30, measured live on ollama 0.34.4): when a
        # prompt EXCEEDS num_ctx, ollama does NOT report a count at/over the window — it silently
        # truncates the prompt to about half the window and reports prompt_eval_count ~= num_ctx/2 + 2
        # (num_ctx 256 -> 130; num_ctx 4096 -> 2,050), with HTTP 200 and done_reason "length" either
        # way. So a count near num_ctx NEVER appears on truncation and the old `>= num_ctx-1` threshold
        # could not fire; the truncation tell is a count near HALF the window. HALT when the reported
        # prompt count reaches num_ctx//2 - 64 (1,984 at 4096) — never score a possibly-truncated prompt.
        # This belt is FAIL-CLOSED: it ALSO halts a legitimate, untruncated prompt of 1,984-4,095 tokens.
        # That is acceptable — the preflight (num_ctx - 64 on the chars/2.5 estimate) and the frozen
        # items stay far below it (worst real frozen item S1-choice-045 is 1,213 tokens), so a true
        # prompt never reaches the belt; it catches an unfrozen item or a wrong num_ctx.
        num_ctx = self.options.get("num_ctx", NUM_CTX)
        belt = num_ctx // 2 - 64
        if isinstance(itok, int) and not isinstance(itok, bool) and itok >= belt:
            raise ArmHalt(f"prompt_eval_count {itok} >= num_ctx//2-64 ({belt}) for "
                          f"{self.serving_tag} on {r.item_id}; possible silent prompt truncation "
                          "(ollama truncates to ~num_ctx/2 on overflow; reviewer 2026-09-30) (§2)")

        extra = {"labels_missing": missing, "ollama_version": self.ollama_version,
                 "options_sent": self.options, "done_reason": resp.get("done_reason"),
                 "pinned_digest": self.pinned_digest,   # note 4: the pin, kept beside the observed digest
                 "probability_source": "top_logprobs over valid labels"}
        if self.precheck is not None:
            extra["precheck_sha256"] = self.precheck["sha256"]
            extra["precheck_findings"] = self.precheck["findings"]
        if self.vram_report is not None:
            extra["vram_report"] = self.vram_report
        return ArmResult(
            arm=self.arm_name, item_id=r.item_id, requested_model=self.serving_tag,
            returned_model=resp.get("model"), model_id=self.model_id, serving_tag=self.serving_tag,
            # note 4: model_digest is the OBSERVED /api/tags digest (updated at each recheck), so the
            # runner's cross-record identity check compares a real value; the pin is in extra.
            model_digest=self.observed_digest, probabilities=probs, raw_probabilities=top,
            raw_sum=raw_sum, abstained=abstained, error=error, latency_s=latency,
            input_tokens=itok if isinstance(itok, int) else None,
            output_tokens=otok if isinstance(otok, int) else None,
            wire_request=body, wire_response=resp, wire_sha256=wire_sha,
            sampling_config=self.sampling_config(), server_latency=server_latency or None,
            extra=extra)

    def numctx_preflight(self, rendered_list: List[Rendered]) -> Dict[str, Any]:
        """Prereg §2: BEFORE any call, refuse the run if any item's estimated LOCAL prompt would
        exceed num_ctx minus a margin. The estimate (chars / 2.5 over the FULL prompt the arm sends —
        canonical text + the format instruction) is conservative AT THE WORST ITEM, not every item:
        2 of 340 items measure more real tokens than the estimate (reviewer F2, 2026-09-30: S2a-037 is
        590 real vs 528 est, +62; S2a-032 +24), which the 64-token margin still covers (S2a-037 by 2
        tokens). Names the worst item and REFUSES (ArmHalt); it never truncates silently. Returns the
        report for the run record."""
        num_ctx = int(self.options.get("num_ctx", NUM_CTX))
        ceiling = num_ctx - NUMCTX_TOKEN_MARGIN
        worst = None                                        # (est_tokens, item_id, chars)
        for r in rendered_list:
            chars = len(self.prompt(r))
            est = math.ceil(chars / NUMCTX_CHARS_PER_TOKEN)
            if worst is None or est > worst[0]:
                worst = (est, r.item_id, chars)
        report = {"num_ctx": num_ctx, "margin_tokens": NUMCTX_TOKEN_MARGIN, "ceiling_tokens": ceiling,
                  "chars_per_token": NUMCTX_CHARS_PER_TOKEN, "n_items": len(rendered_list),
                  "worst_item_id": worst[1] if worst else None,
                  "worst_est_tokens": worst[0] if worst else 0,
                  "worst_prompt_chars": worst[2] if worst else 0}
        if worst is not None and worst[0] > ceiling:
            raise ArmHalt(f"num_ctx preflight: item {worst[1]!r} est {worst[0]} prompt tokens "
                          f"(chars {worst[2]} / {NUMCTX_CHARS_PER_TOKEN}) exceeds num_ctx {num_ctx} "
                          f"- margin {NUMCTX_TOKEN_MARGIN} = {ceiling}; raise num_ctx or shorten the "
                          "item — NEVER truncate silently (§2)")
        return report

    def close(self) -> None:
        """Best-effort model unload (keep_alive:0). The runner calls this in a finally block on any
        arm that has it; it never raises into that finally (the runner swallows), and it is a no-op
        if the transport is unavailable."""
        body = {"model": self.serving_tag, "messages": [], "keep_alive": 0}
        self.transport("POST", _norm_base(self.base_url) + "/api/chat",
                       canonical_json(body).encode("utf-8"),
                       {"Content-Type": "application/json"}, self.timeout)


def _default_run_probe(argv: List[str], timeout: float = 20.0) -> str:
    """Run the VRAM probe argv directly (never through a shell). Returns stdout."""
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise ArmHalt(f"VRAM probe {argv!r} exited {p.returncode}: {p.stderr[:200]}")
    return p.stdout


def _parse_free_mib(stdout: str) -> int:
    """First numeric line of `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`."""
    for line in (stdout or "").splitlines():
        s = line.strip().split()[0] if line.strip() else ""
        if s:
            try:
                return int(float(s))
            except ValueError:
                continue
    raise ArmHalt(f"could not parse free-MiB integer from VRAM probe output {stdout!r}")


def vram_preflight(model_id: str, *, transport: Optional[Transport] = None,
                   run_probe: Optional[Callable[[List[str]], str]] = None,
                   base_url: Optional[str] = None, probe_cmd: Optional[str] = None,
                   timeout: float = 30.0, num_ctx: int = NUM_CTX) -> Dict[str, Any]:
    """Prereg §7: free VRAM >= ceil(model size / 2^20) + KV(num_ctx) + 2048 MiB, checked BEFORE the
    arm is built.

    Prices BOTH the weights (from /api/tags size) AND the KV cache at the pinned `num_ctx` (from
    /api/show model_info, via kv_cache_mib). The old gate priced only the weights, so a run at
    ollama's default context (qwen3:14b => 40,960) spilled KV into shared system RAM and passed
    anyway (E.4 first live LOCAL-P run). Runs JEVCAL_VRAM_PROBE (shlex-split, never a shell), and
    REFUSES (ArmHalt) on: unset base URL, unset probe, an absent tag, no size, MISSING KV metadata
    (fail closed), or too little free VRAM. Every number is returned for the run record
    (`local_preflight`). Tests inject `run_probe`."""
    base = base_url or os.environ.get(BASE_URL_ENV)
    if not base:
        raise ArmHalt(f"{BASE_URL_ENV} is unset; cannot resolve the VRAM precondition host (A2)")
    if model_id not in IDENTITY:
        raise ArmHalt(f"model_id {model_id!r} has no A4.2 identity entry")
    tag, _digest = IDENTITY[model_id]
    tr = transport or urllib_transport
    entry = _find_tag(read_tags(base, tr, timeout), tag)
    if entry is None:
        raise ArmHalt(f"serving tag {tag!r} absent from /api/tags; cannot size the VRAM precondition")
    size = entry.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ArmHalt(f"/api/tags entry for {tag!r} has no positive integer size: {size!r}")
    # §7 KV pricing: read /api/show model_info and size the KV cache at num_ctx (fail closed if a
    # needed field is absent, before the model is loaded).
    show = _read_json_post(base, "/api/show", {"model": tag}, tr, timeout)
    model_info = show.get("model_info") if isinstance(show, dict) else None
    kv = kv_cache_mib(model_info, num_ctx)
    cmd = probe_cmd or os.environ.get(VRAM_PROBE_ENV)
    if not cmd or not cmd.strip():
        raise ArmHalt(f"{VRAM_PROBE_ENV} is unset; the free-VRAM precondition cannot be checked (§7)")
    argv = shlex.split(cmd)
    free_mib = _parse_free_mib((run_probe or _default_run_probe)(argv))
    size_mib = math.ceil(size / (1024 * 1024))
    required = size_mib + kv["kv_mib"] + VRAM_HEADROOM_MIB
    report = {"model_id": model_id, "serving_tag": tag, "model_size_bytes": size,
              "model_size_mib": size_mib, "num_ctx": int(num_ctx), "kv_mib": kv["kv_mib"],
              "kv_formula": kv, "headroom_mib": VRAM_HEADROOM_MIB,
              "required_free_mib": required, "free_mib": free_mib, "probe_argv": argv,
              "ok": free_mib >= required}
    if not report["ok"]:
        raise ArmHalt(f"VRAM precondition failed for {model_id}: free {free_mib} MiB < required "
                      f"{required} MiB (model {size_mib} + KV {kv['kv_mib']} @ num_ctx {num_ctx} + "
                      f"{VRAM_HEADROOM_MIB} MiB headroom, §7)")
    return report


def make_local_arm(arm_name: str, model_id: str) -> Callable[..., "LocalArm"]:
    """Factory for the ARMS registry: fixes arm_name + model_id, passes the runner's kwargs through."""
    def factory(**kw: Any) -> "LocalArm":
        return LocalArm(arm_name=arm_name, model_id=model_id, **kw)
    return factory
