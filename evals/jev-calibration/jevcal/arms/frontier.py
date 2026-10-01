"""FRONTIER arm: claude-opus-5-5 via the subscription CLI, VERBALISED probabilities.

No logprobs are exposed, so the probability source is the model's own stated
per-option probabilities, returned as schema-constrained JSON, validated and
renormalised. The arm is labelled "verbalised" everywhere it is reported
(prereg §2): a known-weaker estimator, disclosed rather than hidden.

Sampling (prereg A4.1): `claude -p` has no temperature control, so every record
logs temperature as exactly TEMP_LABEL. `--effort` is REQUIRED, passed explicitly
on every call and logged per record: there is no default, because a run launched
from ~/ASIF would otherwise silently inherit `xhigh` from ~/ASIF/.claude/settings.json.

Secondary estimator (A4.1, descriptive only, feeds no H-test): `sample_answer`
asks for a single option letter; the runner calls it N=5 independent times at
provider-default sampling on a fixed subset and reports the empirical frequency.

Isolation: `--setting-sources ""` (no user/project hooks or settings),
`--tools ""`, `--strict-mcp-config` (no MCP servers), `--no-session-persistence`,
a fixed `--system-prompt`, and an empty working directory. `--bare` is not used
because it requires an API key and this arm must run on the subscription.

Fail-closed HALT: served model != pinned, CLI error, or any rate-limit event
that is not `allowed` / reports overage use (a spend signal).
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional

from .. import isolation
from ..render import Rendered
from .base import Arm, ArmHalt, ArmResult, canonical_json, normalise, sha256_text

PINNED_MODEL = "claude-opus-5-5"
# The registered FRONTIER arm's effort. Prereg A5.1 fixes it at `high` (documented default
# tier; matches TypeSafe's reference labellers "at high thinking"; xhigh is our CoS pin, not
# a production setting). It is part of the arm's identity: any other value is a different
# arm and is refused before any call. Changing it requires a prereg amendment, not a flag.
PINNED_EFFORT = "high"
TEMP_LABEL = "provider-default (not settable or observable via claude -p)"
MAX_OUTPUT_TOKENS_REQUESTED = 1024  # via CLAUDE_CODE_MAX_OUTPUT_TOKENS; see config note
ACCESS_PATH = "claude-cli-print"  # how the model is reached (GATE F v3 / GATE H); registry-confirmed

SYSTEM_PROMPT = (
    "You are one arm of a calibration evaluation. Read the state, the question and "
    "the lettered options. Report, for every option letter, your probability that "
    "that option is the correct answer. Probabilities must be between 0 and 1 and "
    "sum to 1. Report your honest uncertainty; do not round to 0 or 1 unless you are "
    "certain. Respond only through the required JSON structure."
)
SECONDARY_SYSTEM_PROMPT = (
    "You are one arm of a calibration evaluation. Read the state, the question and "
    "the lettered options, and answer with the single option letter you believe is "
    "correct. Respond only through the required JSON structure."
)


def format_instruction(labels: List[str]) -> str:
    return ("Give your probability that each option is correct, as JSON "
            '{"probabilities": {' + ", ".join(f'"{l}": p' for l in labels) + "}}.")


def secondary_instruction(labels: List[str]) -> str:
    return ('Answer with the single letter of the correct option, as JSON {"answer": "<letter>"}, '
            "where <letter> is one of " + ", ".join(labels) + ".")


def json_schema(labels: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {"probabilities": {
            "type": "object",
            "properties": {l: {"type": "number", "minimum": 0, "maximum": 1} for l in labels},
            "required": list(labels),
            "additionalProperties": False,
        }},
        "required": ["probabilities"],
        "additionalProperties": False,
    }


def secondary_schema(labels: List[str]) -> Dict[str, Any]:
    return {"type": "object", "properties": {"answer": {"type": "string", "enum": list(labels)}},
            "required": ["answer"], "additionalProperties": False}


# (argv, env, cwd, timeout) -> (returncode, stdout, stderr)
Runner = Callable[[List[str], Dict[str, str], str, float], Any]


def subprocess_runner(argv, env, cwd, timeout):
    p = subprocess.run(argv, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def parse_cli_output(stdout: str):
    events = json.loads(stdout)
    if not isinstance(events, list):
        raise ArmHalt("claude -p output is not an event list")
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), None)
    result = next((e for e in events if e.get("type") == "result"), None)
    assistants = [e for e in events if e.get("type") == "assistant"]
    rate = [e for e in events if e.get("type") == "rate_limit_event"]
    return init, result, assistants, rate


class FrontierArm(Arm):
    arm_name = "FRONTIER"
    model_id = PINNED_MODEL     # verbatim sampling-registry key
    serving_tag = "n/a"         # claude -p (A4.2 table)
    family = "anthropic"

    def __init__(self, effort: Optional[str] = None, runner: Optional[Runner] = None,
                 timeout: float = 300.0, claude_bin: str = "claude"):
        if effort != PINNED_EFFORT:
            raise ValueError(f"FRONTIER effort is REQUIRED and must be exactly {PINNED_EFFORT!r} "
                             f"(prereg A5.1); got {effort!r}. There is no default (A4.1)")
        self.effort = effort
        self.runner = runner or subprocess_runner
        self.timeout = timeout
        self.claude_bin = claude_bin
        # One fixed, empty cwd for every child (A5.2 (b)); created under the system
        # temp dir, and CHECKED rather than assumed to be outside ~/ASIF.
        self.cwd = tempfile.mkdtemp(prefix="jevcal-frontier-")
        atexit.register(shutil.rmtree, self.cwd, True)

    def child_env(self) -> Dict[str, str]:
        """The exact env every claude -p child receives (A5.2 (a))."""
        env = {k: v for k, v in os.environ.items() if k != isolation.EFFORT_KEY}
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(MAX_OUTPUT_TOKENS_REQUESTED)
        env.pop("TYPESAFE_API_KEY", None)          # no Jev credential reaches another process
        env.pop("JEV_KEY_FILE", None)
        return env

    def isolation_report(self, env: Optional[Dict[str, str]] = None,
                         cwd: Optional[str] = None) -> Dict[str, Any]:
        rep = isolation.check(self.child_env() if env is None else env,
                              self.cwd if cwd is None else cwd)
        rep["effort_flag"] = self.effort
        return rep

    def sampling_config(self) -> Dict[str, Any]:
        # access_path/sampling_exposed/output_cap_exposed are the typed declaration
        # GATE F v3 / GATE H read: claude -p (print mode) exposes no temperature and
        # its output cap is only env-REQUESTED (unverifiable); the registry confirms
        # both are not-exposed for this access path. claude-opus-5-5 reached via the
        # HTTP API would expose temp/max_tokens — exposure is a property of the path.
        return {"temp": TEMP_LABEL, "top_p": None, "seed": None, "effort": self.effort,
                "max_tokens": None,
                "access_path": ACCESS_PATH,
                "sampling_exposed": False, "output_cap_exposed": False,
                "max_output_tokens_requested": MAX_OUTPUT_TOKENS_REQUESTED,
                "note": "the claude CLI exposes no temperature/top_p/seed; Anthropic documents a "
                        "default temperature of 1.0, which is a documented default, not an "
                        "observation. The output cap is requested via "
                        "CLAUDE_CODE_MAX_OUTPUT_TOKENS and is NOT verifiable from the CLI"}

    def config(self) -> Dict[str, Any]:
        return {"arm": self.arm_name, "model": PINNED_MODEL, "transport": "claude -p (subscription CLI)",
                "probability_source": "verbalised",
                "flags": ["--setting-sources", "", "--tools", "", "--strict-mcp-config",
                          "--no-session-persistence", "--output-format", "json",
                          "--effort", self.effort],
                "effort": self.effort,
                "system_prompt": SYSTEM_PROMPT,
                "format_instruction_template": format_instruction(["<L1>", "<L2>"]),
                "secondary_system_prompt": SECONDARY_SYSTEM_PROMPT,
                "secondary_instruction_template": secondary_instruction(["<L1>", "<L2>"]),
                "max_output_tokens_requested": MAX_OUTPUT_TOKENS_REQUESTED,
                "max_output_tokens_mechanism": "CLAUDE_CODE_MAX_OUTPUT_TOKENS env",
                "max_output_tokens_note": "requested, not verified: the CLI's modelUsage reports "
                                          "maxOutputTokens=128000 regardless",
                "temperature": TEMP_LABEL}

    def prompt(self, r: Rendered) -> str:
        return r.canonical_text + "\n\n" + format_instruction(r.labels)

    def _invoke(self, item_id: str, system_prompt: str, prompt: str, schema: Dict[str, Any]):
        """One claude -p call with every fail-closed check. Returns
        (result_event, modelUsage_for_pin, latency_s, served_model, wire, wire_sha256)."""
        argv = [self.claude_bin, "-p", "--model", PINNED_MODEL, "--setting-sources", "",
                "--tools", "", "--strict-mcp-config", "--no-session-persistence",
                "--effort", self.effort, "--system-prompt", system_prompt,
                "--output-format", "json", "--json-schema", canonical_json(schema), prompt]
        env = self.child_env()
        cwd = self.cwd
        iso = self.isolation_report(env, cwd)       # the EXACT env + cwd handed to the child
        if not iso["ok"]:
            raise ArmHalt(f"FRONTIER isolation failed on item {item_id}: {iso['details']}")
        wire = {"system_prompt": system_prompt, "user_prompt": prompt, "json_schema": schema,
                "model": PINNED_MODEL, "effort": self.effort, "temperature": TEMP_LABEL}
        t0 = time.monotonic()
        rc, out, err = self.runner(argv, env, cwd, self.timeout)
        latency = time.monotonic() - t0
        if rc != 0:
            raise ArmHalt(f"FRONTIER claude -p exit {rc} on item {item_id}: {err[:300]}")
        init, result, assistants, rate = parse_cli_output(out)
        for ev in rate:
            info = ev.get("rate_limit_info") or {}
            if info.get("status") != "allowed" or info.get("isUsingOverage"):
                raise ArmHalt(f"FRONTIER rate-limit/spend signal on item {item_id}: "
                              f"status={info.get('status')} overage={info.get('isUsingOverage')}")
        if result is None or result.get("is_error"):
            raise ArmHalt(f"FRONTIER CLI result error on item {item_id}: "
                          f"{(result or {}).get('result')!r}")
        served = set()
        if init and init.get("model"):
            served.add(init["model"])
        for a in assistants:
            m = (a.get("message") or {}).get("model")
            if m:
                served.add(m)
        served |= set((result.get("modelUsage") or {}).keys())
        if served != {PINNED_MODEL}:
            raise ArmHalt(f"FRONTIER served model(s) {sorted(served)} != pinned {PINNED_MODEL!r} "
                          f"(item {item_id}); the arm is invalidated (prereg §2)")
        mu = (result.get("modelUsage") or {}).get(PINNED_MODEL, {})
        return result, mu, latency, PINNED_MODEL, wire, sha256_text(canonical_json(wire))

    @staticmethod
    def _input_tokens(mu):
        if not mu:
            return None
        return sum(int(mu.get(k) or 0) for k in
                   ("inputTokens", "cacheReadInputTokens", "cacheCreationInputTokens"))

    def call(self, r: Rendered) -> ArmResult:
        result, mu, latency, served, wire, wire_sha = self._invoke(
            r.item_id, SYSTEM_PROMPT, self.prompt(r), json_schema(r.labels))
        so = result.get("structured_output")
        raw = so.get("probabilities") if isinstance(so, dict) else None
        vec, error = None, None
        if not isinstance(raw, dict) or set(raw) != set(r.labels):
            error = f"structured probabilities missing or keys != {r.labels}: {raw!r}"
        else:
            vec = [raw[l] for l in r.labels]
        probs, s = None, None
        if vec is not None:
            probs, s, error = normalise(vec)
        return ArmResult(
            arm=self.arm_name, item_id=r.item_id, requested_model=PINNED_MODEL,
            returned_model=served, model_id=self.model_id, serving_tag=self.serving_tag,
            model_digest="n/a", probabilities=probs, raw_probabilities=raw, raw_sum=s,
            abstained=probs is None, error=error, latency_s=latency,
            input_tokens=self._input_tokens(mu), output_tokens=mu.get("outputTokens"),
            wire_request=wire, wire_response={"result": result.get("result"),
                                              "structured_output": so,
                                              "stop_reason": result.get("stop_reason"),
                                              "modelUsage": result.get("modelUsage"),
                                              "session_id": result.get("session_id")},
            wire_sha256=wire_sha, cost_usd_list=mu.get("costUSD"),
            sampling_config=self.sampling_config(),
            extra={"probability_source": "verbalised", "effort": self.effort,
                   "temperature": TEMP_LABEL, "child_cwd": self.cwd,
                   "child_env_has_effort_key": isolation.EFFORT_KEY in self.child_env(),
                   "observed_cli_maxOutputTokens": mu.get("maxOutputTokens"),
                   "duration_api_ms": result.get("duration_api_ms")})

    def sample_answer(self, r: Rendered, sample_index: int) -> Dict[str, Any]:
        """ONE independent single-letter sample for the A4.1 secondary estimator."""
        prompt = r.canonical_text + "\n\n" + secondary_instruction(r.labels)
        result, mu, latency, served, wire, wire_sha = self._invoke(
            r.item_id, SECONDARY_SYSTEM_PROMPT, prompt, secondary_schema(r.labels))
        so = result.get("structured_output")
        label = so.get("answer") if isinstance(so, dict) else None
        ok = label in r.labels
        return {"item_id": r.item_id, "arm": self.arm_name, "estimator": "secondary-5-sample",
                "sample_index": sample_index, "model_id": self.model_id,
                "serving_tag": self.serving_tag, "model_digest": "n/a",
                "returned_model": served, "effort": self.effort, "temperature": TEMP_LABEL,
                "answer_label": label if ok else None,
                "answer": r.keys[r.labels.index(label)] if ok else None,
                "error": None if ok else f"no valid letter in structured output: {so!r}",
                "latency_s": latency, "input_tokens": self._input_tokens(mu),
                "output_tokens": mu.get("outputTokens"), "cost_usd_list": mu.get("costUSD"),
                "msg_sha256": r.msg_sha256, "wire_sha256": wire_sha, "wire_request": wire,
                "session_id": result.get("session_id")}
