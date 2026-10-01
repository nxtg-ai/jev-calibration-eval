"""Reusable eval-rail LangFuse tracing helper — GATE O v0.1 (FROZEN 2026-07-08).

Wraps an eval's model leg + judge leg in ONE LangFuse trace and returns the
GATE-O trace-completeness fields VERBATIM per the frozen contract
(standards/deterministic-eval-rail.md, GATE O v0.1):

  * on a successfully-traced run  -> {"langfuse_trace_ids": [<id>, ...]}   (>=1 id
    spanning the pinned model + judge calls)
  * on a GENUINE run-time LangFuse outage (health probe fails) ->
    {"langfuse_unreachable_proof": {"checked_at","endpoint","probe_result","attempts"}}

Contract stance (deliberate): fail-OPEN protects a real OUTAGE only. If LangFuse
is REACHABLE but tracing/auth fails, this helper RAISES rather than fabricating an
unreachable proof — "silent non-tracing while LangFuse was reachable" is a GATE-O
FAIL, and a fabricated proof would be exactly that failure in disguise.

Future eval runners import `trace_eval_run` and pass their real model_call /
judge_call callables; the helper owns the trace plumbing + the GATE-O emit shape.

LangFuse v4 Python SDK (targets self-hosted server v3). Keypair resolves from the
env first, else deploy/langfuse/.lf-eval-rail.env (gitignored).
"""
from __future__ import annotations

import datetime
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

_DEFAULT_ENV_FILE = Path(__file__).resolve().parent / ".lf-eval-rail.env"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_keypair(env_file: Path = _DEFAULT_ENV_FILE) -> tuple[str, str, str]:
    """Resolve (public_key, secret_key, base_url). Env vars win; else parse the
    gitignored .lf-eval-rail.env (KEY="value" or KEY=value lines, # comments)."""
    pub = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sec = os.environ.get("LANGFUSE_SECRET_KEY")
    base = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")
    if not (pub and sec and base) and env_file.exists():
        for raw in env_file.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            v = v.strip().strip('"').strip("'")
            k = k.strip()
            if k == "LANGFUSE_PUBLIC_KEY" and not pub:
                pub = v
            elif k == "LANGFUSE_SECRET_KEY" and not sec:
                sec = v
            elif k in ("LANGFUSE_BASE_URL", "LANGFUSE_HOST") and not base:
                base = v
    if not (pub and sec and base):
        raise RuntimeError(
            "eval-rail LangFuse keypair unresolved: set LANGFUSE_PUBLIC_KEY / "
            f"LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL or provide {env_file}"
        )
    return pub, sec, base


def health_probe(base_url: str, attempts: int = 3, timeout: float = 5.0) -> tuple[int, str, str, bool]:
    """Probe /api/public/health. Returns (attempts_used, endpoint, probe_result, reachable)."""
    endpoint = base_url.rstrip("/") + "/api/public/health"
    result = "no-attempt"
    for i in range(1, max(1, attempts) + 1):
        try:
            with urllib.request.urlopen(endpoint, timeout=timeout) as resp:
                code = getattr(resp, "status", resp.getcode())
                if 200 <= int(code) < 300:
                    return i, endpoint, str(code), True
                result = str(code)
        except urllib.error.HTTPError as e:
            result = str(e.code)
        except Exception as e:  # URLError / timeout / conn refused = genuine outage
            result = f"error:{type(e).__name__}"
        time.sleep(1)
    return max(1, attempts), endpoint, result, False


def unreachable_proof(base_url: str, attempts: int = 3) -> dict:
    """Return the GATE-O langfuse_unreachable_proof block (probe run NOW)."""
    checked_at = _now_iso()
    used, endpoint, result, reachable = health_probe(base_url, attempts=attempts)
    if reachable:
        raise RuntimeError(
            f"unreachable_proof requested but LangFuse IS reachable at {endpoint} "
            f"(probe_result={result}) — silent non-tracing while reachable is a GATE-O FAIL"
        )
    return {
        "langfuse_unreachable_proof": {
            "checked_at": checked_at,
            "endpoint": endpoint,
            "probe_result": result,
            "attempts": used,
        }
    }


def trace_eval_run(
    *,
    run_id: str,
    model_id: str,
    task_input: Any,
    model_call: Callable[[], Any],
    judge_call: Callable[[Any], Any],
    judge_id: str = "eval-rail-judge",
    model_parameters: Optional[dict] = None,
    metadata: Optional[dict] = None,
    health_attempts: int = 3,
    env_file: Path = _DEFAULT_ENV_FILE,
) -> dict:
    """Run the model leg + judge leg inside ONE LangFuse trace; return GATE-O fields.

    model_call() -> model_output (the real generation; caller owns the real work).
    judge_call(model_output) -> judge_verdict (the real grading).

    Returns {"langfuse_trace_ids": [trace_id], "trace_url": ..., "model_output": ...,
             "judge_verdict": ...} on success, or {"langfuse_unreachable_proof": {...}}
    if LangFuse is genuinely unreachable AT RUN TIME. Raises if reachable-but-broken.
    """
    pub, sec, base = load_keypair(env_file)

    # Run-time liveness probe FIRST — this decides fail-open vs trace.
    used, endpoint, result, reachable = health_probe(base, attempts=health_attempts)
    checked_at = _now_iso()
    if not reachable:
        return {
            "langfuse_unreachable_proof": {
                "checked_at": checked_at,
                "endpoint": endpoint,
                "probe_result": result,
                "attempts": used,
            }
        }

    from langfuse import Langfuse

    client = Langfuse(public_key=pub, secret_key=sec, host=base)
    if not client.auth_check():
        # Reachable but auth failed => NOT an outage. Fail loud (contract).
        raise RuntimeError(
            f"LangFuse reachable at {endpoint} but auth_check() failed for the "
            "eval-rail keypair — refusing to emit a fake unreachable_proof (GATE-O)"
        )

    trace_id = client.create_trace_id(seed=run_id)
    base_meta = {"run_id": run_id, "gate": "O", "rail": "eval-rail"}
    if metadata:
        base_meta.update(metadata)

    with client.start_as_current_observation(
        name=f"eval-rail:{run_id}",
        as_type="span",
        trace_context={"trace_id": trace_id},
        input=task_input,
        metadata=base_meta,
    ) as root:
        with client.start_as_current_observation(
            name="model-call",
            as_type="generation",
            model=model_id,
            model_parameters=model_parameters or {},
            input=task_input,
        ) as gen:
            model_output = model_call()
            gen.update(output=model_output)

        with client.start_as_current_observation(
            name="judge-call",
            as_type="evaluator",
            input={"task_input": task_input, "model_output": model_output},
            metadata={"judge_id": judge_id},
        ) as jud:
            judge_verdict = judge_call(model_output)
            jud.update(output=judge_verdict)

        root.update(output={"model_output": model_output, "judge_verdict": judge_verdict})

    client.flush()

    try:
        trace_url = client.get_trace_url(trace_id=trace_id)
    except Exception:
        trace_url = f"{base.rstrip('/')}/project/{os.environ.get('LANGFUSE_PROJECT_ID', 'eval-rail')}/traces/{trace_id}"

    return {
        "langfuse_trace_ids": [trace_id],
        "trace_url": trace_url,
        "model_output": model_output,
        "judge_verdict": judge_verdict,
    }
