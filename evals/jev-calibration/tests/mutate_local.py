#!/usr/bin/env python3
"""Mutation proof for the LOCAL arm (E.4b): revert each guard in the SHIPPED file and require the
arm named for it to go RED for the RIGHT reason (canon 2026-09-29: a mutant must reproduce the
DEFECT's mechanism, not merely change the file). Engine: tests/mutation_engine.py, which asserts
each anchor matches exactly once AND that the file's bytes changed. Report: $JEVCAL_LOCAL_MUTATION_REPORT.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mutation_engine import run_suite  # noqa: E402

# ArmNxtgRefused is intentionally NOT here (review note 5): an nxtg- served model is already halted
# by the pinned-tag equality, so reverting the nxtg- branch changes only the diagnostic message, not
# behaviour — there is no mutant that turns it red for a real reason. The guard is labelled
# belt-and-braces in local.py and the ArmNxtgRefused test still runs in the normal suite.
ARMS = ["ArmDigestPin", "ArmServedModel", "ArmNon200", "ArmBaseUrl",
        "ArmUnmappedKey", "ArmNoNumericConfig", "ArmLabelParse", "ArmLatencyNs", "ArmVram",
        "ArmCertifyingPrecheck", "ArmPrecheckBasis", "ArmObservedDigest", "ArmPrecheckVramGate",
        "ArmPrecheckCacheConfound", "ArmCertBindConfoundedRefused", "ArmCertBindValidAccepted",
        "ArmCertBindEditedFindingRefused", "ArmCertBindUndeterminedRefused",
        "ArmNumCtxOption", "ArmNumCtxPreflight", "ArmNumCtxBelt", "ArmVramKv", "ArmSignalUnload"]
LOC = "jevcal/arms/local.py"
PC = "jevcal/local_precheck.py"
BASE = "jevcal/arms/base.py"

MUTATIONS = [
    # accept a construction-time digest that != the A4.2 pin -> ArmDigestPin.construction stops halting
    ("digest-construction-loosened", "ArmDigestPin", LOC,
     "            raise ArmHalt(f\"{self.serving_tag} digest {entry.get('digest')!r} != pin \"\n"
     "                          f\"{self.pinned_digest!r}; a version change invalidates the arm (§2/A4.2)\")",
     "            pass  # mutant: accept a mismatched construction digest"),

    # never re-check the digest mid-run -> a version change lands undetected (ArmDigestPin.mid_run)
    ("digest-recheck-disabled", "ArmDigestPin", LOC,
     "if self._calls == 1 or self._calls % self.digest_recheck_every == 0:",
     "if False:  # mutant: never re-check the digest mid-run"),

    # accept a served model that is not the pinned serving tag (ArmServedModel)
    ("served-model-unchecked", "ArmServedModel", LOC,
     "if served != self.serving_tag:",
     "if False:  # mutant: accept any served model"),

    # record the PIN instead of the observed digest (ArmObservedDigest: cross-record identity check
    # would then compare a constant, not the served weights — review note 4)
    ("record-pin-not-observed-digest", "ArmObservedDigest", LOC,
     "model_digest=self.observed_digest, probabilities=probs",
     "model_digest=self.pinned_digest, probabilities=probs"),

    # accept a non-200 /api/chat status (ArmNon200: a valid body under HTTP 500 slips through)
    ("non-200-accepted", "ArmNon200", LOC,
     "raise ArmHalt(f\"ollama /api/chat HTTP {status} for {self.serving_tag} on item \"\n"
     "                          f\"{body['messages'][0]['content'][:40]!r}: {rbytes[:200].decode('utf-8','replace')}\")",
     "pass  # mutant: accept a non-200 /api/chat"),

    # drop the ns->s conversion on server-reported durations (ArmLatencyNs)
    ("ns-to-s-dropped", "ArmLatencyNs", LOC,
     "float(v) / 1e9",
     "float(v)"),

    # fall back to a hard-coded host instead of refusing an unset base URL (ArmBaseUrl)
    ("base-url-hardcoded-fallback", "ArmBaseUrl", LOC,
     "self.base_url = base_url or os.environ.get(BASE_URL_ENV)",
     "self.base_url = base_url or os.environ.get(BASE_URL_ENV) or \"http://127.0.0.1:11434\""),

    # let an unmapped registry sampling key pass through silently (ArmUnmappedKey)
    ("unmapped-key-passes", "ArmUnmappedKey", LOC,
     "            raise ArmHalt(f\"model {model_id!r}: unmapped registry sampling key {k!r} (A1); refusing \"\n"
     "                          \"rather than passing an unknown knob through silently\")",
     "            mapped[k] = v  # mutant: unmapped key passes silently"),

    # accept a model with no numeric recommended config (ArmNoNumericConfig: gpt-oss stops refusing)
    ("no-numeric-config-accepted", "ArmNoNumericConfig", LOC,
     "        raise ArmHalt(f\"model {model_id!r}: registry holds no numeric recommended sampling config \"\n"
     "                      \"(only a note); refusing until a cited config lands (A1)\")",
     "        pass  # mutant: accept a no-numeric-config model"),

    # stop summing whitespace label variants (ArmLabelParse: ' A' no longer counts for A)
    ("label-strip-dropped", "ArmLabelParse", LOC,
     "stripped = tok.strip()",
     "stripped = tok  # mutant: drop whitespace-variant summing"),

    # force the VRAM precondition to always pass (ArmVram.fail case)
    ("vram-check-always-ok", "ArmVram", LOC,
     "\"ok\": free_mib >= required}",
     "\"ok\": True}"),

    # skip the certifying-mode precheck requirement (ArmCertifyingPrecheck.without_receipt)
    ("certifying-precheck-skipped", "ArmCertifyingPrecheck", LOC,
     "                raise ArmHalt(f\"certifying mode requires a LOCAL precheck receipt for {self.arm_name} \"\n"
     "                              f\"(JEVCAL_LOCAL_PRECHECK_{self.arm_name.replace('-', '_')} or \"\n"
     "                              \"--local-precheck); none given (A1/A4.3)\")",
     "                pass  # mutant: skip the certifying precheck requirement"),

    # break the basis determination in the precheck: always return the eq-label, so a post-temperature
    # / post-sampling-chain difference is mislabelled as equal (ArmPrecheckBasis.post_temperature)
    ("precheck-basis-always-equal", "ArmPrecheckBasis", PC,
     "return eq_label if _eq(xl, yl) else neq_label",
     "return eq_label  # mutant: ignore the logprob comparison"),

    # remove the §7 VRAM preflight from the precheck CLI: the model then loads without the check
    # (ArmPrecheckVramGate — probe-unset/short no longer refuse, and no numbers reach the receipt)
    ("precheck-skips-vram-preflight", "ArmPrecheckVramGate", PC,
     "vram = L.vram_preflight(a.model_id, base_url=a.base_url)",
     "vram = {}  # mutant: skip the §7 VRAM preflight"),

    # revert the temperature-basis baseline from warm a1 to the COLD warm-up: the cache-state delta
    # gets read as a temperature effect again (ArmPrecheckCacheConfound — real 04:03Z receipt data)
    ("precheck-baseline-is-cold-warmup", "ArmPrecheckCacheConfound", PC,
     'basis(a1l, bl, a1, b, "pre-temperature", "post-temperature")',
     'basis(wl, bl, warmup, b, "pre-temperature", "post-temperature")'),

    # RESTORE trust-the-stored-findings: drop BOTH the calls-presence check and the re-derive, so the
    # certifying binding trusts the receipt's stored findings again — the exact PR-#85 defect. The
    # confounded receipt (no warmup, stored allowlisted bases) and the edited-finding receipt then both
    # sail through. Named ArmCertBindConfoundedRefused; ArmCertBindEditedFindingRefused also goes RED
    # (both (a) and (c), verified in the report's red_arms).
    ("bind-trusts-stored-findings", "ArmCertBindConfoundedRefused", LOC,
     "            calls = receipt.get(\"calls\")\n"
     "            if not isinstance(calls, dict):\n"
     "                raise ArmHalt(\"precheck receipt has no 'calls' object; receipt predates the warm-up \"\n"
     "                              \"precheck; re-run it\")\n"
     "            missing_calls = [k for k in (\"warmup\", \"a1\", \"a2\", \"b\", \"c\")\n"
     "                             if not isinstance(calls.get(k), dict)]\n"
     "            if missing_calls:\n"
     "                raise ArmHalt(f\"precheck receipt calls missing {missing_calls}; receipt predates the \"\n"
     "                              \"warm-up precheck; re-run it\")\n"
     "            if not (isinstance(receipt.get(\"labels\"), list) and receipt.get(\"labels\")):\n"
     "                raise ArmHalt(\"precheck receipt has no non-empty 'labels' list; cannot re-derive findings\")\n"
     "            from ..local_precheck import compute_findings   # lazy: local_precheck imports THIS module\n"
     "            rederived = compute_findings(calls[\"warmup\"], calls[\"a1\"], calls[\"a2\"], calls[\"b\"],\n"
     "                                         calls[\"c\"], receipt[\"labels\"])\n"
     "            diff_key = _first_findings_diff(rederived, findings)\n"
     "            if diff_key is not None:\n"
     "                raise ArmHalt(f\"precheck findings do not match a re-derivation from the raw calls \"\n"
     "                              f\"(first differing key {diff_key!r}); the stored findings are not \"\n"
     "                              \"trustworthy; re-run the precheck (A1/A4.3)\")",
     "            pass  # mutant bind-trusts-stored-findings: skip calls-presence check + re-derive"),

    # LOOSEN the basis allowlist to accept ANY string: an undetermined basis (a truthy string) is then
    # accepted (ArmCertBindUndeterminedRefused). (a)/(c) stay refused by the checks above, so this
    # mutant reds ONLY (d).
    ("basis-allowlist-accepts-any", "ArmCertBindUndeterminedRefused", LOC,
     "            tb = findings.get(\"temperature_basis\")\n"
     "            if tb not in (\"pre-temperature\", \"post-temperature\"):\n"
     "                raise ArmHalt(f\"precheck temperature_basis {tb!r} is not a decided basis \"\n"
     "                              \"(pre-temperature/post-temperature); refusing (A1/A4.3)\")\n"
     "            pb = findings.get(\"penalty_basis\")\n"
     "            if pb not in (\"penalties-not-applied-to-logprobs\", \"post-sampling-chain\"):\n"
     "                raise ArmHalt(f\"precheck penalty_basis {pb!r} is not a decided basis \"\n"
     "                              \"(penalties-not-applied-to-logprobs/post-sampling-chain); refusing (A1/A4.3)\")",
     "            pass  # mutant basis-allowlist-accepts-any: accept any basis string"),

    # drop num_ctx from the ollama options -> the arm sends no num_ctx, ollama uses its default
    # context and the option is absent from the record (ArmNumCtxOption)
    ("numctx-option-dropped", "ArmNumCtxOption", LOC,
     'mapped["num_ctx"] = NUM_CTX',
     'mapped.pop("num_ctx", None)  # mutant numctx-option-dropped: back to the ollama default context'),

    # §7 prices only the weights again (ignore the KV cache): the required-free math loses KV, so a
    # run that would spill KV passes anyway (ArmVramKv)
    ("vram-kv-ignored", "ArmVramKv", LOC,
     'required = size_mib + kv["kv_mib"] + VRAM_HEADROOM_MIB',
     'required = size_mib + VRAM_HEADROOM_MIB  # mutant vram-kv-ignored: price only the weights'),

    # remove the oversize num_ctx preflight refusal: an item over the window is no longer refused
    # (ArmNumCtxPreflight) — the raise becomes unreachable, so nothing halts
    ("oversize-preflight-removed", "ArmNumCtxPreflight", LOC,
     "if worst is not None and worst[0] > ceiling:",
     "if False:  # mutant oversize-preflight-removed: never refuse an oversized item"),

    # remove the runtime truncation belt: a truncated prompt no longer halts (ArmNumCtxBelt)
    ("runtime-belt-removed", "ArmNumCtxBelt", LOC,
     "if isinstance(itok, int) and not isinstance(itok, bool) and itok >= belt:",
     "if False:  # mutant runtime-belt-removed: no possible-truncation guard"),

    # restore the OLD num_ctx-1 belt threshold (the F1 fail-open defect): ollama truncates to
    # ~num_ctx/2 (2,050 at 4096), so a count near num_ctx-1 never appears and the belt can never fire
    # -- the measured truncation value (2,050) no longer halts (ArmNumCtxBelt)
    ("belt-threshold-old-num_ctx-minus-1", "ArmNumCtxBelt", LOC,
     "belt = num_ctx // 2 - 64",
     "belt = num_ctx - 1  # mutant belt-threshold-old-num_ctx-minus-1: unreachable on real truncation"),

    # remove the SIGTERM/SIGINT handler install (the shared mechanism the runner + precheck rely on):
    # a SIGTERM then uses the default disposition and kills the process WITHOUT the finally, so the
    # model is left resident (no keep_alive:0 unload) (ArmSignalUnload)
    ("signal-handler-removed", "ArmSignalUnload", BASE,
     "        signal.signal(signal.SIGTERM, _terminate)\n"
     "        signal.signal(signal.SIGINT, _terminate)",
     "        pass  # mutant signal-handler-removed: install no terminate handler"),
]

if __name__ == "__main__":
    sys.exit(run_suite("tests.test_local", ARMS, MUTATIONS, "JEVCAL_LOCAL_MUTATION_REPORT"))
