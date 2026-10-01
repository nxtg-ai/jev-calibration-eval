"""LOCAL arm (E.4b) named arms. Transport is stubbed at the seam only; every line of the adapter
runs for real. The close()-on-error test drives a REAL in-process ollama stub over HTTP (no mocks).

Class names are the mutation-suite targets (tests/mutate_local.py): each must go RED for the right
reason (an assertion on the CONDITION, never on a crash) when its guard is reverted.
"""
import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from jevcal import items as I  # noqa: E402
from jevcal import local_precheck as PC  # noqa: E402
from jevcal import run as R  # noqa: E402
from jevcal.arms import local as L  # noqa: E402
from jevcal.arms.base import ArmHalt  # noqa: E402
from jevcal.render import render  # noqa: E402

FIXTURE = os.path.join(ROOT, "fixtures", "items-fixture.jsonl")
PROBE = os.path.join(ROOT, "fixtures", "local-precheck-probe.jsonl")
CACHE_CONFOUND_FX = os.path.join(ROOT, "tests", "fixtures", "precheck-cache-confound.json")
# The REAL first live receipt, kept in the repo, read-only, as the known-wrong regression subject.
REPO_ROOT = os.path.dirname(os.path.dirname(ROOT))        # evals/jev-calibration -> evals -> repo
CONFOUNDED = os.path.join(REPO_ROOT, "governance", "evals", "jev-calibration", "local-precheck",
                          "qwen3-14b-20260930T040303Z-CONFOUNDED.json")
FROZEN_ITEMS = os.path.join(REPO_ROOT, "governance", "evals", "jev-calibration", "items-v1.jsonl")
PINNED = "bdbd181c33f2"                                   # qwen3-14b A4.2 digest pin
TAG = "qwen3:14b"
NINE_GIB = 9 * 1024 * 1024 * 1024                         # -> ceil 9216 MiB + 2048 headroom = 11264 required


def qwen_tag(digest=PINNED, size=NINE_GIB):
    return {"name": TAG, "digest": "sha256:" + digest + "aabbccddeeff", "size": size}


# qwen3-14b architecture (public config): 40 layers, 40 heads, 8 KV heads, head_dim 128, hidden 5120.
# KV MiB at num_ctx=4096, f16: 2 * 40 * 8 * 128 * 4096 * 2 / 2^20 = 640 MiB (hand-computed literal).
QWEN_MODEL_INFO = {
    "general.architecture": "qwen3",
    "qwen3.block_count": 40,
    "qwen3.attention.head_count": 40,
    "qwen3.attention.head_count_kv": 8,
    "qwen3.attention.key_length": 128,
    "qwen3.attention.value_length": 128,
    "qwen3.embedding_length": 5120,
}
QWEN_KV_MIB_4096 = 640                                    # hand-computed §7 KV oracle for the stub


def qwen_show(model_info=None):
    return {"model_info": QWEN_MODEL_INFO if model_info is None else model_info}


def gptoss_tag():
    return {"name": "gpt-oss:20b", "digest": "sha256:17052f91a42e" + "00" * 10, "size": 13 * 1024 ** 3}


def r3():
    """A 3-option Rendered (labels A,B,C -> keys billing/technical/sales)."""
    it = {it_.item_id: it_ for it_ in I.load_items(FIXTURE)}["FX-003"]
    return render(it)


def top(*pairs):
    """Build a top_logprobs list from (token, prob) pairs; prob -> logprob."""
    return [{"token": t, "logprob": (math.log(p) if p > 0 else -1e9)} for t, p in pairs]


def chat_resp(top_logprobs, model=TAG, total=1_000_000_000, load=200_000_000, ev=50_000_000,
              pe=42, ec=1, cached=None):
    first = top_logprobs[0] if top_logprobs else {"token": "", "logprob": 0.0}
    # cached defaults to a WARM prompt (fully cached): the precheck warm-up call means every
    # compared call hits a warm KV cache. A test that wants a cold call passes cached=0.
    return {"model": model, "message": {"role": "assistant", "content": first["token"]},
            "logprobs": [{"token": first["token"], "logprob": first["logprob"],
                          "top_logprobs": top_logprobs}],
            "done": True, "done_reason": "stop", "total_duration": total, "load_duration": load,
            "eval_duration": ev, "prompt_eval_count": pe, "eval_count": ec,
            "prompt_eval_cached_count": pe if cached is None else cached}


DEFAULT_TOP = top(("A", 0.7), ("B", 0.2), ("C", 0.1))


def warm(top_list, pe=42):
    """A WARM precheck call dict for compute_findings (prompt prefix fully KV-cached)."""
    return {"top_logprobs": top_list, "prompt_eval_count": pe, "prompt_eval_cached_count": pe}


def cold(top_list, pe=42):
    """A COLD precheck call dict (nothing cached) — the confound the warm-only comparison excludes."""
    return {"top_logprobs": top_list, "prompt_eval_count": pe, "prompt_eval_cached_count": 0}


def five_warm_calls(top_list=None):
    """A full precheck calls object (warmup + a1/a2/b/c). warmup is COLD (so cache_state_effect is
    defined); a1/a2/b/c are WARM and byte-identical, so compute_findings decides pre-temperature /
    penalties-not-applied-to-logprobs / repeatable / feasible — all inside the certifying allowlist."""
    t = top_list if top_list is not None else top(("A", 0.7), ("B", 0.2), ("C", 0.1))
    return {"warmup": cold(list(t)), "a1": warm(list(t)), "a2": warm(list(t)),
            "b": warm(list(t)), "c": warm(list(t))}


def findings_for(calls, labels):
    return PC.compute_findings(calls["warmup"], calls["a1"], calls["a2"], calls["b"], calls["c"], labels)


class OllamaStub:
    """A transport callable (method,url,body,headers,timeout)->(status,headers,bytes). Mutable so a
    test can flip the served digest mid-run. Records every request in .seen."""
    def __init__(self, tags=None, version="0.34.4", chat=None, chat_status=200, tags_status=200,
                 model_info=None, show_status=200):
        self.tags = tags if tags is not None else [qwen_tag()]
        self.version = version
        self.chat = chat                                  # dict, callable(body)->dict, or None (default)
        self.chat_status = chat_status
        self.tags_status = tags_status
        self.model_info = QWEN_MODEL_INFO if model_info is None else model_info
        self.show_status = show_status
        self.seen = []

    def __call__(self, method, url, body, headers, timeout):
        parsed = json.loads(body) if body else None
        self.seen.append({"method": method, "url": url, "body": parsed})
        if url.endswith("/api/tags"):
            return self.tags_status, {}, json.dumps({"models": self.tags}).encode()
        if url.endswith("/api/version"):
            return 200, {}, json.dumps({"version": self.version}).encode()
        if url.endswith("/api/show"):
            return self.show_status, {}, json.dumps({"model_info": self.model_info}).encode()
        if url.endswith("/api/chat"):
            if parsed and parsed.get("keep_alive") == 0:
                return 200, {}, json.dumps({"model": parsed["model"], "done_reason": "unload"}).encode()
            resp = self.chat(parsed) if callable(self.chat) else (
                self.chat if self.chat is not None else chat_resp(DEFAULT_TOP))
            return self.chat_status, {}, json.dumps(resp).encode()
        return 404, {}, b"{}"


def arm(stub=None, **kw):
    stub = stub or OllamaStub()
    return L.LocalArm(arm_name="LOCAL-P", model_id="qwen3-14b", transport=stub,
                      base_url="http://stub", **kw), stub


@contextlib.contextmanager
def env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ----------------------------------------------------------------------------------------------
class ArmDigestPin(unittest.TestCase):
    def test_construction_digest_mismatch_halts(self):
        stub = OllamaStub(tags=[qwen_tag(digest="ffffffffffff")])
        with self.assertRaises(ArmHalt):
            arm(stub)

    def test_mid_run_digest_change_halts(self):
        stub = OllamaStub()
        a, _ = arm(stub, digest_recheck_every=1)          # re-check on every call
        a.call(r3())                                      # call 1: digest still the pin -> ok
        stub.tags = [qwen_tag(digest="ffffffffffff")]     # a version change lands mid-run
        with self.assertRaises(ArmHalt):
            a.call(r3())


class ArmObservedDigest(unittest.TestCase):
    def test_record_carries_observed_digest_not_the_pin(self):
        # qwen_tag() digest is "sha256:" + pin + "aabbccddeeff"; the record must carry the OBSERVED
        # normalised digest (pin + suffix), not the pin constant, with the pin kept in extra (note 4).
        a, _ = arm()
        rec = a.call(r3()).to_record(r3())
        self.assertEqual(rec["model_digest"], PINNED + "aabbccddeeff")   # observed, not the pin
        self.assertNotEqual(rec["model_digest"], PINNED)
        self.assertEqual(rec["extra"]["pinned_digest"], PINNED)

    def test_mid_run_change_still_halts_before_recording_it(self):
        # a changed digest is caught at call-start recheck, so it halts BEFORE a record is written
        # (the changed value never reaches a record); confirm the halt behaviour holds (note 4).
        stub = OllamaStub()
        a, _ = arm(stub, digest_recheck_every=1)
        a.call(r3())
        stub.tags = [qwen_tag(digest="ffffffffffff")]
        with self.assertRaises(ArmHalt):
            a.call(r3())


class ArmServedModel(unittest.TestCase):
    def test_served_model_not_serving_tag_halts(self):
        a, _ = arm(OllamaStub(chat=chat_resp(DEFAULT_TOP, model="qwen3:32b")))
        with self.assertRaises(ArmHalt):
            a.call(r3())


class ArmNon200(unittest.TestCase):
    def test_non_200_chat_halts_no_retry(self):
        # a VALID body under a 500 status: only the status check can catch it (proves the check
        # is load-bearing; if it were skipped, the valid body would parse into a normal result).
        stub = OllamaStub(chat_status=500, chat=chat_resp(DEFAULT_TOP))
        a, _ = arm(stub)
        with self.assertRaises(ArmHalt):
            a.call(r3())
        # no retry: exactly one /api/chat POST reached the server
        chats = [s for s in stub.seen if s["url"].endswith("/api/chat") and s["method"] == "POST"]
        self.assertEqual(len(chats), 1)


class ArmNxtgRefused(unittest.TestCase):
    def test_served_nxtg_model_refused_with_reason(self):
        a, _ = arm(OllamaStub(chat=chat_resp(DEFAULT_TOP, model="nxtg-coder-qwen3")))
        # the refusal must be the nxtg- prompt-parity refusal, not merely the generic tag mismatch
        # (the served-name mismatch message also contains "nxtg-...", so match the parity reason)
        with self.assertRaisesRegex(ArmHalt, "prompt parity"):
            a.call(r3())


class ArmBaseUrl(unittest.TestCase):
    def test_base_url_unset_refuses_before_any_call(self):
        stub = OllamaStub()
        with env(JEVCAL_OLLAMA_BASE_URL=None):
            with self.assertRaises(ArmHalt):
                L.LocalArm(arm_name="LOCAL-P", model_id="qwen3-14b", transport=stub)
        # the guard fires before the transport is touched
        self.assertEqual(stub.seen, [])


class ArmUnmappedKey(unittest.TestCase):
    def test_unmapped_registry_key_refuses(self):
        reg = {"models": {"qwen3-14b": {"family": "qwen", "recommended": {
            "non_thinking": {"temperature": 0.7, "banana": 1.0}}}}}
        stub = OllamaStub()
        with self.assertRaisesRegex(ArmHalt, "unmapped"):
            L.LocalArm(arm_name="LOCAL-P", model_id="qwen3-14b", transport=stub,
                       base_url="http://stub", registry=reg)
        self.assertEqual(stub.seen, [])   # options-build refuses before any /api/tags read


class ArmNoNumericConfig(unittest.TestCase):
    def test_no_numeric_config_refuses(self):
        # A registry model whose recommended block carries only a note (no numeric config) -> refuse.
        # (gpt-oss:20b gained a numeric block in #80, so a crafted registry exercises this path now.)
        reg = {"models": {"qwen3-14b": {"family": "qwen", "recommended": {"note": "no numbers yet"}}}}
        stub = OllamaStub()
        with self.assertRaisesRegex(ArmHalt, "numeric"):
            L.LocalArm(arm_name="LOCAL-P", model_id="qwen3-14b", transport=stub,
                       base_url="http://stub", registry=reg)
        self.assertEqual(stub.seen, [])   # options-build refuses before any /api/tags read

    def test_gpt_oss_now_builds_numeric_config(self):
        # Post-#80 gpt-oss:20b has a numeric single-block config (temperature 1.0 / top_p 1.0); the
        # arm now constructs and pins those, rather than refusing.
        a = L.LocalArm(arm_name="LOCAL-C-GPTOSS", model_id="gpt-oss:20b",
                       transport=OllamaStub(tags=[gptoss_tag()]), base_url="http://stub")
        self.assertEqual(a.options["temperature"], 1.0)
        self.assertEqual(a.options["top_p"], 1.0)
        self.assertEqual(a.options["seed"], L.LOCAL_SEED)

    def test_coder_repetition_penalty_maps_to_repeat_penalty(self):
        with open(L.default_registry_path()) as fh:
            reg = json.load(fh)
        opts = L.build_sampling_options("Qwen3-Coder-30B-A3B", reg)
        self.assertEqual(opts.get("repeat_penalty"), 1.05)
        self.assertNotIn("repetition_penalty", opts)


class ArmOptionsLogged(unittest.TestCase):
    def test_options_logged_byte_identical_to_options_sent(self):
        a, stub = arm()
        rec = a.call(r3()).to_record(r3())
        post = next(s for s in stub.seen if s["url"].endswith("/api/chat") and s["method"] == "POST")
        # the options on the wire, in the record, and in extra are the SAME object content
        self.assertEqual(post["body"]["options"], a.options)
        self.assertEqual(rec["wire_request"]["options"], a.options)
        self.assertEqual(rec["extra"]["options_sent"], a.options)
        # and the registry temperature is present as a numeric ollama option + rail-keyed temp
        self.assertEqual(a.options["temperature"], 0.7)
        self.assertEqual(a.sampling_config()["temp"], 0.7)
        self.assertEqual(a.options["seed"], L.LOCAL_SEED)
        self.assertEqual(a.options["num_predict"], 1)


class ArmLabelParse(unittest.TestCase):
    def test_whitespace_variants_summed(self):
        a, _ = arm(OllamaStub(chat=chat_resp(top((" A", 0.3), ("A", 0.3), ("B", 0.4)))))
        res = a.call(r3())
        self.assertAlmostEqual(res.probabilities[0], 0.6, places=6)   # A = 0.3 + 0.3
        self.assertAlmostEqual(res.probabilities[1], 0.4, places=6)
        self.assertAlmostEqual(res.probabilities[2], 0.0, places=6)

    def test_lowercase_label_ignored(self):
        a, _ = arm(OllamaStub(chat=chat_resp(top(("a", 0.9), ("B", 0.1)))))
        res = a.call(r3())
        self.assertAlmostEqual(res.probabilities[1], 1.0, places=6)   # only B counts; a is ignored
        self.assertAlmostEqual(res.probabilities[0], 0.0, places=6)
        self.assertIn("A", res.extra["labels_missing"])

    def test_non_label_token_ignored(self):
        a, _ = arm(OllamaStub(chat=chat_resp(top(("the", 0.8), ("C", 0.2)))))
        res = a.call(r3())
        self.assertAlmostEqual(res.probabilities[2], 1.0, places=6)   # C only

    def test_no_valid_label_is_abstention(self):
        a, _ = arm(OllamaStub(chat=chat_resp(top(("the", 0.5), ("dog", 0.5)))))
        res = a.call(r3())
        self.assertTrue(res.abstained)
        self.assertIsNone(res.probabilities)
        self.assertIsNotNone(res.error)

    def test_no_logprobs_is_abstention(self):
        a, _ = arm(OllamaStub(chat={"model": TAG, "message": {"content": "A"}, "done": True}))
        res = a.call(r3())
        self.assertTrue(res.abstained)
        self.assertIsNone(res.probabilities)


class ArmLatencyNs(unittest.TestCase):
    def test_server_latency_ns_to_s(self):
        a, _ = arm(OllamaStub(chat=chat_resp(DEFAULT_TOP, total=2_000_000_000, load=500_000_000,
                                             ev=250_000_000)))
        res = a.call(r3())
        self.assertAlmostEqual(res.server_latency["total_s"], 2.0, places=9)
        self.assertAlmostEqual(res.server_latency["load_s"], 0.5, places=9)
        self.assertAlmostEqual(res.server_latency["eval_s"], 0.25, places=9)


class ArmVram(unittest.TestCase):
    def test_vram_pass(self):
        rep = L.vram_preflight("qwen3-14b", transport=OllamaStub(), base_url="http://stub",
                               probe_cmd="nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
                               run_probe=lambda argv: "20000\n")
        self.assertTrue(rep["ok"])
        # §7 now prices weights + KV(num_ctx) + headroom (was weights + headroom only)
        self.assertEqual(rep["required_free_mib"], 9216 + QWEN_KV_MIB_4096 + 2048)
        self.assertEqual(rep["kv_mib"], QWEN_KV_MIB_4096)
        self.assertEqual(rep["num_ctx"], L.NUM_CTX)
        self.assertEqual(rep["free_mib"], 20000)

    def test_vram_fail(self):
        with self.assertRaises(ArmHalt):
            L.vram_preflight("qwen3-14b", transport=OllamaStub(), base_url="http://stub",
                             probe_cmd="nvidia-smi", run_probe=lambda argv: "5000")

    def test_vram_probe_unset_refuses(self):
        with env(JEVCAL_VRAM_PROBE=None):
            with self.assertRaises(ArmHalt):
                L.vram_preflight("qwen3-14b", transport=OllamaStub(), base_url="http://stub")


class ArmPrecheckBasis(unittest.TestCase):
    LBL = ["A", "B", "C"]

    def test_repeatable_and_pre_temperature_and_penalty_not_applied(self):
        t = top(("A", 0.7), ("B", 0.2), ("C", 0.1))
        # all warm and identical -> pre-temperature, penalties-not-applied, repeatable
        f = PC.compute_findings(warm(t), warm(list(t)), warm(list(t)), warm(list(t)), warm(list(t)), self.LBL)
        self.assertTrue(f["repeatable"])
        self.assertEqual(f["temperature_basis"], "pre-temperature")
        self.assertEqual(f["penalty_basis"], "penalties-not-applied-to-logprobs")
        self.assertTrue(f["label_first_token_feasible"])

    def test_post_temperature_and_post_sampling_chain_and_not_repeatable(self):
        wu = top(("A", 0.7), ("B", 0.2), ("C", 0.1))     # warm-up == a1's tops here (isolates the mutant)
        a1 = top(("A", 0.7), ("B", 0.2), ("C", 0.1))
        a2 = top(("A", 0.6), ("B", 0.3), ("C", 0.1))     # a1 != a2 -> not repeatable
        b = top(("A", 0.5), ("B", 0.3), ("C", 0.2))      # a1 != b  -> post-temperature
        c = top(("A", 0.4), ("B", 0.4), ("C", 0.2))      # a1 != c  -> post-sampling-chain
        f = PC.compute_findings(warm(wu), warm(a1), warm(a2), warm(b), warm(c), self.LBL)
        self.assertFalse(f["repeatable"])
        self.assertEqual(f["temperature_basis"], "post-temperature")
        self.assertEqual(f["penalty_basis"], "post-sampling-chain")

    def test_undetermined_when_a_compared_call_is_cold(self):
        # a cold b (cache differs) makes temperature_basis undetermined, never a label
        t = top(("A", 0.7), ("B", 0.2), ("C", 0.1))
        f = PC.compute_findings(warm(t), warm(list(t)), warm(list(t)), cold(list(t)), warm(list(t)), self.LBL)
        self.assertTrue(f["temperature_basis"].startswith("undetermined"))
        self.assertEqual(f["penalty_basis"], "penalties-not-applied-to-logprobs")

    def test_label_first_token_not_feasible_when_mass_below_half(self):
        t = top(("A", 0.2), ("the", 0.6), ("dog", 0.2))  # total label mass 0.2 < 0.5
        f = PC.compute_findings(warm(t), warm(list(t)), warm(list(t)), warm(list(t)), warm(list(t)), self.LBL)
        self.assertFalse(f["label_first_token_feasible"])

    def test_run_precheck_receipt_has_findings_and_options(self):
        a, _ = arm()                                     # default stub -> deterministic, feasible
        receipt = PC.run_precheck(a, r3())
        self.assertEqual(receipt["model_id"], "qwen3-14b")
        self.assertEqual(receipt["model_digest"], PINNED)
        self.assertEqual(receipt["options"], a.options)
        self.assertIn("temperature_basis", receipt["findings"])
        self.assertTrue(receipt["findings"]["repeatable"])
        self.assertIn("cache_state_effect", receipt["findings"])
        for k in ("warmup", "a1", "a2", "b", "c"):     # warm-up call recorded alongside the four
            self.assertIn(k, receipt["calls"])
        # every call records its cache counts (review note: prompt_eval_cached_count for every call)
        for k in ("warmup", "a1", "a2", "b", "c"):
            self.assertIn("prompt_eval_cached_count", receipt["calls"][k])

    def test_probe_b_temperature_differs_from_registry(self):
        # note 2: call (b) must probe at a temperature != the registry temperature, else the basis is
        # vacuous. qwen registry temp is 0.7 -> probe at 1.0.
        a, _ = arm()
        receipt = PC.run_precheck(a, r3())
        self.assertEqual(receipt["registry_temperature"], 0.7)
        self.assertEqual(receipt["b_probe_temperature"], 1.0)
        self.assertEqual(receipt["calls"]["b"]["options"]["temperature"], 1.0)

    def test_probe_b_temperature_when_registry_is_one(self):
        # note 2: gpt-oss:20b's registry temperature is 1.0, so (b) must probe at 0.5, not 1.0
        # (a fixed 1.0 probe would read "pre-temperature" by construction).
        stub = OllamaStub(tags=[gptoss_tag()],
                          chat=lambda b: chat_resp(DEFAULT_TOP, model=b["model"]))
        a = L.LocalArm(arm_name="LOCAL-C-GPTOSS", model_id="gpt-oss:20b",
                       transport=stub, base_url="http://stub")
        receipt = PC.run_precheck(a, r3())
        self.assertEqual(receipt["registry_temperature"], 1.0)
        self.assertEqual(receipt["b_probe_temperature"], 0.5)
        self.assertEqual(receipt["calls"]["b"]["options"]["temperature"], 0.5)


class ArmPrecheckCacheConfound(unittest.TestCase):
    """Real-data regression from the first live precheck (qwen3:14b 2026-09-30T04:03Z): a1 was the
    only COLD call (cached 0/139, 15 s load), a2/b/c WARM (138/139) with byte-identical label
    logprobs. The OLD logic used cold a1 as the baseline for every comparison and mislabelled the
    cache-state delta as post-temperature / post-sampling-chain. Fixture:
    tests/fixtures/precheck-cache-confound.json (trimmed copy of that receipt)."""

    def _fx(self):
        with open(os.path.join(ROOT, "tests", "fixtures", "precheck-cache-confound.json")) as fh:
            return json.load(fh)

    def test_old_baseline_mislabels_cold_vs_warm(self):
        # DOCUMENTS THE DEFECT: cold a1 vs warm b differ by cache state -> old logic reads it as a
        # temperature effect. This is exactly what the shipped receipt's findings carried.
        fx = self._fx(); labels = fx["labels"]; c = fx["calls"]
        a1l = PC.label_logprobs(c["a1"]["top_logprobs"], labels)   # a1 COLD
        bl = PC.label_logprobs(c["b"]["top_logprobs"], labels)     # b  WARM
        old = "pre-temperature" if PC._eq(a1l, bl) else "post-temperature"
        self.assertEqual(old, "post-temperature")

    def test_new_logic_warm_only_is_correct(self):
        # map old a1 -> warm-up (cold), old a2 -> a1, duplicate old b as the second warm call.
        fx = self._fx(); labels = fx["labels"]; c = fx["calls"]
        f = PC.compute_findings(c["a1"], c["a2"], c["b"], c["b"], c["c"], labels)
        self.assertEqual(f["temperature_basis"], "pre-temperature")
        self.assertEqual(f["penalty_basis"], "penalties-not-applied-to-logprobs")
        self.assertEqual(f["repeatable"], True)
        # the cold/warm difference is REAL and now surfaced as its own finding, not a false basis
        self.assertGreater(f["cache_state_effect"]["max_delta_logprob"], 0.0)


class ArmCertifyingPrecheck(unittest.TestCase):
    def _receipt(self, a, tmp, **override):
        # A full valid receipt by default: raw calls + labels + findings RE-DERIVED from those calls
        # (the binding recomputes and requires a match). Overrides replace individual fields.
        calls = override.pop("calls", five_warm_calls())
        labels = override.pop("labels", ["A", "B", "C"])
        findings = override.pop("findings", None)
        if findings is None:
            findings = findings_for(calls, labels)
        r = {"model_id": a.model_id, "model_digest": a.pinned_digest, "options": a.options,
             "ollama_version": a.ollama_version, "calls": calls, "labels": labels,
             "findings": findings}
        r.update(override)
        p = os.path.join(tmp, "receipt.json")
        with open(p, "w") as fh:
            json.dump(r, fh)
        return p

    def test_certifying_refuses_without_receipt(self):
        stub = OllamaStub()
        with self.assertRaisesRegex(ArmHalt, "precheck"):
            L.LocalArm(arm_name="LOCAL-P", model_id="qwen3-14b", transport=stub,
                       base_url="http://stub", require_precheck=True)

    def test_certifying_accepts_matching_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            a, _ = arm()
            p = self._receipt(a, d)
            a2, _ = arm(precheck_path=p, require_precheck=True)
            self.assertEqual(a2.precheck["findings"]["label_first_token_feasible"], True)

    def test_certifying_refuses_infeasible_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            a, _ = arm()
            p = self._receipt(a, d, findings={"label_first_token_feasible": False})
            with self.assertRaisesRegex(ArmHalt, "feasible"):
                arm(precheck_path=p, require_precheck=True)

    def test_certifying_refuses_option_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            a, _ = arm()
            p = self._receipt(a, d, options={"temperature": 0.1})
            with self.assertRaisesRegex(ArmHalt, "options"):
                arm(precheck_path=p, require_precheck=True)

    def test_certifying_refuses_ollama_version_mismatch(self):
        # every field matches except the ollama_version -> refuse (note 1): the findings are a
        # property of the ollama build, so a receipt from a different /api/version cannot apply.
        with tempfile.TemporaryDirectory() as d:
            a, _ = arm()                                   # arm's /api/version is the stub's 0.34.4
            p = self._receipt(a, d, ollama_version="0.33.0")
            with self.assertRaisesRegex(ArmHalt, "ollama_version"):
                arm(precheck_path=p, require_precheck=True)

    def test_certifying_missing_receipt_file_refuses_cleanly(self):
        # note 3: a specified-but-missing receipt is a clean ArmHalt, not an uncaught FileNotFoundError
        with self.assertRaisesRegex(ArmHalt, "could not be read"):
            arm(precheck_path="/nonexistent/receipt.json", require_precheck=True)

    def test_certifying_bad_json_receipt_refuses_cleanly(self):
        # note 3: an unparseable receipt is handled the same way as a missing one
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "receipt.json")
            with open(p, "w") as fh:
                fh.write("{not valid json")
            with self.assertRaisesRegex(ArmHalt, "could not be read"):
                arm(precheck_path=p, require_precheck=True)


# ----------------------------------------------------------------------------------------------
# The certifying binding RE-DERIVES the findings from the raw calls instead of trusting the stored
# `findings` block (2026-09-30). These four arms are the regression: (a) the real confounded receipt
# is refused, (b) a valid 5-call receipt is accepted, (c) an edited finding is refused, (d) an
# undetermined basis is refused. They are the named targets of tests/mutate_local.py.
class ArmCertBindConfoundedRefused(unittest.TestCase):
    """The certifying binding refuses a receipt that predates the warm-up precheck (no `warmup` call).
    Two refusals now apply. Since num_ctx was pinned into the options, the REAL first-live receipt
    (…-CONFOUNDED.json, read READ-ONLY) predates it, so its options no longer match and it is refused
    at the OPTIONS gate (intended: a fresh precheck must run). A synthetic receipt whose options DO
    match (num_ctx present) but that carries no warmup is refused at the warm-up/calls gate — the gate
    the `bind-trusts-stored-findings` mutant removes, so this arm reds under it."""

    def test_real_confounded_receipt_refused_on_num_ctx_options(self):
        a, _ = arm()                                       # arm pins num_ctx; the receipt predates it
        with open(CONFOUNDED) as fh:                       # read-only; governance file is never edited
            rc = json.load(fh)
        self.assertNotIn("num_ctx", rc["options"])         # pre-num_ctx receipt
        self.assertEqual(a.options["num_ctx"], L.NUM_CTX)  # the arm pins it -> options no longer match
        self.assertNotIn("warmup", rc["calls"])            # also the old no-warmup shape
        with self.assertRaisesRegex(ArmHalt, "options"):
            a._load_precheck(CONFOUNDED, True)

    def test_no_warmup_receipt_refused_at_warmup_gate(self):
        # A receipt whose options DO match (num_ctx present) but with NO warmup call is refused at the
        # warm-up/calls gate. Its stored findings are allowlisted, so if that gate were removed the
        # receipt would sail through (which is exactly what bind-trusts-stored-findings makes happen).
        with tempfile.TemporaryDirectory() as d:
            a, _ = arm()
            full = five_warm_calls()
            labels = ["A", "B", "C"]
            findings = findings_for(full, labels)          # allowlisted (pre-temperature, ...)
            calls = {k: v for k, v in full.items() if k != "warmup"}   # drop the warm-up call
            r = {"model_id": a.model_id, "model_digest": a.pinned_digest, "options": a.options,
                 "ollama_version": a.ollama_version, "calls": calls, "labels": labels,
                 "findings": findings}
            p = os.path.join(d, "no_warmup.json")
            with open(p, "w") as fh:
                json.dump(r, fh)
            with self.assertRaisesRegex(ArmHalt, "warm-up precheck"):
                a._load_precheck(p, True)


class ArmCertBindValidAccepted(unittest.TestCase):
    """(b) A valid 5-call receipt SYNTHESIZED FROM THE REAL fixture calls is ACCEPTED. Per the task:
    warmup = real a1 (cold), a1 = real a2, a2 = real a2, b = real b, c = real c. Its findings come
    from compute_findings; the binding re-derives them and they match, so the receipt passes."""

    def _synth_from_real(self, a, tmp):
        with open(CACHE_CONFOUND_FX) as fh:
            fx = json.load(fh)
        c, labels = fx["calls"], fx["labels"]
        calls = {"warmup": c["a1"], "a1": c["a2"], "a2": c["a2"], "b": c["b"], "c": c["c"]}
        findings = findings_for(calls, labels)
        r = {"model_id": a.model_id, "model_digest": a.pinned_digest, "options": a.options,
             "ollama_version": a.ollama_version, "calls": calls, "labels": labels, "findings": findings}
        p = os.path.join(tmp, "valid.json")
        with open(p, "w") as fh:
            json.dump(r, fh)
        return p, findings

    def test_valid_synth_receipt_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            a, _ = arm()
            p, findings = self._synth_from_real(a, d)
            info = a._load_precheck(p, True)               # ACCEPTED -> returns info, no ArmHalt
            self.assertEqual(info["findings"], findings)


class ArmCertBindEditedFindingRefused(unittest.TestCase):
    """(c) The same valid receipt with ONE stored finding edited (pre-temperature ->
    post-temperature — still an allowlisted value) is REFUSED because the binding re-derives the
    findings and the stored block no longer matches."""

    def test_edited_finding_refused(self):
        with tempfile.TemporaryDirectory() as d:
            a, _ = arm()
            with open(CACHE_CONFOUND_FX) as fh:
                fx = json.load(fh)
            c, labels = fx["calls"], fx["labels"]
            calls = {"warmup": c["a1"], "a1": c["a2"], "a2": c["a2"], "b": c["b"], "c": c["c"]}
            findings = findings_for(calls, labels)
            self.assertEqual(findings["temperature_basis"], "pre-temperature")  # what re-derive yields
            findings["temperature_basis"] = "post-temperature"                  # the single edit
            r = {"model_id": a.model_id, "model_digest": a.pinned_digest, "options": a.options,
                 "ollama_version": a.ollama_version, "calls": calls, "labels": labels,
                 "findings": findings}
            p = os.path.join(d, "edited.json")
            with open(p, "w") as fh:
                json.dump(r, fh)
            with self.assertRaisesRegex(ArmHalt, "differing key 'temperature_basis'"):
                a._load_precheck(p, True)


class ArmCertBindUndeterminedRefused(unittest.TestCase):
    """(d) A receipt whose basis is 'undetermined (cache state differs)' and whose findings are
    OTHERWISE CONSISTENT (they DO re-derive from the calls) is REFUSED by the basis allowlist — an
    undetermined basis is a truthy string, not a decided basis."""

    def test_undetermined_basis_refused(self):
        with tempfile.TemporaryDirectory() as d:
            a, _ = arm()
            t = top(("A", 0.7), ("B", 0.2), ("C", 0.1))
            # b is COLD -> compute_findings marks temperature_basis undetermined (warm-vs-cold)
            calls = {"warmup": cold(list(t)), "a1": warm(list(t)), "a2": warm(list(t)),
                     "b": cold(list(t)), "c": warm(list(t))}
            labels = ["A", "B", "C"]
            findings = findings_for(calls, labels)
            self.assertTrue(str(findings["temperature_basis"]).startswith("undetermined"))
            r = {"model_id": a.model_id, "model_digest": a.pinned_digest, "options": a.options,
                 "ollama_version": a.ollama_version, "calls": calls, "labels": labels,
                 "findings": findings}
            p = os.path.join(d, "undetermined.json")
            with open(p, "w") as fh:
                json.dump(r, fh)
            with self.assertRaisesRegex(ArmHalt, "temperature_basis.*not a decided basis"):
                a._load_precheck(p, True)


# ----------------------------------------------------------------------------------------------
class _StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):      # silence
        pass

    def _send(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/api/tags"):
            self.server.reqs.append(("GET", self.path, None))
            self._send({"models": [qwen_tag()]})
        elif self.path.endswith("/api/version"):
            self._send({"version": "0.34.4"})
        else:
            self._send({}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n)) if n else None
        self.server.reqs.append(("POST", self.path, body))
        if self.path.endswith("/api/show"):
            self._send({"model_info": QWEN_MODEL_INFO})   # §7 KV pricing reads /api/show
        elif self.path.endswith("/api/chat"):
            self._send({"error": "stub 500"}, self.server.chat_status)
        else:
            self._send({}, 404)


class ArmClose(unittest.TestCase):
    def test_close_unloads_via_keep_alive_zero(self):
        a, stub = arm()
        a.close()
        unloads = [s for s in stub.seen if s["url"].endswith("/api/chat") and s["method"] == "POST"
                   and (s["body"] or {}).get("keep_alive") == 0]
        self.assertEqual(len(unloads), 1)

    def test_close_called_on_runner_error_path(self):
        # A REAL ollama stub: /api/tags + /api/version 200, /api/chat -> 500 so the first call HALTs.
        # main() then returns 3 and its finally block must still fire close() (the unload POST).
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        srv.reqs = []
        srv.chat_status = 500
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            with env(JEVCAL_OLLAMA_BASE_URL=base, JEVCAL_VRAM_PROBE="printf 20000",
                     LANGFUSE_PUBLIC_KEY="pk-test", LANGFUSE_SECRET_KEY="sk-test",
                     LANGFUSE_BASE_URL="http://127.0.0.1:9", JEV_KEY_FILE=None):
                with tempfile.TemporaryDirectory() as d:
                    rc = R.main(["--items", FIXTURE, "--arms", "LOCAL-P", "--mode", "fixture",
                                 "--evals-dir", d, "--run-id", "cls-err", "--purpose", "close-on-error"])
            self.assertEqual(rc, 3)                        # ArmHalt on the chat 500 -> exit 3
            unloads = [r for r in srv.reqs if r[0] == "POST" and r[1].endswith("/api/chat")
                       and (r[2] or {}).get("keep_alive") == 0]
            self.assertTrue(unloads, "close() unload POST must reach the server on the error path")
        finally:
            srv.shutdown()


class _OkOllamaHandler(BaseHTTPRequestHandler):
    """A REAL in-process ollama stub that serves VALID responses (tags with size, version, and chat
    with first-token top_logprobs). Records every request in server.reqs."""
    def log_message(self, *a):
        pass

    def _send(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.server.reqs.append(("GET", self.path, None))
        if self.path.endswith("/api/tags"):
            self._send({"models": [qwen_tag()]})
        elif self.path.endswith("/api/version"):
            self._send({"version": "0.34.4"})
        else:
            self._send({}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n)) if n else {}
        self.server.reqs.append(("POST", self.path, body))
        if self.path.endswith("/api/show"):
            self._send({"model_info": QWEN_MODEL_INFO})   # §7 KV pricing reads /api/show
        elif self.path.endswith("/api/chat"):
            if body.get("keep_alive") == 0:
                self._send({"model": body["model"], "done_reason": "unload"})
            else:
                self._send(chat_resp(DEFAULT_TOP, model=body["model"]))
        else:
            self._send({}, 404)


class ArmPrecheckVramGate(unittest.TestCase):
    """Prereg §7: `python3 -m jevcal.local_precheck` must run the same VRAM preflight as the runner
    BEFORE loading the model. Driven end-to-end against a REAL in-process ollama stub (no mocks)."""

    def _server(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _OkOllamaHandler)
        srv.reqs = []
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)                  # LIFO: shutdown then close the socket
        self.addCleanup(srv.shutdown)
        return srv, f"http://127.0.0.1:{srv.server_address[1]}"

    @staticmethod
    def _chats(srv):
        return [r for r in srv.reqs if r[0] == "POST" and r[1].endswith("/api/chat")]

    def test_probe_unset_refuses_with_no_model_load(self):
        srv, base = self._server()
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "r.json")
            with env(JEVCAL_OLLAMA_BASE_URL=base, JEVCAL_VRAM_PROBE=None):
                rc = PC.main(["--model-id", "qwen3-14b", "--out", out])
            self.assertNotEqual(rc, 0)                     # clean refusal
            self.assertFalse(os.path.exists(out))          # no receipt written
        self.assertEqual(self._chats(srv), [])             # NO /api/chat -> no model load

    def test_short_reading_refuses_with_no_model_load(self):
        srv, base = self._server()
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "r.json")
            with env(JEVCAL_OLLAMA_BASE_URL=base, JEVCAL_VRAM_PROBE="printf 100"):
                rc = PC.main(["--model-id", "qwen3-14b", "--out", out])
            self.assertNotEqual(rc, 0)                     # 100 MiB < required -> refuse
            self.assertFalse(os.path.exists(out))
        self.assertEqual(self._chats(srv), [])             # no model load

    def test_sufficient_reading_proceeds_and_records_numbers(self):
        srv, base = self._server()
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "r.json")
            with env(JEVCAL_OLLAMA_BASE_URL=base, JEVCAL_VRAM_PROBE="printf 20000"):
                rc = PC.main(["--model-id", "qwen3-14b", "--out", out])
            self.assertEqual(rc, 0)
            with open(out) as fh:
                receipt = json.load(fh)
        vp = receipt["vram_preflight"]                     # §7 numbers recorded
        self.assertEqual(vp["free_mib"], 20000)
        self.assertEqual(vp["required_free_mib"], 9216 + QWEN_KV_MIB_4096 + 2048)   # weights + KV + headroom
        self.assertEqual(vp["kv_mib"], QWEN_KV_MIB_4096)
        self.assertEqual(vp["num_ctx"], L.NUM_CTX)
        self.assertEqual(vp["model_size_bytes"], 9 * 1024 * 1024 * 1024)
        self.assertEqual(vp["probe_argv"], ["printf", "20000"])
        self.assertTrue(vp["ok"])
        self.assertTrue(self._chats(srv))                  # the model WAS loaded (calls happened)


# ----------------------------------------------------------------------------------------------
# num_ctx pin + §7 KV pricing + signal-safe unload (2026-09-30). These five arms are the named
# targets of the four new mutations in tests/mutate_local.py (drop num_ctx from the options, §7
# ignores KV, remove the signal handler, remove the oversize preflight) plus the runtime belt.
class ArmNumCtxOption(unittest.TestCase):
    def test_num_ctx_in_options_on_wire_and_in_config(self):
        a, stub = arm()
        self.assertIn("num_ctx", a.options)
        self.assertEqual(a.options.get("num_ctx"), L.NUM_CTX)
        rec = a.call(r3()).to_record(r3())
        post = next(s for s in stub.seen if s["url"].endswith("/api/chat") and s["method"] == "POST")
        self.assertEqual(post["body"]["options"].get("num_ctx"), L.NUM_CTX)   # exactly as sent
        self.assertEqual(rec["extra"]["options_sent"].get("num_ctx"), L.NUM_CTX)
        self.assertEqual(a.config()["options"].get("num_ctx"), L.NUM_CTX)     # in the hashed config

    def test_build_sampling_options_carries_num_ctx(self):
        with open(L.default_registry_path()) as fh:
            reg = json.load(fh)
        opts = L.build_sampling_options("qwen3-14b", reg)
        self.assertEqual(opts.get("num_ctx"), L.NUM_CTX)


class ArmNumCtxPreflight(unittest.TestCase):
    def test_frozen_set_passes_at_num_ctx(self):
        # read the FROZEN items READ-ONLY in certifying mode; the preflight must PASS at num_ctx.
        items = I.load_items(FROZEN_ITEMS, "certifying")
        a, _ = arm()
        rep = a.numctx_preflight([render(it) for it in items])
        self.assertEqual(rep["num_ctx"], L.NUM_CTX)
        self.assertEqual(rep["n_items"], len(items))
        self.assertLessEqual(rep["worst_est_tokens"], rep["ceiling_tokens"])
        self.assertEqual(rep["worst_item_id"], "S1-choice-045")   # the measured worst item

    def test_oversize_item_refuses_and_names_it(self):
        a, _ = arm()
        # a synthetic item whose full LOCAL prompt is far over num_ctx - margin (never truncated)
        big = types.SimpleNamespace(item_id="BIG-001", canonical_text="x" * (L.NUM_CTX * 3))
        with self.assertRaisesRegex(ArmHalt, "BIG-001"):
            a.numctx_preflight([big])

    def test_refusal_forbids_silent_truncation(self):
        a, _ = arm()
        big = types.SimpleNamespace(item_id="BIG-002", canonical_text="y" * (L.NUM_CTX * 3))
        with self.assertRaisesRegex(ArmHalt, "NEVER truncate"):
            a.numctx_preflight([big])


class ArmNumCtxBelt(unittest.TestCase):
    # independent review 2026-09-30 (ollama 0.34.4): an over-num_ctx prompt is silently truncated to
    # ~num_ctx/2 + 2 (at num_ctx 4096 -> prompt_eval_count 2,050), HTTP 200 -- so the belt fires at
    # num_ctx//2 - 64 (1,984), never near num_ctx. The worst real frozen item (S1-choice-045) is
    # 1,213 tokens, well below the belt, so it does not halt.
    def test_belt_halts_on_truncated_prompt(self):
        # the MEASURED ollama truncation count at num_ctx 4096 => possible silent truncation => HALT
        a, _ = arm(OllamaStub(chat=chat_resp(DEFAULT_TOP, pe=2050)))
        with self.assertRaisesRegex(ArmHalt, "truncation"):
            a.call(r3())

    def test_belt_allows_worst_real_frozen_prompt(self):
        # 1,213 = the worst real frozen item (S1-choice-045); below the 1,984 belt => no halt
        a, _ = arm(OllamaStub(chat=chat_resp(DEFAULT_TOP, pe=1213)))
        res = a.call(r3())
        self.assertFalse(res.abstained)

    def test_belt_threshold_is_half_window_minus_64(self):
        belt = L.NUM_CTX // 2 - 64                          # 1,984 at num_ctx 4096
        a, _ = arm(OllamaStub(chat=chat_resp(DEFAULT_TOP, pe=belt)))
        with self.assertRaisesRegex(ArmHalt, "truncation"):
            a.call(r3())                                    # exactly at the belt => HALT
        a2, _ = arm(OllamaStub(chat=chat_resp(DEFAULT_TOP, pe=belt - 1)))
        res = a2.call(r3())                                 # one below => no halt
        self.assertFalse(res.abstained)


class ArmVramKv(unittest.TestCase):
    def test_kv_math_matches_hand_value(self):
        kv = L.kv_cache_mib(QWEN_MODEL_INFO, 4096)
        # 2 * 40 * 8 * 128 * 4096 * 2 / 2^20 = 640 MiB, computed by hand
        self.assertEqual(kv["kv_mib"], QWEN_KV_MIB_4096)
        self.assertEqual(kv["block_count"], 40)
        self.assertEqual(kv["head_count_kv"], 8)
        self.assertEqual(kv["head_dim"], 128)
        self.assertEqual(kv["bytes_per_elem"], 2)
        # KV scales linearly with num_ctx
        self.assertEqual(L.kv_cache_mib(QWEN_MODEL_INFO, 8192)["kv_mib"], 2 * QWEN_KV_MIB_4096)

    def test_kv_head_dim_falls_back_to_embedding_over_head_count(self):
        mi = {k: v for k, v in QWEN_MODEL_INFO.items() if not k.endswith("key_length")}
        kv = L.kv_cache_mib(mi, 4096)
        self.assertEqual(kv["head_dim"], 5120 // 40)       # 128 via embedding_length / head_count
        self.assertEqual(kv["head_dim_source"], "embedding_length/attention.head_count")
        self.assertEqual(kv["kv_mib"], QWEN_KV_MIB_4096)

    def test_kv_refuses_missing_block_count(self):
        mi = {k: v for k, v in QWEN_MODEL_INFO.items() if not k.endswith("block_count")}
        with self.assertRaisesRegex(ArmHalt, "block_count"):
            L.kv_cache_mib(mi, 4096)

    def test_kv_refuses_missing_head_dim_sources(self):
        mi = {"qwen3.block_count": 40, "qwen3.attention.head_count_kv": 8}   # no key_length, no emb/heads
        with self.assertRaisesRegex(ArmHalt, "key_length"):
            L.kv_cache_mib(mi, 4096)

    def test_kv_refuses_empty_model_info(self):
        with self.assertRaisesRegex(ArmHalt, "model_info"):
            L.kv_cache_mib({}, 4096)

    def test_vram_preflight_prices_kv(self):
        rep = L.vram_preflight("qwen3-14b", transport=OllamaStub(), base_url="http://stub",
                               probe_cmd="x", run_probe=lambda argv: "20000")
        self.assertEqual(rep["kv_mib"], QWEN_KV_MIB_4096)
        self.assertEqual(rep["num_ctx"], L.NUM_CTX)
        self.assertEqual(rep["required_free_mib"], rep["model_size_mib"] + QWEN_KV_MIB_4096 + 2048)

    def test_vram_preflight_refuses_on_missing_metadata(self):
        stub = OllamaStub(model_info={})                   # /api/show returns an empty model_info
        with self.assertRaisesRegex(ArmHalt, "KV cache"):
            L.vram_preflight("qwen3-14b", transport=stub, base_url="http://stub",
                             probe_cmd="x", run_probe=lambda argv: "20000")


# A child that exercises the SHIPPED install_terminate_handlers() + a REAL LocalArm's close(), the
# same primitive the runner and precheck CLI rely on, through a stub transport (the file's seam). The
# transport BLOCKS in the first /api/chat, so a SIGTERM lands mid-call. If the signal unwinds through
# the try/finally (because the handler is installed), the arm's close() fires a keep_alive:0 call and
# the transport writes CLOSE. It does NOT drive run.main: the mutation engine leaves the tree dirty,
# and run.main's committed-tree guard would REFUSE before the call — a false RED on every mutation.
# The `signal-handler-removed` mutation is on install_terminate_handlers itself, the real mechanism.
_SIGTERM_CHILD = r'''
import json, os, sys, time
ROOT, READY, CLOSE = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, ROOT)
from jevcal.arms import local as L
from jevcal.arms.base import install_terminate_handlers
from jevcal.render import render
from jevcal.items import load_items
TAG = "qwen3:14b"
def transport(method, url, body, headers, timeout):
    parsed = json.loads(body) if body else None
    if url.endswith("/api/tags"):
        return 200, {}, json.dumps({"models": [{"name": TAG,
                "digest": "sha256:bdbd181c33f2aabbccddeeff", "size": 9 * 1024 ** 3}]}).encode()
    if url.endswith("/api/version"):
        return 200, {}, json.dumps({"version": "0.34.4"}).encode()
    if url.endswith("/api/chat"):
        if parsed and parsed.get("keep_alive") == 0:                # the unload close() fires
            open(CLOSE, "w").write("closed")
            return 200, {}, json.dumps({"model": TAG, "done_reason": "unload"}).encode()
        open(READY, "w").write("ready")                             # signal we are about to block
        time.sleep(30)                                             # block so SIGTERM lands mid-call
        tl = [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.0}]
        return 200, {}, json.dumps({"model": TAG, "message": {"role": "assistant", "content": "A"},
                "logprobs": [{"token": "A", "logprob": -0.1, "top_logprobs": tl}], "done": True,
                "prompt_eval_count": 42, "eval_count": 1}).encode()
    return 404, {}, b"{}"
os.environ["JEVCAL_OLLAMA_BASE_URL"] = "http://stub"
install_terminate_handlers()                                        # the mechanism under test
a = L.LocalArm(arm_name="LOCAL-P", model_id="qwen3-14b", transport=transport, base_url="http://stub")
r = render(load_items(os.path.join(ROOT, "fixtures", "items-fixture.jsonl"))[0])
try:
    a.call(r)                                                       # blocks in the stub /api/chat
finally:
    a.close()                                                       # keep_alive:0 -> writes CLOSE
'''


class ArmSignalUnload(unittest.TestCase):
    """A subprocess installs the SHIPPED SIGTERM/SIGINT handler and blocks in a real LocalArm call; a
    SIGTERM must unwind through the try/finally so the arm's close() (keep_alive:0) still fires. No
    live model — the transport is a stub. Removing the handler (mutation) leaves the model resident."""

    def test_sigterm_still_unloads_via_close(self):
        with tempfile.TemporaryDirectory() as d:
            ready, close = os.path.join(d, "ready"), os.path.join(d, "close")
            script = os.path.join(d, "sigterm_child.py")
            with open(script, "w") as fh:
                fh.write(_SIGTERM_CHILD)
            p = subprocess.Popen([sys.executable, script, ROOT, ready, close])
            try:
                for _ in range(600):                       # up to 60s to reach the blocking call
                    if os.path.exists(ready):
                        break
                    time.sleep(0.1)
                self.assertTrue(os.path.exists(ready), "child never reached the blocking call")
                p.send_signal(signal.SIGTERM)
                p.wait(timeout=30)
            finally:
                if p.poll() is None:
                    p.kill()
                    p.wait(timeout=10)
            self.assertTrue(os.path.exists(close),
                            "SIGTERM must unwind through the finally and call close() (keep_alive:0); "
                            "no unload marker was written -- the model would stay resident")


if __name__ == "__main__":
    unittest.main()
