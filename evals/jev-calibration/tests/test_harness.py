"""Renderer, items schema, arm adapters and runner guards.

External services (TypeSafe HTTP, the claude CLI) are stubbed at the transport
seam only; every line of OUR adapter code runs for real. The live calls are made
by the fixture run (fixture-evals-dir/), not here.
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from jevcal import items as I  # noqa: E402
from jevcal import run as R  # noqa: E402
from jevcal.arms.base import ArmHalt  # noqa: E402
from jevcal.arms.frontier import FrontierArm  # noqa: E402
from jevcal.arms.jev import JevArm, build_question  # noqa: E402
from jevcal.render import render  # noqa: E402

FIXTURE = os.path.join(ROOT, "fixtures", "items-fixture.jsonl")


def items():
    return {it.item_id: it for it in I.load_items(FIXTURE)}


class KeyFile:
    def __enter__(self):
        self.d = tempfile.mkdtemp()
        self.path = os.path.join(self.d, "k.env")
        with open(self.path, "w") as fh:
            fh.write("TYPESAFE_API_KEY=test-key-not-real\n")
        self.old = os.environ.get("JEV_KEY_FILE")
        os.environ["JEV_KEY_FILE"] = self.path
        return self

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop("JEV_KEY_FILE", None)
        else:
            os.environ["JEV_KEY_FILE"] = self.old


def jev_transport(status=200, body=None, headers=None, seen=None):
    def t(url, data, hdrs, timeout):
        if seen is not None:
            seen.append({"url": url, "body": json.loads(data), "headers": dict(hdrs)})
        return status, headers or {"content-type": "application/json"}, json.dumps(body).encode()
    return t


class TestItemsSchema(unittest.TestCase):
    def test_fixture_loads(self):
        self.assertEqual(len(items()), 6)

    def test_refuses_the_real_frozen_items_file(self):
        # E.2 froze governance/evals/jev-calibration/items-v1.jsonl on origin/main; the guard
        # refuses it by path BEFORE opening it, so no E.3 code path can feed it to an arm.
        real = os.path.join(R.REPO, "governance", "evals", "jev-calibration", "items-v1.jsonl")
        with self.assertRaises(I.EvalDataRefused):
            I.load_items(real)

    def test_refuses_eval_data_by_name_and_dir(self):
        with self.assertRaises(I.EvalDataRefused):
            I.load_items("/anywhere/items-v1.jsonl")
        with self.assertRaises(I.EvalDataRefused):
            I.load_items("/x/governance/evals/jev-calibration/other.jsonl")

    def test_rejects_bad_rows(self):
        with open(FIXTURE) as fh:
            good = json.loads(fh.readline())
        for mutate, why in [
            (lambda r: r.update(gold="nobody"), "gold not an option"),
            (lambda r: r.update(options=["a"]), "one option"),
            (lambda r: r.update(options=[f"o{i}" for i in range(21)], gold="o0"), "21 options"),
            (lambda r: r.update(question_type="freeform"), "bad type"),
            (lambda r: r.update(split="train"), "bad split"),
            (lambda r: r.update(state="w " * 301), "state over 300 words"),
            (lambda r: r.update(extra=1), "unknown field"),
            (lambda r: r.pop("source_ref"), "missing source_ref"),
            (lambda r: r.update(criteria={"nobody": "x"}), "criteria key not an option"),
        ]:
            row = dict(good)
            mutate(row)
            with self.assertRaises(I.ItemError, msg=why):
                I.parse_item(row, 1)


class TestRenderParity(unittest.TestCase):
    def test_every_arm_sees_the_same_canonical_text(self):
        it = items()["FX-003"]
        r = render(it)
        q = build_question(r)
        self.assertEqual(q["instructions"], it.instructions)
        self.assertEqual(list(q["criteria"]), list(it.options))           # same order
        self.assertEqual(list(q["criteria"].values()), [d for _, _, d in r.options])
        fp = FrontierArm(effort="high").prompt(r)
        self.assertTrue(fp.startswith(r.canonical_text))
        for _, key, desc in r.options:
            self.assertIn(key, r.canonical_text)
            self.assertIn(desc, r.canonical_text)                          # descriptions shown to all arms

    def test_digest_is_stable_and_content_bound(self):
        a, b = items()["FX-001"], items()["FX-002"]
        self.assertEqual(render(a).msg_sha256, render(a).msg_sha256)
        self.assertNotEqual(render(a).msg_sha256, render(b).msg_sha256)

    def test_noul_and_score_wire_shapes(self):
        q = build_question(render(items()["FX-004"]))
        self.assertEqual(q["type"], "noul")
        self.assertEqual(set(q["criteria"]), {"true", "false"})
        q = build_question(render(items()["FX-006"]))
        self.assertEqual(q["criteria"], ["not urgent — Can wait a week or more",
                                         "somewhat urgent — Should be handled within days",
                                         "very urgent — Needs action today"])


class TestJevAdapter(unittest.TestCase):
    def ok_choice(self):
        return {"model": "jev-1.13.0", "answers": {"q": {"type": "choice", "choice": "billing",
                "probabilities": {"billing": .88, "technical": .12, "sales": 0.0}, "confidence": .81}},
                "usage": {"input_tokens": 318, "output_tokens": 34}}

    def test_choice_parse_and_key_never_logged(self):
        seen = []
        with KeyFile():
            res = JevArm(transport=jev_transport(body=self.ok_choice(), seen=seen)).call(render(items()["FX-003"]))
        self.assertEqual(res.probabilities, [.88, .12, 0.0])
        self.assertEqual(res.returned_model, "jev-1.13.0")
        self.assertEqual(seen[0]["headers"]["Authorization"], "Bearer test-key-not-real")
        blob = json.dumps(res.to_record(render(items()["FX-003"])))
        self.assertNotIn("test-key-not-real", blob)
        self.assertNotIn("Authorization", blob)
        self.assertEqual(seen[0]["body"]["model"], "jev-1.13.0")

    def test_noul_maps_to_first_option(self):
        body = {"model": "jev-1.13.0", "answers": {"q": {"type": "noul", "noul": 0.45}},
                "usage": {"input_tokens": 10, "output_tokens": 2}}
        with KeyFile():
            res = JevArm(transport=jev_transport(body=body)).call(render(items()["FX-004"]))
        self.assertAlmostEqual(res.probabilities[0], 0.45)
        self.assertAlmostEqual(res.probabilities[1], 0.55)

    def test_score_levels_by_index(self):
        body = {"model": "jev-1.13.0", "answers": {"q": {"type": "score", "score": 1.9,
                "legend": {}, "probabilities": {"0": .0, "1": .1, "2": .9}, "confidence": .8}},
                "usage": {"input_tokens": 10, "output_tokens": 2}}
        with KeyFile():
            res = JevArm(transport=jev_transport(body=body)).call(render(items()["FX-006"]))
        self.assertEqual(res.probabilities, [.0, .1, .9])

    def test_halts(self):
        r = render(items()["FX-003"])
        with KeyFile():
            for status in (401, 402, 422, 429, 529):
                with self.assertRaises(ArmHalt, msg=str(status)):
                    JevArm(transport=jev_transport(status=status, body={"error": "x"})).call(r)
            wrong = dict(self.ok_choice(), model="jev-1.14.0")
            with self.assertRaises(ArmHalt):
                JevArm(transport=jev_transport(body=wrong)).call(r)
            with self.assertRaises(ArmHalt):
                JevArm(transport=jev_transport(body=self.ok_choice(),
                                               headers={"X-Billing-Status": "overage"})).call(r)
            with self.assertRaises(ArmHalt):
                JevArm(transport=jev_transport(body=dict(self.ok_choice(), credits_remaining=0))).call(r)
        old = os.environ.pop("JEV_KEY_FILE", None)
        try:
            with self.assertRaises(ArmHalt):
                JevArm(transport=jev_transport(body=self.ok_choice())).call(r)
        finally:
            if old:
                os.environ["JEV_KEY_FILE"] = old

    def test_mismatched_keys_is_an_abstention_not_a_guess(self):
        body = self.ok_choice()
        body["answers"]["q"]["probabilities"] = {"billing": 1.0}
        with KeyFile():
            res = JevArm(transport=jev_transport(body=body)).call(render(items()["FX-003"]))
        self.assertTrue(res.abstained)
        self.assertIsNone(res.probabilities)


def cli_events(model="claude-opus-5-5", probs=None, rate="allowed", overage=False, is_error=False,
               so=None):
    structured = so if so is not None else ({"probabilities": probs} if probs is not None else None)
    return json.dumps([
        {"type": "system", "subtype": "init", "model": model},
        {"type": "assistant", "message": {"model": model, "content": []}},
        {"type": "rate_limit_event", "rate_limit_info": {"status": rate, "isUsingOverage": overage}},
        {"type": "result", "subtype": "success", "is_error": is_error, "result": "",
         "structured_output": structured,
         "modelUsage": {model: {"inputTokens": 5, "outputTokens": 40, "cacheReadInputTokens": 0,
                                "cacheCreationInputTokens": 900, "costUSD": 0.01,
                                "maxOutputTokens": 128000}}},
    ])


def runner(stdout, rc=0, seen=None):
    def r(argv, env, cwd, timeout):
        if seen is not None:
            seen.append({"argv": argv, "env": env, "cwd": cwd})
        return rc, stdout, ""
    return r


class TestFrontierAdapter(unittest.TestCase):
    def test_verbalised_parse_renormalise_and_isolation_flags(self):
        seen = []
        r = render(items()["FX-003"])
        os.environ["JEV_KEY_FILE"] = "/nonexistent/should-be-stripped"
        try:
            res = FrontierArm(effort="high", runner=runner(cli_events(probs={"A": .6, "B": .2, "C": .4}), seen=seen)).call(r)
        finally:
            os.environ.pop("JEV_KEY_FILE", None)
        self.assertEqual([round(x, 6) for x in res.probabilities], [0.5, 0.166667, 0.333333])
        self.assertTrue(res.error.startswith("WARN"))                  # sum 1.2 flagged, renormalised
        argv = seen[0]["argv"]
        for flag in ("--setting-sources", "--tools", "--strict-mcp-config", "--no-session-persistence",
                     "--system-prompt", "--json-schema"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--model") + 1], "claude-opus-5-5")
        self.assertNotIn("JEV_KEY_FILE", seen[0]["env"])
        self.assertNotIn("TYPESAFE_API_KEY", seen[0]["env"])
        self.assertEqual(res.extra["probability_source"], "verbalised")

    def test_halts_on_model_mismatch_error_and_spend(self):
        r = render(items()["FX-001"])
        p = {"A": .9, "B": .05, "C": .05}
        for out, why in [(cli_events(model="claude-sonnet-5", probs=p), "model"),
                         (cli_events(probs=p, rate="rejected"), "rate"),
                         (cli_events(probs=p, overage=True), "overage"),
                         (cli_events(probs=p, is_error=True), "error")]:
            with self.assertRaises(ArmHalt, msg=why):
                FrontierArm(effort="high", runner=runner(out)).call(r)
        with self.assertRaises(ArmHalt):
            FrontierArm(effort="high", runner=runner("", rc=1)).call(r)

    def test_missing_or_invalid_probabilities_abstain(self):
        r = render(items()["FX-001"])
        for p in (None, {"A": 1.0}, {"A": -1, "B": 1, "C": 1}, {"A": 0, "B": 0, "C": 0}):
            res = FrontierArm(effort="high", runner=runner(cli_events(probs=p))).call(r)
            self.assertTrue(res.abstained, msg=repr(p))

    def test_effort_is_required_with_no_default(self):
        for bad in (None, "", "ultracode", "HIGH"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                FrontierArm(effort=bad)

    def test_a4_identity_temperature_and_effort_logged_per_record(self):
        seen = []
        r = render(items()["FX-001"])
        os.environ["CLAUDE_CODE_EFFORT_LEVEL"] = "xhigh"   # the silent-inheritance trap
        try:
            res = FrontierArm(effort="high", runner=runner(
                cli_events(probs={"A": .9, "B": .05, "C": .05}), seen=seen)).call(r)
        finally:
            os.environ.pop("CLAUDE_CODE_EFFORT_LEVEL", None)
        argv = seen[0]["argv"]
        self.assertEqual(argv[argv.index("--effort") + 1], "high")
        self.assertNotIn("CLAUDE_CODE_EFFORT_LEVEL", seen[0]["env"])
        rec = res.to_record(r)
        self.assertEqual((rec["model_id"], rec["serving_tag"], rec["model_digest"]),
                         ("claude-opus-5-5", "n/a", "n/a"))
        self.assertEqual(rec["sampling_config"]["temp"],
                         "provider-default (not settable or observable via claude -p)")
        self.assertEqual(rec["sampling_config"]["effort"], "high")
        self.assertEqual(rec["extra"]["effort"], "high")

    def test_sampling_config_declares_not_exposed_typed_fields(self):
        # Rail change (internal): GATE F v3 / GATE H read typed access_path + sampling_exposed +
        # output_cap_exposed from each arm's sampling_config() (these flow into
        # pins.per_model_gen_params via run.py). JEV (hosted API) and FRONTIER (claude -p) both
        # expose no caller-settable sampling knobs and no verifiable output cap. assertIs(..., False)
        # asserts a real bool (a string "false" would be rejected by the gate as not-a-bool).
        jev = JevArm().sampling_config()
        self.assertEqual(jev["access_path"], "typesafe-systemone-api")
        self.assertIs(jev["sampling_exposed"], False)
        self.assertIs(jev["output_cap_exposed"], False)
        self.assertIn("no sampling parameters", jev["note"])
        fr = FrontierArm(effort="high").sampling_config()
        self.assertEqual(fr["access_path"], "claude-cli-print")
        self.assertIs(fr["sampling_exposed"], False)
        self.assertIs(fr["output_cap_exposed"], False)
        self.assertEqual(fr["max_output_tokens_requested"], 1024)  # descriptive field preserved

    def test_secondary_sample_is_a_single_letter(self):
        seen = []
        r = render(items()["FX-001"])
        s = FrontierArm(effort="high", runner=runner(cli_events(so={"answer": "B"}), seen=seen)).sample_answer(r, 0)
        self.assertEqual((s["answer_label"], s["answer"]), ("B", "wolf"))
        self.assertEqual(s["temperature"], "provider-default (not settable or observable via claude -p)")
        schema = json.loads(seen[0]["argv"][seen[0]["argv"].index("--json-schema") + 1])
        self.assertEqual(schema["properties"]["answer"]["enum"], ["A", "B", "C"])
        bad = FrontierArm(effort="high", runner=runner(cli_events(so={"answer": "Z"}))).sample_answer(r, 1)
        self.assertIsNone(bad["answer"])
        self.assertIsNotNone(bad["error"])


class TestModelIdentityPreflight(unittest.TestCase):
    def test_jev_record_carries_three_identity_fields(self):
        body = {"model": "jev-1.13.0", "answers": {"q": {"type": "choice", "choice": "kestrel",
                "probabilities": {"kestrel": 1.0, "wolf": 0.0, "dx3-pm": 0.0}, "confidence": 1.0}},
                "usage": {"input_tokens": 10, "output_tokens": 2}}
        r = render(items()["FX-001"])
        with KeyFile():
            rec = JevArm(transport=jev_transport(body=body)).call(r).to_record(r)
        self.assertEqual((rec["model_id"], rec["serving_tag"], rec["model_digest"]),
                         ("jev-1.13.0", "n/a", "jev-1.13.0"))

    def test_preflight_accepts_shipped_arms_against_the_real_registry(self):
        R.preflight_registry([JevArm(), FrontierArm(effort="high")])

    def test_preflight_rejects_a_non_registry_model_id_generically(self):
        class Tagged(JevArm):
            model_id = "qwen3:14b"      # a serving tag, not the registry key qwen3-14b
        with self.assertRaises(SystemExit):
            R.preflight_registry([JevArm(), Tagged()])

    def test_runner_refuses_frontier_without_effort_before_any_call(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as cm:
                R.main(["--items", FIXTURE, "--evals-dir", d, "--run-id", "x", "--purpose", "p",
                        "--arms", "FRONTIER", "--mode", "fixture"])
            self.assertIn("effort", str(cm.exception))
            self.assertEqual(os.listdir(d), [])


class TestRunnerGuards(unittest.TestCase):
    def test_refuses_real_eval_store(self):
        with self.assertRaises(SystemExit):
            R.refuse_real_evals_dir(os.path.join(R.REPO, "governance", "evals"))

    def test_refuses_eval_items_before_any_call(self):
        # E.4: --mode fixture still refuses the frozen items by name, before any file read
        # or arm construction. (Certifying-mode allow-on-pinned-hash is covered in
        # test_codex_fixes.py's E.4 arms.)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(I.EvalDataRefused):
                R.main(["--items", os.path.join(d, "items-v1.jsonl"), "--evals-dir", d,
                        "--run-id", "x", "--purpose", "p", "--mode", "fixture"])


if __name__ == "__main__":
    unittest.main()
