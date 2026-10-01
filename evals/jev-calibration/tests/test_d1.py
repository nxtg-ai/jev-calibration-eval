"""Arms for Jev eval step D1 (three-arm run): pair-derived GATE E labels, reads["h2"] (prereg §5 H2
per slice, test split, draw 1), the certifying arm-set refusals, and jevcal/compare_choices.py.

Real harness code; only external seams are replaced: the TypeSafe transport (jev_stub), the claude
subprocess (frontier_stub_runner), ollama (an in-process HTTP stub on 127.0.0.1, as ArmClose does),
LangFuse (patched to a proven outage) and `git` (so a mutation-dirtied tree does not refuse the run).
No GPU, no hosted API, no network beyond localhost. tests/mutate_d1.py reverts each guard in the
SHIPPED file and requires the arm named for it to go RED.
"""
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from jevcal import compare_choices as CC  # noqa: E402
from jevcal import run as R  # noqa: E402
from jevcal import scorer as S  # noqa: E402
from jevcal.arms.frontier import FrontierArm  # noqa: E402
from jevcal.arms.jev import JevArm  # noqa: E402
from test_codex_fixes import (FIXTURE, KeyFile, certifying_pin, fake_git,  # noqa: E402
                              frontier_stub_runner, jev_stub)
from test_local import _OkOllamaHandler, env  # noqa: E402

PROOF = {"checked_at": "x", "endpoint": "e", "probe_result": "error:URLError", "attempts": 3}


@contextmanager
def ollama_stub():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _OkOllamaHandler)
    srv.reqs = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", srv
    finally:
        srv.shutdown()


def fixture_run(arms, evals_dir, run_id="d1"):
    """--mode fixture over the 6 fixture items with every arm stubbed at its transport seam.
    Returns (rc, run record, raw-log path, calls per arm)."""
    jev_calls, fr_calls = [], []

    def fr_runner(argv, env_, cwd, timeout):
        fr_calls.append(argv)
        return frontier_stub_runner(argv, env_, cwd, timeout)
    with ollama_stub() as (base, srv), \
            env(JEVCAL_OLLAMA_BASE_URL=base, JEVCAL_VRAM_PROBE="printf 20000"), KeyFile(), \
            mock.patch.object(R, "git", side_effect=fake_git), \
            mock.patch.dict(R.ARMS, {"JEV": lambda **kw: JevArm(transport=jev_stub(jev_calls)),
                                     "FRONTIER": lambda **kw: FrontierArm(effort="high", runner=fr_runner)}), \
            mock.patch.object(R, "langfuse_setup", return_value={"mode": "unreachable", "proof": PROOF}):
        rc = R.main(["--items", FIXTURE, "--arms", arms, "--mode", "fixture", "--frontier-effort", "high",
                     "--evals-dir", evals_dir, "--run-id", run_id, "--purpose", "D1 fixture proof"])
        local_chats = [r for r in srv.reqs if r[0] == "POST" and r[1].endswith("/api/chat")
                       and (r[2] or {}).get("keep_alive") != 0]
    with open(os.path.join(evals_dir, "runs", f"{run_id}.json")) as fh:
        art = json.load(fh)
    return rc, art, os.path.join(evals_dir, "runs", f"{run_id}.raw.jsonl"), \
        {"JEV": len(jev_calls), "FRONTIER": len(fr_calls), "LOCAL-P": len(local_chats)}


_CACHE = {}


def three_arm():
    """The item-6 proof run, shared by the arms below (one run per interpreter)."""
    if "three" not in _CACHE:
        d = tempfile.mkdtemp()
        _CACHE["three"] = fixture_run("LOCAL-P,JEV,FRONTIER", d, "d1-three-arm") + (d,)
    return _CACHE["three"]


class ArmThreeArmFixtureRun(unittest.TestCase):
    """Item 6: a fixture-mode LOCAL-P,JEV,FRONTIER run has non-null win_gap, parity and h2."""

    def test_reads_all_non_null(self):
        rc, art, _raw, calls, _d = three_arm()
        self.assertEqual(rc, 0)
        self.assertEqual((art["mode"], art["certifying"]), ("fixture", False))
        for read in ("win_gap", "parity", "h2"):
            self.assertIsNotNone(art["reads"].get(read), read)
        self.assertEqual(calls, {"JEV": 6, "FRONTIER": 6, "LOCAL-P": 6})


class ArmPairLabel(unittest.TestCase):
    """Item 1: the GATE E labels are derived from the actual pair and never state a falsehood."""

    def test_three_arm_pair_names_the_local_arm(self):
        _rc, art, *_ = three_arm()
        for read in ("win_gap", "parity"):
            r = art["reads"][read]
            self.assertEqual(r["pair"]["pair"], ["LOCAL-P", "JEV"])
            self.assertEqual(r["pair"]["local_arms_in_pair"], ["LOCAL-P"])
            self.assertIn("LOCAL arm in this pair: LOCAL-P", r["metric"])
            self.assertIn("LOCAL arm(s) in this run: LOCAL-P", r["metric"])
            self.assertNotIn("absent", r["metric"])

    def test_label_without_a_local_arm(self):
        pl = R.pair_label(["JEV", "FRONTIER"], ["JEV", "FRONTIER"])
        self.assertEqual(pl["text"], "no LOCAL arm in this pair; no LOCAL arm in this run")
        pl = R.pair_label(["JEV", "FRONTIER"], ["JEV", "FRONTIER", "LOCAL-P"])
        self.assertEqual(pl["text"], "no LOCAL arm in this pair; LOCAL arm(s) in this run: LOCAL-P")


class ArmLocalMaxTokens(unittest.TestCase):
    """GATE H: the LOCAL arm caps output with ollama num_predict, so its run record must carry that
    cap as an integer max_tokens, in BOTH places it appears: the top-level max_tokens map and
    pins.per_model_gen_params (the one the internal certifier's GATE H reads). The expected value is
    read off the LOCAL-P wire requests in the raw log, never a constant."""

    def test_local_max_tokens_is_the_num_predict_sent(self):
        rc, art, raw, *_ = three_arm()
        self.assertEqual(rc, 0)
        sent = set()
        with open(raw) as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("arm") == "LOCAL-P":
                    sent.add(rec["wire_request"]["options"]["num_predict"])
        self.assertEqual(len(sent), 1, f"LOCAL-P wire num_predict not single-valued: {sent}")
        num_predict = sent.pop()
        self.assertIs(type(num_predict), int)
        model = art["arm_configs"]["LOCAL-P"]["model"]
        for where, value in (("max_tokens", art["max_tokens"].get(model)),
                             ("pins.per_model_gen_params",
                              art["pins"]["per_model_gen_params"][model].get("max_tokens"))):
            self.assertIs(type(value), int, f"{where}[{model}] is {value!r}, not an int")
            self.assertEqual(value, num_predict, where)


def dr(draw, correct, ids, p=0.9):
    """A 2-option DrawRun, gold 0: a correct item puts p on option 0, a wrong one puts p on option 1."""
    return S.DrawRun(draw, [[p, 1 - p] if c else [1 - p, p] for c in correct], [0] * len(ids), ids,
                     [["a", "b"]] * len(ids), [False] * len(ids))


class ArmH2BestByAccuracy(unittest.TestCase):
    """Item 2: the comparator is the non-local arm with the highest accuracy on the slice, never a
    fixed arm. Two slices, one won by each non-local arm, so a mutant fixed to either arm dies."""
    ids = [f"i{j:02d}" for j in range(10)]

    def test_jev_best_slice(self):
        v = S.h2_for_draw(dr(1, [True] * 9 + [False], self.ids),
                          {"JEV": dr(1, [True] * 10, self.ids), "FRONTIER": dr(1, [True] * 5 + [False] * 5, self.ids)},
                          seed=7)
        self.assertEqual(v["best_non_local"], "JEV")
        self.assertEqual(v["tied_at_best"], ["JEV"])
        self.assertAlmostEqual(v["accuracy_gap_local_minus_best"], -0.1)
        self.assertEqual((v["accuracy_within_3_points"], v["h2"]), (False, "does not"))
        self.assertEqual(v["accuracy"]["LOCAL-P"]["point"], 0.9)
        self.assertEqual(len(v["accuracy"]["FRONTIER"]["ci95"]), 2)

    def test_frontier_best_slice(self):
        v = S.h2_for_draw(dr(1, [True] * 10, self.ids),
                          {"JEV": dr(1, [True] * 5 + [False] * 5, self.ids), "FRONTIER": dr(1, [True] * 10, self.ids)},
                          seed=7)
        self.assertEqual(v["best_non_local"], "FRONTIER")
        self.assertEqual(v["accuracy_gap_local_minus_best"], 0.0)
        self.assertTrue(v["brier_upper_ci_le_jev"])
        self.assertEqual(v["h2"], "matches on this class")
        self.assertLessEqual(v["brier_upper_ci"]["LOCAL-P"], v["brier_upper_ci"]["JEV"])

    def test_tie_is_verdict_neutral_and_recorded(self):
        local = dr(1, [True] * 8 + [False] * 2, self.ids)
        jev, fr = dr(1, [True] * 10, self.ids), dr(1, [True] * 10, self.ids, p=0.6)
        both = S.h2_for_draw(local, {"JEV": jev, "FRONTIER": fr}, seed=7)
        jev_only = S.h2_for_draw(local, {"JEV": jev}, seed=7)
        self.assertEqual(both["tied_at_best"], ["FRONTIER", "JEV"])
        self.assertEqual(both["best_non_local"], "FRONTIER")
        self.assertIn("no tie-break can make the bar easier or harder", both["tie_rule"])
        for k in ("h2", "accuracy_within_3_points", "brier_upper_ci_le_jev", "accuracy_gap_local_minus_best"):
            self.assertEqual(both[k], jev_only[k], k)


class ArmH2Runner(unittest.TestCase):
    """Item 2 in the run record: per slice, TEST split only, draw 1, both conditions + verdict words."""

    def test_h2_block_per_slice_on_the_test_split(self):
        _rc, art, *_ = three_arm()
        h2 = art["reads"]["h2"]
        self.assertTrue(h2["complete"])
        self.assertEqual((h2["population"], h2["draw"], h2["local_arm"]), ("test", 1, "LOCAL-P"))
        self.assertEqual(h2["non_local_arms_considered"], ["JEV", "FRONTIER"])
        bs = h2["by_slice"]
        self.assertEqual(sorted(bs), ["FX-ownership", "FX-routing"])
        # FX-002 is the only calibration item: FX-ownership has 4 items in all, 3 on test
        self.assertEqual((bs["FX-ownership"]["n_items"], bs["FX-routing"]["n_items"]), (3, 2))
        self.assertEqual(bs["FX-ownership"]["item_ids"], ["FX-001", "FX-004", "FX-005"])
        for sl, v in bs.items():
            self.assertIn(v["verdict"], ("matches on this class", "does not"))
            self.assertEqual(sorted(v["accuracy"]), ["FRONTIER", "JEV", "LOCAL-P"])
            self.assertEqual(sorted(v["brier_upper_ci"]), ["JEV", "LOCAL-P"])
            best = max(("FRONTIER", "JEV"), key=lambda n: (v["accuracy"][n]["point"], n == "FRONTIER"))
            self.assertEqual(v["best_non_local"], best)
            want = (v["accuracy_within_3_points"] and v["brier_upper_ci_le_jev"])
            self.assertEqual(v["verdict"] == "matches on this class", want)


class ArmH2Withheld(unittest.TestCase):
    """Item 2: a fixture run without FRONTIER cannot answer §5 (best of BOTH non-local arms): the block
    is computed, but every verdict is null and the reason is recorded."""

    def test_two_arm_fixture_withholds_the_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            rc, art, *_ = fixture_run("LOCAL-P,JEV", d, "d1-two-arm")
        self.assertEqual(rc, 0)
        h2 = art["reads"]["h2"]
        self.assertFalse(h2["complete"])
        self.assertIn("FRONTIER", h2["verdict_withheld_reason"])
        for v in h2["by_slice"].values():
            self.assertIsNone(v["verdict"])
            self.assertNotIn("h2", v)


@contextmanager
def spies():
    """Factories + VRAM probe that record any use; the refusals must fire before either is touched."""
    built, jev_calls = [], []

    def spy(name):
        def f(**kw):
            built.append(name)
            if name == "JEV":
                return JevArm(transport=jev_stub(jev_calls))
            raise SystemExit(f"spy: {name} was built")
        return f
    with mock.patch.dict(R.ARMS, {n: spy(n) for n in ("JEV", "FRONTIER", "LOCAL-P")}), \
            mock.patch.object(R, "vram_preflight", side_effect=lambda m: built.append("vram") or {}), \
            mock.patch.object(R, "git", side_effect=fake_git), KeyFile(), \
            mock.patch.object(R, "langfuse_setup", return_value={"mode": "unreachable", "proof": PROOF}):
        yield built, jev_calls


def certifying(arms, d):
    with certifying_pin(FIXTURE):
        return R.main(["--items", FIXTURE, "--arms", arms, "--mode", "certifying", "--draws", "3",
                       "--frontier-effort", "high", "--evals-dir", d, "--run-id", "c", "--purpose", "p"])


class ArmRefuseSingleArm(unittest.TestCase):
    """Item 3(a): certifying mode refuses fewer than 2 arms before any arm is built or called."""

    def check(self, arms):
        with tempfile.TemporaryDirectory() as d, spies() as (built, calls):
            with self.assertRaisesRegex(SystemExit, r"GATE E needs two arms") as cm:
                certifying(arms, d)
            self.assertEqual(os.listdir(d), [])
        self.assertEqual((built, calls), ([], []))
        self.assertNotEqual(cm.exception.code, 0)

    def test_single_jev_refused(self):
        self.check("JEV")

    def test_single_local_p_refused(self):
        self.check("LOCAL-P")


class ArmRefuseLocalWithoutBoth(unittest.TestCase):
    """Item 3(b): a LOCAL arm without BOTH JEV and FRONTIER is refused before any arm is built."""

    def test_local_p_jev_refused(self):
        with tempfile.TemporaryDirectory() as d, spies() as (built, calls):
            with self.assertRaisesRegex(SystemExit, r"without both JEV and FRONTIER.*missing: \['FRONTIER'\]"):
                certifying("LOCAL-P,JEV", d)
            self.assertEqual(os.listdir(d), [])
        self.assertEqual((built, calls), ([], []))

    def test_local_p_frontier_refused(self):
        with tempfile.TemporaryDirectory() as d, spies() as (built, calls):
            with self.assertRaisesRegex(SystemExit, r"missing: \['JEV'\]"):
                certifying("LOCAL-P,FRONTIER", d)
        self.assertEqual(built, [])

    def test_controls_pass_the_arm_set_check(self):
        for ok in (["LOCAL-P", "JEV", "FRONTIER"], ["JEV", "FRONTIER"]):
            self.assertIsNone(R.certifying_arm_set(ok))


def write_raw(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def row(iid, answer, draw=1, arm="LOCAL-P", abstained=False):
    return {"item_id": iid, "arm": arm, "draw": draw, "answer": answer, "abstained": abstained}


class ArmCompareChoices(unittest.TestCase):
    """Item 4: per-item argmax-label agreement over shared ids, draw 1 only, row counts beside counts."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.a = os.path.join(self.d, "a.raw.jsonl")
        self.b = os.path.join(self.d, "b.raw.jsonl")
        write_raw(self.a, [row("i1", "x"), row("i2", "y"), row("i3", "z"), row("i1", "JEVANS", arm="JEV"),
                           row("i1", "q", draw=2), row("i1", "y", draw=3),      # re-draws differ
                           {"halt": True, "arm": "LOCAL-P", "item_id": "i9", "draw": 1, "reason": "r"}])
        write_raw(self.b, [row("i1", "x"), row("i2", "w"), row("i4", None, abstained=True)])

    def test_counts_and_mismatches(self):
        res = CC.compare(self.a, self.b, "LOCAL-P")
        fa, fb = res["files"]["a"], res["files"]["b"]
        self.assertEqual((fa["rows"], fa["halt_rows"], fa["other_arm_rows"], fa["arm_rows_all_draws"],
                          fa["arm_rows_selected_draw"]), (7, 1, 1, 5, 3))
        self.assertEqual((fb["rows"], fb["arm_rows_selected_draw"]), (3, 3))
        self.assertEqual((res["shared_items"], res["matches"], res["match_rate"]), (2, 1, 0.5))
        self.assertEqual(res["mismatches"], [{"item_id": "i2", "a": "y", "b": "w"}])
        self.assertEqual((res["only_in_a"], res["only_in_b"]), (["i3"], ["i4"]))
        text = CC.render(res)
        self.assertIn("argmax-label matches: 1 of 2 shared (A rows=7, B rows=3)", text)
        self.assertIn("halt rows=1 of 7", text)

    def test_duplicate_primary_refused(self):
        write_raw(self.b, [row("i1", "x"), row("i1", "y")])
        with self.assertRaisesRegex(SystemExit, "second LOCAL-P row"):
            CC.compare(self.a, self.b, "LOCAL-P")


if __name__ == "__main__":
    unittest.main()
