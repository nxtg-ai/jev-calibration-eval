"""Arms for prereg A7 (jev-1.13.0 is not deterministic): draw 1 is the primary, per-arm
jitter across draws, draw-sensitive verdicts, and a k-averaged variant no H-test may read.
tests/mutate_draws.py breaks each rule in the shipped code and requires its arm to go RED.
"""
import json
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from jevcal import run as R  # noqa: E402
from jevcal import scorer as S  # noqa: E402
from jevcal.arms.frontier import FrontierArm  # noqa: E402
from jevcal.arms.jev import JevArm  # noqa: E402
from test_codex_fixes import (FIXTURE, KeyFile, certifying_pin, fake_git,  # noqa: E402
                              frontier_companion, frontier_stub_runner)

DRAW_P = [0.9, 0.6, 0.3]   # probability the stub puts on the first option, by draw


class ArmDrawJitter(unittest.TestCase):
    def test_hand_values(self):
        # item 1: draws [.9,.1] [.6,.4] [.3,.7]; pairwise |dp| per option: .3,.6,.3 (x2 options)
        #   -> diffs [.3,.3,.6,.6,.3,.3], answers A,A,B -> flips
        # item 2: draws [.8,.2] [.8,.2] [.7,.3]; diffs [0,0,.1,.1,.1,.1], answers A,A,A
        d = [[[.9, .1], [.6, .4], [.3, .7]], [[.8, .2], [.8, .2], [.7, .3]]]
        j = S.jitter(d)
        self.assertEqual(j["items_with_draws"], 2)
        self.assertAlmostEqual(j["mean_abs_dp"], (0.3 * 4 + 0.6 * 2 + 0.1 * 4) / 12)
        self.assertAlmostEqual(j["max_abs_dp"], 0.6)
        self.assertAlmostEqual(j["answer_flip_share"], 0.5)

    def test_abstained_draw_counts_as_a_flip_and_is_excluded_from_dp(self):
        j = S.jitter([[[.9, .1], [.5, .5], [.9, .1]]], [[False, True, False]])
        self.assertEqual(j["answer_flip_share"], 1.0)
        self.assertEqual(j["max_abs_dp"], 0.0)


class ArmDrawSensitivity(unittest.TestCase):
    def test_claim_only_if_identical_on_every_draw(self):
        self.assertEqual(S.claim_across_draws(["calibrated"] * 3), "calibrated")
        self.assertEqual(S.claim_across_draws(["calibrated", "calibrated", "roughly calibrated"]),
                         "draw-sensitive")
        self.assertEqual(S.claim_across_draws(["roughly calibrated", "calibrated", "calibrated"]),
                         "draw-sensitive")

    def test_h1_across_draws_uses_each_single_draw(self):
        gold = [0] * 20
        ids = [f"i{j:02d}" for j in range(20)]
        runs = [S.DrawRun(1, [[.97, .03]] * 20, gold, ids), S.DrawRun(2, [[.7, .3]] * 20, gold, ids)]
        out = S.h1_across_draws(runs)
        self.assertEqual([p["draw"] for p in out["per_draw"]], [1, 2])
        self.assertEqual([p["h1"] for p in out["per_draw"]],
                         ["calibrated", "not calibrated as claimed"])   # ECE .03 vs .30
        self.assertEqual(out["h1"], "draw-sensitive")


class ArmKAveragedIsolation(unittest.TestCase):
    def test_h_verdict_refuses_the_k_averaged_variant(self):
        gold = [0] * 10
        runs = [S.DrawRun(d, [[p, 1 - p]] * 10, gold, [f"i{j}" for j in range(10)]) for d, p in ((1, .9), (2, .6))]
        k_avg = S.KAveragedRun(runs)
        self.assertTrue(k_avg.metrics()["descriptive_only"])
        with self.assertRaises(TypeError):
            S.h1_for_draw(k_avg)
        with self.assertRaises(TypeError):
            S.h1_across_draws(runs + [k_avg])
        self.assertEqual(S.h1_for_draw(runs[0])["draw"], 1)
        with self.assertRaises(TypeError):
            S.h2_for_draw(k_avg, {"JEV": runs[0]})
        with self.assertRaises(TypeError):
            S.h2_for_draw(runs[0], {"JEV": k_avg})

    def test_codex_discriminator_no_verdict_materialized(self):
        # CODEX re-grade P0-2: draw 1 = twenty [.9,.1] items, all correct (ECE .1 -> NOT
        # calibrated); draws 2-3 = [1,0]. The mean of draws is ~[.967,.033], which WOULD
        # read "calibrated" if any verdict were computed on it. None may appear anywhere.
        ids, gold = [f"i{j:02d}" for j in range(20)], [0] * 20
        runs = [S.DrawRun(1, [[.9, .1]] * 20, gold, ids),
                S.DrawRun(2, [[1.0, 0.0]] * 20, gold, ids), S.DrawRun(3, [[1.0, 0.0]] * 20, gold, ids)]
        block = S.KAveragedRun(runs).metrics()
        self.assertEqual(verdict_like(block), [])
        self.assertEqual(S.h1_for_draw(runs[0])["h1"], "roughly calibrated")    # the real, draw-1 verdict

    def test_harness_hook_is_verdict_free(self):
        self.assertEqual(verdict_like({"a": [{"ok": 1}], "h1": "x"}), ["$.h1"])      # the walker CAN see one
        self.assertEqual(verdict_like({"b": "roughly calibrated"}), ["$.b"])


VERDICT_KEY = ("h1", "h2", "verdict", "band_")
VERDICT_WORDS = ("calibrated", "matches on this class", "does not", "binning-sensitive", "draw-sensitive")


def verdict_like(obj, path="$"):
    """Every path under `obj` whose KEY looks like an H verdict, or whose string VALUE is one."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}"
            if any(t in str(k).lower() for t in VERDICT_KEY):
                hits.append(p)
            hits.extend(verdict_like(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(verdict_like(v, f"{path}[{i}]"))
    elif isinstance(obj, str) and any(w in obj.lower() for w in VERDICT_WORDS):
        hits.append(path)
    return hits


def varying_jev(calls):
    """TypeSafe stand-in whose answer drifts by draw (A7's non-determinism, exaggerated)."""
    seen = {}

    def t(url, data, headers, timeout):
        body = json.loads(data)
        calls.append(body)
        key = json.dumps(body, sort_keys=True)
        seen[key] = seen.get(key, 0) + 1
        p = DRAW_P[min(seen[key], 3) - 1]
        q = body["questions"]["q"]
        if q["type"] == "noul":
            ans = {"type": "noul", "noul": p}
        elif q["type"] == "choice":
            ks = list(q["criteria"])
            ans = {"type": "choice", "choice": ks[0], "confidence": .5,
                   "probabilities": {k: (p if i == 0 else (1 - p) / (len(ks) - 1)) for i, k in enumerate(ks)}}
        else:
            n = len(q["criteria"])
            ans = {"type": "score", "score": 0, "confidence": .5, "legend": {},
                   "probabilities": {str(i): (p if i == 0 else (1 - p) / (n - 1)) for i in range(n)}}
        out = {"model": "jev-1.13.0", "answers": {"q": ans}, "usage": {"input_tokens": 10, "output_tokens": 2}}
        return 200, {}, json.dumps(out).encode()
    return t


class ArmDrawPrimary(unittest.TestCase):
    """A7.1/A7.2 end to end: k=3 on 2 items; every metric reads draw 1."""

    def test_runner_k3_primary_is_draw_1(self):
        proof = {"checked_at": "x", "endpoint": "e", "probe_result": "error:URLError", "attempts": 3}
        calls = []
        with tempfile.TemporaryDirectory() as d, KeyFile(), \
                mock.patch.object(R, "git", side_effect=fake_git), \
                mock.patch.dict(R.ARMS, {"JEV": lambda **kw: JevArm(transport=varying_jev(calls))}), \
                mock.patch.object(R, "langfuse_setup", return_value={"mode": "unreachable", "proof": proof}):
            rc = R.main(["--items", FIXTURE, "--arms", "JEV", "--evals-dir", d, "--run-id", "k3",
                         "--purpose", "t", "--mode", "fixture", "--draws", "3", "--draw-items", "FX-001,FX-003"])
            with open(os.path.join(d, "runs", "k3.json")) as fh:
                art = json.load(fh)
            with open(os.path.join(d, "runs", "k3.raw.jsonl")) as fh:
                raw = [json.loads(l) for l in fh]
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 6 + 2 * 2)
        self.assertEqual(sorted((r["item_id"], r["draw"]) for r in raw if r["draw"] > 1),
                         [("FX-001", 2), ("FX-001", 3), ("FX-003", 2), ("FX-003", 3)])
        # the primary metrics are exactly those of the draw-1 records
        items = {it.item_id: it for it in R.load_items(FIXTURE)}
        order = [it.item_id for it in R.load_items(FIXTURE)]
        d1 = {r["item_id"]: r for r in raw if r["draw"] == 1}
        probs = [R.record_vector(d1[i], items[i].options) for i in order]
        gold = [items[i].gold_index for i in order]
        self.assertEqual(art["metrics"]["JEV"]["raw"]["brier"], S.brier(probs, gold))
        dr = art["metrics"]["JEV"]["draws"]
        self.assertEqual((dr["k"], dr["items_with_k_draws"]), (3, 2))
        self.assertAlmostEqual(dr["jitter"]["max_abs_dp"], 0.6)
        self.assertEqual(dr["jitter"]["answer_flip_share"], 1.0)        # .3 on the first option flips it
        self.assertEqual([p["draw"] for p in dr["h1_draw_sensitivity"]["per_draw"]], [1, 2, 3])
        self.assertTrue(dr["k_averaged_descriptive"]["descriptive_only"])
        self.assertEqual(verdict_like(dr["k_averaged_descriptive"]), [])   # A7.5 in the real record
        self.assertEqual((art["mode"], art["certifying"], art["a7_compliant"]), ("fixture", False, False))
        self.assertEqual(art["draws_config"]["k"], 3)


def run_mode(args, git_side_effect=fake_git, lfs=None):
    """Drive R.main with a stub JEV (spy on its calls); return (rc or SystemExit, record, ledger?, calls)."""
    proof = {"checked_at": "x", "endpoint": "e", "probe_result": "error:URLError", "attempts": 3}
    calls = []
    if git_side_effect == "companion":      # D1: certifying needs two arms; add a stubbed FRONTIER
        with frontier_companion() as tracked:
            return run_mode(args, tracked, lfs)
    with tempfile.TemporaryDirectory() as d, KeyFile(), \
            mock.patch.object(R, "git", side_effect=git_side_effect), \
            mock.patch.dict(R.ARMS, {"JEV": lambda **kw: JevArm(transport=varying_jev(calls))}), \
            mock.patch.object(R, "langfuse_setup",
                              return_value=lfs or {"mode": "unreachable", "proof": proof}):
        # certifying runs need FIXTURE to be the pinned content (F1); fixture runs must not pin it (F2).
        is_cert = "fixture" not in args
        try:
            with (certifying_pin(FIXTURE) if is_cert else nullcontext()):
                out = R.main(["--items", FIXTURE, "--evals-dir", d, "--run-id", "m", "--purpose", "t"] + args)
        except SystemExit as e:
            out = e
        rec = None
        p = os.path.join(d, "runs", "m.json")
        if os.path.exists(p):
            with open(p) as fh:
                rec = json.load(fh)
        ledger = os.path.exists(os.path.join(d, "metric-history.jsonl"))
    return out, rec, ledger, calls


def subset_file(n):
    fd, p = tempfile.mkstemp(suffix=".txt", dir=os.path.join(ROOT, "config"))
    with os.fdopen(fd, "w") as fh:
        fh.write("".join(f"{i}\n" for i in [it.item_id for it in R.load_items(FIXTURE)][:n]))
    return p


class ArmRunMode(unittest.TestCase):
    """CODEX re-grade P1-3: certifying mode enforces A7 before any call; fixture mode is
    labelled non-certifying and never writes a cert-ledger row."""

    def test_certifying_refuses_k1_before_any_call(self):
        out, rec, ledger, calls = run_mode(["--arms", "JEV,FRONTIER", "--frontier-effort", "high", "--draws", "1"], "companion")
        self.assertIsInstance(out, SystemExit)
        self.assertIn("exactly 3", str(out))
        self.assertEqual((calls, rec, ledger), ([], None, False))

    def test_certifying_refuses_a_partial_redraw(self):
        out, rec, ledger, calls = run_mode(["--arms", "JEV,FRONTIER", "--frontier-effort", "high", "--draws", "3", "--draw-items", "FX-001"],
                                            "companion")
        self.assertIsInstance(out, SystemExit)
        self.assertIn("ALL items", str(out))
        self.assertEqual(calls, [])

    def test_certifying_reads_the_committed_a8_subset(self):
        self.assertEqual(os.path.relpath(R.FRONTIER_SUBSET_PATH, R.REPO),
                         "governance/evals/jev-calibration/frontier-subset-v1.txt")

    def test_certifying_frontier_needs_a_committed_100_item_subset(self):
        base = ["--arms", "JEV,FRONTIER", "--frontier-effort", "high", "--draws", "3"]
        with mock.patch.object(R, "FRONTIER_SUBSET_PATH", "/nonexistent/frontier-subset.txt"):
            out, *_ = run_mode(base)
        self.assertIn("missing", str(out))
        p = subset_file(6)
        try:
            rel = os.path.relpath(p, ROOT)
            with mock.patch.object(R, "FRONTIER_SUBSET_PATH", p):
                out, *_ = run_mode(base)                                    # untracked
                self.assertIn("committed", str(out))

                def tracked(*args):
                    return rel if args[:1] == ("ls-files",) else fake_git(*args)
                out, rec, ledger, calls = run_mode(base, tracked)
            self.assertIn("exactly 100", str(out))                          # 6 ids, not 100
            self.assertEqual(calls, [])
        finally:
            os.remove(p)

    def test_certifying_refuses_a_frontier_redraw_set_that_is_not_the_subset(self):
        p = subset_file(6)
        try:
            rel = os.path.relpath(p, ROOT)

            def tracked(*args):
                return rel if args[:1] == ("ls-files",) else fake_git(*args)
            with mock.patch.object(R, "FRONTIER_SUBSET_PATH", p), \
                    mock.patch.object(R, "CERT_FRONTIER_SUBSET_N", 6):
                out, rec, ledger, calls = run_mode(["--arms", "JEV,FRONTIER", "--frontier-effort", "high",
                                                    "--draws", "3", "--frontier-draw-items", "FX-001"], tracked)
            self.assertIn("exactly the fixed subset", str(out))
            self.assertEqual(calls, [])
        finally:
            os.remove(p)

    def test_fixture_mode_is_labelled_and_never_ledgered(self):
        out, rec, ledger, calls = run_mode(["--arms", "JEV", "--mode", "fixture", "--draws", "1"])
        self.assertEqual(out, 0)
        self.assertEqual((rec["mode"], rec["certifying"], rec["a7_compliant"]), ("fixture", False, False))
        self.assertFalse(ledger)
        self.assertEqual(len(calls), 6)

    def test_certifying_k3_all_items_is_ledgered(self):
        out, rec, ledger, calls = run_mode(["--arms", "JEV,FRONTIER", "--frontier-effort", "high", "--draws", "3"], "companion")
        self.assertEqual(out, 0)
        self.assertEqual((rec["mode"], rec["certifying"]), ("certifying", True))
        self.assertTrue(ledger)
        self.assertEqual(len(calls), 18)


class ArmTestPopulation(unittest.TestCase):
    """Advisor review P1: the ledger headline, win_gap, parity and the temperature-scaled
    comparison all use the TEST population; the pooled block is labelled all_splits."""

    def test_headline_reads_and_scaled_comparison_are_test_split(self):
        items = R.load_items(FIXTURE)
        test_ids = [it.item_id for it in items if it.split == "test"]
        fd, p = tempfile.mkstemp(suffix=".txt", dir=os.path.join(ROOT, "config"))
        with os.fdopen(fd, "w") as fh:
            fh.write("".join(i + "\n" for i in test_ids))
        rel = os.path.relpath(p, ROOT)

        def tracked(*args):
            return rel if args[:1] == ("ls-files",) else fake_git(*args)
        proof = {"checked_at": "x", "endpoint": "e", "probe_result": "error:URLError", "attempts": 3}
        calls = []
        try:
            with certifying_pin(FIXTURE), tempfile.TemporaryDirectory() as d, KeyFile(), \
                    mock.patch.object(R, "git", side_effect=tracked), \
                    mock.patch.object(R, "FRONTIER_SUBSET_PATH", p), \
                    mock.patch.object(R, "CERT_FRONTIER_SUBSET_N", len(test_ids)), \
                    mock.patch.dict(R.ARMS, {
                        "JEV": lambda **kw: JevArm(transport=varying_jev(calls)),
                        "FRONTIER": lambda **kw: FrontierArm(effort="high", runner=frontier_stub_runner)}), \
                    mock.patch.object(R, "langfuse_setup", return_value={"mode": "unreachable", "proof": proof}):
                rc = R.main(["--items", FIXTURE, "--evals-dir", d, "--run-id", "pop", "--purpose", "t",
                             "--arms", "JEV,FRONTIER", "--frontier-effort", "high", "--draws", "3"])
                with open(os.path.join(d, "runs", "pop.json")) as fh:
                    art = json.load(fh)
                with open(os.path.join(d, "metric-history.jsonl")) as fh:
                    row = json.loads(fh.readline())
        finally:
            os.remove(p)
        self.assertEqual(rc, 0)
        jev = art["metrics"]["JEV"]
        self.assertEqual((jev["raw"]["population"], jev["raw"]["n"]), ("all_splits", 6))
        self.assertEqual((jev["test"]["population"], jev["test"]["n"]), ("test", len(test_ids)))
        self.assertIn("brier", jev["test"]["ci95"])
        # ledger headline = the test block, not the pooled one (they differ on this fixture)
        self.assertEqual((row["population"], row["N_test"]), ("test", len(test_ids)))
        for key in ("accuracy", "brier", "ece_equal_mass_15"):
            self.assertEqual(row["arms"]["JEV"][key], jev["test"][key])
        self.assertNotEqual(jev["test"]["brier"], jev["raw"]["brier"])
        self.assertEqual(row["arms"]["JEV"]["brier_ci95"],
                         [jev["test"]["ci95"]["brier"]["lo"], jev["test"]["ci95"]["brier"]["hi"]])
        self.assertEqual(row["arms"]["JEV"]["all_splits"]["population"], "all_splits")
        # both reads on the test ids only
        for read in ("win_gap", "parity"):
            self.assertEqual(sorted(art["reads"][read]["item_ids"]), sorted(test_ids))
            self.assertEqual(art["reads"][read]["population"], "test")
            self.assertEqual(art["reads"][read]["n_items"], len(test_ids))   # what was actually resampled
        # temperature-scaled secondary is reported on the same test population as metrics.test
        ts = jev["temperature_scaled_secondary"]
        self.assertEqual(ts["n_test"], jev["test"]["n"])
        self.assertTrue(ts["population"].startswith("test"))


if __name__ == "__main__":
    unittest.main()
