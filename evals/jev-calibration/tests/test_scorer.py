"""Named test ARMS for the shipped scorer (jevcal/scorer.py). Real code, no mocks.

Each class is one arm. tests/mutate_scorer.py breaks one scorer line at a time
and requires the arm named for that line to go RED; expected values here are
hand-derived (shown in comments), never recomputed by a second implementation.
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jevcal import scorer as S  # noqa: E402
from jevcal import synth  # noqa: E402


class ArmAccuracy(unittest.TestCase):
    def test_hand_values(self):
        # preds [0,1,0] vs gold [0,0,1] -> 1 of 3 correct
        self.assertAlmostEqual(S.accuracy([[.7, .3], [.2, .8], [.6, .4]], [0, 0, 1]), 1 / 3)

    def test_tie_breaks_to_lowest_index(self):
        self.assertEqual(S.accuracy([[.5, .5]], [0]), 1.0)
        self.assertEqual(S.accuracy([[.5, .5]], [1]), 0.0)


class ArmBrier(unittest.TestCase):
    def test_hand_values(self):
        # [.7,.3] g0: .09+.09=.18 ; [.2,.8] g0: .64+.64=1.28 ; [.5,.3,.2] g2: .25+.09+.64=.98
        per = S.brier_per_item([[.7, .3], [.2, .8], [.5, .3, .2]], [0, 0, 2])
        np.testing.assert_allclose(per, [.18, 1.28, .98], atol=1e-12)
        self.assertAlmostEqual(S.brier([[.7, .3], [.2, .8], [.5, .3, .2]], [0, 0, 2]), (.18 + 1.28 + .98) / 3)

    def test_perfect_and_worst(self):
        self.assertEqual(S.brier([[1.0, 0.0]], [0]), 0.0)
        self.assertEqual(S.brier([[1.0, 0.0]], [1]), 2.0)

    def test_agrees_with_sklearn_multiclass_brier(self):
        # independent implementation as oracle (skipped where sklearn is absent)
        try:
            from sklearn.metrics import brier_score_loss
        except ImportError:
            self.skipTest("sklearn not installed")
        p, g = synth.calibrated_set(500, 5, seed=41)
        self.assertAlmostEqual(S.brier(p, g), brier_score_loss(g, np.array(p), labels=list(range(5))),
                               places=10)


class ArmECE(unittest.TestCase):
    def test_fifteen_singleton_bins(self):
        # n=15 -> 15 equal-mass bins of ONE item each, so ECE = mean |y_i - c_i|.
        # c_i = .93 + .004 i ; first 5 wrong (y=0), last 10 right (y=1).
        c = [0.93 + 0.004 * i for i in range(15)]
        y = [0.0] * 5 + [1.0] * 10
        want = (sum(c[:5]) + sum(1 - ci for ci in c[5:])) / 15
        self.assertAlmostEqual(S.ece(c, y), want, places=12)
        # equal-WIDTH 15 bins over [0,1] would pool all 15 into two bins -> different value
        self.assertEqual(len(S.reliability(c, y)), 15)

    def test_thirty_items_two_per_bin(self):
        # 30 items, confidences all distinct, bin b = sorted items {2b, 2b+1}.
        # Every bin pairs one right and one wrong item with confs (x, x+.001):
        # |acc - conf| = |0.5 - (x+.0005)|.
        c = [0.5 + 0.01 * b + 0.001 * j for b in range(15) for j in range(2)]
        y = [1.0, 0.0] * 15
        want = sum(abs(0.5 - (0.5 + 0.01 * b + 0.0005)) for b in range(15)) / 15
        self.assertAlmostEqual(S.ece(c, y), want, places=12)
        curve = S.reliability(c, y)
        self.assertEqual([p["n"] for p in curve], [2] * 15)

    def test_perfectly_calibrated_is_zero(self):
        # 15 bins of 4 items, each bin conf .75 with 3 of 4 correct
        c = [0.75] * 60
        y = [1.0, 1.0, 1.0, 0.0] * 15
        self.assertAlmostEqual(S.ece(c, y), 0.0, places=12)


class ArmNLL(unittest.TestCase):
    def test_hand_values(self):
        self.assertAlmostEqual(S.nll([[.7, .3]], [0]), -math.log(.7))
        self.assertAlmostEqual(S.nll([[.7, .3]], [1]), -math.log(.3))

    def test_zero_probability_is_clipped_not_infinite(self):
        self.assertAlmostEqual(S.nll([[1.0, 0.0]], [1]), -math.log(1e-15), places=6)


class ArmMacroF1(unittest.TestCase):
    def test_hand_values(self):
        # a: tp1 fp1 fn0 -> 2/3 ; b: tp1 fp0 fn1 -> 2/3 ; c: tp1 -> 1 ; macro = 7/9 (micro would be 3/4)
        self.assertAlmostEqual(S.macro_f1(["a", "a", "b", "c"], ["a", "b", "b", "c"]), 7 / 9)

    def test_unpredicted_class_counts_zero(self):
        # gold has class b never predicted: a: tp1 fp1 -> 2/3 ; b: tp0 fn1 -> 0 ; macro = 1/3
        self.assertAlmostEqual(S.macro_f1(["a", "a"], ["a", "b"]), 1 / 3)


class ArmCoverage(unittest.TestCase):
    def test_exactly_at_target_is_accepted(self):
        # 20 distinct confidences, the lowest one wrong: full coverage has precision 19/20 = .95
        c = [1.0 - 0.01 * i for i in range(20)]
        y = [1.0] * 19 + [0.0]
        self.assertAlmostEqual(S.coverage_at_precision(c, y), 1.0)

    def test_cut_only_between_distinct_confidences(self):
        # [.99 right, .8 right, .8 wrong]: the two .8 items cannot be split -> coverage 1/3
        self.assertAlmostEqual(S.coverage_at_precision([.99, .8, .8], [1, 1, 0]), 1 / 3)

    def test_none_reach_target(self):
        self.assertEqual(S.coverage_at_precision([.9, .8], [0, 1]), 0.0)


class ArmAURC(unittest.TestCase):
    def test_hand_values(self):
        # desc conf order right,right,wrong: risks 0, 0, 1/3 -> AURC 1/9
        self.assertAlmostEqual(S.aurc([.9, .8, .7], [1, 1, 0]), 1 / 9)
        # desc order right,wrong,right: risks 0, 1/2, 1/3 -> 5/18
        self.assertAlmostEqual(S.aurc([.9, .8, .7], [1, 0, 1]), 5 / 18)


class ArmBootstrap(unittest.TestCase):
    def test_width_matches_normal_theory(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0.0, 1.0, size=400)
        ci = S.bootstrap_ci(len(x), lambda idx: float(x[idx].mean()), seed=11)
        theory = 2 * 1.959964 * x.std() / math.sqrt(len(x))
        self.assertLess(abs((ci["hi"] - ci["lo"]) / theory - 1.0), 0.08)
        self.assertLess(ci["lo"], x.mean())
        self.assertGreater(ci["hi"], x.mean())
        self.assertEqual(ci["n_resamples"], 2000)

    def test_seeded_determinism(self):
        x = np.arange(50, dtype=float)
        f = lambda idx: float(x[idx].mean())  # noqa: E731
        self.assertEqual(S.bootstrap_ci(50, f, seed=3), S.bootstrap_ci(50, f, seed=3))


class ArmPairedBootstrap(unittest.TestCase):
    def test_constant_shift_has_zero_width_interval(self):
        # b = a + .01 on every item: every paired resample's diff is exactly -.01
        rng = np.random.default_rng(5)
        a = rng.uniform(0, 2, size=200)
        b = a + 0.01
        ids = [f"i{j:03d}" for j in range(200)]
        r = S.paired_bootstrap(a, b, seed=1, ids_a=ids, ids_b=ids)
        self.assertAlmostEqual(r["mean_diff"], -0.01, places=12)
        self.assertAlmostEqual(r["lo"], -0.01, places=9)
        self.assertAlmostEqual(r["hi"], -0.01, places=9)
        self.assertEqual(r["n_resamples"], 2000)

    def test_joins_by_item_id_not_by_position(self):
        # b lists the same items in REVERSE order; the join by id must pair them correctly
        a, ids = [0.1, 0.5, 0.9], ["x", "y", "z"]
        r = S.paired_bootstrap(a, [v + 0.01 for v in a][::-1], seed=0, ids_a=ids, ids_b=ids[::-1])
        self.assertAlmostEqual(r["lo"], -0.01, places=9)
        self.assertAlmostEqual(r["hi"], -0.01, places=9)

    def test_rejects_differing_id_sets(self):
        with self.assertRaises(ValueError):
            S.paired_bootstrap([1.0, 2.0], [1.0], seed=0, ids_a=["a", "b"], ids_b=["a"])
        with self.assertRaises(ValueError):
            S.paired_bootstrap([1.0, 2.0], [1.0, 2.0], seed=0, ids_a=["a", "b"], ids_b=["a", "c"])


class ArmTemperature(unittest.TestCase):
    def test_recovers_known_temperature(self):
        # calibrated set sharpened x2 == T=0.5 on log p; the NLL-optimal fix is T ~= 2
        p, g = synth.calibrated_set(4000, 4, seed=21)
        T = S.fit_temperature(synth.sharpen(p, 2.0), g)
        self.assertLess(abs(T - 2.0), 0.25, f"fitted T={T}")

    def test_fit_uses_calibration_split_only(self):
        p, g = synth.calibrated_set(600, 4, seed=22)
        p = synth.sharpen(p, 2.0)
        splits = ["calibration"] * 200 + ["test"] * 400
        base = S.temperature_scaled(p, g, splits)
        g2 = np.array(g).copy()
        g2[200:] = synth.flip_correct_labels(p[200:], g[200:])  # rewrite TEST labels only
        moved = S.temperature_scaled(p, g2, splits)
        self.assertTrue(base["fitted"])
        self.assertEqual(base["temperature"], moved["temperature"])
        self.assertEqual((base["n_cal"], base["n_test"]), (200, 400))


class ArmECEEqualWidth(unittest.TestCase):
    """prereg A6.2 secondary: 15 equal-width bins on [0,1]."""

    def test_hand_values(self):
        # floor(c*15): .05->0, .1->1, .95->14, 1.0->15 capped to 14. Bins {.05},{.1},{.95,1.0}.
        # ECE = 1/4*|0-.05| + 1/4*|1-.1| + 2/4*|.5-.975| = .0125 + .225 + .2375 = .475
        c, y = [.05, .1, .95, 1.0], [0.0, 1.0, 1.0, 0.0]
        self.assertAlmostEqual(S.ece_equal_width(c, y), 0.475, places=12)
        self.assertEqual([len(b) for b in S.equal_width_bins(np.asarray(c))], [1, 1, 2])
        # the primary on the same items: 4 distinct values -> 4 singleton bins -> .5
        self.assertAlmostEqual(S.ece(c, y), 0.5, places=12)


class ArmH1Verdict(unittest.TestCase):
    """prereg §5 H1 bands; A6.2 'binning-sensitive' when the two ECEs disagree."""

    def test_bands(self):
        self.assertEqual(S.h1_band(0.05), "calibrated")
        self.assertEqual(S.h1_band(0.0500001), "roughly calibrated")
        self.assertEqual(S.h1_band(0.10), "roughly calibrated")
        self.assertEqual(S.h1_band(0.1000001), "not calibrated as claimed")

    def test_binning_sensitive(self):
        self.assertEqual(S._combine_h1_bands(0.04, 0.06)["h1"], "binning-sensitive")
        self.assertEqual(S._combine_h1_bands(0.12, 0.09)["h1"], "binning-sensitive")
        self.assertEqual(S._combine_h1_bands(0.04, 0.03)["h1"], "calibrated")
        # the public verdict path: h1_for_draw on a single-draw run. 20 items at .9 with 18
        # right -> both ECEs are exactly 0 -> calibrated on both binnings.
        run = S.DrawRun(1, [[.9, .1]] * 20, [0] * 18 + [1] * 2, [f"i{j:02d}" for j in range(20)])
        v = S.h1_for_draw(run)
        self.assertEqual((v["band_equal_mass_15"], v["band_equal_width_15"], v["h1"]),
                         ("calibrated", "calibrated", "calibrated"))
        m = S.point_metrics([[.9, .1]] * 20, [0] * 18 + [1] * 2)
        self.assertNotIn("h1", m)                                               # metrics only
        self.assertEqual((m["ece_bins_realized"], m["ece_bin_n"]), (1, [20]))   # A6.1 reported


class ArmTieInvariance(unittest.TestCase):
    """CODEX HOLD finding 1: a tie group must never be split
    across bins, and no metric may depend on item order."""

    @staticmethod
    def metrics(c, y):
        return (S.ece(c, y), S.reliability(c, y), S.aurc(c, y), S.coverage_at_precision(c, y))

    def assert_permutation_invariant(self, c, y, seeds=60):
        c, y = np.asarray(c, dtype=float), np.asarray(y, dtype=float)
        base = self.metrics(c, y)
        for seed in range(seeds):
            p = np.random.default_rng(seed).permutation(len(c))
            self.assertEqual(self.metrics(c[p], y[p]), base, f"seed {seed}")  # bitwise equality
        return base

    def test_codex_sixteen_items_all_at_half(self):
        # CODEX's fixture: 16 items all at confidence .5, 8 right. One tie group ->
        # one bin; ECE = |8/16 - .5| = 0 exactly; AURC = risk of accepting all = .5.
        c, y = [0.5] * 16, [1.0, 0.0] * 8
        ece, curve, aurc, cov = self.assert_permutation_invariant(c, y)
        self.assertEqual(ece, 0.0)
        self.assertEqual(aurc, 0.5)
        self.assertEqual([b["n"] for b in curve], [16])

    def test_tie_group_straddling_a_bin_edge(self):
        # 3 bins over 6 items (edges would fall after items 2 and 4); the .2 group
        # spans both edges, so it stays whole: bins {.1,.2,.2,.2} and {.3,.4}.
        c = np.array([.1, .2, .2, .2, .3, .4])
        self.assertEqual([len(b) for b in S.equal_mass_bins(c, 3)], [4, 2])
        # mixed fixture at the default 15 bins: 30 items, a tie group of 3 at .6 on an edge
        c = [round(0.30 + 0.01 * i, 2) for i in range(27)] + [0.6] * 3
        c[26] = 0.6
        y = ([1.0, 0.0, 1.0] * 10)[:30]
        self.assert_permutation_invariant(c, y)
        bins = S.equal_mass_bins(np.asarray(c))
        seen = [set(np.asarray(c)[b].tolist()) for b in bins]
        for i in range(len(seen)):
            for j in range(i + 1, len(seen)):
                self.assertFalse(seen[i] & seen[j], "a confidence value appears in two bins")

    def test_untied_set_bin_sums_are_order_free(self):
        # 600 distinct confidences -> 40 per bin: a bin mean summed in item order would
        # differ in its last bits between permutations; the scorer's must not.
        p, g = synth.calibrated_set(600, 4, seed=52)
        self.assert_permutation_invariant(S.confidences(p), S.correctness(p, g), seeds=30)

    def test_every_metric_is_bit_identical_under_permutation(self):
        # A6.3: Brier, NLL, accuracy, macro-F1, both ECEs, reliability, coverage, AURC,
        # realized bins and the H1 verdict: all from point_metrics, compared as a whole.
        rng = np.random.default_rng(53)
        for n, tie_heavy in ((240, True), (240, False)):
            if tie_heavy:   # 4 distinct confidences over 240 items, 3 options
                top = rng.choice([0.4, 0.55, 0.7, 0.9], size=n)
                probs = [[t, (1 - t) * 0.7, (1 - t) * 0.3] for t in top]
            else:
                probs = [list(p) for p in rng.dirichlet(np.ones(3), size=n)]
            gold = rng.integers(0, 3, size=n)
            keys = [["a", "b", "c"]] * n
            ab = rng.random(n) < 0.05
            base = S.point_metrics(probs, gold, keys, list(ab))
            for seed in range(12):
                p = np.random.default_rng(1000 + seed).permutation(n)
                got = S.point_metrics([probs[i] for i in p], gold[p], [keys[i] for i in p], list(ab[p]))
                self.assertEqual(got, base, f"tie_heavy={tie_heavy} seed={seed}")

    def test_score_arm_cis_and_paired_bootstrap_bit_identical(self):
        # CODEX re-grade P0-1: EVERY score_arm output, all bootstrap CIs included, and the
        # paired bootstrap (bounds and mean_diff) must not move under a permutation of the
        # items with the same seeds. 200 items, tie-heavy, 3 options, some abstentions.
        rng = np.random.default_rng(61)
        n = 200
        top = rng.choice([0.4, 0.55, 0.7, 0.9], size=n)
        probs = [[t, (1 - t) * 0.7, (1 - t) * 0.3] for t in top]
        gold = rng.integers(0, 3, size=n)
        keys = [["a", "b", "c"]] * n
        ab = list(rng.random(n) < 0.05)
        ids = [f"item-{j:04d}" for j in range(n)]
        other = S.brier_per_item([[1 / 3] * 3] * n, gold)          # a second arm, per item
        base = S.score_arm(probs, gold, ids, keys, seed=7, n_resamples=300, abstain=ab)
        mine = S.brier_per_item(probs, gold)
        base_pb = S.paired_bootstrap(mine, other, seed=9, ids_a=ids, ids_b=ids, n_resamples=300)
        for seed in range(5):
            p = np.random.default_rng(2000 + seed).permutation(n)
            got = S.score_arm([probs[i] for i in p], gold[p], [ids[i] for i in p], [keys[i] for i in p],
                              seed=7, n_resamples=300, abstain=[ab[i] for i in p])
            self.assertEqual(got, base, f"score_arm moved under shuffle {seed}")
            q = np.random.default_rng(3000 + seed).permutation(n)       # arms shuffled independently
            pb = S.paired_bootstrap(mine[p], other[q], seed=9, ids_a=[ids[i] for i in p],
                                    ids_b=[ids[i] for i in q], n_resamples=300)
            self.assertEqual(pb, base_pb, f"paired bootstrap moved under shuffle {seed}")

    def test_heavily_tied_calibrated_set(self):
        p, g = synth.calibrated_set(300, 4, seed=51)
        c = np.round(S.confidences(p), 1)                    # ~7 distinct values over 300 items
        y = S.correctness(p, g)
        self.assert_permutation_invariant(c, y)


class ArmAbstention(unittest.TestCase):
    def test_abstention_is_uniform_and_incorrect(self):
        # item 0 abstained, scored uniform [.5,.5] with gold 0: argmax would tie to 0
        # ("correct"), the rubric forces it INCORRECT. Item 1 is a normal right answer.
        probs, gold, keys = [[.5, .5], [.9, .1]], [0, 0], [["a", "b"], ["a", "b"]]
        m = S.point_metrics(probs, gold, keys, abstain=[True, False])
        self.assertEqual(m["abstentions"], 1)
        self.assertAlmostEqual(m["accuracy"], 0.5)
        # For F1 the abstained item's predicted class is "<abstain>", never the tie-broken
        # argmax. (This arm asserts the abstention RULE only; the Brier and macro-F1
        # formulas are ArmBrier's and ArmMacroF1's, so a formula break isolates there.)
        self.assertEqual(S._pred_keys(probs, keys, [True, False]), ["<abstain>", "a"])
        self.assertEqual(S._pred_keys([[.9, .1], [.2, .8]], keys, None), ["a", "b"])


class ArmMutationProperties(unittest.TestCase):
    """prereg §6: a scorer that cannot fail is not a scorer."""

    def test_sharpening_x2_raises_ece_and_brier(self):
        p, g = synth.calibrated_set(3000, 4, seed=31)
        sp = synth.sharpen(p, 2.0)
        conf, corr = S.confidences(p), S.correctness(p, g)
        sconf, scorr = S.confidences(sp), S.correctness(sp, g)
        # the mechanism first: sharpening must raise stated confidence without changing
        # any answer, so the calibration gap can only open upward
        self.assertGreater(sconf.mean(), conf.mean() + 0.05)
        np.testing.assert_array_equal(scorr, corr)
        self.assertGreater(S.ece(sconf, scorr), S.ece(conf, corr))
        self.assertGreater(S.brier(sp, g), S.brier(p, g))

    def test_label_flip_drops_accuracy(self):
        p, g = synth.calibrated_set(3000, 4, seed=32)
        self.assertLess(S.accuracy(p, synth.flip_correct_labels(p, g)), S.accuracy(p, g))


if __name__ == "__main__":
    unittest.main()
