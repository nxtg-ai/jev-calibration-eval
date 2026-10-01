#!/usr/bin/env python3
"""Mutation proof on the SHIPPED scorer (prereg §6): break one scorer line at a
time, require the arm named for that line to go RED, then restore byte-for-byte.
Engine and discipline: tests/mutation_engine.py. Report path: $JEVCAL_MUTATION_REPORT.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mutation_engine import run_suite  # noqa: E402

ARMS = ["ArmAccuracy", "ArmBrier", "ArmECE", "ArmNLL", "ArmMacroF1", "ArmCoverage",
        "ArmAURC", "ArmTieInvariance", "ArmECEEqualWidth", "ArmH1Verdict", "ArmBootstrap", "ArmPairedBootstrap", "ArmTemperature",
        "ArmAbstention", "ArmMutationProperties"]

T = "jevcal/scorer.py"
# (mutation id, named arm, target, anchor, replacement)
MUTATIONS = [
    ("M01-correctness-eq-to-ne", "ArmAccuracy", T,
     "return (predictions(probs) == np.asarray(gold, dtype=int)).astype(float)",
     "return (predictions(probs) != np.asarray(gold, dtype=int)).astype(float)"),
    ("M18-argmax-ties-to-highest-index", "ArmAccuracy", T,
     "return np.array([int(np.argmax(p)) for p in _as_list(probs)], dtype=int)",
     "return np.array([int(len(p) - 1 - np.argmax(np.asarray(p)[::-1])) for p in _as_list(probs)], dtype=int)"),
    ("M02-brier-square-to-abs", "ArmBrier", T,
     "out.append(float(np.sum((p - y) ** 2)))",
     "out.append(float(np.sum(np.abs(p - y))))"),
    ("M20-brier-halved-range-0-to-1", "ArmBrier", T,
     "out.append(float(np.sum((p - y) ** 2)))",
     "out.append(float(0.5 * np.sum((p - y) ** 2)))"),
    ("M21-brier-mean-to-median", "ArmBrier", T,
     "return _mean(brier_per_item(probs, gold))",
     "return float(np.median(brier_per_item(probs, gold)))"),
    ("M03-ece-drop-abs", "ArmECE", T,
     "for b in equal_mass_bins(conf, n_bins):\n        total += (len(b) / n) * abs(_bin_acc(correct, b) - _bin_conf_mean(conf, b))",
     "for b in equal_mass_bins(conf, n_bins):\n        total += (len(b) / n) * (_bin_acc(correct, b) - _bin_conf_mean(conf, b))"),
    ("M04-ece-equal-width-bins", "ArmECE", T,
     "return [np.flatnonzero(np.isin(inv, gs)) for gs in bins_groups]",
     "w = np.minimum((conf * n_bins).astype(int), n_bins - 1); "
     "return [np.where(w == k)[0] for k in range(n_bins) if np.any(w == k)]"),
    ("M05-ece-10-bins", "ArmECE", T,
     "N_BINS = 15",
     "N_BINS = 10"),
    ("M06-nll-uses-max-not-gold", "ArmNLL", T,
     "return np.array([-np.log(max(float(p[int(g)]), EPS)) for p, g in zip(_as_list(probs), gold)])",
     "return np.array([-np.log(max(float(np.max(p)), EPS)) for p, g in zip(_as_list(probs), gold)])"),
    ("M19-nll-clip-1e-9", "ArmNLL", T,
     "EPS = 1e-15",
     "EPS = 1e-9"),
    ("M07-macro-f1-to-micro", "ArmMacroF1", T,
     "return float(np.mean(f1s)) if f1s else 0.0",
     "return float(sum(1 for p, g in zip(pred_keys, gold_keys) if p == g) / len(gold_keys))"),
    ("M08-coverage-ge-to-gt", "ArmCoverage", T,
     "if boundary and (cum / (i + 1)) >= target:",
     "if boundary and (cum / (i + 1)) > target:"),
    ("M09-coverage-ignore-ties", "ArmCoverage", T,
     "boundary = (i == n - 1) or (c_sorted[i + 1] != c_sorted[i])",
     "boundary = True"),
    ("M10-aurc-ascending-order", "ArmAURC", T,
     "vals, inv = np.unique(-conf, return_inverse=True)",
     "vals, inv = np.unique(conf, return_inverse=True)"),
    ("M22-bins-split-tie-groups", "ArmTieInvariance", T,
     "return [np.flatnonzero(np.isin(inv, gs)) for gs in bins_groups]",
     "return [b for b in np.array_split(np.argsort(conf, kind='stable'), n_bins) if len(b) > 0]"),
    ("M23-aurc-per-item-order", "ArmTieInvariance", T,
     "return float(np.sum(n_g * risk_g) / n)",
     "return float(np.mean(1.0 - np.cumsum(correct[np.argsort(-conf, kind='stable')]) / np.arange(1, n + 1)))"),
    ("M24-bin-mean-item-order-sum", "ArmTieInvariance", T,
     "return _mean(conf[b])                                     # fsum: order-independent bits",
     "return float(conf[b].sum() / len(b))"),
    ("M28-mean-item-order-sum", "ArmTieInvariance", T,
     "return math.fsum(xs) / len(xs)",
     "return float(np.mean(xs))"),
    ("M29-score-arm-resamples-in-caller-order", "ArmTieInvariance", T,
     "    order = canonical_order(item_ids)\n    probs = [_as_list(probs)[i] for i in order]",
     "    order = np.arange(len(item_ids))\n    probs = [_as_list(probs)[i] for i in order]"),
    ("M30-paired-joined-in-caller-order", "ArmTieInvariance", T,
     "    joined = sorted(da)",
     "    joined = list(da)"),
    ("M25-equal-width-no-cap-at-1", "ArmECEEqualWidth", T,
     "idx = np.minimum(np.floor(conf * n_bins).astype(int), n_bins - 1)",
     "idx = np.floor(conf * n_bins).astype(int)"),
    ("M26-h1-ignores-equal-width", "ArmH1Verdict", T,
     '"h1": a if a == b else "binning-sensitive"}',
     '"h1": a}'),
    ("M27-h1-band-edge-exclusive", "ArmH1Verdict", T,
     "        if ece_value <= upper:",
     "        if ece_value < upper:"),
    ("M11-bootstrap-90pct-interval", "ArmBootstrap", T,
     "CI_LO, CI_HI = 2.5, 97.5",
     "CI_LO, CI_HI = 5.0, 95.0"),
    ("M12-bootstrap-200-resamples", "ArmBootstrap", T,
     "N_RESAMPLES = 2000",
     "N_RESAMPLES = 200"),
    ("M13-paired-becomes-unpaired", "ArmPairedBootstrap", T,
     "diffs[i] = a[idx].mean() - b[idx].mean()",
     "diffs[i] = a[idx].mean() - b[rng.integers(0, n, size=n)].mean()"),
    ("M14-temperature-multiplies", "ArmTemperature", T,
     "z = np.log(np.clip(p, EPS, 1.0)) / T",
     "z = np.log(np.clip(p, EPS, 1.0)) * T"),
    ("M15-temperature-fit-on-all-items", "ArmTemperature", T,
     "T = fit_temperature([probs[i] for i in cal], gold[cal])",
     "T = fit_temperature(probs, gold)"),
    ("M17-abstention-not-forced-incorrect", "ArmAbstention", T,
     "return corr * (1.0 - np.asarray(abstain, dtype=float))",
     "return corr"),
    ("M16-confidence-mean-not-max", "ArmMutationProperties", T,
     "return np.array([float(np.max(p)) for p in _as_list(probs)], dtype=float)",
     "return np.array([float(np.mean(p)) for p in _as_list(probs)], dtype=float)"),
]

if __name__ == "__main__":
    sys.exit(run_suite("tests.test_scorer", ARMS, MUTATIONS, "JEVCAL_MUTATION_REPORT"))
