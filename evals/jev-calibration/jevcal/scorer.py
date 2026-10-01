"""Deterministic scorer (prereg §4, §6). stdlib + numpy only. Not a model.

Inputs are per-item probability vectors (variable option count allowed) and the
gold option index. All metrics are computed on RAW probabilities; temperature
scaling is a separate, secondary transform fitted on the calibration split only.

Conventions (fixed here so a reviewer can reproduce them):
  - prediction = argmax, ties broken to the LOWEST option index;
  - confidence = max probability;
  - multiclass Brier = sum_k (p_k - y_k)^2 per item (range 0..2), then the mean;
  - ECE = up to 15 EQUAL-MASS bins over confidence, TIE-INVARIANT: bin edges are
    placed ONLY between distinct confidence values, at the first distinct-value
    boundary where the cumulative count reaches j*n/15. A tie group is never split
    across bins, so ties can make bins unequal or reduce their number (all items at
    one confidence -> one bin). ECE = sum_b (n_b/n) * |acc_b - conf_b|, with conf_b
    summed over the bin's SORTED values so the result is bitwise independent of item
    order;
  - NLL = mean of -log(max(p_gold, 1e-15));
  - macro-F1 over the union of gold and predicted class keys, F1 = 0 for a class
    with no support and no predictions (sklearn zero_division=0 behaviour);
  - coverage at >=95% precision: items sorted by confidence (desc); a threshold
    can only cut BETWEEN distinct confidence values; coverage is the largest
    accepted fraction whose precision is >= 0.95 (0.0 if none);
  - AURC over DISTINCT-CONFIDENCE thresholds (group-wise): for each distinct
    confidence c (descending), accept every item with confidence >= c; the risk at
    that threshold (1 - precision) is weighted by the tie group's share n_c/n. With
    no ties this equals the mean over k=1..n of the top-k risk; with ties it never
    depends on item order;
  - every metric above is invariant to item order, bitwise (tests/test_scorer.py
    ArmTieInvariance permutes item order over many seeds);
  - bootstrap: 2000 resamples of items with replacement, numpy default_rng(seed),
    95% percentile interval (2.5, 97.5);
  - paired bootstrap: ONE index resample shared by both arms, on per-item Brier
    differences (arm A minus arm B);
  - SECONDARY (prereg A6.2): ECE over 15 EQUAL-WIDTH bins on [0,1] (bin k =
    [k/15, (k+1)/15), last bin closed at 1.0), reported beside the primary. The realized
    equal-mass bin count and per-bin n are reported (A6.1). If the two ECEs place an
    arm in different H1 bands (<=0.05 calibrated, (0.05,0.10] roughly, >0.10 not), that
    arm's H1 is "binning-sensitive" and claimed neither way;
  - DRAWS (prereg A7): the PRIMARY is draw 1, the first call per item, never best-of-k;
    every metric and the paired Brier bootstrap read draw 1. Draws 2..k give per-arm
    jitter (mean and max |dp| over options and draw pairs, and the answer-flip share)
    and draw-sensitivity: an H verdict is claimed only if identical on every
    single-draw run, else "draw-sensitive". The k-averaged variant (KAveragedRun) is
    descriptive only and h1_for_draw refuses it;
  - ABSTENTION RUBRIC (prereg A6.4; config/abstention-rubric.json): an abstention is
    never dropped. It is scored as the uniform distribution for Brier and NLL, and as
    incorrect at confidence 1/K for ECE, the reliability curve, accuracy, macro-F1 and
    selective prediction. Abstention counts are reported per arm. (Mechanically: the
    caller passes the uniform vector, whose max is 1/K, plus `abstain`, which forces
    correctness to 0 and the predicted class to "<abstain>".)
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

N_BINS = 15
EPS = 1e-15
N_RESAMPLES = 2000
CI_LO, CI_HI = 2.5, 97.5
PRECISION_TARGET = 0.95


def _as_list(probs: Sequence[Sequence[float]]) -> List[np.ndarray]:
    return [np.asarray(p, dtype=float) for p in probs]


def predictions(probs) -> np.ndarray:
    return np.array([int(np.argmax(p)) for p in _as_list(probs)], dtype=int)


def confidences(probs) -> np.ndarray:
    return np.array([float(np.max(p)) for p in _as_list(probs)], dtype=float)


def correctness(probs, gold: Sequence[int]) -> np.ndarray:
    return (predictions(probs) == np.asarray(gold, dtype=int)).astype(float)


def accuracy(probs, gold) -> float:
    return float(np.mean(correctness(probs, gold)))


def brier_per_item(probs, gold) -> np.ndarray:
    out = []
    for p, g in zip(_as_list(probs), gold):
        y = np.zeros_like(p)
        y[int(g)] = 1.0
        out.append(float(np.sum((p - y) ** 2)))
    return np.array(out, dtype=float)


def _mean(xs) -> float:
    """Exactly rounded mean (math.fsum): bitwise independent of item order."""
    xs = [float(x) for x in xs]
    return math.fsum(xs) / len(xs)


def brier(probs, gold) -> float:
    return _mean(brier_per_item(probs, gold))


def nll_per_item(probs, gold) -> np.ndarray:
    return np.array([-np.log(max(float(p[int(g)]), EPS)) for p, g in zip(_as_list(probs), gold)])


def nll(probs, gold) -> float:
    return _mean(nll_per_item(probs, gold))


def equal_mass_bins(conf: np.ndarray, n_bins: int = N_BINS) -> List[np.ndarray]:
    """Tie-invariant equal-mass bins: edges only BETWEEN distinct confidence values."""
    conf = np.asarray(conf, dtype=float)
    n = len(conf)
    if n == 0:
        return []
    vals, inv = np.unique(conf, return_inverse=True)       # ascending distinct values
    counts = np.bincount(inv, minlength=len(vals))
    bins_groups, cur, cum, j = [], [], 0, 1
    for g, c in enumerate(counts):
        cur.append(g)
        cum += int(c)
        if cum >= j * n / n_bins and g < len(vals) - 1:
            bins_groups.append(cur)
            cur = []
            while j < n_bins and j * n / n_bins <= cum:
                j += 1
    if cur:
        bins_groups.append(cur)
    return [np.flatnonzero(np.isin(inv, gs)) for gs in bins_groups]


def _bin_conf_mean(conf: np.ndarray, b: np.ndarray) -> float:
    return _mean(conf[b])                                     # fsum: order-independent bits


def _bin_acc(correct: np.ndarray, b: np.ndarray) -> float:
    return _mean(correct[b])


def equal_width_bins(conf: np.ndarray, n_bins: int = N_BINS) -> List[np.ndarray]:
    """A6.2 secondary: n_bins equal-width bins on [0,1]; bin k = [k/n, (k+1)/n), the
    last bin closed at 1.0. Empty bins are dropped. Order-invariant by construction."""
    conf = np.asarray(conf, dtype=float)
    idx = np.minimum(np.floor(conf * n_bins).astype(int), n_bins - 1)
    return [np.flatnonzero(idx == k) for k in range(n_bins) if np.any(idx == k)]


def ece_equal_width(conf, correct, n_bins: int = N_BINS) -> float:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(conf)
    total = 0.0
    for b in equal_width_bins(conf, n_bins):
        total += (len(b) / n) * abs(_bin_acc(correct, b) - _bin_conf_mean(conf, b))
    return float(total)


H1_BANDS = ((0.05, "calibrated"), (0.10, "roughly calibrated"))  # prereg §5 H1; >0.10 below


def h1_band(ece_value: float) -> str:
    for upper, name in H1_BANDS:
        if ece_value <= upper:
            return name
    return "not calibrated as claimed"


def _combine_h1_bands(ece_equal_mass_value: float, ece_equal_width_value: float) -> Dict[str, str]:
    """A6.2: if the two ECEs put an arm in different H1 bands, H1 is reported as
    'binning-sensitive' and claimed neither way. PRIVATE: the only public verdict path
    is h1_for_draw(DrawRun), which refuses the k-averaged variant (A7.5)."""
    a, b = h1_band(ece_equal_mass_value), h1_band(ece_equal_width_value)
    return {"band_equal_mass_15": a, "band_equal_width_15": b,
            "h1": a if a == b else "binning-sensitive"}


def reliability(conf, correct, n_bins: int = N_BINS) -> List[Dict[str, float]]:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    curve = []
    for b in equal_mass_bins(conf, n_bins):
        curve.append({"n": int(len(b)), "conf_mean": _bin_conf_mean(conf, b),
                      "acc": _bin_acc(correct, b),
                      "conf_lo": float(conf[b].min()), "conf_hi": float(conf[b].max())})
    return curve


def ece(conf, correct, n_bins: int = N_BINS) -> float:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(conf)
    total = 0.0
    for b in equal_mass_bins(conf, n_bins):
        total += (len(b) / n) * abs(_bin_acc(correct, b) - _bin_conf_mean(conf, b))
    return float(total)


def macro_f1(pred_keys: Sequence[str], gold_keys: Sequence[str]) -> float:
    classes = sorted(set(gold_keys) | set(pred_keys))
    f1s = []
    for c in classes:
        tp = sum(1 for p, g in zip(pred_keys, gold_keys) if p == c and g == c)
        fp = sum(1 for p, g in zip(pred_keys, gold_keys) if p == c and g != c)
        fn = sum(1 for p, g in zip(pred_keys, gold_keys) if p != c and g == c)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def coverage_at_precision(conf, correct, target: float = PRECISION_TARGET) -> float:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(conf)
    order = np.argsort(-conf, kind="stable")
    c_sorted = conf[order]
    k_sorted = correct[order]
    best = 0.0
    cum = 0.0
    for i in range(n):
        cum += k_sorted[i]
        boundary = (i == n - 1) or (c_sorted[i + 1] != c_sorted[i])
        if boundary and (cum / (i + 1)) >= target:
            best = (i + 1) / n
    return float(best)


def aurc(conf, correct) -> float:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(conf)
    vals, inv = np.unique(-conf, return_inverse=True)       # distinct thresholds, highest conf first
    n_g = np.bincount(inv, minlength=len(vals)).astype(float)
    k_g = np.bincount(inv, weights=correct, minlength=len(vals))  # 0/1 sums: exact in any order
    risk_g = 1.0 - np.cumsum(k_g) / np.cumsum(n_g)           # risk at each distinct threshold
    return float(np.sum(n_g * risk_g) / n)


def bootstrap_ci(n_items: int, stat: Callable[[np.ndarray], float], seed: int,
                 n_resamples: int = N_RESAMPLES) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    vals = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n_items, size=n_items)
        vals[i] = stat(idx)
    return {"lo": float(np.percentile(vals, CI_LO)), "hi": float(np.percentile(vals, CI_HI)),
            "n_resamples": int(n_resamples), "seed": int(seed)}


def canonical_order(item_ids: Sequence[str]) -> np.ndarray:
    """Indices that sort items by item_id. Every resampling runs in THIS order, so the
    same seed draws the same items whatever order the caller passed (CODEX re-grade P0)."""
    ids = [str(x) for x in item_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("item_ids must be unique to define a canonical resampling order")
    return np.array(sorted(range(len(ids)), key=lambda i: ids[i]), dtype=int)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], seed: int,
                     ids_a: Sequence[str], ids_b: Sequence[str],
                     n_resamples: int = N_RESAMPLES) -> Dict[str, float]:
    """Paired bootstrap of mean(a) - mean(b). The two arms are JOINED BY item_id and the
    joined pairs are sorted by item_id before any index is drawn; differing id sets are
    refused (a pairing that is not on the same items is not paired)."""
    if len(a) != len(ids_a) or len(b) != len(ids_b):
        raise ValueError("each value needs its item_id")
    da = dict(zip(map(str, ids_a), map(float, a)))
    db = dict(zip(map(str, ids_b), map(float, b)))
    if len(da) != len(ids_a) or len(db) != len(ids_b):
        raise ValueError("duplicate item_ids")
    if set(da) != set(db):
        raise ValueError("paired bootstrap needs per-item values on the SAME items (id sets differ)")
    joined = sorted(da)
    a = np.array([da[i] for i in joined], dtype=float)
    b = np.array([db[i] for i in joined], dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    return {"mean_diff": _mean(a) - _mean(b),
            "lo": float(np.percentile(diffs, CI_LO)), "hi": float(np.percentile(diffs, CI_HI)),
            "frac_resamples_a_lower": float(np.mean(diffs < 0)),
            "n_resamples": int(n_resamples), "seed": int(seed), "n_items": int(n)}


# ---------------- temperature scaling (secondary, calibration split only) ----------------

def apply_temperature(probs, T: float) -> List[np.ndarray]:
    out = []
    for p in _as_list(probs):
        z = np.log(np.clip(p, EPS, 1.0)) / T
        z = z - z.max()
        e = np.exp(z)
        out.append(e / e.sum())
    return out


def fit_temperature(cal_probs, cal_gold, lo: float = -3.0, hi: float = 3.0,
                    iters: int = 100) -> float:
    """Golden-section search over log T minimising NLL on the CALIBRATION split."""
    def loss(logt):
        return nll(apply_temperature(cal_probs, float(np.exp(logt))), cal_gold)
    gr = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = loss(d)
    return float(np.exp((a + b) / 2))


def temperature_scaled(probs, gold, splits: Sequence[str],
                       abstain: Optional[Sequence[bool]] = None) -> Dict:
    """Secondary read (prereg §4): fit T on split=='calibration' ONLY, report the
    test-split metrics after scaling. Every arm gets this identical treatment."""
    probs = _as_list(probs)
    gold = np.asarray(gold, dtype=int)
    cal = [i for i, s in enumerate(splits) if s == "calibration"]
    test = [i for i, s in enumerate(splits) if s == "test"]
    if not cal or not test:
        return {"fitted": False, "reason": "needs both a calibration and a test split",
                "n_cal": len(cal), "n_test": len(test)}
    T = fit_temperature([probs[i] for i in cal], gold[cal])
    scaled = apply_temperature([probs[i] for i in test], T)
    ab = None if abstain is None else [bool(abstain[i]) for i in test]
    out = point_metrics(scaled, gold[test], abstain=ab)
    out.update({"fitted": True, "temperature": T, "n_cal": len(cal), "n_test": len(test)})
    return out


# ---------------- aggregate ----------------

ABSTAIN_KEY = "<abstain>"


def _mask(corr: np.ndarray, abstain) -> np.ndarray:
    if abstain is None:
        return corr
    return corr * (1.0 - np.asarray(abstain, dtype=float))


def _pred_keys(probs, keys_per_item, abstain) -> List[str]:
    pred = predictions(probs)
    ab = [False] * len(probs) if abstain is None else list(abstain)
    return [ABSTAIN_KEY if ab[i] else keys_per_item[i][pred[i]] for i in range(len(probs))]


def point_metrics(probs, gold, keys_per_item: Optional[Sequence[Sequence[str]]] = None,
                  abstain: Optional[Sequence[bool]] = None) -> Dict:
    probs = _as_list(probs)
    gold = np.asarray(gold, dtype=int)
    conf = confidences(probs)
    corr = _mask(correctness(probs, gold), abstain)
    out = {
        "n": int(len(probs)),
        "abstentions": int(sum(abstain)) if abstain is not None else 0,
        "accuracy": _mean(corr),
        "brier": brier(probs, gold),
        "ece_equal_mass_15": ece(conf, corr),
        "ece_equal_width_15": ece_equal_width(conf, corr),   # A6.2 pre-declared secondary
        "nll": nll(probs, gold),
        "coverage_at_95_precision": coverage_at_precision(conf, corr),
        "aurc": aurc(conf, corr),
        "reliability_curve": reliability(conf, corr),
        "ece_degenerate": bool(len(probs) < N_BINS),
    }
    curve = out["reliability_curve"]
    out["ece_bins_realized"] = len(curve)                     # A6.1: can be < 15 when ties sit on an edge
    out["ece_bin_n"] = [b["n"] for b in curve]
    # METRICS ONLY. No verdict is computed here (CODEX re-grade P0-2): H verdicts come
    # only from h1_for_draw / h2_for_draw, which accept a single-draw DrawRun.
    if keys_per_item is not None:
        pk = _pred_keys(probs, keys_per_item, abstain)
        gk = [keys_per_item[i][gold[i]] for i in range(len(probs))]
        out["macro_f1"] = macro_f1(pk, gk)
    return out


def score_arm(probs, gold, item_ids: Sequence[str], keys_per_item=None, seed: int = 0,
              n_resamples: int = N_RESAMPLES, abstain: Optional[Sequence[bool]] = None) -> Dict:
    """Point metrics + 2000-resample bootstrap 95% CIs for every scalar metric. Items are
    put in CANONICAL order (sorted by item_id) before any resample is drawn, so every
    output, CIs included, is bit-identical under a permutation of the input."""
    order = canonical_order(item_ids)
    probs = [_as_list(probs)[i] for i in order]
    gold = np.asarray(gold, dtype=int)[order]
    keys_per_item = None if keys_per_item is None else [keys_per_item[i] for i in order]
    abstain = None if abstain is None else [bool(abstain[i]) for i in order]
    res = point_metrics(probs, gold, keys_per_item, abstain)
    n = len(probs)
    conf = confidences(probs)
    corr = _mask(correctness(probs, gold), abstain)
    bper = brier_per_item(probs, gold)
    nper = nll_per_item(probs, gold)
    stats = {
        "accuracy": lambda idx: float(corr[idx].mean()),
        "brier": lambda idx: float(bper[idx].mean()),
        "nll": lambda idx: float(nper[idx].mean()),
        "ece_equal_mass_15": lambda idx: ece(conf[idx], corr[idx]),
        "ece_equal_width_15": lambda idx: ece_equal_width(conf[idx], corr[idx]),
        "coverage_at_95_precision": lambda idx: coverage_at_precision(conf[idx], corr[idx]),
        "aurc": lambda idx: aurc(conf[idx], corr[idx]),
    }
    if keys_per_item is not None:
        pk = _pred_keys(probs, keys_per_item, abstain)
        gk = [keys_per_item[i][gold[i]] for i in range(n)]
        stats["macro_f1"] = lambda idx: macro_f1([pk[i] for i in idx], [gk[i] for i in idx])
    res["ci95"] = {k: bootstrap_ci(n, f, seed=seed + j, n_resamples=n_resamples)
                   for j, (k, f) in enumerate(sorted(stats.items()))}
    return res


# ---------------- draws: jitter, draw-sensitivity, k-averaged (prereg A7) ----------------
# jev-1.13.0 is not deterministic (A7). The PRIMARY measurement is draw 1, the first call
# per item (A7.1). Draws 2..k exist only to measure jitter and draw-sensitivity.

class DrawRun:
    """One single-draw run over a fixed item set: the ONLY input an H-verdict may read."""

    def __init__(self, draw: int, probs, gold, item_ids: Sequence[str], keys_per_item=None,
                 abstain=None):
        if not (isinstance(draw, int) and draw >= 1):
            raise ValueError(f"draw must be an int >= 1, got {draw!r}")
        if len(item_ids) != len(probs):
            raise ValueError("DrawRun needs one item_id per item")
        self.draw, self.probs, self.gold = draw, _as_list(probs), np.asarray(gold, dtype=int)
        self.item_ids = [str(x) for x in item_ids]
        self.keys, self.abstain = keys_per_item, abstain


class KAveragedRun:
    """Mean-of-k probabilities (A7.5). DESCRIPTIVE ONLY: no H-test may consume it."""
    descriptive_only = True

    def __init__(self, draw_runs: Sequence[DrawRun]):
        n = len(draw_runs[0].probs)
        self.draws = [r.draw for r in draw_runs]
        self.probs = [np.mean([r.probs[i] for r in draw_runs], axis=0) for i in range(n)]
        self.gold, self.keys = draw_runs[0].gold, draw_runs[0].keys
        # an item abstained on every draw stays an abstention; otherwise the mean is of
        # the draws' vectors (an abstained draw contributes its uniform vector)
        ab = [r.abstain if r.abstain is not None else [False] * n for r in draw_runs]
        self.abstain = [all(a[i] for a in ab) for i in range(n)]

    def metrics(self) -> Dict:
        """Metrics ONLY; point_metrics materializes no verdict (A7.5, CODEX re-grade P0-2)."""
        out = point_metrics(self.probs, self.gold, self.keys, self.abstain)
        out.update({"descriptive_only": True, "draws": self.draws})
        return out


def _require_draw_run(run) -> None:
    if not isinstance(run, DrawRun):
        raise TypeError(f"H-verdicts read single-draw runs only (prereg A7.5); got {type(run).__name__}")


def h1_for_draw(run) -> Dict:
    """The H1 verdict (with the A6.2 binning check) of ONE single-draw run. Type-checked:
    anything that is not a DrawRun, including a KAveragedRun, is refused (A7.5)."""
    _require_draw_run(run)
    m = point_metrics(run.probs, run.gold, run.keys, run.abstain)
    v = _combine_h1_bands(m["ece_equal_mass_15"], m["ece_equal_width_15"])
    return {"draw": run.draw, "ece_equal_mass_15": m["ece_equal_mass_15"],
            "ece_equal_width_15": m["ece_equal_width_15"],
            "band_equal_mass_15": v["band_equal_mass_15"],
            "band_equal_width_15": v["band_equal_width_15"], "h1": v["h1"]}


H2_ACCURACY_POINTS = 0.03  # prereg §5 H2


def h2_for_draw(local, non_local: Dict[str, "DrawRun"], jev_arm: str = "JEV",
                seed: int = 0, local_arm: str = "LOCAL-P") -> Dict:
    """Prereg §5 H2 on ONE draw, per class: LOCAL accuracy within 3 points of the best
    non-local arm AND LOCAL's Brier upper-CI <= JEV's Brier upper-CI. Same type barrier
    as h1_for_draw: every input must be a DrawRun over the same item_ids."""
    _require_draw_run(local)
    for r in non_local.values():
        _require_draw_run(r)
    if jev_arm not in non_local:
        raise ValueError(f"H2 needs the {jev_arm} arm among the non-local arms")
    for name, r in non_local.items():
        if set(r.item_ids) != set(local.item_ids):
            raise ValueError(f"H2 compares arms on the SAME items; {name} differs")

    # ONE score_arm per arm (same seed as the runner's per-arm blocks): the accuracy point
    # and CI and the Brier upper CI below are the numbers the verdict is computed from.
    def scored(r):
        return score_arm(r.probs, r.gold, r.item_ids, r.keys, seed=seed, abstain=r.abstain)
    names = sorted(non_local)
    sc = {n: scored(non_local[n]) for n in names}
    sc_local = scored(local)
    acc = {n: sc[n]["accuracy"] for n in names}
    top = max(acc.values())
    tied = [n for n in names if acc[n] == top]
    best_name = tied[0]
    gap = sc_local["accuracy"] - acc[best_name]
    acc_ok = sc_local["accuracy"] >= acc[best_name] - H2_ACCURACY_POINTS
    local_hi = sc_local["ci95"]["brier"]["hi"]
    jev_hi = sc[jev_arm]["ci95"]["brier"]["hi"]
    brier_ok = local_hi <= jev_hi
    return {"draw": local.draw, "n_items": len(local.item_ids),
            "accuracy": {local_arm: {"point": sc_local["accuracy"],
                                     "ci95": [sc_local["ci95"]["accuracy"]["lo"],
                                              sc_local["ci95"]["accuracy"]["hi"]]},
                         **{n: {"point": acc[n], "ci95": [sc[n]["ci95"]["accuracy"]["lo"],
                                                          sc[n]["ci95"]["accuracy"]["hi"]]}
                            for n in names}},
            "best_non_local": best_name,
            "best_non_local_rule": "the non-local arm with the highest accuracy on these items (prereg §5)",
            "tied_at_best": tied,
            "tie_rule": ("On an accuracy tie the accuracy condition is identical for every tied arm "
                         "(it reads only the best accuracy value) and the Brier condition always reads "
                         f"{jev_arm}, so no tie-break can make the bar easier or harder; the verdict is "
                         "the same whichever tied arm is named. The first tied arm by name is named."),
            "accuracy_gap_local_minus_best": gap, "accuracy_band": H2_ACCURACY_POINTS,
            "brier_upper_ci": {local_arm: local_hi, jev_arm: jev_hi},
            "accuracy_within_3_points": bool(acc_ok), "brier_upper_ci_le_jev": bool(brier_ok),
            "h2": "matches on this class" if (acc_ok and brier_ok) else "does not"}


def claim_across_draws(verdicts: Sequence[str]) -> str:
    """A7.4: a verdict is claimed only if it is identical on every single-draw run."""
    vs = list(verdicts)
    if not vs:
        raise ValueError("no draws")
    return vs[0] if all(v == vs[0] for v in vs) else "draw-sensitive"


def h1_across_draws(draw_runs: Sequence[DrawRun]) -> Dict:
    per = [h1_for_draw(r) for r in draw_runs]
    return {"per_draw": per, "h1": claim_across_draws([p["h1"] for p in per]),
            "draws": [p["draw"] for p in per]}


def jitter(draw_probs: Sequence[Sequence[Sequence[float]]],
           draw_abstain: Optional[Sequence[Sequence[bool]]] = None) -> Dict:
    """A7.3 per-arm jitter. draw_probs[i] = the list of that item's probability vectors,
    one per draw (>= 2). |dp| is taken over every option and every PAIR of draws;
    an abstained draw is excluded from |dp| and counts as the answer '<abstain>'.
    flip share = share of items whose predicted answer is not identical on all draws."""
    diffs, flips, n_items = [], 0, 0
    for i, vecs in enumerate(draw_probs):
        if len(vecs) < 2:
            continue
        n_items += 1
        ab = list(draw_abstain[i]) if draw_abstain is not None else [False] * len(vecs)
        answers = ["<abstain>" if ab[d] else int(np.argmax(v)) for d, v in enumerate(vecs)]
        if len(set(answers)) > 1:
            flips += 1
        ok = [np.asarray(v, dtype=float) for d, v in enumerate(vecs) if not ab[d]]
        for a in range(len(ok)):
            for b in range(a + 1, len(ok)):
                diffs.extend(np.abs(ok[a] - ok[b]).tolist())
    return {"items_with_draws": n_items,
            "mean_abs_dp": _mean(diffs) if diffs else None,
            "max_abs_dp": float(max(diffs)) if diffs else None,
            "answer_flip_share": (flips / n_items) if n_items else None}
