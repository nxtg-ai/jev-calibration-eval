"""Synthetic, seeded, CALIBRATED probability fixtures for the scorer's own tests.

Calibrated by construction: each item's probability vector p is drawn from a
Dirichlet, then its gold label is drawn from Categorical(p). Sharpening such a
set can only move it AWAY from calibration, which is what makes the prereg §6
mutation property (sharpen x2 must raise ECE and Brier) a valid test rather than
a coincidence of one fixture. No model and no real record is involved.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def calibrated_set(n: int, k: int, seed: int, alpha: float = 1.0) -> Tuple[List[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.full(k, alpha), size=n)
    gold = np.array([rng.choice(k, p=p) for p in probs], dtype=int)
    return [p for p in probs], gold


def sharpen(probs, factor: float = 2.0) -> List[np.ndarray]:
    """p -> p**factor, renormalised (factor 2 == temperature 0.5 on log p)."""
    out = []
    for p in probs:
        q = np.asarray(p, dtype=float) ** factor
        out.append(q / q.sum())
    return out


def flip_correct_labels(probs, gold) -> np.ndarray:
    """Move the gold label off the argmax on every currently-correct item."""
    g = np.array(gold, dtype=int).copy()
    for i, p in enumerate(probs):
        pred = int(np.argmax(p))
        if g[i] == pred:
            g[i] = (pred + 1) % len(p)
    return g
