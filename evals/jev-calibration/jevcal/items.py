"""Items schema (E.2's, the study author), loader, and the eval-data guard.

One JSONL row per item, exactly these fields (E.2 freezes items-v1.jsonl to them):

    item_id        str   unique within the file
    slice          str   "S1" | "S2a" | "S2b" | "S2c" (fixtures use "FX-*")
    question_type  str   "choice" | "noul" | "score"
    instructions   str   the canonical instruction text (user-seen, identical for every arm)
    options        list  ordered option keys (str), 2..20. Order is fixed per item and
                         identical across arms (prereg §2).
                         noul  -> exactly 2, ordered [yes-side, no-side]; Jev's noul
                                  value is the probability of options[0]
                         score -> ordered levels, lowest first
    criteria       obj|null  option descriptions (carried to Jev as `criteria`, and shown to
                         every other arm beside the lettered option, so no arm sees text
                         another does not). choice/score: {option_key: str|null}.
                         noul: {"true": str, "false": str} or keyed by the two option keys.
    state          str   the content the decision is about, <= 300 words (prereg / E.2)
    gold           str   one of `options`; from a human or human-ruled source, never a model
    split          str   "calibration" | "test" (the frozen file's "cal" is canonicalised to
                         "calibration" on parse — see SPLIT_ALIASES)
    source_ref     str   provenance of the item and its gold
    tags           list  of str (e.g. "counting", "date-order", "numeric" sub-slice tags)

The frozen items-v1.jsonl also carries two provenance-only keys for build audit —
`build_seed` and `leak_strings` (PROVENANCE_ONLY_FIELDS). parse_item accepts and DROPS them;
they never reach Item, render, or any arm prompt. Any OTHER unknown key is still refused.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

MAX_OPTIONS = 20      # prereg §3: option count <= 20 (single-token letter labels A..T)
MAX_STATE_WORDS = 300  # E.2 item construction: passage window <= 300 words
TYPES = ("choice", "noul", "score")
SPLITS = ("calibration", "test")
# The E.2 frozen items-v1.jsonl abbreviates the calibration split as "cal" (67 of 340 rows);
# canonicalise it to "calibration" at parse so every consumer that keys on the split label works
# — notably scorer.temperature_scaled, which fits T on split=="calibration" ONLY (prereg §4).
# Widening SPLITS to accept "cal" instead would load the file but leave that split silently empty.
SPLIT_ALIASES = {"cal": "calibration"}
FIELDS = ("item_id", "slice", "question_type", "instructions", "options", "criteria",
          "state", "gold", "split", "source_ref", "tags")
# Provenance-only keys the frozen items-v1.jsonl carries for build audit. parse_item ACCEPTS
# exactly these two and DROPS them: they never reach Item, render, or any arm prompt. Every
# other unknown key is still refused, exactly as before. (leak_strings are the answer-leak
# phrases the build guaranteed absent from `state`; surfacing them would leak the gold, so
# tests/test_codex_fixes.py's (i) arm asserts none appear in any rendered prompt.)
PROVENANCE_ONLY_FIELDS = ("build_seed", "leak_strings")

# The eval-data guard. E.3 was fixture-only. E.4 (this go-live change) lets a reviewed
# CERTIFYING run use the frozen items, but ONLY when the items file's sha256 equals the single
# committed pin (identity is CONTENT, not path: --mode certifying accepts only the pinned bytes
# at any path — F1; --mode fixture refuses the pinned bytes at any path, symlink/copy included —
# F2). These basenames/dirs are the belt-and-braces path refusal. No env-var escape hatch.
FORBIDDEN_ITEM_BASENAMES = ("items-v1.jsonl",)
FORBIDDEN_ITEM_DIRS = (os.path.join("governance", "evals", "jev-calibration"),)
# The pinned frozen-items hash lives ONLY in this one committed file (sha256sum format).
# The guard reads it; the value is never hard-coded in a second place.
PIN_REL = os.path.join("governance", "evals", "jev-calibration", "items-v1.sha256")
# The committed frozen items themselves. N2 (independent review of PR #89): a whole-file hash cannot
# match a SUBSET, so --mode fixture reads this file (verified against PIN_REL) and refuses any items
# file that reuses a frozen row's item_id OR state — a `head -3 items-v1.jsonl` a copy-hash misses.
FROZEN_REL = os.path.join("governance", "evals", "jev-calibration", "items-v1.jsonl")


class ItemError(ValueError):
    pass


class EvalDataRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class Item:
    item_id: str
    slice: str
    question_type: str
    instructions: str
    options: Tuple[str, ...]
    criteria: Optional[Dict[str, Optional[str]]]
    state: str
    gold: str
    split: str
    source_ref: str
    tags: Tuple[str, ...]

    @property
    def gold_index(self) -> int:
        return self.options.index(self.gold)

    def description(self, i: int) -> str:
        """Option i's description from `criteria`, or "" when none is given."""
        c = self.criteria or {}
        key = self.options[i]
        if self.question_type == "noul" and key not in c:
            key = "true" if i == 0 else "false"
        d = c.get(key)
        return d or ""


def _need_str(row, key, idx):
    v = row.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ItemError(f"row {idx}: '{key}' must be a non-empty string")
    return v


def parse_item(row: dict, idx: int = 0) -> Item:
    if not isinstance(row, dict):
        raise ItemError(f"row {idx}: not a JSON object")
    extra = set(row) - set(FIELDS) - set(PROVENANCE_ONLY_FIELDS)
    if extra:
        raise ItemError(f"row {idx}: unknown fields {sorted(extra)}")
    iid = _need_str(row, "item_id", idx)
    sl = _need_str(row, "slice", idx)
    typ = _need_str(row, "question_type", idx)
    if typ not in TYPES:
        raise ItemError(f"row {idx} ({iid}): question_type {typ!r} not in {TYPES}")
    instr = _need_str(row, "instructions", idx)
    state = _need_str(row, "state", idx)
    if len(state.split()) > MAX_STATE_WORDS:
        raise ItemError(f"row {idx} ({iid}): state is {len(state.split())} words > {MAX_STATE_WORDS}")
    opts = row.get("options")
    if not isinstance(opts, list) or not (2 <= len(opts) <= MAX_OPTIONS):
        raise ItemError(f"row {idx} ({iid}): options must be a list of 2..{MAX_OPTIONS}")
    if not all(isinstance(o, str) and o.strip() for o in opts):
        raise ItemError(f"row {idx} ({iid}): every option must be a non-empty string")
    if len(set(opts)) != len(opts):
        raise ItemError(f"row {idx} ({iid}): duplicate options")
    if typ == "noul" and len(opts) != 2:
        raise ItemError(f"row {idx} ({iid}): noul needs exactly 2 options [yes-side, no-side]")
    crit = row.get("criteria")
    if crit is not None:
        if not isinstance(crit, dict):
            raise ItemError(f"row {idx} ({iid}): criteria must be an object or null")
        allowed = set(opts) | ({"true", "false"} if typ == "noul" else set())
        bad = set(crit) - allowed
        if bad:
            raise ItemError(f"row {idx} ({iid}): criteria keys {sorted(bad)} are not options")
        if not all(v is None or isinstance(v, str) for v in crit.values()):
            raise ItemError(f"row {idx} ({iid}): criteria values must be str or null")
    gold = _need_str(row, "gold", idx)
    if gold not in opts:
        raise ItemError(f"row {idx} ({iid}): gold {gold!r} is not an option")
    split = _need_str(row, "split", idx)
    split = SPLIT_ALIASES.get(split, split)   # frozen "cal" -> canonical "calibration"
    if split not in SPLITS:
        raise ItemError(f"row {idx} ({iid}): split {split!r} not in {SPLITS}")
    src = _need_str(row, "source_ref", idx)
    tags = row.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise ItemError(f"row {idx} ({iid}): tags must be a list of strings")
    return Item(iid, sl, typ, instr, tuple(opts), dict(crit) if crit else None, state, gold,
                split, src, tuple(tags))


def _repo_root() -> str:
    # items.py is evals/jev-calibration/jevcal/items.py; four parents up is the repo root.
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _read_pin(mode: str = "fixture") -> Optional[str]:
    """The frozen items' pinned sha256, read from the one committed pin file
    (governance/evals/jev-calibration/items-v1.sha256, sha256sum format). A missing or empty
    file returns None and the caller refuses.

    N1 (independent review of PR #89): in --mode certifying the pin is authoritative — it is what
    lets a run write a cert-ledger row over the frozen items — so it is read ONLY from the repo
    that ships items.py (_repo_root) and ASIF_ROOT is IGNORED. ASIF_ROOT is a real operational
    variable; honouring it here let a certifying run be pointed at a directory whose pin matches an
    arbitrary SUBSET, which the guard would then accept. Tests inject an alternate repo root by
    monkeypatching _repo_root — a mechanism production code cannot reach from the environment."""
    root = _repo_root() if mode == "certifying" else (os.environ.get("ASIF_ROOT") or _repo_root())
    pin_path = os.path.join(root, PIN_REL)
    if not os.path.isfile(pin_path):
        return None
    with open(pin_path, "r", encoding="utf-8") as fh:
        first = fh.readline().split()
    return first[0].strip().lower() if first else None


def _content_refuse(path: str, mode: str, pinned: Optional[str], actual: Optional[str]) -> bool:
    """F1 + F2 CONTENT GATE (independent review of PR #89): the frozen eval items are identified by
    their committed sha256, NOT by their path. --mode certifying accepts ONLY the pinned content,
    at ANY name or location (F1); --mode fixture refuses the pinned content wherever it sits — a
    symlink or a byte-copy at any path included (F2). Raises EvalDataRefused to refuse; returns
    True when the input is the pinned certifying content (guard then allows and returns); returns
    False to fall through to the belt-and-braces path gate."""
    if mode == "certifying":
        if pinned is None:
            raise EvalDataRefused(
                f"refusing {path}: the committed items pin ({PIN_REL}) is missing or empty; a "
                "certifying run requires it")
        if actual != pinned:
            raise EvalDataRefused(
                f"refusing {path}: items sha256 {actual} does not match the pinned frozen-items "
                f"hash {pinned} in {PIN_REL}; --mode certifying accepts only the pinned items, at "
                "any path, and refuses before any arm is constructed")
        return True
    if pinned is not None and actual == pinned:
        raise EvalDataRefused(
            f"refusing {path}: --mode fixture refuses the frozen eval items (committed sha256 "
            f"{pinned}) wherever they sit — a symlink or byte-copy included; they require "
            "--mode certifying")
    return False


def _frozen_row_overlap(path: str) -> Optional[str]:
    """N2 (independent review of PR #89): a whole-file hash cannot catch a PARTIAL copy of the frozen
    eval set (`head -3 items-v1.jsonl > x.jsonl`). Load the committed frozen set, VERIFIED against
    its pin, and return a message naming the first row of `path` whose item_id OR state matches a
    frozen row, else None. Read the frozen set + pin from _repo_root() only (ignore ASIF_ROOT — the
    committed frozen set is authoritative; N1). No committed frozen set present -> None (nothing to
    protect); a frozen set whose bytes do not match its pin -> refuse (never check against an
    unverified set)."""
    root = _repo_root()
    pin_path = os.path.join(root, PIN_REL)
    frozen = os.path.join(root, FROZEN_REL)
    if not (os.path.isfile(pin_path) and os.path.isfile(frozen)):
        return None
    with open(pin_path, "r", encoding="utf-8") as fh:
        first = fh.readline().split()
    pinned = first[0].strip().lower() if first else None
    if pinned is None:
        return None
    if file_sha256(frozen) != pinned:
        raise EvalDataRefused(
            f"refusing {path}: the committed frozen items ({FROZEN_REL}) do not match their pin "
            f"({PIN_REL}); the per-row fixture guard will not run against an unverified frozen set")
    ids, states = set(), set()
    with open(frozen, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                ids.add(r.get("item_id"))
                states.add(r.get("state"))
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = r.get("item_id")
            if iid in ids:
                return f"row {iid!r} is a frozen eval item (item_id matches the pinned frozen set)"
            if r.get("state") in states:
                return f"row {iid!r} reuses a frozen eval item's state (state matches the pinned frozen set)"
    return None


def guard_not_eval_data(path: str, mode: str = "fixture") -> None:
    """Gate the items file BEFORE any arm is constructed or called. The frozen eval items are
    identified by CONTENT (the committed pin) first — so a rename, symlink or copy cannot slip
    them past (F2), and a certifying run cannot run on arbitrary items (F1) — then by name/dir
    as belt-and-braces (the original E.3 refusal), then row-by-row against the frozen set so a
    PARTIAL copy a whole-file hash cannot catch is refused too (N2)."""
    ap = os.path.abspath(path)
    pinned = _read_pin(mode)
    actual = file_sha256(ap) if os.path.isfile(ap) else None
    if _content_refuse(path, mode, pinned, actual):
        return
    looks_frozen = os.path.basename(ap) in FORBIDDEN_ITEM_BASENAMES or any(
        (os.sep + d + os.sep) in ap for d in FORBIDDEN_ITEM_DIRS)
    if looks_frozen:
        raise EvalDataRefused(
            f"refusing {path}: --mode fixture refuses the frozen eval items by name/dir; they "
            "require --mode certifying (which verifies the committed pin)")
    # Reaching here means --mode fixture with non-pinned content (certifying already returned or
    # raised in _content_refuse). Refuse a partial/renamed copy of the frozen rows BEFORE any arm.
    overlap = _frozen_row_overlap(ap) if os.path.isfile(ap) else None
    if overlap is not None:
        raise EvalDataRefused(
            f"refusing {path}: {overlap}; --mode fixture refuses any file containing a row of the "
            "frozen eval set (a partial or renamed copy a whole-file hash cannot catch) — use "
            "--mode certifying")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_items(path: str, mode: str = "fixture") -> List[Item]:
    guard_not_eval_data(path, mode)
    items, seen = [], set()
    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, 1):
            if not line.strip():
                continue
            it = parse_item(json.loads(line), idx)
            if it.item_id in seen:
                raise ItemError(f"row {idx}: duplicate item_id {it.item_id!r}")
            seen.add(it.item_id)
            items.append(it)
    if not items:
        raise ItemError(f"{path}: no items")
    return items
