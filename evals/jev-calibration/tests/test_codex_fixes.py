"""Arms for CODEX's PR 72 HOLD findings 2-5. Real harness code;
external seams only are replaced: the TypeSafe HTTP transport, the claude subprocess,
the LangFuse helper, and `git` (so the runner's clean-tree check does not depend on
whether this checkout is mid-edit or mid-mutation). tests/mutate_harness.py reverts
each fix and requires its arm to go RED.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from jevcal import items as ITEMS  # noqa: E402
from jevcal import run as R  # noqa: E402
from jevcal.arms import frontier as FR  # noqa: E402
from jevcal.arms.frontier import FrontierArm  # noqa: E402
from jevcal.arms.jev import JevArm, build_question  # noqa: E402
from jevcal.arms import local as LOCAL  # noqa: E402
from jevcal.items import (EvalDataRefused, ItemError, file_sha256,  # noqa: E402
                          guard_not_eval_data, load_items, parse_item)
from jevcal.render import render  # noqa: E402

FIXTURE = os.path.join(ROOT, "fixtures", "items-fixture.jsonl")
REPO = os.path.dirname(os.path.dirname(ROOT))
REAL_ITEMS_V1 = os.path.join(REPO, "governance", "evals", "jev-calibration", "items-v1.jsonl")
REAL_PIN_V1 = os.path.join(REPO, "governance", "evals", "jev-calibration", "items-v1.sha256")
# Public release: the pinned frozen file here holds the 160 S1 rows (31 of them calibration split)
# of the 340-row private set (67 calibration). The S2 rows are private and are not released.
FROZEN_N, FROZEN_CAL_N = 160, 31


@contextmanager
def certifying_pin(items_file):
    """Point items._repo_root at an isolated repo whose committed pin == sha256(items_file), so a
    --mode certifying run over that file passes the F1 content gate without touching the real store.
    Used by the certifying-mode machinery tests that exercise the fixture rather than the real frozen
    items. Monkeypatches _repo_root — NOT ASIF_ROOT — because certifying now reads the pin only from
    the repo root and ignores ASIF_ROOT (N1). The monkeypatch is a mechanism production code cannot
    reach from the environment."""
    d = tempfile.mkdtemp()
    sub = os.path.join(d, "governance", "evals", "jev-calibration")
    os.makedirs(sub)
    with open(os.path.join(sub, "items-v1.sha256"), "w") as fh:
        fh.write(file_sha256(items_file) + "  " + os.path.basename(items_file) + "\n")
    with mock.patch.object(ITEMS, "_repo_root", return_value=d):
        try:
            yield d
        finally:
            shutil.rmtree(d, ignore_errors=True)


def fake_git(*args):
    return "0" * 40 if args[:1] == ("rev-parse",) else ""


class KeyFile:
    def __enter__(self):
        self.d = tempfile.mkdtemp()
        p = os.path.join(self.d, "k.env")
        with open(p, "w") as fh:
            fh.write("TYPESAFE_API_KEY=test-key-not-real\n")
        self.old = os.environ.get("JEV_KEY_FILE")
        os.environ["JEV_KEY_FILE"] = p

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop("JEV_KEY_FILE", None)
        else:
            os.environ["JEV_KEY_FILE"] = self.old


def jev_stub(calls):
    """A TypeSafe stand-in answering any fixture question with a valid distribution."""
    def t(url, data, headers, timeout):
        body = json.loads(data)
        calls.append(body)
        q = body["questions"]["q"]
        if q["type"] == "noul":
            ans = {"type": "noul", "noul": 0.8}
        elif q["type"] == "choice":
            keys = list(q["criteria"])
            ans = {"type": "choice", "choice": keys[0], "confidence": 0.5,
                   "probabilities": {k: (0.8 if i == 0 else 0.2 / (len(keys) - 1)) for i, k in enumerate(keys)}}
        else:
            n = len(q["criteria"])
            ans = {"type": "score", "score": 0, "confidence": 0.5, "legend": {},
                   "probabilities": {str(i): (0.8 if i == n - 1 else 0.2 / (n - 1)) for i in range(n)}}
        out = {"model": "jev-1.13.0", "answers": {"q": ans}, "usage": {"input_tokens": 10, "output_tokens": 2}}
        return 200, {"content-type": "application/json"}, json.dumps(out).encode()
    return t


class FakeLF:
    """LangFuse helper stand-in: traces the first call, then fails every later one."""
    def __init__(self, fail_from=0):
        self.n, self.fail_from = 0, fail_from

    def trace_eval_run(self, *, run_id, model_id, task_input, model_call, judge_call, judge_id, **kw):
        self.n += 1
        if self.n > self.fail_from:
            raise RuntimeError("injected LangFuse helper failure")
        out = model_call()
        judge_call(out)
        return {"langfuse_trace_ids": [f"trace-{self.n}"]}


def frontier_stub_runner(argv, env, cwd, timeout):
    """claude -p stand-in: verbalised probabilities, or a single letter for the A4.1 secondary."""
    schema = json.loads(argv[argv.index("--json-schema") + 1])
    if "answer" in schema["properties"]:
        so = {"answer": schema["properties"]["answer"]["enum"][0]}
    else:
        labels = list(schema["properties"]["probabilities"]["properties"])
        so = {"probabilities": {l: (0.7 if i == 0 else 0.3 / (len(labels) - 1)) for i, l in enumerate(labels)}}
    return 0, json.dumps([
        {"type": "system", "subtype": "init", "model": "claude-opus-5-5"},
        {"type": "result", "is_error": False, "result": "", "structured_output": so,
         "modelUsage": {"claude-opus-5-5": {"outputTokens": 5}}}]), ""


@contextmanager
def frontier_companion(runner=frontier_stub_runner):
    """D1: certifying mode refuses a single-arm run (GATE E needs two arms). Certifying-mode
    machinery tests that used to run JEV alone add a stubbed FRONTIER through this: the fixture's
    6 ids stand in for the committed A8 subset (git reports the temp file as tracked). Yields the
    git side effect the caller patches R.git with."""
    ids = [it.item_id for it in load_items(FIXTURE)]
    fd, p = tempfile.mkstemp(suffix=".txt", dir=os.path.join(ROOT, "config"))
    with os.fdopen(fd, "w") as fh:
        fh.write("".join(i + "\n" for i in ids))
    rel = os.path.relpath(p, ROOT)

    def tracked(*args):
        return rel if args[:1] == ("ls-files",) else fake_git(*args)
    try:
        with mock.patch.object(R, "FRONTIER_SUBSET_PATH", p), \
                mock.patch.object(R, "CERT_FRONTIER_SUBSET_N", len(ids)), \
                mock.patch.dict(R.ARMS, {"FRONTIER": lambda **kw: FrontierArm(effort="high", runner=runner)}):
            yield tracked
    finally:
        os.remove(p)


def run_jev_only(evals_dir, lfs):
    """JEV (spied) plus a stubbed FRONTIER companion: D1 refuses a single-arm certifying run."""
    calls = []
    # certifying_pin makes FIXTURE the pinned content (F1), so certifying accepts it here.
    with certifying_pin(FIXTURE), KeyFile(), frontier_companion() as tracked, \
            mock.patch.object(R, "git", side_effect=tracked), \
            mock.patch.dict(R.ARMS, {"JEV": lambda **kw: JevArm(transport=jev_stub(calls))}), \
            mock.patch.object(R, "langfuse_setup", return_value=lfs):
        rc = R.main(["--items", FIXTURE, "--arms", "JEV,FRONTIER", "--frontier-effort", "high",
                     "--evals-dir", evals_dir,
                     "--run-id", "t", "--purpose", "test", "--mode", "certifying", "--draws", "3"])
    rec_path = os.path.join(evals_dir, "runs", "t.json")
    rec = None
    if os.path.exists(rec_path):
        with open(rec_path) as fh:
            rec = json.load(fh)
    ledger = os.path.exists(os.path.join(evals_dir, "metric-history.jsonl"))
    return rc, rec, ledger, calls


class ArmEffortPin(unittest.TestCase):
    """Finding 2: the registered FRONTIER arm accepts ONLY prereg A5.1's 'high'."""

    def test_non_pinned_effort_refused_before_any_spawn(self):
        with mock.patch.object(FR.subprocess, "run") as spy:
            for bad in ("low", "xhigh", "medium", "max", None):
                with self.assertRaises(ValueError, msg=repr(bad)):
                    FrontierArm(effort=bad)
            with tempfile.TemporaryDirectory() as d:
                for bad in ("low", "xhigh"):
                    with self.assertRaises(SystemExit):
                        # fixture mode: the effort guard (mode-independent) fires before any call;
                        # certifying would refuse the fixture at the F1 content gate first.
                        R.main(["--items", FIXTURE, "--arms", "FRONTIER", "--frontier-effort", bad,
                                "--evals-dir", d, "--run-id", "x", "--purpose", "p", "--mode", "fixture"])
                self.assertEqual(os.listdir(d), [])
            self.assertEqual(spy.call_count, 0)

    def test_pinned_effort_reaches_the_child(self):
        self.assertEqual(FR.PINNED_EFFORT, "high")
        events = json.dumps([
            {"type": "system", "subtype": "init", "model": "claude-opus-5-5"},
            {"type": "result", "is_error": False, "result": "",
             "structured_output": {"probabilities": {"A": .8, "B": .1, "C": .1}},
             "modelUsage": {"claude-opus-5-5": {"outputTokens": 5}}}])
        done = subprocess.CompletedProcess(args=[], returncode=0, stdout=events, stderr="")
        with mock.patch.object(FR.subprocess, "run", return_value=done) as spy:
            FrontierArm(effort="high").call(render(load_items(FIXTURE)[0]))
        argv = spy.call_args[0][0]
        self.assertEqual(argv[argv.index("--effort") + 1], "high")


def all_strings(obj):
    if isinstance(obj, dict):
        return [s for k, v in obj.items() for s in [str(k)] + all_strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in all_strings(v)]
    return [obj] if isinstance(obj, str) else []


class ArmPromptParity(unittest.TestCase):
    """Finding 3: every option key AND description reaches every arm, for every type."""

    def test_keys_and_descriptions_reach_every_arm(self):
        seen_types = set()
        frontier = FrontierArm(effort="high")
        for it in load_items(FIXTURE):
            r = render(it)
            seen_types.add(r.type)
            inputs = {
                "JEV": "\n".join(all_strings({"state": r.state, "question": build_question(r)})),
                "FRONTIER": frontier.prompt(r),
            }
            for arm, text in inputs.items():
                for _, key, desc in r.options:
                    self.assertIn(key, text, f"{arm} {it.item_id} ({r.type}) lacks option key {key!r}")
                    if desc:
                        self.assertIn(desc, text, f"{arm} {it.item_id} ({r.type}) lacks description {desc!r}")
                self.assertIn(r.instruction, text)
        self.assertEqual(seen_types, {"choice", "noul", "score"})


class ArmLangfuseFailClosed(unittest.TestCase):
    """Finding 4: no trace ids and no valid unreachable proof -> INVALID record, rc != 0."""

    def test_helper_error_aborts_before_any_call(self):
        with tempfile.TemporaryDirectory() as d:
            rc, rec, ledger, calls = run_jev_only(d, {"mode": "error", "error": "injected helper failure"})
        self.assertEqual(rc, R.EXIT_INVALID)
        self.assertFalse(rec["valid"])
        self.assertIn("injected helper failure", rec["invalid_reason"])
        self.assertEqual(calls, [])
        self.assertFalse(ledger)

    def test_tracing_that_fails_midrun_invalidates_the_record(self):
        with tempfile.TemporaryDirectory() as d:
            rc, rec, ledger, calls = run_jev_only(d, {"mode": "trace", "lf": FakeLF(fail_from=1),
                                                     "env_file": None})
        self.assertEqual(rc, R.EXIT_INVALID)
        self.assertFalse(rec["valid"])
        self.assertIn("tracing incomplete", rec["invalid_reason"])
        self.assertEqual(rec["langfuse_trace_ids"], ["trace-1"])   # partial tracing is still invalid
        self.assertFalse(ledger)

    def test_control_proven_outage_is_valid(self):
        proof = {"checked_at": "2026-09-30T00:00:00Z", "endpoint": "http://x/api/public/health",
                 "probe_result": "error:URLError", "attempts": 3}
        with tempfile.TemporaryDirectory() as d:
            rc, rec, ledger, calls = run_jev_only(d, {"mode": "unreachable", "proof": proof})
        self.assertEqual(rc, 0)
        self.assertTrue(rec["valid"])
        self.assertTrue(ledger)
        self.assertEqual(len(calls), 18)            # certifying: k=3 over all 6 items
        # the run record carries exactly what the committed config says; the harness never sets it
        self.assertIs(rec["pins"]["abstention_rubric"]["operator_endorsed"],
                      R.load_abstention_rubric()["operator_endorsed"])
        self.assertEqual(rec["pins"]["abstention_rubric"]["config_path"],
                         "evals/jev-calibration/config/abstention-rubric.json")


class ArmRubricConfig(unittest.TestCase):
    """Finding 5 / A6.4: the rubric and operator_endorsed are READ from git-tracked config."""

    def write(self, obj):
        fd, p = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh)
        return p

    def test_committed_config_endorsement_is_bound_to_the_rubric_text(self):
        # Endorsed 2026-09-30 by the study author under a delegated sign-off (prereg A8.3 status note).
        # An endorsement must name its signer and the sha256 of the exact rubric text it endorsed,
        # so any later edit to the rubric text fails this test until someone re-endorses.
        import hashlib
        r = R.load_abstention_rubric()
        self.assertIsInstance(r["operator_endorsed"], bool)
        if r["operator_endorsed"]:
            self.assertTrue(isinstance(r["endorsed_by"], str) and r["endorsed_by"].strip())
            self.assertIn(hashlib.sha256(r["definition"].encode()).hexdigest(), r["endorsement_ref"] or "")
        self.assertIn("uniform distribution for Brier and NLL", r["definition"])
        self.assertIn("incorrect at confidence 1/K", r["definition"])

    def test_missing_field_defaults_false_and_value_is_never_rewritten(self):
        self.assertIs(R.load_abstention_rubric(self.write({"rubric": "x"}))["operator_endorsed"], False)
        self.assertIs(R.load_abstention_rubric(
            self.write({"rubric": "x", "operator_endorsed": True}))["operator_endorsed"], True)

    def test_non_bool_refused(self):
        with self.assertRaises(SystemExit):
            R.load_abstention_rubric(self.write({"rubric": "x", "operator_endorsed": "yes"}))


class E4Env:
    """Isolated ASIF_ROOT for the E.4 guards. Optionally writes the committed pin (the real
    frozen-items hash). Helpers drop an items file at a chosen basename with chosen content (real /
    tampered / arbitrary) as a regular file or a symlink to the real frozen file, so the guard sees
    it by CONTENT and by PATH. No writes to the real governance/evals; no live calls."""

    def __init__(self, with_pin=True):
        self.with_pin = with_pin

    def __enter__(self):
        self.d = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.d, "governance", "evals", "jev-calibration"))
        if self.with_pin:
            shutil.copyfile(REAL_PIN_V1, os.path.join(
                self.d, "governance", "evals", "jev-calibration", "items-v1.sha256"))
        self.evals_dir = os.path.join(self.d, "governance", "evals")
        self._old = os.environ.get("ASIF_ROOT")
        os.environ["ASIF_ROOT"] = self.d
        return self

    def __exit__(self, *a):
        if self._old is None:
            os.environ.pop("ASIF_ROOT", None)
        else:
            os.environ["ASIF_ROOT"] = self._old
        shutil.rmtree(self.d, ignore_errors=True)

    def real_at(self, basename):          # exact copy of the frozen file: content == pin
        p = os.path.join(self.d, basename)
        shutil.copyfile(REAL_ITEMS_V1, p)
        return p

    def symlink_at(self, basename):       # symlink to the frozen file: content == pin via target
        p = os.path.join(self.d, basename)
        os.symlink(REAL_ITEMS_V1, p)
        return p

    def tampered_at(self, basename):      # frozen copy with one byte flipped: content != pin
        p = self.real_at(basename)
        with open(p, "r+b") as fh:
            b = fh.read(1)
            fh.seek(0)
            fh.write(bytes([b[0] ^ 0x01]))
        return p

    def arbitrary_at(self, basename):     # small non-frozen content: content != pin
        p = os.path.join(self.d, basename)
        with open(p, "w") as fh:
            fh.write('{"item_id": "x"}\n')
        return p


class ArmE4FixtureRefusesFrozen(unittest.TestCase):
    """E.4 (a): --mode fixture refuses a file at the frozen NAME/DIR (belt-and-braces path gate),
    even when its content is not the pinned bytes. mutate_harness reverts the path gate -> RED."""

    def test_fixture_mode_refuses_the_frozen_items_path(self):
        with E4Env(with_pin=True) as env:
            f = env.arbitrary_at("items-v1.jsonl")            # frozen basename, non-pinned content
            with self.assertRaises(EvalDataRefused) as cm:
                guard_not_eval_data(f, mode="fixture")
            self.assertIn("name/dir", str(cm.exception))


class ArmE4HashMismatch(unittest.TestCase):
    """E.4 (c): --mode certifying refuses a tampered frozen items file, naming the hash mismatch.
    mutate_harness drops the certifying content-match check -> RED."""

    def test_certifying_refuses_tampered_items_naming_the_mismatch(self):
        with E4Env(with_pin=True) as env:
            f = env.tampered_at("items-v1.jsonl")
            with self.assertRaises(EvalDataRefused) as cm:
                guard_not_eval_data(f, mode="certifying")
            msg = str(cm.exception)
            self.assertIn("does not match", msg)
            self.assertIn("pinned frozen-items hash", msg)


class ArmE4FixtureRefusesRealStore(unittest.TestCase):
    """E.4 (d): --mode fixture refuses the real eval store. mutate_harness reverts the certifying
    gate in refuse_real_evals_dir -> RED."""

    def test_fixture_mode_refuses_the_real_eval_store(self):
        with E4Env(with_pin=True) as env:
            with self.assertRaises(SystemExit):
                R.refuse_real_evals_dir(env.evals_dir, mode="fixture")


class ArmE4MissingPin(unittest.TestCase):
    """E.4 (f): --mode certifying refuses when the committed pin is missing, naming it, even for a
    valid-hash items file. mutate_harness treats a missing pin as OK -> RED. Monkeypatches _repo_root
    at a pin-less repo (certifying reads the pin only from the repo root and ignores ASIF_ROOT, N1)."""

    def test_certifying_refuses_when_pin_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "items-v1.jsonl")
            shutil.copyfile(REAL_ITEMS_V1, f)                  # valid-hash items, but no pin in this repo
            with mock.patch.object(ITEMS, "_repo_root", return_value=d):
                with self.assertRaises(EvalDataRefused) as cm:
                    guard_not_eval_data(f, mode="certifying")
            self.assertIn("missing", str(cm.exception))


class ArmE4CertifyingRequiresPinnedContent(unittest.TestCase):
    """E.4 F1 (reviewer): --mode certifying refuses an UNPINNED file at a non-frozen path, so a
    certifying run cannot write a cert-ledger row over arbitrary items. mutate_harness deletes the
    content gate (restoring path-only) -> RED (together with the two F2 arms below)."""

    def test_certifying_refuses_an_unpinned_file_at_any_path(self):
        with E4Env(with_pin=True) as env:
            f = env.arbitrary_at("decoy.jsonl")               # non-frozen path, content != pin
            with self.assertRaises(EvalDataRefused) as cm:
                guard_not_eval_data(f, mode="certifying")
            self.assertIn("does not match", str(cm.exception))


class ArmE4FixtureRefusesSymlink(unittest.TestCase):
    """E.4 F2 (reviewer): --mode fixture refuses a SYMLINK to the frozen file at a non-frozen path
    (content identity). mutate_harness deletes the content gate -> RED."""

    def test_fixture_mode_refuses_a_symlink_to_the_frozen_file(self):
        with E4Env(with_pin=True) as env:
            f = env.symlink_at("link.jsonl")                  # non-frozen basename, pinned content
            with self.assertRaises(EvalDataRefused) as cm:
                guard_not_eval_data(f, mode="fixture")
            self.assertIn("committed sha256", str(cm.exception))


class ArmE4FixtureRefusesCopy(unittest.TestCase):
    """E.4 F2 (reviewer): --mode fixture refuses a byte-COPY of the frozen file at a non-frozen path
    (content identity). mutate_harness deletes the content gate -> RED."""

    def test_fixture_mode_refuses_a_byte_copy_at_another_path(self):
        with E4Env(with_pin=True) as env:
            f = env.real_at("copy.jsonl")                     # non-frozen basename, pinned content
            with self.assertRaises(EvalDataRefused) as cm:
                guard_not_eval_data(f, mode="fixture")
            self.assertIn("committed sha256", str(cm.exception))


class ArmE4ProvenanceUnknownKeyRefused(unittest.TestCase):
    """E.4 (h): parse_item accepts EXACTLY build_seed + leak_strings and drops them; any OTHER
    unknown key is still refused. mutate_harness accepts any unknown key -> RED."""

    def test_provenance_dropped_but_other_unknown_key_refused(self):
        with open(FIXTURE) as fh:
            good = json.loads(fh.readline())
        it = parse_item(dict(good, build_seed=20260929, leak_strings=["x"]), 1)   # accepted + dropped
        self.assertFalse(hasattr(it, "build_seed") or hasattr(it, "leak_strings"))
        with self.assertRaises(ItemError):
            parse_item(dict(good, some_other_key=1), 1)                            # still refused


class ArmE4NoLeakInPrompt(unittest.TestCase):
    """E.4 (i): for EVERY frozen item, no leak_string appears in the rendered STATE — the shipped
    `leaks_in_state` invariant (internal build notes, BUILD.md:77, not released: word-boundary, case-insensitive). The check is scoped
    to STATE, not the whole prompt: the QUESTION/options deliberately name the clause type being
    asked about (internal BUILD.md:129-130), so a whole-prompt check false-positives on every noul item
    (measured: 341 hits) while STATE is clean (0). The state renders verbatim into every arm's
    prompt (asserted below), so a state-clean check covers what each arm consumes.
    mutate_harness surfaces leak_strings into the rendered state -> RED."""

    def test_no_leak_string_in_the_rendered_state(self):
        leaks = {}
        with open(REAL_ITEMS_V1, encoding="utf-8") as fh:      # read-only; no writes to governance/
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    leaks[row["item_id"]] = [s for s in (row.get("leak_strings") or []) if s]
        items = load_items(REAL_ITEMS_V1, mode="certifying")
        self.assertEqual(len(items), FROZEN_N)
        checked = 0
        for it in items:
            r = render(it)
            self.assertIn(it.state, r.canonical_text)          # state renders verbatim into every arm's prompt
            self.assertIn(it.state, LOCAL.LocalArm.prompt(None, r))
            for s in leaks.get(it.item_id, []):
                checked += 1
                self.assertIsNone(
                    re.search(r"\b" + re.escape(s) + r"\b", r.state, re.IGNORECASE),
                    f"leak_string in rendered state for {it.item_id}: {s!r}")
        self.assertGreater(checked, 0)                         # positive control: leak_strings were checked


class ArmE4CertifyingAllows(unittest.TestCase):
    """E.4 (b) + (e) + (g): the allow paths (not mutation arms). (b) pinned content passes the item
    guard in certifying at ANY path; (e) certifying allows the real store; (g) the real frozen file
    loads 340 items in certifying (proves parse_item drops the provenance keys)."""

    def test_b_certifying_allows_pinned_content_at_any_path(self):
        with E4Env(with_pin=True) as env:
            self.assertIsNone(guard_not_eval_data(env.real_at("anywhere.jsonl"), mode="certifying"))

    def test_e_certifying_allows_the_real_eval_store(self):
        with E4Env(with_pin=True) as env:
            self.assertIsNone(R.refuse_real_evals_dir(env.evals_dir, mode="certifying"))

    def test_g_real_frozen_file_loads_340_in_certifying(self):
        old = os.environ.pop("ASIF_ROOT", None)   # use the repo-root pin, which matches the real file
        try:
            items = load_items(REAL_ITEMS_V1, mode="certifying")
        finally:
            if old is not None:
                os.environ["ASIF_ROOT"] = old
        self.assertEqual(len(items), FROZEN_N)
        # provenance keys dropped, and the frozen "cal" split canonicalised to "calibration" so the
        # calibration split is NOT silently empty (scorer.temperature_scaled fits T on it, prereg §4).
        self.assertEqual({it.split for it in items}, {"calibration", "test"})
        self.assertEqual(sum(it.split == "calibration" for it in items), FROZEN_CAL_N)


class ArmE4CertIgnoresAsifRoot(unittest.TestCase):
    """N1 (independent review of PR #89): --mode certifying reads the pin ONLY from the repo that ships
    items.py and IGNORES ASIF_ROOT, so a certifying run cannot be pointed at a directory whose pin
    matches an arbitrary subset. Uses the REAL repo pin (no monkeypatch) and proves the env override
    is inert. mutate_harness restores the ASIF_ROOT pin redirect -> RED."""

    def test_certifying_ignores_an_asif_root_subset_pin(self):
        with tempfile.TemporaryDirectory() as d:
            subset = os.path.join(d, "subset.jsonl")          # a 3-row head copy: sha != the real pin
            with open(REAL_ITEMS_V1) as src:
                head = [ln for _, ln in zip(range(3), src)]
                src.seek(0)
                first_id = json.loads(src.readline())["item_id"]
            with open(subset, "w") as fh:
                fh.writelines(head)
            root = os.path.join(d, "rogue")                    # ASIF_ROOT whose pin == sha(subset)
            sub = os.path.join(root, "governance", "evals", "jev-calibration")
            os.makedirs(sub)
            with open(os.path.join(sub, "items-v1.sha256"), "w") as fh:
                fh.write(file_sha256(subset) + "  subset.jsonl\n")
            old = os.environ.get("ASIF_ROOT")
            os.environ["ASIF_ROOT"] = root
            try:
                with self.assertRaises(EvalDataRefused) as cm:
                    guard_not_eval_data(subset, mode="certifying")
                self.assertIn("does not match", str(cm.exception))
            finally:
                if old is None:
                    os.environ.pop("ASIF_ROOT", None)
                else:
                    os.environ["ASIF_ROOT"] = old
        # the subset carried a real frozen item_id; certifying still refused it on the real repo pin
        self.assertTrue(first_id)


class ArmE4FixtureRefusesPartialFrozenById(unittest.TestCase):
    """N2 (independent review of PR #89): --mode fixture refuses a PARTIAL copy of the frozen set (a
    head-3 copy whose sha cannot match the whole-file pin) at a NON-frozen path, naming the offending
    item_id. mutate_harness drops the per-row check -> RED."""

    def test_fixture_refuses_a_head_copy_of_the_frozen_items(self):
        with open(REAL_ITEMS_V1) as src:
            head = [ln for _, ln in zip(range(3), src)]
        first_id = json.loads(head[0])["item_id"]
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "subset.jsonl")                # non-frozen basename: path gate must not fire
            with open(f, "w") as fh:
                fh.writelines(head)
            with self.assertRaises(EvalDataRefused) as cm:
                guard_not_eval_data(f, mode="fixture")
            msg = str(cm.exception)
            self.assertIn("frozen eval", msg)
            self.assertIn(first_id, msg)                       # names the offending item_id


class ArmE4FixtureRefusesPartialFrozenByState(unittest.TestCase):
    """N2 (independent review of PR #89): --mode fixture refuses a file that reuses a frozen item's STATE
    even with a changed item_id (a rename cannot launder frozen eval content). mutate_harness drops
    the per-row check -> RED."""

    def test_fixture_refuses_a_frozen_state_under_a_changed_id(self):
        with open(REAL_ITEMS_V1) as src:
            row = json.loads(src.readline())
        row["item_id"] = "RENAMED-000"                         # changed id; state identical to a frozen row
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "renamed.jsonl")
            with open(f, "w") as fh:
                fh.write(json.dumps(row) + "\n")
            with self.assertRaises(EvalDataRefused) as cm:
                guard_not_eval_data(f, mode="fixture")
            msg = str(cm.exception)
            self.assertIn("state", msg)
            self.assertIn("RENAMED-000", msg)


class ArmE4FixturesStillLoad(unittest.TestCase):
    """N2 (independent review of PR #89): the two shipped fixtures share no item_id or state with the
    frozen set, so the per-row guard passes and they still load in --mode fixture (the fix must not
    break legitimate fixtures)."""

    def test_both_shipped_fixtures_load_in_fixture_mode(self):
        self.assertEqual(len(load_items(FIXTURE, mode="fixture")), 6)
        probe = os.path.join(ROOT, "fixtures", "local-precheck-probe.jsonl")
        self.assertEqual(len(load_items(probe, mode="fixture")), 1)


if __name__ == "__main__":
    unittest.main()
