"""Named arms for the FRONTIER isolation preflight (prereg A5.2). Real code, real
files on disk (temp trees), no mocks. tests/mutate_isolation.py reverts each check
and requires its arm to go RED.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jevcal import isolation as ISO  # noqa: E402
from jevcal import run as R  # noqa: E402
from jevcal.arms.base import ArmHalt  # noqa: E402
from jevcal.arms.frontier import FrontierArm  # noqa: E402
from jevcal.items import load_items  # noqa: E402
from jevcal.render import render  # noqa: E402

K = ISO.EFFORT_KEY
FIXTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "fixtures", "items-fixture.jsonl")


def tree(settings=None, name="settings.json", raw=None):
    """A temp root holding .claude/<name> with `settings` (or raw bytes), and a deep cwd."""
    root = tempfile.mkdtemp(prefix="jevcal-iso-")
    if settings is not None or raw is not None:
        os.makedirs(os.path.join(root, ".claude"))
        with open(os.path.join(root, ".claude", name), "w") as fh:
            fh.write(raw if raw is not None else json.dumps(settings))
    cwd = os.path.join(root, "a", "b")
    os.makedirs(cwd)
    return root, cwd


class EnvGuard:
    def __init__(self, **kv):
        self.kv = kv

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.kv}
        for k, v in self.kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class ArmIsolationEnv(unittest.TestCase):
    """(a) the child env must not carry CLAUDE_CODE_EFFORT_LEVEL."""

    def test_check_env(self):
        self.assertFalse(ISO.check_env({K: "xhigh"})[0])
        self.assertTrue(ISO.check_env({"PATH": "/usr/bin"})[0])

    def test_arm_child_env_strips_an_inherited_pin(self):
        with EnvGuard(**{K: "xhigh"}):
            arm = FrontierArm(effort="high")
            self.assertNotIn(K, arm.child_env())
            self.assertTrue(arm.isolation_report()["env_ok"])
            inherited = dict(os.environ)                     # what a naive launcher would pass
            self.assertFalse(arm.isolation_report(env=inherited)["env_ok"])


class ArmIsolationOutsideAsif(unittest.TestCase):
    """(b) the child cwd must be outside ~/ASIF (and $ASIF_ROOT)."""

    def test_inside_asif_root_fails_even_with_no_settings_file(self):
        root, cwd = tree()                                   # no .claude anywhere
        with EnvGuard(ASIF_ROOT=root):
            rep = ISO.check({}, cwd)
        self.assertFalse(rep["cwd_outside_asif_ok"])
        self.assertTrue(rep["ancestor_settings_ok"])         # separated: only (b)-outside fires
        self.assertFalse(rep["ok"])

    def test_outside_passes(self):
        other, cwd = tree()
        root, _ = tree()
        with EnvGuard(ASIF_ROOT=root):
            self.assertTrue(ISO.check_outside_asif(cwd)[0])


class ArmIsolationAncestors(unittest.TestCase):
    """(b) no .claude/settings*.json on the ANCESTOR chain may set the key."""

    def test_pin_two_levels_up_fails(self):
        _, cwd = tree({"env": {K: "xhigh"}})
        rep = ISO.check({}, cwd)
        self.assertFalse(rep["ancestor_settings_ok"])
        self.assertTrue(rep["cwd_outside_asif_ok"])          # separated: only (b)-ancestors fires
        self.assertFalse(rep["ok"])

    def test_settings_local_pin_fails(self):
        _, cwd = tree({"env": {K: "max"}}, name="settings.local.json")
        self.assertFalse(ISO.check_ancestor_settings(cwd)[0])

    def test_unreadable_settings_fails_closed(self):
        _, cwd = tree(raw="{not json")
        self.assertFalse(ISO.check_ancestor_settings(cwd)[0])

    def test_settings_without_the_key_pass(self):
        _, cwd = tree({"env": {"OTHER": "1"}, "effortLevel": "high"})
        ok, _msg, checked = ISO.check_ancestor_settings(cwd)
        self.assertTrue(ok)
        self.assertEqual(len(checked), 1)


class ArmIsolationEnforced(unittest.TestCase):
    """The checks bind: the arm halts before invoking the child, the runner before any call."""

    def test_arm_halts_before_the_child_runs(self):
        _, cwd = tree({"env": {K: "xhigh"}})
        calls = []
        arm = FrontierArm(effort="high", runner=lambda *a: calls.append(a) or (0, "[]", ""))
        arm.cwd = cwd
        r = render(load_items(FIXTURE)[0])
        with self.assertRaises(ArmHalt):
            arm.call(r)
        self.assertEqual(calls, [])

    def test_runner_preflight_refuses(self):
        _, cwd = tree({"env": {K: "xhigh"}})
        arm = FrontierArm(effort="high")
        arm.cwd = cwd
        with self.assertRaises(SystemExit):
            R.preflight_isolation([arm])
        R.preflight_isolation([FrontierArm(effort="high")])  # the real, isolated arm passes


if __name__ == "__main__":
    unittest.main()
