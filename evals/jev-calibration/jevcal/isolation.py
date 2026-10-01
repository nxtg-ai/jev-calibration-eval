"""FRONTIER child-process isolation preflight (prereg A5.2).

Passing `--effort high` does not prove it took effect. An ~/ASIF Claude pane
exports CLAUDE_CODE_EFFORT_LEVEL=xhigh to every child, and ~/ASIF/.claude/
settings.json pins it at project scope. So the FRONTIER child must receive:

  (a) an env WITHOUT CLAUDE_CODE_EFFORT_LEVEL;
  (b) a cwd OUTSIDE ~/ASIF (and outside $ASIF_ROOT when set) whose ENTIRE
      ancestor chain holds no .claude/settings.json or .claude/settings.local.json
      that sets env.CLAUDE_CODE_EFFORT_LEVEL. An unreadable settings file on the
      chain fails closed (it cannot be shown not to set the key).

`check(env, cwd)` is evaluated on the EXACT env dict and cwd the child will be
given. The runner calls it before the first eval call (and the arm re-checks it
before every call); a failure aborts the run.

Positive control:  python3 -m jevcal.isolation --cwd ~/ASIF --inherit-env
                   python3 -m jevcal.isolation --as-arm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

EFFORT_KEY = "CLAUDE_CODE_EFFORT_LEVEL"
SETTINGS_FILES = ("settings.json", "settings.local.json")


def asif_roots() -> List[str]:
    roots = {os.path.realpath(os.path.expanduser("~/ASIF"))}
    if os.environ.get("ASIF_ROOT"):
        roots.add(os.path.realpath(os.path.expanduser(os.environ["ASIF_ROOT"])))
    return sorted(roots)


def check_env(env: Dict[str, str]) -> Tuple[bool, str]:
    if EFFORT_KEY in env:
        return False, f"(a) FAIL: child env carries {EFFORT_KEY}={env[EFFORT_KEY]!r}"
    return True, f"(a) PASS: child env has no {EFFORT_KEY}"


def check_outside_asif(cwd: str) -> Tuple[bool, str]:
    rp = os.path.realpath(cwd)
    inside = [r for r in asif_roots() if rp == r or rp.startswith(r + os.sep)]
    if inside:
        return False, f"(b) FAIL: cwd {rp} is inside {inside[0]}"
    return True, f"(b) PASS: cwd {rp} is outside {asif_roots()}"


def ancestors(path: str) -> List[str]:
    out, cur = [], os.path.realpath(path)
    while True:
        out.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            return out
        cur = parent


def check_ancestor_settings(cwd: str) -> Tuple[bool, str, List[str]]:
    checked, bad = [], []
    for d in ancestors(cwd):
        for name in SETTINGS_FILES:
            p = os.path.join(d, ".claude", name)
            if not os.path.isfile(p):
                continue
            checked.append(p)
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, UnicodeError, json.JSONDecodeError) as e:
                bad.append(f"{p} unreadable ({type(e).__name__}); cannot show it does not pin {EFFORT_KEY}")
                continue
            env = data.get("env") if isinstance(data, dict) else None
            if isinstance(env, dict) and EFFORT_KEY in env:
                bad.append(f"{p} sets env.{EFFORT_KEY}={env[EFFORT_KEY]!r}")
    if bad:
        return False, "(b) FAIL: ancestor settings pin effort: " + "; ".join(bad), checked
    return True, (f"(b) PASS: {len(checked)} settings file(s) on the ancestor chain, none sets "
                  f"{EFFORT_KEY}"), checked


def check(env: Dict[str, str], cwd: str) -> Dict:
    a_ok, a_msg = check_env(env)
    o_ok, o_msg = check_outside_asif(cwd)
    s_ok, s_msg, checked = check_ancestor_settings(cwd)
    return {"ok": a_ok and o_ok and s_ok, "env_ok": a_ok, "cwd_outside_asif_ok": o_ok,
            "ancestor_settings_ok": s_ok, "cwd": os.path.realpath(cwd),
            "env_has_effort_key": EFFORT_KEY in env, "settings_files_checked": checked,
            "asif_roots": asif_roots(), "details": [a_msg, o_msg, s_msg]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FRONTIER isolation preflight (prereg A5.2)")
    ap.add_argument("--cwd", help="evaluate this cwd")
    ap.add_argument("--inherit-env", action="store_true", help="evaluate this process's env as-is")
    ap.add_argument("--set-env", action="append", default=[], metavar="K=V",
                    help="add K=V to the evaluated env (e.g. to reproduce an ~/ASIF pane child env)")
    ap.add_argument("--as-arm", action="store_true",
                    help="evaluate the exact env and cwd a FrontierArm(effort='high') child receives")
    a = ap.parse_args(argv)
    if a.as_arm:
        from .arms.frontier import FrontierArm
        rep = FrontierArm(effort="high").isolation_report()
        mode = "as-arm (the exact child env + cwd FrontierArm builds)"
    else:
        env = dict(os.environ) if a.inherit_env else {}
        for kv in a.set_env:
            k, _, v = kv.partition("=")
            env[k] = v
        rep = check(env, a.cwd or os.getcwd())
        mode = f"cwd={a.cwd or os.getcwd()} inherit_env={a.inherit_env} set_env={a.set_env}"
    print(f"isolation preflight — {mode}")
    for d in rep["details"]:
        print("  " + d)
    print("PASS" if rep["ok"] else "FAIL")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
