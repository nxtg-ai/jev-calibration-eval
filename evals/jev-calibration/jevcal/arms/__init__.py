"""Arm registry: name -> factory. The runner builds arms only from here, passing per-arm keyword
options (FRONTIER requires `effort`; LOCAL arms take precheck/vram/transport kwargs).

E.4b (the study author) adds three LOCAL arms, one `LocalArm` class parameterised by the sampling-registry
key: LOCAL-P (qwen3-14b), LOCAL-C-CODER (Qwen3-Coder-30B-A3B), LOCAL-C-GPTOSS (gpt-oss:20b). LOCAL-P
and LOCAL-C-CODER share family "qwen", so the runner's family-uniqueness rule refuses them together
(intended: they cannot share the 4090's VRAM). See jevcal/arms/local.py and README.md.
"""
from typing import Callable, Dict

from .base import Arm
from .frontier import FrontierArm
from .jev import JevArm
from .local import LOCAL_ARMS, make_local_arm, vram_preflight  # noqa: F401 (re-exported for the runner)

ARMS: Dict[str, Callable[..., Arm]] = {
    "JEV": JevArm,
    "FRONTIER": FrontierArm,
}
for _name, _model_id in LOCAL_ARMS.items():
    ARMS[_name] = make_local_arm(_name, _model_id)
