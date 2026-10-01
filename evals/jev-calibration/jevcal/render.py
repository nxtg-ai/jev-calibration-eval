"""The ONE prompt renderer (prereg §2, prompt parity).

render(item) produces the canonical user-seen text and the ordered option table.
Every arm adapter is built from this object and nothing else, so no arm can see
content another arm does not. The wire shapes differ by design (JEV takes
`instructions` + a `criteria` map; FRONTIER/LOCAL take lettered options), so the
canonical digest `msg_sha256` is taken over the rendered canonical text, never
over any one wire payload. Each adapter also logs its own wire digest.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Tuple

from .items import Item

LABELS = tuple("ABCDEFGHIJKLMNOPQRST")  # 20 single-token labels, prereg §2 (A–T)


@dataclass(frozen=True)
class Rendered:
    item_id: str
    type: str                                   # question_type
    state: str
    instruction: str                            # canonical instruction text (item.instructions)
    options: Tuple[Tuple[str, str, str], ...]   # (label, key, description-or-"")
    canonical_text: str
    msg_sha256: str

    @property
    def labels(self) -> List[str]:
        return [o[0] for o in self.options]

    @property
    def keys(self) -> List[str]:
        return [o[1] for o in self.options]

    @property
    def option_texts(self) -> List[str]:
        """'key — description' (or 'key'): the ONE rendering of an option. The lettered
        lines every arm reads are built from it, and so are the JEV noul/score criteria,
        whose wire cannot carry option keys natively (CODEX PR 72 finding 3)."""
        return [option_text(key, desc) for _, key, desc in self.options]


def option_text(key: str, desc: str) -> str:
    return f"{key} — {desc}" if desc else key


def render(item: Item) -> Rendered:
    opts = tuple((LABELS[i], key, item.description(i)) for i, key in enumerate(item.options))
    lines = ["State:", item.state, "", "Question:", item.instructions, "", "Options:"]
    for label, key, desc in opts:
        lines.append(f"{label}. {option_text(key, desc)}")
    text = "\n".join(lines)
    return Rendered(item_id=item.item_id, type=item.question_type, state=item.state,
                    instruction=item.instructions, options=opts, canonical_text=text,
                    msg_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
