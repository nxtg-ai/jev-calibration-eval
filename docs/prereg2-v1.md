# Does context beat model choice on routing decisions? Pre-registration (Jev study, part 2) — public copy

> **About this copy.** This is a public copy of NXTG.AI's internal pre-registration for part 2 of the Jev study. **No question, arm, item rule, retrieval rule, leakage control, bar or verdict rule has been changed.** Internal file paths, ledger ids and internal team and agent names are replaced with neutral wording. The private S2 items are described without quoting any of their content.
>
> **Registration timestamp.** Unlike part 1, this copy is public **before any context-arm call**: the commit that adds this file to this repository is the public timestamp. The internal original was filed on 2026-10-01; amendment A1 was filed the same day, also before any context-arm call.
>
> Part 1: [`prereg-v1.md`](prereg-v1.md) and [`results-v1.md`](results-v1.md).

## 1. Question

In part 1 no model could route our internal work from the bare text. On the 48 test items per class, the best accuracy any arm reached was 0.21 (which agent team wrote this coordination post, S2a), 0.58 (which program a work item belongs to, S2b) and 0.44 (which team owns a tracked commitment, S2c). Those items deliberately stripped signatures, mentions and IDs, so no arm had any history to go on.

**RQ5:** if each question comes with the most similar *earlier* records from our memory store, and their known labels, does accuracy rise, and does the choice of model still matter?

This plan is written knowing the part 1 no-context results, and says so.

## 2. Arms

| Arm | What it gets | Model |
|---|---|---|
| **JEV+CTX** | the part 1 prompt plus k retrieved neighbours with their labels | `jev-1.13.0` |
| **JEV** | the part 1 prompt, no context (concurrent comparator, A1.1) | `jev-1.13.0` |
| **FRONTIER+CTX** | same as JEV+CTX | `claude-opus-5-5`, `claude -p`, effort high |
| **FRONTIER** | no context (concurrent comparator, A1.1) | `claude-opus-5-5` |
| **RETRIEVAL-ONLY** | no model: the answer is the option label held by the most neighbours (ties → the nearest neighbour's label); confidence = that label's count / k | none |
| **LOCAL-P+CTX** | same as JEV+CTX, only if every full prompt fits the local model's context limit (A1.6); otherwise not run, and that is reported | `qwen3:14b` |

Models, options and prompt text are identical to part 1 apart from the added context block.

## 3. Items and retrieval

- **Items:** the S2 test items of the frozen part 1 item set (S2a 48, S2b 48, S2c 48 = 144). Calibration-split items are used only to check the pipeline end to end before the run.
- **Corpus per class**, each from the record type the item came from: S2a = coordination posts with their author field; S2b = program work-breakdown nodes with their program prefix; S2c = tracked commitments with an explicitly set owning team. Each corpus is frozen with a SHA-256 before the first call.
- **Retrieval:** k = 5 nearest neighbours by embedding similarity of the item's text, using the embedding model of our memory store (Dx3) with exact kNN over the frozen corpus file (A1.5). Neighbour text is stripped the same way as the item (signatures, mentions and IDs removed), truncated to 400 characters, and followed by its label.
- **Leakage controls, all enforced in code and tested before the run:**
  1. the item's own source record is excluded;
  2. any record whose stripped text is identical or near-identical to the item (character-trigram Jaccard ≥ 0.8) is excluded;
  3. **only records created strictly before the item's source record are eligible** (a real router only has the past);
  4. the item's leak strings are stripped from neighbour text exactly as from the item, and every option's name is redacted symmetrically (A1.2).
- **What this is and is not.** Dx3 is three stores (relational, vector, graph). This test uses **one of them: vector retrieval**. A result here is a claim about *retrieved context from our memory store*, never about Dx3 as a whole, and the write-up will say so.

## 4. Hypotheses and bars (fixed now)

| ID | Claim | Bar | Words |
|---|---|---|---|
| **H5** | Context helps a model route | Per class and model: accuracy with context minus without, paired by item on the same 48 items; claimed if the paired-bootstrap 95% CI (2,000 resamples, seed 20260929) is entirely above 0 **and** the gain is ≥ 10 points | *context helps* / *no clear effect* / *context hurts* (CI entirely below 0) |
| **H6** | Memory alone rivals the models | Per class: RETRIEVAL-ONLY accuracy vs the best no-context model arm | *beats* (CI of the paired difference entirely above 0) / *matches* (point within 3 points) / *trails* |
| **H7** | With context, model choice still matters | Per class: FRONTIER+CTX minus JEV+CTX accuracy, paired | descriptive: the difference and its CI, no verdict word |

Calibration (ECE, Brier) and cost/latency are reported for every arm, descriptive only. A missed bar is reported as a miss. No bar moves after data.

## 5. Run discipline

- **One certifying run, one draw** (draw 1 is primary, as in part 1).
- **Spend:** Jev on its early-access allocation; the frontier arm on an existing subscription. No purchase. If Jev usage turns billable, the run halts.
- **Certification:** the run must pass the same automated certification gates as part 1 before any number is reported. The harness change lands with tests and mutation proofs and is verified by a reviewer who did not write it.
- **Independent check:** a second reviewer reproduces the H5 and H6 tables from the raw records before any public word.
- **Stopping rule (carried over from part 1, A8.3):** no amendment except a defect that would make a reported number or verdict wrong. One run. If the run fails certification for a recording defect, one re-run; nothing more without a new decision.

## 6. What each outcome means for us (written before the data)

- **H5 "context helps" with large gains, H6 "matches/beats":** the bottleneck for routing our work is memory, not model size. That supports building the routing tier on retrieval plus the cheapest adequate model.
- **H5 "no clear effect":** five similar past records are not enough context, or this task is not retrievable from text. Then routing needs structure (the graph store: who owns what, who replied to whom), which this test does not use. That becomes the next test, not a claim.
- **H5 "context hurts":** retrieval is pulling misleading neighbours; we report it plainly.

## 7. Amendments

**A1 · 2026-10-01 · filed before any context-arm call** (an independent review of this plan). Every item below overrides the text above where they differ; the tables above already reflect it.

- **A1.1 · Concurrent no-context comparators.** The run includes no-context JEV and FRONTIER arms, interleaved per item with the context arms. **They are the H5 comparator.** Part 1's draw-1 records become a secondary read only. Reason: in part 1, the answer changed across three draws on 5.3% of items for JEV and 21% for FRONTIER, so a cross-run comparison could mistake re-ask noise or version drift for a context effect.
- **A1.2 · Symmetric redaction.** In neighbour text, the names and aliases of **every** option in the item are redacted the same way, not only the gold answer's. Otherwise the one redacted name points at the answer. Neighbour labels stay visible.
- **A1.3 · RETRIEVAL-ONLY stays inside the option set.** The vote counts only neighbour labels that are among the item's options. If no neighbour's label is an option, the arm abstains, scored under the part 1 abstention rubric. Confidence = winning label's count / k.
- **A1.4 · Timestamps.** S2a: the post's timestamp. S2c: the commitment's creation time (else the earliest recorded time for that id). S2b: the author time of the commit that first added the node to the work-breakdown file. A record with no derivable time is excluded from the corpus. "Strictly earlier" compares against the item's source record time derived the same way.
- **A1.5 · What "Dx3" means here.** Neighbours come from Dx3's embedding model with exact kNN over the frozen corpus file, not from a live Dx3 vector-store query.
- **A1.6 · LOCAL-P+CTX feasibility.** It runs only if every full context prompt measures **under 1,984 tokens** with the Qwen3 tokenizer and chat template (the harness halts at 1,984 prompt tokens). This is checked before any GPU time is requested. The local context window stays 4096, because changing it would invalidate the part 1 precheck. Otherwise LOCAL-P+CTX is reported as not run.
- **A1.7 · Arm order:** `JEV+CTX, JEV, FRONTIER+CTX, FRONTIER, RETRIEVAL-ONLY`, then `LOCAL-P+CTX` if A1.6 passes. The first two form the certification's paired-arm gate, so its gap is Jev's H5 read. A model-free arm must leave every other certification gate valid, shown on a fixture before the run.

**A2 · 2026-10-01 · filed before any certifying context-arm call** (on review of the built harness; its two smoke calls were on one calibration item and produce no reported number). Overrides the text above where they differ.

- **A2.1 · H6 comparator.** The primary H6 comparator is the best **concurrent** no-context model arm (JEV or FRONTIER, best = higher accuracy on that class's items). The A1.1 reasoning applies to H6 exactly as to H5. The best part 1 draw-1 arm is reported as a secondary read.
- **A2.2 · H6 words, replacing the §4 order.** *beats* if the paired CI is entirely above 0; *trails* if it is entirely below 0; otherwise *matches* if the point difference is within 3 points, and *no clear difference* if it is not. Reason: read literally, §4 calls a gain above 3 points whose CI includes 0 "trails", which would misstate the result against retrieval.
- **A2.3 · Disclosed harness readings, accepted as built:** certification gates cover the context arms; the concurrent no-context arms are comparators, held to the same model id and sampling settings by a pre-call refusal, and the readout refuses any paired read with a missing item. RETRIEVAL-ONLY puts count/5 on the winner (k fixed at 5) and spreads the remainder evenly over the other options, which sets its Brier and ECE. Work-item times are the first commit adding that item's id line. Commitment records are merged per id, taking the earliest non-empty value per field; the record time is the earliest creation time.
