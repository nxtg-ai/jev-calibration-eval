# Jev calibration eval — results (v1) — public redacted copy

> **About this copy.** These are the results of the pre-registered study in [`prereg-v1.md`](prereg-v1.md), written 2026-10-01. Every number below is the number in NXTG.AI's internal results document. What was changed: internal paths, ids, commit hashes and team/agent names are replaced with neutral wording. **S2 is a private set of internal operating text.** Its aggregate accuracy, calibration and verdicts are published here; no S2 item, record or example text is.
>
> **Which numbers you can reproduce from this repo:** every S1 row (H1, H2, H3, the sub-slices, the S1 per-arm metrics), by running `python3 scripts/reproduce_s1.py` over the released S1 raw records. Pooled numbers, the S2 rows, H4 and the pooled temperature-scaling and jitter figures mix in S2 and cannot be reproduced from public data. See the root [`README.md`](../README.md).

**Reported run:** `jevcal-d2b-3arm-20261001T043407Z`. Certified 17/17 by our internal eval-rail certifier.
**Independent checks, and their scope:** the H2 and H1 tables were reproduced by an independent reviewer from the raw files, and an independent cross-vendor reviewer's (OpenAI Codex) method review passed clean (divergence, binning, scorer). **H3, H4, the pairwise Brier, the sub-slices, jitter and the recommendations are single-author and not independently reproduced.**

## Answers in one paragraph

- **RQ1 (is Jev's confidence calibrated?)** No. Pooled test ECE is 0.156, and Jev is "not calibrated as claimed" on 4 of 5 decision classes in every draw. It is overconfident: mean confidence 0.66 against accuracy 0.53. The fifth class (S1-noul, yes/no) is draw-sensitive. No class reaches "calibrated" (ECE 0.05 or less) in any draw or binning.
- **RQ2 (does a local model match?)** qwen3:14b, read this way, does not match on any class, in any draw. That was the only local model tested. Its accuracy is close to Jev's (0.51 vs 0.53), but its confidence is far worse (ECE 0.40; mean confidence 0.91 against 0.51 accuracy).
- **RQ3 (what does a decision cost?)** Jev is very fast and very cheap, about 44× faster and about 570× cheaper per decision than our frontier arm. Our frontier arm is not the model Jev compares itself to, so H4 is "partially" reproduced.
- **What this decides:** under the pre-registered rule, **no decision class moves to the local tier**. The per-class recommendation below (Jev or frontier) is the author's judgment, not a pre-registered verdict.

## The run

- **Arms, in declared order:** LOCAL-P = `qwen3:14b` (ollama on an RTX 4090, label read from the first token's logprobs); JEV = `jev-1.13.0`; FRONTIER = `claude-opus-5-5` via `claude -p`, effort high, verbalised probabilities.
- **Items:** 340 frozen items, 273 of them in the test split. Slices: S1-choice (CUAD contract-clause type, 20 options, n=46), S1-noul (CUAD yes/no, n=83), and three private slices: S2a (which agent team wrote this internal coordination post, 20 options, n=48), S2b (which internal program owns this work item, ~11 options, n=48), S2c (which team owns this tracked commitment, ~9 options, n=48).
- **Draws:** 3 per item for LOCAL-P and JEV; 3 for FRONTIER on the fixed 100-item subset (1 elsewhere). Draw 1 is primary (A7.1).
- **Integrity:** 0 hard errors in 2,580 records. 127 records carry a warning that the returned probabilities summed to 0.99–1.04 and were renormalised. No GPU memory spill (checked by the GPU host's operator). The public release carries the 1,214 S1 records of the 2,580 (LOCAL-P 480, JEV 480, FRONTIER 254) and 235 of the 500 FRONTIER secondary samples.
- **Determinism (declared in advance):** LOCAL-P's answers match the earlier single-arm run on 340/340 items in each draw.

## H1 — Jev's calibration (test split, raw probabilities)

ECE with 15 equal-mass bins (primary) and 15 equal-width bins (A6.2). Bands: ≤0.05 calibrated, 0.05–0.10 roughly, >0.10 not calibrated as claimed.

| Class | n | Draw 1 ECE mass / width | Draw 2 | Draw 3 | Bins (equal-mass) | **Claimed word** |
|---|---|---|---|---|---|---|
| Pooled | 273 | 0.156 / 0.145 | 0.132 / 0.124 | 0.132 / 0.132 | 14 | **not calibrated as claimed** |
| S1-choice | 46 | 0.200 / 0.203 | 0.198 / 0.207 | 0.188 / 0.199 | 9 | **not calibrated as claimed** |
| S1-noul | 83 | 0.078 / 0.098 | **0.102** / 0.086 | 0.077 / 0.099 | 11 | **draw-sensitive** |
| S2a | 48 | 0.250 / 0.264 | 0.256 / 0.256 | 0.247 / 0.277 | 15 | **not calibrated as claimed** |
| S2b | 48 | 0.305 / 0.284 | 0.334 / 0.261 | 0.313 / 0.293 | 14 | **not calibrated as claimed** |
| S2c | 48 | 0.167 / 0.109 | 0.180 / 0.128 | 0.215 / 0.158 | 14 | **not calibrated as claimed** |

- **S1-noul was decided by 0.002.** Draws 1 and 3 read "roughly calibrated". On draw 2 the equal-mass ECE is 0.1023, 0.0023 above the 0.10 line (0.0016 under the independent reviewer's equal-mass cut rule). Equal-width reads 0.086, so that draw is binning-sensitive, and the class is draw-sensitive. Had draw 2 landed 0.003 lower, Jev would be claimed "roughly calibrated" on S1-noul.
- **The S2 identifier read** (excluding S2a-023, S2c-016, S2c-046) gives the same words.
- **Bins:** Jev returns probabilities in steps of 0.01, so ties cut the equal-mass bin count to 9–15 per class (the cross-vendor reviewer matched every bin count and every per-bin value from the raw data). Per-bin n and values are in the run record's `reliability_curve` arrays (A6.1).
- **Recalibration does not rescue it.** Temperature scaling fitted on the calibration split (secondary, §4) moves Jev's pooled equal-mass ECE only from 0.156 to 0.150.

## H2 — does the local model match? (per class)

Bar: LOCAL-P accuracy within 3 points of the best non-local arm **and** LOCAL-P's Brier upper-CI ≤ Jev's. Draw 1 shown; the verdict is the same in all three draws.

| Class | n | Accuracy LOCAL-P | JEV | FRONTIER | Chance | Brier upper-CI LOCAL-P / JEV | **H2** |
|---|---|---|---|---|---|---|---|
| S1-choice | 46 | 0.630 | 0.739 | 0.739 | 0.05 | 0.955 / 0.642 | **does not** |
| S1-noul | 83 | 0.892 | 0.867 | 0.952 | 0.50 | 0.334 / 0.291 | **does not** † |
| S2a | 48 | 0.062 | 0.104 | 0.208 | 0.05 | 1.731 / 1.092 | **does not** |
| S2b | 48 | 0.458 | 0.396 | 0.583 | 0.09 | 1.249 / 0.966 | **does not** |
| S2c | 48 | 0.208 | 0.312 | 0.438 | 0.11 | 1.464 / 0.919 | **does not** |

† On the 30 S1-noul items in the FRONTIER subset, LOCAL-P **matches** in every draw. The all-items read governs (a rule fixed in the pre-registration, labelled there as written after the data): S1-noul does not match on all 83 test items in any draw, and matches on the 30-item subset in every draw. The difference comes from the population, not the draws.

Pairwise calibration-plus-accuracy (paired per-item Brier, test, draw 1; negative = first arm better):

| Pair | Mean difference | 95% CI |
|---|---|---|
| LOCAL-P − JEV | +0.262 | [+0.195, +0.335] |
| JEV − FRONTIER | +0.136 | [+0.092, +0.179] |
| LOCAL-P − FRONTIER | +0.398 | [+0.326, +0.474] |

So FRONTIER beats Jev, and Jev beats LOCAL-P, with every interval excluding zero. Accuracy alone is closer: LOCAL-P − JEV = −0.026, 95% CI [−0.070, +0.018].

**Where confidence and accuracy diverge** (the cross-vendor reviewer's independent table, mean confidence minus accuracy, test, draw 1): LOCAL-P is overconfident on every class (pooled +0.40; S2a +0.76). Jev is overconfident on every class (pooled +0.13; worst S2a/S2b at +0.25/+0.26). FRONTIER is close to neutral pooled (−0.02) but its errors cancel within classes: S1-choice has a gap of −0.005 and an ECE of 0.159.

**LOCAL-P's miscalibration is a scale problem, not a ranking problem.** One temperature fitted on the calibration split (T ≈ 10) brings its pooled ECE from 0.40 to 0.067 and its Brier from 0.858 to 0.595. Given the same treatment, Jev moves only from 0.156 to 0.150 ECE and from 0.596 to 0.610 Brier, so after scaling LOCAL-P is slightly better calibrated than Jev (secondary only; no H-test uses it). Its accuracy is the harder limit.

## H3 — is confidence usable for routing? (descriptive)

Coverage at ≥95% precision: the share of items a confidence threshold could auto-accept while keeping 95% of them right.

| | Pooled | S1-choice | S1-noul | S2a | S2b | S2c |
|---|---|---|---|---|---|---|
| LOCAL-P | 0.21 | 0.09 | 0.82 | 0.00 | 0.13 | 0.02 |
| JEV | 0.32 | 0.48 | 0.81 | 0.00 | 0.08 | 0.02 |
| FRONTIER | 0.44 | 0.46 | 1.00 | 0.06 | 0.10 | 0.08 |

Confidence gating works on yes/no contract checks for every arm. It does almost nothing on the internal routing classes, because no arm is accurate there.

## H4 — speed and cost (Jev's claims: ~25× faster, ~76× cheaper)

| Arm | p50 / p95 latency | Cost per decision (list price) |
|---|---|---|
| JEV | 0.149 s / 0.203 s | $0.0000387 |
| FRONTIER (Opus 5.5, effort high) | 6.53 s / 10.64 s | $0.0220 |
| LOCAL-P | 0.39 s / 0.55 s | $0 marginal (our GPU) |

Population: draw 1, all 340 items, the same population as the run record's `cost_usd_list_total` (JEV $0.0131, FRONTIER $7.49) and latency fields. Test-only draw-1 figures differ by under 3%.

- Jev vs our frontier arm: **~44× faster, ~570× cheaper.**
- **Verdict: partially reproduced.** The direction holds and the size exceeds the claim. But the ~25×/~76× claim the pre-registration names is against GPT-5.6 "Terra" on TypeSafe's own four-workflow eval ($0.0004 and 0.4 s per case vs $0.0304 and 10.1 s), as reported by third-party write-ups ([opentweet.io/jev/vs-gpt](https://opentweet.io/jev/vs-gpt), [orcarouter.ai](https://www.orcarouter.ai/blog/jev-typesafe-system-one-what-we-know)); TypeSafe's launch blog itself was not opened when the bar was written. This run had no GPT-5.6 arm.
- **What TypeSafe itself publishes** (primary sources read 2026-10-01): the homepage says "193.6x Faster, 444.6x Cheaper" on "workflows for System One tasks", and the launch blog says those numbers come from workflow evals against "the average of GPT-6 Astra and Fable 5.1", are "on the higher end of real world gains", and that typical speed-ups "range from 40x-200x faster". **Neither page states 25× or 76×**; those multiples are third-party arithmetic on TypeSafe's GPT-5.6 Terra demo. So the pre-registration's H4 bar quoted a secondary figure, not a published claim. Against our Opus 5.5 arm: the cost gap (~570×) exceeds TypeSafe's 444.6×; the speed gap (~44×) is well below its 193.6× headline and at the bottom of its own 40–200× range. Our comparator is neither of theirs, so none of this is scored as a verdict. Against the model they named, the claim is untested.
- FRONTIER latency includes `claude -p` start-up, and its cost prices ~2,700 input and ~490 output tokens per call at list price; it actually runs on a subscription.

## §3 sub-slice — items Jev is known to be weak on (test, draw 1)

| Tag | n | LOCAL-P | JEV | FRONTIER |
|---|---|---|---|---|
| dates | 27 | 0.667 | 0.704 | 0.778 |
| numeric | 18 | 0.833 | 0.833 | 0.944 |

All of these items sit in S1. Accuracy is a little lower than each arm's S1 average on date items; the numeric items are not harder here.

## Jitter across draws (A7.3)

| Arm | Items | Mean prob. change | Max change | Answer flips |
|---|---|---|---|---|
| LOCAL-P | 340 | 0 | 0 | 0% |
| JEV | 340 | 0.005 | 0.14 | 5.3% |
| FRONTIER | 100 | 0.014 | 0.43 | **21%** |

FRONTIER changes its answer on one item in five when asked again, so a single frontier call is not a stable answer either.

## Recommendation per class (author's judgment; the pre-registration sets no bar for this)

| Class | Local tier (H2) | Recommendation | Why |
|---|---|---|---|
| S1-noul (yes/no field checks) | no | **Jev**, with a confidence gate | All arms ≥0.87 accurate; Jev's 0.81 coverage at 95% precision, at ~1/450 of frontier cost (S1-noul's measured ratio is 454×, independently reproduced). Re-measure calibration before trusting the gate threshold, since S1-noul is draw-sensitive. |
| S1-choice (20-way clause type) | no | **Jev**, don't trust its confidence | Jev and frontier are equally accurate (0.74); Jev's ECE is 0.20, so its probability can't set a routing threshold. |
| S2a/S2b/S2c (internal routing: author, program, owner) | no | **Neither tier is good enough yet** | Best accuracy is 0.21/0.58/0.44 (frontier). This is a context problem before it is a model problem: the items deliberately strip signatures, mentions and IDs. Re-test with the context a real router would have (retrieval over our internal knowledge store) before choosing a model. |

**Build vs buy:** buy (Jev) for S1-shaped classification, where it matches frontier accuracy at a tiny fraction of the cost and latency. Don't use its probabilities as calibrated without per-class measurement. Don't build a local tier on qwen3:14b. Its probabilities could be made usable with one temperature, but its accuracy is below the best non-local arm on all 5 classes (and below Jev's on 3 of 5; it beats Jev on S1-noul and S2b).

## Limitations and follow-ups

1. **One local model.** gpt-oss:20b was infeasible as designed, and Qwen3-Coder-30B never ran because it needs 19,279 MiB free on the GPU. An infrastructure change has since freed GPU memory, so re-check whether Coder-30B fits. "No local tier" is a result about qwen3:14b read this way, not about local models.
2. **Where the harness ran.** §7 said the GPU host; it ran on a separate machine and reached ollama over an SSH tunnel (A2). Latency is client wall-clock, and LOCAL cold load is not separated from warm calls (§4 asked for both). LOCAL-P's 0.39 s p50 here vs 0.15 s in the single-arm run likely reflects interleaving with the other two arms; it was not isolated.
3. **Jev spend (§8).** All 1,020 Jev calls succeeded, at a list-price value of $0.013. We have no account-side statement of the early-access allocation balance, so "stayed inside the allocation" is not verified.
4. **FRONTIER is verbalised and unstable:** probabilities are stated, not measured, and 21% of its answers flip between draws.
5. **S1 contamination:** CUAD is public and may be in every arm's training data. S2 is private.
6. **S2 measures a stripped task.** Signatures, @mentions and IDs were removed on purpose. Low S2 accuracy says what a model can infer from the bare text, not how a real internal router would do.
7. **Small classes:** 46–83 test items each. Accuracy CIs are about ±0.13.
8. **Equal-mass cut placement** (A6.1) can be implemented two ways that both fit the text (the author's and the independent reviewer's). Verdicts are identical under both; individual ECE values differ slightly (e.g. pooled 0.156 vs 0.150).
9. **The stated LOCAL options in a §9 disclosure were wrong** (`temperature 0`); corrected in the pre-registration. No number changed.
10. **Two earlier runs exist and are disclosed, not reported:** `jevcal-e4b-local-p-20260930T193843Z` (single arm, cannot certify) and `jevcal-d2-3arm-20260930T230651Z` (failed GATE H on a recording defect).
11. **Public-release scope.** The S2 items, records and the full run record (which embeds every user-seen prompt) are not released, so the S2 rows and pooled figures above can be checked only by NXTG.AI. The released S1 records reproduce every S1 number exactly (see the root README).

## Provenance

- Run record, raw data, certifier output, per-draw H1/H2 tables and scripts: kept in NXTG.AI's private repository. The S1 portion of the raw data is in [`../results/jevcal-d2b-3arm-20261001T043407Z-S1/`](../results/jevcal-d2b-3arm-20261001T043407Z-S1/).
- Method notes fixed before computing (per-draw H2, per-draw H1) and the Read C rule are in the pre-registration's §9, with their timestamps.
- Reviews: an independent reviewer reproduced the H2 and H1 tables and the S1-noul cost ratio; the cross-vendor method review passed clean.
