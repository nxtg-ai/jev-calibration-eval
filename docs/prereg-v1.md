# Jev calibration eval: pre-registration v1 (three arms) — public redacted copy

> **About this copy.** This is a public copy of NXTG.AI's internal pre-registration. **No question, arm, item rule, metric, bar, verdict rule or amendment has been changed.** What was changed: internal file paths, ticket and ledger ids, commit hashes, internal team and agent names, and verbatim internal chat quotes are replaced with neutral wording, and the private S2 slices are described without quoting any of their content.
>
> **Registration timestamp.** The pre-registration was registered on 2026-09-29, before any eval-data call, by committing it to NXTG.AI's private repository; each amendment below carries the time it was filed. Those commits are private, so readers cannot verify the timestamps independently. We say so rather than imply otherwise.
>
> Results: [`results-v1.md`](results-v1.md). Code: [`../evals/jev-calibration/`](../evals/jev-calibration/).

**Registered:** 2026-09-29, **before any eval-data call.** The study was approved by NXTG.AI's founder on 2026-09-19. On 2026-09-29 he added the local-model arm. The local arm also informs an internal decision: which typed decisions could run on a local "permanence floor" tier that keeps working without hosted models.

## 0. What happened before registration (disclosed)

Two **access smoke calls**. Neither used eval data, and neither informs any bar below:
- **Jev:** `POST https://api.typesafe.ai/v1/systemone` → HTTP 200 in 0.23 s, returned `model: jev-1.13.0`, answered a noul + a 3-option choice with probabilities (the docs' own example text).
- **Local:** ollama 0.34.4 on our GPU host, `qwen3:14b`, `think:false`, `temperature:0`, `logprobs:true`, `top_logprobs:5` → a single-token letter answer with per-option logprobs. That smoke returned **A = 0.0 vs B = −29.2**, a near-certainty. Local models are often *over*-confident under logprobs, and RQ1/RQ2 exist to measure exactly that. First call was 43 s including a cold 9.3 GB load (`keep_alive:0`).

## 1. Questions

- **RQ1:** Is Jev's reported confidence **calibrated** on typed decisions whose correct answer is known from human ground truth?
- **RQ2:** On the same items, does a **local open-weights model on our own GPU** match Jev and a frontier baseline on accuracy and calibration, **per decision type**?
- **RQ3:** What does each decision actually cost in latency and money, measured on our hardware and accounts rather than taken from vendor claims?

**Decision this eval drives:** which typed-decision classes can run on the local tier (the degraded-mode floor), which stay on Jev, and which need a frontier model. A class moves to the local tier only if it meets H2 on that class.

## 2. Arms (pinned; a version change mid-run invalidates that arm and it re-runs)

| Arm | Model | Probability source | Notes |
|---|---|---|---|
| **JEV** | `jev-1.13.0` (the returned `model` string, logged per call) | Returned `probabilities` (choice/score) or `noul` | Hosted only. Criteria map carries the option descriptions |
| **LOCAL-P** (primary) | `qwen3:14b` via ollama on an RTX 4090 host | `top_logprobs` over the valid option-label tokens, renormalised over valid labels only | `think:false`, `temperature:0`, `num_predict:1`, single-token labels A–T (sampling superseded by A1) |
| **LOCAL-C** (challengers) | `Qwen3-Coder-30B-A3B UD-Q4_K_XL`, `gpt-oss:20b` | same | Run only when the free-VRAM precondition in §7 holds. gpt-oss is a second model family, included to expose family bias |
| **FRONTIER** | `claude-opus-5-5` via `claude -p` (existing subscription; no new spend) | **Verbalised** per-option probabilities as JSON. Secondary: 5-sample empirical frequency at T=1 on a fixed 100-item subset (redefined by A4.1) | No logprobs exposed, so the arm is labelled *verbalised*. That is a known-weaker estimator, disclosed rather than hidden |

**Prompt parity:** one canonical instruction text per task. JEV receives it as `instructions` plus a `criteria` map; LOCAL and FRONTIER receive the same text with lettered options. No arm gets examples the others do not. The option order is fixed per item and identical across arms.

## 3. Items (step E.2 assembles them; this section fixes admission rules and slices)

**Gold must come from humans or from human-ruled records. No model-generated labels, ever.** That is the whole differentiator from the market's published benchmarks.

| Slice | What | Gold source | Target |
|---|---|---|---|
| **S1 (market anchor)** | Document / contract field classification typed as choice / noul / score, the task shape Jev published a number for | A public, **human-labelled** dataset whose licence permits evaluation use; the licence is checked and recorded before admission (CUAD v1, CC BY 4.0) | ≥150 |
| **S2a** (private) | *Which agent team authored this internal coordination post?* (choice over the active team roster) | The post's author field, which is captured by the system, not typed by a person. Signatures, @mentions and self-names are stripped from the text | ≥50 |
| **S2b** (private) | *Which internal program does this work item belong to?* (choice over the top-level programs) | The program prefix of the item's id, placed by the program manager's ruling. The text is the item's title plus entry/exit criteria, with ids stripped | ≥50 |
| **S2c** (private) | *Which team owns this tracked commitment?* (choice) | The owner field, only where the person creating the record **set it explicitly** (derived or blank owners are excluded) | ≥50 |

**Total ≥300.** S2 is the internal half, and it is the decision set our own operations actually need: routing, placement, ownership. **S2 is private operating text and is not released in any form**; only aggregate S2 metrics are published.

**Admission rules:**
- Option count ≤20, which keeps single-token letter labels on the local arm fair.
- Items that need counting, date ordering or numeric precision (Jev's own published weak spots, per the `jev-1.13` jaggedness notes) are **tagged and reported as their own sub-slice**, never silently dropped and never pooled so as to hide a loss.
- De-duplicate near-identical items.
- Freeze `items-v1.jsonl` with a SHA-256 before the first eval call. (The full 340-row frozen file has sha256 `87b5417f9e2c2a5bc5df01c886cf64634cdd13fb7dea96a8d7cee38a414e067d`. The public repo carries only its 160 S1 rows, byte-identical lines, with their own pin; the full-file hash cannot be checked publicly.)
- The S1 contamination risk (public data may sit in training sets) is disclosed. S2 is private.

**Split:** a seeded 20% calibration split and an 80% test split, identical for all arms. The seed is recorded in the frozen items file.

## 4. Metrics

- **Correctness:** accuracy and macro-F1, per slice.
- **Calibration (primary):** multiclass Brier; **ECE with 15 equal-mass bins**; a reliability curve; NLL. All reported on **raw** probabilities.
  - Secondary: the same metrics after **temperature scaling fitted on the calibration split only**. Every arm gets the identical treatment.
- **Routing value:** selective prediction, meaning coverage at ≥95% precision and AURC. This is how we would actually use confidence (the confidence-gated routing pattern).
- **Cost/latency:** p50/p95 wall-clock per decision; tokens per decision.
  - Dollars at list price for comparison only.
  - LOCAL is reported as GPU wall-clock, with cold load separated from warm calls.
- **Uncertainty:** 2,000-resample bootstrap 95% CIs. Arm comparisons use a **paired** bootstrap on per-item Brier differences.

## 5. Hypotheses and bars (committed here, before results)

| ID | Claim tested | Bar | Verdict words |
|---|---|---|---|
| **H1** | Jev's confidence is calibrated | Test-split raw ECE ≤0.05 → *calibrated*; 0.05–0.10 → *roughly calibrated*; >0.10 → *not calibrated as claimed*. Reported per slice and pooled. **A pooled pass does not override a slice fail** | as stated |
| **H2** | The local model matches on a decision class | Per slice: LOCAL-P accuracy within **3 points** of the best non-local arm **AND** its Brier upper-CI ≤ Jev's Brier upper-CI | *matches on this class* / *does not* |
| **H3** | Confidence is usable for routing | Coverage at ≥95% precision, reported per arm and slice. No bar, descriptive only | — |
| **H4** | Vendor speed/cost claims | Independently measured p50 latency and cost versus Jev's published tier-appropriate claims (~25× faster, ~76× cheaper versus a frontier model) | *reproduced* / *not reproduced* / *partially* |

A result that misses a bar is reported as a miss, with the numbers. A bar is **never moved after data is seen**. Any post-registration change goes in §9 with a timestamp and reason, and results under a changed protocol carry that label.

## 6. Instruments, independence and certification

- **Scorer:** deterministic code, not a model. Its correctness is **mutation-proven**: inject a known miscalibration (probabilities sharpened ×2) into a fixture, and ECE and Brier must rise. Inject a label flip, and accuracy must fall. **A scorer that cannot fail is not a scorer.**
- **Per-item raw log (JSONL):** item id, arm, returned model string, every option probability, chosen answer, latency, tokens, error.
- **Every run carries:** runner git sha, `items-v1.jsonl` hash, arm config hash, `max_tokens`, `graded_by: deterministic-scorer@<sha>`, an abstention rubric, the full user-seen payload, and LangFuse trace IDs or a valid unreachable proof.
  - This list comes directly from why our earlier small-model eval runs failed our internal eval-rail certifier (missing required fields). This run is **born certifiable**; UNKNOWN never renders as PASS.
- **Grader independence:** no arm grades any arm. Gold labels come from human-ruled records or public human labels.
- **Step E.5 method review:** an independent cross-vendor reviewer (OpenAI Codex), with the request in neutral engineering language: *"show where confidence and accuracy diverge; check the reliability curve's binning; reproduce the scorer on the fixtures"*.

## 7. Where it runs, and the shared-GPU rule

- **Serving:** the GPU host (RTX 4090), the only machine with a GPU. The other development machine has none, so a 14–30B model would run CPU-only in 19 GB of RAM, which is impractical.
- **Harness:** co-located on the GPU host. It calls ollama on loopback, plus Jev and the frontier model over the network (superseded by A2).
- **Why no network change:** the ollama listener is bound to loopback. **No listener change is made to the GPU host's shared server.** Any remote development access uses an SSH tunnel.
- **VRAM precondition before loading any local model:** `nvidia-smi` free memory ≥ model size + 2 GB, else wait. Probed 2026-09-29: ~10.6 GB already held by another workload, so `qwen3:14b` fits and the 17 GB Qwen3-Coder does not.
- **Run window:** `keep_alive` holds the model loaded during the window only, then unloads. Nothing runs while the GPU's other workload is active.

## 8. Spend and publication

- **No purchase.** Jev usage runs on the early-access allocation. **If TypeSafe usage becomes billable beyond that allocation, the run halts and asks the founder** (purchasing is his decision). FRONTIER runs through the existing Claude subscription CLI.
- **Publication** was pre-authorized by the founder. Fairness rules are fixed now: publish Jev's real wins (speed and cost are real), cite their own public claims, report per slice, and link this pre-registration so readers can see the bars predate the results.

## 9. Amendments

**A1–A3 · 2026-09-29 ~23:10Z · filed BEFORE any eval-data call.** They correct defects the author found by reading our eval-rail standard. No results existed to steer them.

- **A1: Sampling validity (the rail's sampling-validity gate, GATE F). REPLACES §2 "temperature:0" for the LOCAL arms.** A blind cross-family `temperature:0` would make every Qwen3 arm uncertifiable. The sampling registry (`governance/evals/sampling-registry.json` in this repo, public subset) has `qwen3-14b: greedy_ok=false`, with non-thinking config `temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5`, cited to the model card.
  - Each local arm now pins its registry-recommended config plus a fixed seed.
  - Probabilities are read from `top_logprobs` over the valid labels.
  - A pre-run check (no eval data) establishes whether ollama's returned logprobs are computed before or after the sampling temperature is applied. The finding is recorded and applied identically to every local arm.
  - The author had written the sampling-validity rule and still wrote `temperature:0` into this pre-registration. That is exactly why a written rule is not a control.
- **A2: Harness host. REPLACES §7 "harness co-located on the GPU host".** The harness runs on a separate machine and reaches the GPU host's ollama through an SSH tunnel, so the TypeSafe key stays on one machine.
  - LOCAL latency is taken from ollama's own server-side `total_duration`, `load_duration` and `eval_duration` fields, so the tunnel adds nothing to the local arm's measured latency.
  - Client wall-clock is reported separately.
- **A3: Certification prerequisites made explicit (GATE F / GATE O).** Before step E.4, the sampling registry must carry card-cited entries for `jev-1.13.0` (no sampling parameters exposed), `claude-opus-5-5`, `gpt-oss:20b` and `Qwen3-Coder-30B-A3B`.
  - An unknown model means uncertified.
  - Every run carries LangFuse trace IDs or a recorded `langfuse_unreachable_proof`.

**A4 · 2026-09-29 ~23:31Z · filed BEFORE any eval-data call.** It answers three open items from the registry owner's review. No results existed.

- **A4.1: FRONTIER secondary estimator. REPLACES §2 "5-sample empirical frequency at T=1".** `claude -p` exposes no temperature control; `claude --help` offers only `--effort`. "T=1" therefore cannot be pinned.
  - New definition: 5 independent `claude -p` samples at provider-default sampling on the fixed 100-item subset.
  - Each record logs temperature as `provider-default (not settable or observable via claude -p)`. Anthropic documents a default of 1.0, but that is a documented default, not an observation, and the receipt says so.
  - `--effort` is pinned explicitly and logged per call. Our internal operations repo's project settings pin `xhigh`, so a run launched from there would otherwise inherit it silently.
  - The secondary stays descriptive only and feeds no H-test.
- **A4.2: Model identity (GATE F key match).** GATE F looks up `pins.model_id` in the registry verbatim. Two of the three local registry keys are not the ollama serving tags, so every record carries three fields:
  - `model_id` = the registry key;
  - `serving_tag` = the ollama tag;
  - `model_digest` from `/api/tags`.
  - A digest change mid-run counts as a §2 version change. A run whose `model_id` is not a verbatim registry key fails the preflight before the first eval call, not at certification.
  - Mapping, from the GPU host's ollama 0.34.4 `/api/tags` probe of 2026-09-29 ~23:31Z:

  | `model_id` (registry key) | `serving_tag` | digest | arm |
  |---|---|---|---|
  | `qwen3-14b` | `qwen3:14b` (Q4_K_M) | `bdbd181c33f2` | LOCAL-P |
  | `Qwen3-Coder-30B-A3B` | `hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:UD-Q4_K_XL` | `683e1639bbff` | LOCAL-C |
  | `gpt-oss:20b` | `gpt-oss:20b` (MXFP4) | `17052f91a42e` | LOCAL-C |
  | `claude-opus-5-5` | n/a (`claude -p`) | n/a | FRONTIER |
  | `jev-1.13.0` | n/a (hosted) | returned `model` string, logged per call | JEV |

  - **Use the base HF tag, never the derived in-house models on the same host.** Those may carry a system prompt or parameters, which would contaminate prompt parity.
- **A4.3: Penalties are covered by A1's pre-run check, not only temperature.** The registry owner reported that `presence_penalty 1.5` is AWQ-derived and unconfirmed for GGUF, and called it moot under a one-token read. We do not accept "moot" without the probe. Every option letter A–T appears in the prompt, so a context-window repetition penalty could shift label logprobs non-uniformly, for example where a letter also occurs in the passage.
  - The A1 check therefore runs a non-eval prompt twice: once with the registry config, once with penalties neutralised.
  - If `top_logprobs` differ, the read is post-sampling-chain. That is disclosed, and the config is applied identically to every local arm.
  - Each record logs the options object exactly as sent, plus the ollama version.
- **A4.4: gpt-oss:20b sampling now has a primary-source cite.** The `openai/gpt-oss` README, §"Recommended Sampling Parameters", says verbatim: *"We recommend sampling with `temperature=1.0` and `top_p=1.0`."*
  - `greedy_ok` stays FALSE, because there is still no positive greedy cite.
  - The arm pins `temperature 1.0`, `top_p 1.0` and a seed.

**A5 · 2026-09-29 ~23:40Z · filed BEFORE any eval-data call.** It answers an independent reviewer's point: A4.1 said to pin `--effort` but named no value. No results existed.

- **A5.1: The FRONTIER effort value is `high`.** Three reasons:
  1. It is the documented default tier of the effort ladder (`low · medium · high · xhigh · max`).
  2. It matches how TypeSafe's own reference labellers were run: *"both at high thinking"* (evals.typesafe.ai).
  3. `xhigh` is our internal operating pin for coordination agents, not a representative production setting.
  - Effort sensitivity is out of scope, and that is disclosed.
  - The value is part of the arm's identity, so a mid-run change counts as a §2 version change.
- **A5.2: Passing the flag does not prove it took effect.** Measured 2026-09-29:
  - `CLAUDE_CODE_EFFORT_LEVEL=xhigh` is present in the environment of every tool subprocess launched from an agent session in our internal operations repo, because settings `env` is exported to children.
  - That repo's project settings also pin it at project scope.
  - A harness started from such a session would hand `claude -p` a pinned env var **and** project settings. Whether `--effort` outranks either one is unverified.
  - The FRONTIER arm therefore runs `claude -p` in a strict order:
    1. remove `CLAUDE_CODE_EFFORT_LEVEL` from the child environment;
    2. set the working directory to one with no `.claude/settings*.json` pinning effort (outside that repo);
    3. pass `--effort high`.
  - The preflight asserts both of the first two conditions and logs them per run. If it fails, the run aborts before the first eval call.
- **A5.3: GATE F and GATE H for providers that expose no sampling control.** Step E.4 does not start until a versioned eval-rail change lands: a not-exposed-sampling path, accepted only with a cited registry field and fail-closed otherwise, with the standard updated in the same change. This way every receipt is certification-shaped from its first call. No JEV or FRONTIER receipt is certified by bending GATE F inside the harness change.

**A6 · 2026-09-30 ~00:00Z · filed BEFORE any eval-data call.** It answers the independent cross-vendor reviewer's HOLD on the harness change. No results existed.

- **A6.1: Tie groups are atomic in every binned or thresholded metric. This amends §4 "ECE with 15 equal-mass bins".**
  - The reviewer's P0: the shipped scorer split tie groups by item order, so ECE and AURC changed under a pure permutation. A second reviewer reproduced it: 16 items, all at confidence 0.5, give ECE 0.5 or 0.4375 and AURC 0.1686 or 0.4368, depending only on order.
  - **Rule:** bin edges fall only between distinct confidence values. The target is 15 near-equal-mass bins. The realized bin count and the per-bin n are reported per arm, and the count can fall below 15 when ties sit on a boundary.
  - AURC is computed over distinct-confidence thresholds, so a tie group enters the risk-coverage curve all at once.
  - Coverage at ≥95% precision already cut only between distinct values in the shipped scorer (`coverage_at_precision`), so it is consistent and unchanged.
  - Brier and NLL are per-item and unaffected.
- **A6.2: 15 equal-width ECE bins are added as a pre-declared secondary.** The tie rule alone introduces a cross-arm confound:
  - Tie-preserving equal-mass binning makes the realized bin count depend on the arm. JEV's quantized outputs will produce fewer, larger bins than a continuous local arm.
  - The small-sample upward bias of binned ECE depends on items per bin, so the confound leans toward the quantized arm.
  - Equal-width bins on [0,1] are order-invariant by construction and comparable with the literature.
  - Both ECEs are reported. If an arm's H1 verdict differs between them, that arm's H1 is reported as **binning-sensitive** and claimed neither way.
  - Cross-arm comparison stays on the paired per-item Brier bootstrap (§4), which has no bins.
- **A6.3: Permutation invariance becomes a scorer test.** Every metric must be bit-identical across several shuffled item orders, including a tie-heavy fixture. Mutation proof: reverting the tie fix must turn this arm red.
  - Note: the pre-fix suite killed 18 of 18 mutants and still could not see this defect. Mutation testing only covers the fault classes its mutations encode.
- **A6.4: The abstention rubric (GATE M) is endorsed by the founder, not by the study author.**
  - The rubric (harness README §Abstention and the `scorer.py` docstring): an abstention is never dropped. It is scored as the uniform distribution for Brier and NLL, and as incorrect at confidence 1/K for ECE, the reliability curve, accuracy, macro-F1 and selective prediction. Abstention counts are reported per arm.
  - The study author wrote this pre-registration. Endorsing its rubric would be the generator grading itself: the endorsement would carry no independent information.
  - Until the founder endorses the frozen rubric, GATE M reading uncertified is the correct state.

**A7 · 2026-09-30 ~00:15Z · filed BEFORE any eval-data call.** It answers an independent reviewer's finding: **`jev-1.13.0` is not deterministic.**
- **The evidence** is 28 synthetic probes, each sent twice as an identical request, with the pinned model confirmed on all 56 calls. No seed or temperature is exposed, and probabilities are quantized to 0.01.
- **The author's independent recompute from the raw log:** 20 of 28 pairs differ; the maximum spread is 0.06 on a Choice option and 0.05 on a Noul; the winning option flips in 0 of 28 pairs.
- The reviewer's count was 22 of 28 and about 0.07. That difference is a population question and does not change this amendment.

- **A7.1: The primary measurement is the deployed behaviour: one draw per item.**
  - A caller makes one call and acts on the number it returns. Averaging several draws would measure an ensemble nobody deploys, and averaging lowers variance, so it would flatter calibration.
  - The primary draw is **draw 1**, the first call for each item. That is fixed now; it is never the best of k.
  - The paired Brier bootstrap uses draw 1 of each arm.
- **A7.2: Each arm is re-drawn to measure its own jitter, with k = 3.**
  - **JEV and LOCAL:** all items. For JEV that is about 1,020 calls, roughly $0.014 at the measured rate. The §8 halt-if-billable rule still applies.
  - **FRONTIER:** the fixed 100-item subset only; its A4.1 samples serve. Subscription latency makes k draws over all items infeasible. This asymmetry is disclosed.
- **A7.3: Per-arm jitter is reported.** Three figures per arm, across draws:
  - the mean absolute probability difference;
  - the maximum absolute probability difference;
  - the share of items whose predicted answer flips between draws.
- **A7.4: A draw-sensitivity rule, parallel to A6.2's binning-sensitive rule.**
  - Every metric is computed separately on each single-draw run (draw 1, 2 and 3).
  - An H1 or H2 verdict is claimed only if it is the same on every single-draw run. Otherwise it is reported as **draw-sensitive** and claimed neither way.
  - For FRONTIER, draw-sensitivity is assessed on the 100-item subset.
- **A7.5: The mean-of-k probabilities are descriptive only.** They are reported as a "k-averaged" variant, and no H-test uses them.
- **A7.6: "No sampling parameters exposed" does not mean "deterministic".** The registry should record *measured* determinism (the repeat-probe result plus its evidence), never infer it from the absence of knobs.

**A8 · 2026-09-30 ~00:40Z · filed BEFORE any eval-data call. This is the last planned amendment (see A8.3).**

- **A8.1: The FRONTIER fixed subset is defined.**
  - The list is `frontier-subset-v1.txt` (sha256 `4fcee7ab3655d98f…`), produced by `scripts/jev-eval-frontier-subset.py`.
  - Population: the test split only. Strata: slice, plus gold within S1-noul. Quotas: largest-remainder proportional. Order within a stratum: sha256 with a salt.
  - Result: S1-choice 17, S1-noul 30 (14 yes / 16 no), S2a 18, S2b 18, S2c 17. `--check` reproduces it.
  - The harness's certifying mode accepts no other FRONTIER redraw set.
  - *(Public copy: the repo carries the 47 S1 members as `frontier-subset-v1-s1.txt`. The quotas were computed over all five slices, so re-running the script on the S1 rows alone selects a different set.)*
- **A8.2: Data sent to the hosted arms.**
  - The S2 slices are internal operating text. A secret scan of the items came back clean.
  - TypeSafe's privacy policy (https://typesafe.ai/privacy, raw page checked 2026-09-30) says verbatim: *"We will not train or fine tune any artificial intelligence or machine learning models on your prompts or other Input"*, and it *"will not disclose any Input to a third party other than our service providers."*
  - On retention, it says only *"for as long as reasonably necessary to provide you with the Services, or otherwise in support of our business or commercial purposes"*, with no fixed period. That is accepted for this data class. It is disclosed here so the choice is on the record.
  - FRONTIER goes to Anthropic through the subscription already used for this work.
- **A8.3: FREEZE and a stopping rule.** The founder asked, on 2026-09-30, that validation not turn into open-ended loops that add no value.
  - **The pre-registration is frozen at A8.** A further amendment is allowed only for a defect that would make a reported number or verdict wrong. Everything else goes to a post-run *Limitations and follow-ups* section, not into the plan.
  - **Harness gate:** a review finding blocks the harness merge only if it changes a reported number, a verdict or certification. Anything else is logged as a follow-up and does not hold the merge.
    - For the record, the eight cross-vendor review findings so far all met that bar: permutation-dependent metrics and CIs, a forbidden verdict path, unenforced k, and the rest.
  - **Exit conditions for step E.4, the complete list:**
    1. the harness change merged;
    2. the GATE F/H eval-rail change landed;
    3. endorsement of the abstention rubric (GATE M);
    4. the VRAM window for LOCAL.
  - Nothing else gates the run.
  - **Status note, 2026-09-30 (a correction of fact, not an amendment).**
    - A6.4 named the founder as the endorser. The founder then delegated every sign-off for this study to the study author.
    - The study author endorsed the rubric under that delegation, in `evals/jev-calibration/config/abstention-rubric.json`.
    - The config records the study author as the signer, never the founder. The record therefore does not show a founder act that did not happen.
    - Exit condition 3 is met.
  - **Pre-run disclosures, 2026-09-30, written BEFORE any eval-data call (not amendments; no primary metric, bar or verdict rule changes).**
    - **The gpt-oss arm is infeasible as designed.**
      - In its precheck receipt, the first generated token is the harmony `<|channel|>` header at p = 1.0 and label mass is 0.
      - LOCAL-C-GPTOSS does not run. The eval loses its second local model family, and that loss is reported under Limitations.
      - A final-channel read would measure a different quantity. It is a follow-up design, not a patch.
    - **Identifier-embedded gold names (found in an independent review of the items).** The frozen leak rule (word-boundary, case-insensitive) treats `_` as a word character. So three TEST items in the private S2 slices (S2a-023, S2c-016, S2c-046) keep their gold team's name inside a code identifier in the item text. (The identifiers themselves are S2 content and are not quoted here.)
    - **Why they stay, and what is added.**
      - These identifiers are arguably legitimate routing evidence, not leaks. The frozen set cannot change.
      - The primary analysis is unchanged and includes them.
      - Declared now, before any call: every S2 headline metric is ALSO reported as a secondary sensitivity read that excludes these 3 items. If a verdict differs between the two reads, the report says so beside the verdict.
    - **LOCAL arms pin ollama `num_ctx` = 4096, and the §7 VRAM gate now also prices the KV cache.** Written after the aborted LOCAL-P attempt and before the certifying LOCAL-P run.
      - **What happened.** The first live LOCAL-P run (2026-09-30T17:15Z) used ollama's default context, 40,960 tokens for qwen3:14b. Its KV cache (about 6,400 MiB at f16) was not priced by the §7 gate, so it spilled into shared system memory. That would have confounded LOCAL latency, a reported RQ3/H4 metric. The run was aborted after 58 raw records. They are kept privately as evidence of the spill, no score was computed from them, and they are excluded from every reported number.
      - **The change.** `num_ctx` = 4096 is in the LOCAL `options`, so it is in every record. Measured with the Qwen3 tokenizer and chat template by an independent reviewer, the largest of the 340 prompts is S1-choice-045 at 1,213 tokens, 30% of the window. A preflight refuses the run before any call if any item's estimate (characters / 2.5) exceeds 4,032. ollama 0.34.4 truncates an overflowing prompt to about half the window, so a runtime check halts the run at `prompt_eval_count` ≥ 1,984, and a possibly truncated prompt is never scored.
      - **The §7 gate, made stricter, not looser.** Free VRAM must now be at least weights + KV at `num_ctx` + 2 GB. KV is computed from the model's own `/api/show` metadata, and missing metadata refuses. For qwen3:14b at 4096 that adds 640 MiB.
      - **Effect on reported numbers.** Every prompt fits well inside the window, so answers and probabilities are not expected to change. The pin exists so that LOCAL latency measures the model, not a PCIe spill.
      - **Precheck binding.** The earlier valid qwen3 precheck receipt was taken without `num_ctx`, and the binding requires equal options. So LOCAL-P needs a new precheck with `num_ctx` before its certifying run.
    - **The single-arm LOCAL-P run is disclosed, not primary; the reported run is all three arms (2026-09-30 ~22:05Z, BEFORE the paired run).**
      - **What exists.** Run `jevcal-e4b-local-p-20260930T193843Z` (LOCAL-P only, 1,020 records, 0 errors, no VRAM spill). Our certifier returns uncertified on GATE E (`win_gap`, `parity` missing), because the runner computes those reads only with two or more arms. It can never certify. It is kept and disclosed, and none of its numbers is a reported result.
      - **What H2 needs.** §5 H2 compares LOCAL-P with the *best non-local arm* per slice. A two-arm run can answer that only if the absent arm happens to be worse, and choosing the comparator after seeing data is what §5 forbids. So the reported run carries **all three arms, in this declared order: `LOCAL-P,JEV,FRONTIER`**. The GATE E pair is the first two (LOCAL-P versus JEV, matching RQ2's "match Jev"). H2 is computed against **both** non-local arms per slice, and the §5 rule is applied as written: best non-local arm by test accuracy on that slice, then both bar conditions.
      - **Determinism check, declared in advance.** Same items digest, seed and LOCAL options (`temperature 0`, `num_predict 1`, `num_ctx 4096`) as run 193843Z. Prediction: LOCAL-P's per-item argmax label matches 193843Z on **100%** of items (the qwen3 precheck found warm calls exactly repeatable). The exact match rate is reported. Any mismatch is investigated and reported, never dropped.
      - **Not an amendment.** No bar, metric or verdict rule changes. The three-arm design is §2 as registered. This only fixes which run is reported and names the GATE E pair in advance. A8.3's freeze stands.
      - **Correction of fact (2026-10-01 ~21:20Z, found writing the results).** The line above lists the LOCAL options as `temperature 0`. That is wrong: every LOCAL-P record carries the A1 registry config (`temp 0.7`, `top_p 0.8`, `top_k 20`, `min_p 0`, `presence_penalty 1.5`, `seed 20260929`) plus `num_predict 1` and `num_ctx 4096`. The prediction still held (340/340 per draw) because the label is read from the first token's logprobs, which on ollama 0.34.4 are the **pre-sampler** distribution; sampling settings do not move them. No number or verdict changes.
    - **The first three-arm run is disclosed, not primary: it fails GATE H on a recording defect (2026-10-01 ~01:20Z, BEFORE the re-run).**
      - **What exists.** Run `jevcal-d2-3arm-20260930T230651Z`: all three arms, 2,580 raw records plus 500 secondary samples, exit code 0, no VRAM spill. Our certifier passes 16 gates and fails one: GATE H, "contestant qwen3-14b missing max_tokens". The LOCAL arm caps output with `num_predict` = 1 on every call, but its `sampling_config()` never reported it, so the run record carries `null`. The single-arm run never reached GATE H, which reads n/a when not head-to-head.
      - **What is reported.** A re-run of the identical invocation after the reporting fix. Nothing that reaches the model changes: options, prompts, items and arm order are all the same. Its numbers are the reported ones. None of 230651Z's numbers is a reported result.
      - **Determinism, measured, and the declared prediction met.** LOCAL-P's per-item argmax label matches run 193843Z on 340 of 340 items on each of draws 1, 2 and 3 (match rate 1.0, `jevcal.compare_choices`). Prediction for the re-run: 100% again on LOCAL-P. JEV is not deterministic (A7), so no match is predicted for it.
      - **Not an amendment.** No bar, metric or verdict rule changes.
    - **How A7.4 is computed for H2 (2026-10-01 ~06:45Z, written AFTER the reported run `jevcal-d2b-3arm-20261001T043407Z` certified 17/17 and BEFORE any per-draw H2 is computed).**
      - **Why this is needed.** The runner computes H2 for draw 1 only (`reads.h2`); its `draw_analysis` says per-draw H2 is not computed. A7.4 says every metric is computed per single-draw run, and that FRONTIER's draw-sensitivity uses the 100-item subset. It does not say which items the per-draw H2 uses when the comparator includes FRONTIER. This note fixes that before the numbers exist.
      - **What the author had already seen.** The draw-1 `reads.h2` block in the run record, opened while locating the analysis code: the S1-choice verdict ("does not") and the start of S1-noul. Nothing from draws 2 or 3.
      - **Read A (FRONTIER subset, per A7.4).** For each draw d in 1, 2, 3, per slice: H2 on the test items in the FRONTIER subset, with all three arms at draw d. Every arm is compared on the same items in every draw.
      - **Read C (all test items for LOCAL-P and JEV, per A7.2).** For each draw d: per slice, LOCAL-P and JEV at draw d on every test item; FRONTIER held at draw 1, because it has only one draw outside the subset. Read C at d = 1 is the primary `reads.h2`.
      - **The judge.** The shipped `scorer.h2_for_draw`, with inputs built exactly as `run.h2_read` builds them. Before any per-draw number is used, the script must reproduce `reads.h2` exactly at d = 1 under Read C, for every slice: points, CIs and verdicts. If it does not, the script is wrong and nothing is reported from it.
      - **The claim rule.** H2 is claimed for a slice only if the verdict is the same in all three draws under Read C, and the same in all three draws under Read A. Otherwise that slice is reported as **draw-sensitive** and claimed neither way. Read A can differ from Read C only because its population is smaller; that is reported, not reconciled.
      - **The S2 identifier read (declared before the run, above)** is computed for the primary H2 too: excluding S2a-023, S2c-016 and S2c-046. A verdict that differs is reported beside the primary one.
      - **Independent check.** Before any H2 words are published, an independent reviewer reproduces the per-draw table from the raw files and this text, without the author's script.
      - **Which read is claimed when both are draw-stable and disagree (added 2026-10-01 ~07:10Z, AFTER both tables existed; the independent reviewer found the gap).** Read C governs. It is the primary population (all test items, A7.1), its n is larger, and choosing it cannot move a class onto the local tier. Read A is published beside it. The only slice this touches is S1-noul, where C says "does not" in every draw and A says "matches" in every draw. This line is written after the data was seen, so it is labelled that way; it picks the read that makes the bar harder to pass, never easier.
      - **Bootstrap seed.** The rule text names no seed. The reported Brier upper CIs use the run's seed, 20260929, as `reads.h2` does. The independent reviewer recomputed every verdict under 20 seeds: 0 flips in 48 rows.
      - **Not an amendment.** No bar, metric or verdict rule changes. This fixes how an already-declared rule is computed.

    - **How A7.4 is computed for H1 (2026-10-01 ~07:30Z, written BEFORE the per-slice H1 for draws 2 and 3 is computed).**
      - **Why.** The runner reports H1 per slice for draw 1 only (`metrics.JEV.test_split_h1`) and its per-draw H1 (`metrics.JEV.draws.h1_draw_sensitivity`) for the pooled test split only. A7.4 needs every slice per draw.
      - **What the author had already seen.** Draw 1 per slice and pooled, and the pooled verdict for draws 2 and 3 (all three "not calibrated as claimed").
      - **Population.** JEV only (H1 is about Jev). All test items, per slice and pooled, at each draw d in 1, 2, 3; JEV was re-drawn on every item (A7.2), so no subset is needed.
      - **The judge.** The shipped `scorer.h1_for_draw` on a `DrawRun` built as `run.py` builds it for `test_split_h1`, which applies both ECE binnings and the A6.2 binning-sensitive rule. The script must first reproduce the draw-1 per-slice and pooled `h1_verdict` exactly, and the pooled draws 2 and 3 in `h1_draw_sensitivity`, or nothing is reported from it.
      - **Claim rule (A7.4 + A6.2 + §5, unchanged).** A slice's H1 word is claimed only if `h1_for_draw` gives the same word on all three draws; otherwise **draw-sensitive**. A binning-sensitive draw is not a stable word. A pooled pass does not override a slice fail.
      - **S2 identifier read.** Draw 1 also excluding S2a-023, S2c-016, S2c-046 (S2 slices and pooled); a differing verdict is reported beside it.
      - **Not an amendment.** No bar, metric or verdict rule changes.

## 10. Next steps (sequence)

1. **E.2:** assemble and freeze `items-v1.jsonl` (S1 licence check first; S2 extracted by deterministic script from system-captured fields).
2. **E.3:** wire the three arms into the eval rail. Mutation-prove the scorer.
3. **E.4:** run in one VRAM-safe window.
4. **E.5:** independent cross-vendor method review.
5. **Synthesis:** build-vs-buy recommendation and the local-tier decision per class.
6. **E.6:** publish.
