# Jev calibration eval harness (study step E.3)

Governed by the pre-registration, public redacted copy at [`../../docs/prereg-v1.md`](../../docs/prereg-v1.md) (bars, metrics, arms and amendments). This harness changes none of them.

> **Public release note.** This README is the harness's own build README, lightly redacted. Statements about "this build" describe the harness as it was built internally (E.3 = fixture-only build, E.4 = live run); the repo root [`README.md`](../../README.md) says what is released and how to reproduce the S1 numbers. In this repo the pinned `governance/evals/jev-calibration/items-v1.jsonl` holds the **160 S1 rows only**; the S2 rows are private.

**Fixture-only in this build.** `jevcal/items.py` refuses any file named `items-v1.jsonl` and anything under `governance/evals/jev-calibration/`. E.4 lifts that in a reviewed commit. There is no env-var escape hatch.

## Layout

| Path | What |
|---|---|
| `jevcal/items.py` | Items schema + loader + the eval-data guard |
| `jevcal/render.py` | The ONE prompt renderer (prompt parity) |
| `jevcal/arms/base.py` | The arm contract (`Arm`, `ArmResult`, `ArmHalt`) |
| `jevcal/arms/jev.py`, `frontier.py` | JEV and FRONTIER adapters |
| `jevcal/arms/__init__.py` | Arm registry (`ARMS`). E.4b adds `LOCAL-P` here |
| `jevcal/scorer.py` | Deterministic scorer (stdlib + numpy) |
| `jevcal/run.py` | The ONE runner |
| `tests/test_scorer.py` | Named scorer arms (hand-derived expected values) |
| `tests/mutate_scorer.py`, `tests/mutate_isolation.py`, `tests/mutate_harness.py`, `tests/mutation_engine.py` | Mutation proofs on the shipped scorer, isolation checks and CODEX-fix guards (one engine) |
| `tests/test_codex_fixes.py`, `config/abstention-rubric.json` | Arms for CODEX PR 72 findings 2-5; the read-only abstention rubric (GATE M input) |
| `tests/test_draws.py`, `tests/mutate_draws.py` | Prereg A7 draws: primary = draw 1, jitter, draw-sensitivity, k-averaged isolation |
| `jevcal/compare_choices.py` | Post-hoc: one arm's per-item argmax-label agreement across two `raw.jsonl` logs (draw 1; row counts beside every count) |
| `tests/test_d1.py`, `tests/mutate_d1.py` | D1 three-arm run: pair-derived GATE E labels, `reads.h2` (§5 H2 per slice, test split, draw 1), certifying arm-set refusals, `compare_choices` |
| `jevcal/isolation.py`, `tests/test_isolation.py` | FRONTIER child isolation preflight (A5.2) and its arms |
| `tests/test_harness.py` | Schema, renderer, adapters (transport stubbed), runner guards |
| `fixtures/items-fixture.jsonl` | 6 synthetic items (no real ledger rows) |
| `fixture-evals-dir/` | The committed fixture run: run record, raw log, ledger row, registry copy, certifier outputs. **Not included in the public release** (internal certifier output and paths) |

## Items schema (one JSONL row per item; E.2 freezes to this)

| Field | Type | Rule |
|---|---|---|
| `item_id` | str | unique in the file |
| `slice` | str | `S1`, `S2a`, `S2b`, `S2c` (fixtures use `FX-*`) |
| `question_type` | str | `choice` \| `noul` \| `score` |
| `instructions` | str | the canonical instruction text, identical for every arm |
| `options` | list[str] | 2–20 ordered keys. The order is fixed and identical across arms. noul: exactly 2, `[yes-side, no-side]`. score: levels, lowest first |
| `criteria` | object \| null | option descriptions. choice/score: `{option: str\|null}`. noul: `{"true","false"}` or keyed by the two options. Sent to Jev as `criteria`, and shown to every other arm next to its lettered option |
| `state` | str | ≤300 words |
| `gold` | str | one of `options`. Human or human-ruled, never a model |
| `split` | str | `calibration` \| `test` |
| `source_ref` | str | provenance of the item and its gold |
| `tags` | list[str] | e.g. `counting`, `date-order`, `numeric` (reported as sub-slices) |

Unknown fields are rejected.

## Arm interface (items JSONL in, per-item records out)

An arm implements `Arm` in `jevcal/arms/base.py`:

- `arm_name`, `family` (unique per run; it keys `pins.contestants`).
- `model_id` **must be a verbatim key of `governance/evals/sampling-registry.json`** (A4.2). The runner PREFLIGHT refuses the run before the first call of any arm if it is not. The check is generic across arms.
- `serving_tag`: what the server is asked for (`"n/a"` for hosted or CLI arms).
- `config()` returns every knob that can change an answer. Its SHA-256 is the arm config hash.
- `sampling_config()` is what GATE F reads. Use the rail key `temp`. Log what cannot be set as a labelled string (FRONTIER: exactly `provider-default (not settable or observable via claude -p)`), never an invented number.
- `call(rendered) -> ArmResult` receives only the `Rendered` from `render()`, never the item. It returns `probabilities` aligned to `rendered.keys`, or `None` for an abstention.
- `call()` raises `ArmHalt` on anything that must stop the run: non-200, a served model that is not the pin, or a billing, quota or spend signal. **Never retry.**

The runner writes one record per (item, arm) to `runs/<run-id>.raw.jsonl` via `ArmResult.to_record()`. Each record carries:

- `item_id`, `arm`
- **Model identity (A4.2):** `model_id` (registry key), `serving_tag`, `model_digest`. For JEV: `jev-1.13.0`, `n/a`, and the returned `model` string, checked against the pin on every call. For FRONTIER: `claude-opus-5-5`, `n/a`, `n/a`.
- `requested_model`, `returned_model`
- `options`, `labels`, `probabilities` (option→p, renormalised), `raw_probabilities`, `raw_sum`
- `answer`, `abstained`, `error`
- `latency_s` (client wall-clock) and `server_latency` (server-reported, optional)
- `input_tokens`, `output_tokens`, `sampling_config`
- `msg_sha256` (canonical render), `wire_sha256`, `wire_request` (the body; credentials never included), `wire_response`
- `cost_usd_list`, plus `slice`, `split`, `gold`, `question_type`, `tags`

### FRONTIER sampling (A4.1)

- `claude -p` has no temperature control. Every record logs `temp` as exactly `provider-default (not settable or observable via claude -p)`. Anthropic documents a default of 1.0, but that is documented, not observed.
- `--effort` is **required, with no default, and hard-pinned to `high`** (prereg A5.1; the single constant `PINNED_EFFORT` in `jevcal/arms/frontier.py`). It comes from `--frontier-effort`, which accepts only `high`: any other value raises before any subprocess is spawned. It is passed explicitly on every call and logged per record in `sampling_config.effort` and `extra.effort`.
  - Why: a run launched from `~/ASIF` (NXTG.AI's internal operations repo) would otherwise silently inherit `xhigh` from its project `.claude/settings.json`.
  - Effort is part of the arm's identity: the runner invalidates the arm if the effort (or a model digest) differs across records.
- **Isolation, not just the flag (A5.2, `jevcal/isolation.py`).** The child env has no `CLAUDE_CODE_EFFORT_LEVEL`, and the child cwd is a fixed temp dir. The preflight checks that this dir is **outside `~/ASIF`/`$ASIF_ROOT`** and that its **whole ancestor chain** has no `.claude/settings*.json` setting that key (an unreadable file fails closed).
  - These checks run on the **exact** env dict and cwd handed to the child: in the runner before the first call of any arm, and in the arm before every call.
  - Both results are logged in the run record (`frontier_isolation`), and cwd/env status per record. Positive control: `python3 -m jevcal.isolation --cwd ~/ASIF --inherit-env` (its receipt, `fixture-evals-dir/receipts/isolation-positive-control.txt`, is not included in the public release).
- **Secondary estimator** (`--frontier-secondary-items <ids>`, `--frontier-secondary-n 5`): N independent single-letter `claude -p` samples per listed item, at provider-default sampling.
  - Output goes to `runs/<id>.secondary.jsonl`, plus an empirical answer frequency in the run record's `frontier_secondary`.
  - It is **descriptive only and feeds no H-test**.

### LOCAL slot (E.4b). Built here: `jevcal/arms/local.py`

`LocalArm(Arm)` is one class parameterised by the registry key. Three arms are registered in
`jevcal/arms/__init__.py` (`ARMS`):

| arm | `model_id` (registry key) | `serving_tag` | `model_digest` pin | family |
|---|---|---|---|---|
| `LOCAL-P` | `qwen3-14b` | `qwen3:14b` (Q4_K_M) | `bdbd181c33f2` | `qwen` |
| `LOCAL-C-CODER` | `Qwen3-Coder-30B-A3B` | `hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:UD-Q4_K_XL` | `683e1639bbff` | `qwen` |
| `LOCAL-C-GPTOSS` | `gpt-oss:20b` | `gpt-oss:20b` (MXFP4) | `17052f91a42e` | `openai-gpt-oss` |

**Family rule.** `LOCAL-P` and `LOCAL-C-CODER` share family `qwen`, so the runner's family-uniqueness
rule (`pins.contestants` is keyed by family) **refuses the two in one run**. That is intended and kept:
they cannot share the 4090's VRAM either (§7). A run picks one `qwen` arm plus, optionally, the
different-family `LOCAL-C-GPTOSS`.

**No-numeric-config refuses.** A model whose registry `recommended` block holds only a note (no
numeric config) **raises `ArmHalt` at construction** ("registry holds no numeric recommended sampling
config") rather than guessing values. `gpt-oss:20b` was that case until an internal registry change landed its card-cited
single block (`temperature 1.0 / top_p 1.0`); `LOCAL-C-GPTOSS` now constructs and pins those. The
refusal path stays exercised by the tests via a crafted registry, so the guard is still load-bearing.

**Environment (no hard-coded host; prereg A2).**
- `JEVCAL_OLLAMA_BASE_URL` — the ollama base URL (E.4 tunnelled from the harness host to the GPU host over SSH). Unset ⇒ refuse.
- `JEVCAL_VRAM_PROBE` — the free-VRAM probe command, e.g. `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits` (or the same behind `ssh -o ConnectTimeout=8 <user>@<gpu-host>`). Split with `shlex`, never a shell. Unset ⇒ refuse.
- `JEVCAL_LOCAL_PRECHECK_<ARM>` (hyphens→underscores, e.g. `JEVCAL_LOCAL_PRECHECK_LOCAL_P`) or `--local-precheck NAME=PATH` — binds an A1/A4.3 precheck receipt. **Required in `--mode certifying`.**

**Behaviour.**
- **Identity (A4.2):** the `/api/tags` digest is read at construction and re-checked at least every N calls and on the first call; a change mid-run is a version change → `ArmHalt` (§2). Each record carries the **observed** digest (updated at each recheck), with the pin kept in `extra.pinned_digest`, so the runner's cross-record identity check compares a real value. A served model that is not the serving tag → `ArmHalt` (an `nxtg-` served model is caught here too; the explicit `nxtg-` branch is belt-and-braces for a clearer diagnostic).
- **Sampling (A1/A4.3):** the registry's numeric recommended config (`recommended.non_thinking`) is mapped key-by-key onto ollama `options` — `temperature/top_p/top_k/min_p/presence_penalty/frequency_penalty` pass through, `repetition_penalty`→`repeat_penalty`, **an unmapped key refuses** — plus a fixed module-constant seed, `num_predict:1`, and a pinned **`num_ctx` (4096)** context window. `sampling_config()` returns the rail-keyed view (`temp`, numeric) GATE F reads. Because `num_ctx` is in `options`, it is in every record and part of the certifying precheck-binding options equality (a receipt taken at a different `num_ctx` no longer matches).
- **`num_ctx` pin (4096) — why (2026-09-30):** ollama's default context is the model's trained window (`qwen3:14b` → 40,960), whose KV cache is ~10× the pinned one. The first live `LOCAL-P` run loaded at 40,960, its KV made the model resident at 15,167 MiB, and on the Windows/WDDM host `llama-server` held 11,186 MiB **dedicated** plus 5,034 MiB **shared** (system RAM over PCIe) — so LOCAL latency (2.2–3.2 s/call), a pre-registered reported metric (RQ3), was spill-confounded and the run was aborted. We pin a small window sized to the frozen items: the largest rendered LOCAL prompt is **6,736 chars** (`S1-choice-045`, canonical 6,626 + the format instruction), **~2,695 tokens** at chars/2.5 — far under 4096.
  - **Preflight (before any call, never truncate silently):** the runner estimates every item's full LOCAL prompt at `chars / 2.5` and **REFUSES** the run, naming the worst item, if any would exceed `num_ctx − 64`. Recorded under `local_preflight.<arm>.numctx`.
  - **Runtime belt (independent reviewer, 2026-09-30, measured on ollama 0.34.4):** on a prompt that exceeds `num_ctx`, ollama does **not** report a count at the window — it silently truncates to about half the window (`prompt_eval_count ≈ num_ctx/2 + 2`: `num_ctx 256 → 130`, `4096 → 2,050`), HTTP 200, `done_reason "length"`. The belt therefore reads the server's **real** `prompt_eval_count` and raises `ArmHalt` when it reaches **`num_ctx//2 − 64` (1,984 at 4096)** — the truncation tell — not the old `≥ num_ctx − 1`, which could never fire. It is **fail-closed** (it would also halt a genuine 1,984–4,095-token prompt), which is acceptable because the preflight and the frozen items stay far below it: the worst real frozen item, `S1-choice-045`, is **1,213 tokens** (well under 1,984, though its chars/2.5 preflight estimate is the larger 2,695).
- **Probabilities:** the first generated token's `top_logprobs` are read over `rendered.labels` only; whitespace label variants (`A`, ` A`) are summed in probability space, then renormalised over the valid labels. No valid label ⇒ an **abstention** (never a drop). The verbatim `top_logprobs` go in `raw_probabilities`; `labels_missing` in `extra`.
- **Latency:** ollama's `total_duration`/`load_duration`/`eval_duration` (ns→s) go in `server_latency`; `latency_s` stays client wall-clock.
- **Logged per record:** the ollama `options` object **exactly as sent**, the ollama version (`/api/version`), and the precheck sha256 + findings, in `wire_request` / `extra`.
- **VRAM precondition (§7), weights AND KV:** checked by `vram_preflight()` **before** the arm is constructed —
  `required_free_mib = ⌈size/2²⁰⌉ + KV_MiB(num_ctx) + 2048`,
  where the model `size` is from `/api/tags` and the KV cache is priced from `/api/show` `model_info`:
  `KV_MiB = ⌈ 2 · block_count · head_count_kv · head_dim · num_ctx · bytes_per_elem / 2²⁰ ⌉` (the `2` is K and V; `head_dim` is `attention.key_length`, else `embedding_length / attention.head_count`; `bytes_per_elem` = 2 for f16). **A missing metadata field REFUSES (fail closed)**, never guesses. Every number — `kv_mib`, `num_ctx`, the formula inputs, `required_free_mib`, `free_mib` — is logged in the run record (`local_preflight`). The old gate priced **only the weights**, so a run at the default 40,960 context spilled KV into shared system RAM and passed anyway (above); pricing KV at the pinned `num_ctx` is what makes the precondition real. For `qwen3:14b` at `num_ctx 4096`, KV = **640 MiB** (`2·40·8·128·4096·2 / 2²⁰`).
  - ⚠️ **WDDM caveat:** on a Windows host, ollama's `/api/ps` reports `size_vram == size` even when part of the model is in shared system RAM — it is **not trustworthy** there. The instrument for spill is **free VRAM before vs after load** (`nvidia-smi memory.free`), which is what `JEVCAL_VRAM_PROBE` reads.
- **`close()`:** unloads the model (`keep_alive:0`); the runner (and the precheck CLI) call it in a `finally` on any arm that has one. A SIGTERM/SIGINT is converted to a clean exit (`install_terminate_handlers()` raises `SystemExit`) so that `finally` runs and the model is not left resident on an aborted run.

**Pre-run check (A1/A4.3), no eval data:**

```bash
python3 -m jevcal.local_precheck --model-id qwen3-14b --out <receipt.json>
```

The precheck loads the model (four `/api/chat` calls), so it enforces the **§7 free-VRAM
precondition first**, exactly as the runner's `build_arm` path does: it runs `vram_preflight()` with
the same `JEVCAL_VRAM_PROBE` env var **before** constructing the arm, and an unset probe or a short
reading is a clean **REFUSED** — a non-zero exit and no model load. The preflight numbers (free MiB,
required MiB, model bytes, probe argv) are recorded in the receipt under `vram_preflight`.

It then runs the committed synthetic probe (`fixtures/local-precheck-probe.jsonl`, whose passage
repeats option letters so a repetition penalty has letters to act on) four times — registry config
twice (repeatability), a temperature that **differs from the registry temperature** (1.0, or 0.5 when
the registry temperature is already 1.0 as it is for gpt-oss; recorded as `b_probe_temperature`) for
the temperature basis, and penalties neutralised for the penalty basis — and records
`temperature_basis`, `penalty_basis`, `repeatable` and `label_first_token_feasible`. In
`--mode certifying` the runner requires a receipt whose `model_id`, digest, **`ollama_version`** and
options match the arm and whose `label_first_token_feasible` is true, before the first call.

**The certifying binding RE-DERIVES the findings; it does not trust the stored block.** A stored
`findings` block is the author's claim, not an instrument — the first live receipt
(`…-CONFOUNDED.json`) carried post-cold-baseline findings and would have passed a gate that trusted
them. So certifying additionally requires the receipt to carry the **raw calls** it recorded
(`calls.warmup`, `calls.a1`, `calls.a2`, `calls.b`, `calls.c`) and a non-empty `labels` list, then
**recomputes the findings with `compute_findings` and requires an exact match** (strings and bools
exactly; the `cache_state_effect` floats within `1e-9`). A receipt from before the warm-up precheck
lacks `warmup` and is refused ("re-run it"). Each basis must also be a **decided** value from an
allowlist — `temperature_basis ∈ {pre-temperature, post-temperature}`,
`penalty_basis ∈ {penalties-not-applied-to-logprobs, post-sampling-chain}` — so
`undetermined (cache state differs)` is refused.

> **Consumer caveat — `repeatable` (and a basis) can be the truthy string
> `"undetermined (cache state differs)"`, not just `true`/`false`.** A non-warm compared call makes
> the finding undetermined, and a non-empty string is truthy, so `if findings["repeatable"]:` reads an
> undetermined result as repeatable. **Test `findings["repeatable"] is True`** (and compare a basis by
> value), never for truthiness. The certifying binding does not trust the stored findings at all — it
> re-derives them from the raw calls.

**Cache state (why the precheck warms up first).** ollama returns the first token's logprobs computed
against whatever prompt prefix is **KV-cached**, so a COLD call (nothing cached) and a WARM call
(prefix cached) differ *by cache state*, not by sampling — the first live receipt (qwen3:14b
2026-09-30T04:03Z) had a cold a1 and warm a2/b/c and read `post-temperature` purely from that. The
precheck therefore fires a **discarded warm-up call** so every compared call is warm, compares
**warm-vs-warm only** (a non-warm compared call makes the basis `undetermined (cache state differs)`,
which certifying refuses), and reports the cold→warm delta separately as `cache_state_effect`. Each
call records its `prompt_eval_cached_count`. The same caveat applies to the eval itself: **LOCAL draw
1 per item is computed with whatever prefix is cached, so LOCAL draw-to-draw jitter reflects KV-cache
state, not sampling.** `wire_response` carries `prompt_eval_cached_count` on every record, so the
draws analysis can stratify by it.

> **One unverified assumption (flagged for the first live E.4 call):** the exact JSON placement of
> ollama's per-token `top_logprobs` in the `/api/chat` response. This build cannot call `/api/chat`
> live (that loads a model onto a shared GPU, which its operator gates), so the parser reads the documented
> native shape (top-level `logprobs[0].top_logprobs`) and falls back to `message.logprobs` /
> OpenAI-compat `choices[0].logprobs.content`, validating structure before use and **abstaining
> (never guessing) if none is present**. The operator confirms the shape on the first real call; a 100%
> abstention rate is the loud signal if it differs.

## Scorer

`jevcal/scorer.py` states every convention in its header:

- **Correctness:** accuracy and macro-F1.
- **Calibration:** multiclass Brier (sum over options, range 0–2), the reliability curve, and NLL (clipped at 1e-15). ECE uses **up to 15 equal-mass, tie-preserving bins** (prereg A6.1):
  - Bin edges fall **only between distinct confidence values**, at the first distinct-value boundary where the cumulative count reaches j·n/15.
  - A tie group is never split. Ties can therefore make bins unequal or reduce their number; all items at one confidence make a single bin.
  - The **realized bin count and per-bin n** are reported per arm and per slice (`ece_bins_realized`, `ece_bin_n`).
- **Selective prediction:** coverage at ≥95% precision (cut only between distinct confidences), and AURC over **distinct-confidence thresholds**, where a tie group enters the risk-coverage curve all at once.
- **Order invariance:** every metric is bit-identical under any permutation of item order. Means use `math.fsum`, and bins are formed from distinct values. `ArmTieInvariance` checks this over shuffled orders, including CODEX's 16-item all-0.5 fixture (ECE 0.0, AURC 0.5) and a tie-heavy fixture.
- **Uncertainty:** 2,000-resample bootstrap 95% CIs, and a paired bootstrap on per-item Brier differences (the cross-arm comparison; it has no bins). Resampling runs in **canonical item_id order**, and paired inputs are **joined by item_id** (differing id sets are refused), so every CI is bit-identical under any permutation of the items.
- **Pre-declared secondaries:**
  - **ECE over 15 equal-width bins on [0,1]** (`ece_equal_width_15`, A6.2), reported next to the primary for every arm and slice. If the two ECEs place an arm in different H1 bands (≤0.05 calibrated, (0.05, 0.10] roughly calibrated, >0.10 not calibrated as claimed), that arm's H1 is reported as **`binning-sensitive`** and claimed neither way (`h1` in the scored output).
  - Temperature scaling, fitted on the calibration split only.

### Draws (prereg A7: `jev-1.13.0` is not deterministic)

- **The primary is draw 1**, the first call per item, never best-of-k. Every metric, the paired Brier bootstrap and the certifier `tasks` read draw 1. Every per-item record carries `draw`.
- **Re-draws:** `--draws k` (E.4: 3).
  - JEV and LOCAL are re-drawn over `--draw-items`, which defaults to all items.
  - FRONTIER is re-drawn over `--frontier-draw-items`, which defaults to its A4.1 subset. Subscription latency is why FRONTIER does not re-draw every item; this asymmetry is disclosed.
  - FRONTIER re-draws are **verbalised re-calls**. The A4.1 single-letter samples carry no probabilities, so they cannot give probability jitter; they remain the separate descriptive estimator.
  - `k` and the subsets are logged in `draws_config`.
- **Per-arm output** (`metrics.<arm>.draws`):
  - `jitter`: mean and max |dp| over options and draw pairs, plus the answer-flip share;
  - `h1_draw_sensitivity`: H1 on each single-draw run on the test split. It is claimed only if identical on every draw, otherwise it reads **`draw-sensitive`**. Verdicts come **only** from `h1_for_draw` / `h2_for_draw`, which accept a single-draw `DrawRun`; `point_metrics` returns metrics only;
  - `k_averaged_descriptive`: **descriptive only, and carries no verdict of any kind** (a `KAveragedRun` is refused by the type check). `tests/test_draws.py` walks the block for any verdict-like key or value.

### Run mode (E.4 go-live; CODEX re-grade P1-3, prereg A7.2 and A8.1)

The frozen eval items and the real eval store (`governance/evals`) are gated by **content**, not path, and only `--mode certifying` opens them (all refusals fire **before any arm is constructed**):

- **`--mode certifying` (default)** runs the frozen items **only** when the items file's sha256 equals the single committed pin `governance/evals/jev-calibration/items-v1.sha256` — at any name or location; a mismatch or a missing pin refuses (F1: a certifying run cannot execute over arbitrary items). The real store is allowed so the run record and cert-ledger row land where the internal eval-rail certifier reads them. It also enforces A7 and refuses, before the first call:
  - any `--draws` other than 3;
  - any `--draw-items`: JEV and LOCAL redraw ALL items;
  - a FRONTIER redraw set other than exactly the committed 100-item subset at `governance/evals/jev-calibration/frontier-subset-v1.txt` (private, because it lists S2 item ids; the public repo carries its 47 S1 members as `frontier-subset-v1-s1.txt`, so certifying runs with FRONTIER cannot start from the public repo). That file is read by repo path, never copied, and its sha256 goes into the record.
  - (D1, before any arm is built) fewer than 2 arms: GATE E needs `reads.win_gap` and `reads.parity`, which exist only for a pair, so such a run could never certify;
  - (D1, before any arm is built) any LOCAL arm without BOTH JEV and FRONTIER: H2 compares LOCAL-P with the best non-local arm per slice. The declared order is `LOCAL-P,JEV,FRONTIER` (prereg §9); the GATE E pair is the first two arms, and the labels on `win_gap`/`parity` name the pair and any LOCAL arm in it. `reads.h2` holds the per-slice H2 read (test split, draw 1); its verdict is null unless both JEV and FRONTIER ran.
- **`--mode fixture`** refuses the frozen eval items by **content** (the pinned bytes at any path — a symlink or byte-copy included, F2) **and** by name/dir, and refuses the real eval store. Otherwise it allows any k or subset; the record says `mode: fixture`, `certifying: false`, `a7_compliant: false`, and **no cert-ledger row is written**, so GATE A of the internal certifier refuses the run.

The frozen `items-v1.jsonl` carries two provenance-only keys (`build_seed`, `leak_strings`) that `parse_item` accepts and drops, and abbreviates the calibration split as `cal` (canonicalised to `calibration` on parse). `leak_strings` are the answer-leak phrases the build guaranteed absent from `state` (word-boundary, case-insensitive); they legitimately appear in the question/options (the clause type being asked about).

### Abstention

This is the rubric of prereg A6.4, read at run time from `config/abstention-rubric.json`:

> An abstention is never dropped. It is scored as the uniform distribution for Brier and NLL, and as incorrect at confidence 1/K for ECE, the reliability curve, accuracy, macro-F1 and selective prediction. Abstention counts are reported per arm.

**`operator_endorsed` is read-only in the harness.** It defaults to `false` and is never written by code. Endorsement is an operator act recorded outside the harness (in v1, a delegated sign-off; see `config/abstention-rubric.json`). Until then, GATE M reads UNCERTIFIED, and that is the correct state.

### Observability fails closed

LangFuse is fail-open **only for a proven outage**:
- If the helper cannot even resolve (not traceable and not provably unreachable), the run aborts before the first call.
- If a run ends with tracing incomplete while LangFuse was reachable, or with neither trace IDs nor a valid `langfuse_unreachable_proof`, the run record is written with `valid: false` and an `invalid_reason`. No cert-ledger row is written, and the exit code is 4.

**LangFuse SDK requirement.** The eval-rail tracer (`deploy/langfuse/eval_rail_trace.py`) imports the `langfuse` Python SDK. On the runner's Python 3.14, SDK v3.x fails to import (a pydantic v1 `ConfigError`); v4.16.0 works — a live smoke on 2026-09-30 wrote 6 traces with 0 errors. Install it with:

```bash
python3 -m pip install --user 'langfuse>=4,<5'
```

Off-the-shelf sweep (probed on scikit-learn 1.8.0, 2026-09-29): `brier_score_loss` does support multiclass, and it agrees with this scorer (0.19333 on the same 3-item probe; `tests/test_scorer.py` cross-checks it when sklearn is installed). scikit-learn has no ECE function, and `calibration_curve(strategy="quantile")` is binary-only. The scorer therefore stays stdlib + numpy, with sklearn used only as an independent oracle in tests.

## Run

```bash
cd evals/jev-calibration
python3 -m unittest discover -s tests -t .            # all arms + adapters + guards
python3 tests/mutate_scorer.py && python3 tests/mutate_isolation.py && python3 tests/mutate_harness.py && python3 tests/mutate_draws.py   # mutation proofs (targets committed + clean)
python3 -m jevcal.isolation --as-arm                  # A5.2 preflight on the exact FRONTIER child env + cwd
JEV_KEY_FILE=<path to a file holding TYPESAFE_API_KEY=...> \
JEVCAL_LANGFUSE_ENV_FILE=<optional LangFuse keypair file> \
python3 -m jevcal.run --mode fixture --items fixtures/items-fixture.jsonl --arms JEV,FRONTIER \
  --frontier-effort high \
  [--frontier-secondary-items <item_id,...>] [--draws 3 --draw-items <ids> --frontier-draw-items <ids>] \
  --evals-dir <scratch dir> --run-id <id> --purpose "<registered purpose>"
# certification (GATES A-O) is done by NXTG.AI's internal eval-rail certifier, which is not part of this release
```
