# Jev calibration study (v1)

A pre-registered, reproducible test of whether **Jev** (`jev-1.13.0`, TypeSafe's hosted "System One" decision model) reports **calibrated** probabilities, compared with a **local open-weights model** (`qwen3:14b` on one RTX 4090) and a **frontier model** (`claude-opus-5-5`, verbalised probabilities), on decisions whose correct answer comes from humans, never from a model.

By [NXTG.AI](https://nxtg.ai). Run 2026-10-01.

## Results in brief

- **Jev is not calibrated as claimed.** Pooled test ECE 0.156; overconfident (mean confidence 0.66 vs accuracy 0.53). "Not calibrated as claimed" on 4 of 5 decision classes in every draw; the fifth (yes/no contract checks) is draw-sensitive, decided by 0.002.
- **The local model does not match on any class.** qwen3:14b is about as accurate as Jev (0.51 vs 0.53) but far worse calibrated (ECE 0.40). One temperature fitted on the calibration split fixes most of that; accuracy is the harder limit.
- **Jev is fast and cheap.** About 44× faster and 570× cheaper per decision than our frontier arm. That partially reproduces the vendor's speed/cost claim (our frontier arm is not the model their claim names).
- **On contract-clause checks, Jev matches frontier accuracy** (0.74 on 20-way clause type) at a tiny fraction of the cost, but its probabilities should not set a routing threshold without per-class measurement.

Full write-up: [`docs/results-v1.md`](docs/results-v1.md). Bars, metrics and every amendment, all fixed before the data: [`docs/prereg-v1.md`](docs/prereg-v1.md).

## What is in this repo, and what is private

The study has two halves.

| Slice | What | Released? |
|---|---|---|
| **S1** (160 items, 129 test) | CUAD v1 contract passages: yes/no "does this passage contain a *X* clause?" (S1-noul) and 20-way "which clause type?" (S1-choice). Gold = CUAD's human labels | **Yes**: items, every per-call record from the reported run, and a script that reproduces every S1 number |
| **S2** (180 items, 144 test) | NXTG.AI's own internal operating text: which agent team wrote a coordination post, which program owns a work item, which team owns a tracked commitment. Gold = system-captured fields | **No.** Aggregate S2 metrics and verdicts are published in the results; no item, record, prompt or example text is |

**Why S2 is private:** it is verbatim internal communication and planning text. It was chosen because it is the decision set our own operations need, and because private data cannot be in any model's training set. Publishing it would end both properties. The cost is that the S2 rows and pooled figures in the results can be checked only by us; we say that rather than hide it.

## Layout

The directory layout mirrors the internal repository, because the harness locates its pinned data by repo-relative path.

| Path | What |
|---|---|
| [`evals/jev-calibration/`](evals/jev-calibration/) | The harness: items schema + guard, one prompt renderer, the JEV / FRONTIER / LOCAL arms, the deterministic scorer, the runner, tests and mutation proofs. Its own [README](evals/jev-calibration/README.md) documents every convention |
| [`governance/evals/jev-calibration/items-v1.jsonl`](governance/evals/jev-calibration/items-v1.jsonl) | The 160 S1 items: byte-identical lines of the frozen 340-row set. Pinned by `items-v1.sha256` (`552bf6f7…`). The full 340-row set's pin (`87b5417f…`) cannot be checked publicly |
| [`governance/evals/jev-calibration/frontier-subset-v1-s1.txt`](governance/evals/jev-calibration/frontier-subset-v1-s1.txt) | The 47 S1 members of the fixed 100-item FRONTIER subset (prereg A8.1). Re-running `scripts/jev-eval-frontier-subset.py` on S1 alone selects a different set, because the quotas were computed over all five slices |
| [`governance/evals/sampling-registry.json`](governance/evals/sampling-registry.json) | Documented sampling config per model (the five models of this study), read by the harness |
| [`governance/evals/jev-calibration/local-precheck/`](governance/evals/jev-calibration/local-precheck/) | One LOCAL precheck receipt, kept as a test fixture (the cache-confounded receipt the certifying binding must refuse) |
| [`results/jevcal-d2b-3arm-20261001T043407Z-S1/`](results/jevcal-d2b-3arm-20261001T043407Z-S1/) | The reported run's S1 records: `raw-s1.jsonl` (1,214 per-call records: LOCAL-P 480, JEV 480, FRONTIER 254) and `secondary-s1.jsonl` (235 FRONTIER single-letter samples), plus the reproducer's output |
| [`scripts/reproduce_s1.py`](scripts/reproduce_s1.py) | Recomputes every S1 number from the records with the shipped scorer |
| [`deploy/langfuse/eval_rail_trace.py`](deploy/langfuse/eval_rail_trace.py) | The optional LangFuse tracer the runner imports (keys from env only) |
| [`docs/`](docs/) | Public copies of the pre-registration and the results |

## Reproduce

Python 3.10+ and numpy. scikit-learn is optional (one test uses it as an independent oracle and is skipped without it).

```bash
pip install numpy               # optional: scikit-learn

# 1. Every S1 number in docs/results-v1.md, from the released records
python3 scripts/reproduce_s1.py                       # tables
python3 scripts/reproduce_s1.py --json s1.json        # every number as JSON

# 2. The harness test suite (no network, no GPU, no API keys)
cd evals/jev-calibration
python3 -m unittest discover -s tests -t .            # 191 tests: 190 pass, 1 skipped without scikit-learn

# 3. Mutation proofs: each suite breaks the shipped code in place, requires the tests to go red,
#    then restores the files. Run them on a clean, committed checkout, one at a time, and never
#    while anything else (tests included) is running in the same tree.
python3 tests/mutate_scorer.py && python3 tests/mutate_isolation.py && python3 tests/mutate_harness.py \
  && python3 tests/mutate_draws.py && python3 tests/mutate_local.py && python3 tests/mutate_d1.py
```

`reproduce_s1.py` prints H1 per draw (JEV, both ECE binnings), H2 under both pre-registered reads, every arm's per-slice test metrics (H3 = coverage at ≥95% precision), and the tagged sub-slices. We checked its output against the private certified run record and the per-draw tables: 110 of 110 compared values match exactly. It also prints S1-only pairwise Brier, latency, cost and jitter, labelled as S1-only, because the results document's versions of those pool S1 with S2.

**Running the arms live** needs your own access: a TypeSafe API key (`JEV_KEY_FILE`), the `claude` CLI for FRONTIER, and an ollama server for LOCAL (`JEVCAL_OLLAMA_BASE_URL`, `JEVCAL_VRAM_PROBE`). `--mode fixture` runs any items file except the pinned set (for example `fixtures/items-fixture.jsonl`, or your own items). `--mode certifying`, the mode the reported run used, also needs the private 100-item FRONTIER subset and NXTG.AI's internal certifier, so the certified run itself cannot be re-executed from this repo. See the harness README.

## Changes made for the public release

- **Data:** only S1 rows and S1 records. The run record (`.json`) is not released, because it embeds the full user-seen prompt of every item, S2 included.
- **Record redactions** (outside the hashed request bodies, so `msg_sha256` and `wire_sha256` still match the prompts): the SSH user and host in LOCAL-P's `extra.vram_report.probe_argv` became `<user>@<gpu-host>` (480 records), and the `claude -p` `session_id` became `<redacted>` (254 FRONTIER records, 235 secondary samples).
- **Code:** comments and docstrings that named internal reviewers, tickets, hosts or paths were reworded. Two test assertions that counted the private 340-row file now count the public 160-row file (`FROZEN_N`, `FROZEN_CAL_N` in `tests/test_codex_fixes.py`). No test was dropped and no harness logic changed. Two recorded strings did change: the `PREREG` path in `jevcal/run.py` (written into every run record) now names `docs/prereg-v1.md`, and one `ArmHalt` message in `jevcal/arms/local.py` no longer names the reviewer. The isolation check still refuses to run the FRONTIER child inside `~/ASIF` (NXTG.AI's internal operations repository, whose settings pin a different effort level); `ASIF_ROOT` is that check's environment variable.
- **Registry:** only the five models of this study, with internal notes rewritten; values and citations unchanged.
- **Docs:** see the note at the top of each file in `docs/`.

## Licence

- **Code** (`evals/`, `scripts/`, `deploy/`): Apache License 2.0, see [`LICENSE`](LICENSE).
- **Data** (`governance/evals/jev-calibration/items-v1.jsonl`, `results/`): derived from **CUAD v1 © The Atticus Project, licensed CC BY 4.0**. Our changes are listed in [`NOTICE`](NOTICE). The CUAD licence was checked against the Zenodo record for CUAD v1 (`https://zenodo.org/api/records/4595826`, `license.id = cc-by-4.0`) and The Atticus Project's dataset page (`https://www.atticusprojectai.org/cuad`, "CC BY 4.0").

If you use CUAD, cite Hendrycks, Burns, Chen and Ball, *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*, arXiv:2103.06268 (2021).
