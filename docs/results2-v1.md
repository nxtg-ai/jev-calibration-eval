# Does context beat model choice on routing decisions? Results (Jev study, part 2) — public copy

> **About this copy.** This is the public version of NXTG.AI's internal results for part 2. The plan was made public in this repository before any context-arm call: [`prereg2-v1.md`](prereg2-v1.md), commit `2f7e29e`, with amendment A2 at `25c5f36`. The S2 items, their text and the memory records used as context are private and are not released. Every accuracy, ECE, interval and verdict word below can be recomputed from the released per-item file with `scripts/reproduce_s2_context.py`. That file holds no text and no labels.

## Results in brief

Each model was given the five most similar **earlier** records from our memory store, with their known labels, before it answered. The effect depended on the class:

| Class | What the model decides | Context effect, Jev | Context effect, Opus |
|---|---|---|---|
| S2a | which agent team wrote a coordination post (20 options) | **+25 points, context helps** | **+50 points, context helps** |
| S2b | which program a work item belongs to (11 options) | +10 points, no clear effect | +4 points, no clear effect |
| S2c | which team owns a tracked commitment (9 options) | +4 points, no clear effect | **+23 points, context helps** |

- **Memory alone.** We also took a majority vote of the five neighbours' labels, with no model. It **matches** Opus-without-context on S2a and **trails** it on S2b; S2c shows **no clear difference**. On S2a both score about 0.25, five times chance (0.05), and both are low. Opus *with* the same five records scored 0.73.
- **Opus vs Jev with context.** With context, Opus scored higher than Jev in every class: +35, +12.5 and +23 points. The S2b interval's lower bound is exactly 0. By pre-registration this comparison carries no verdict word. At list price Opus cost about 470× as much on these items.
- **Jev's confidence.** On S2c, context raised Jev's mean confidence by 25 points and its accuracy by 4. Its calibration error roughly doubled on that class (ECE 0.171 → 0.335). Part 1 found that Jev's confidence should not be trusted without per-class measurement, and that still holds with context.
- **What this tests.** It tests one of our memory store's three stores: vector retrieval, through the store's own embedding model. A null result is not a test of the whole store. S2b's question (which program owns a node) is what the graph store holds, and that is the next test.

## 1. The run

- **One certifying draw.** 144 S2 test items (48 per class) × 6 arms = 864 records, with no failed calls and no abstentions.
- **Certification.** The run passed every one of the automated certification gates used in part 1 (17 of 17). A second reviewer, working only from the raw records and the plan, reproduced every one of the 24 hypothesis rows below.
- **Arms** (plan §2):

  | Arm | Model | Context |
  |---|---|---|
  | JEV+CTX | `jev-1.13.0` | yes |
  | JEV | `jev-1.13.0` | no |
  | FRONTIER+CTX | `claude-opus-5-5`, effort high | yes |
  | FRONTIER | `claude-opus-5-5`, effort high | no |
  | RETRIEVAL-ONLY | no model: majority label of the 5 neighbours | — |
  | LOCAL-P+CTX | `qwen3:14b` on one RTX 4090 | yes |

  The no-context JEV and FRONTIER arms ran in the same run, interleaved per item (A1.1). They are the comparators.
- **Retrieval** (plan §3):
  - k = 5 nearest neighbours, using `nomic-embed-text-v1.5` truncated to 384 dimensions (our memory store's own embedding model), with exact kNN over frozen per-class corpora of earlier records.
  - Only records created strictly before the item's source record were eligible. Every option's name was redacted from neighbour text, and each neighbour was cut to 400 characters.
  - One S2c item had only 1 eligible earlier neighbour, not 5.
- **LOCAL-P+CTX feasibility.** It ran because every prompt passed the A1.6 token gate: the largest was 1,751 tokens against a 1,984 ceiling.

## 2. Primary results (pre-registered)

Differences are in accuracy points, n = 48 paired items each, with 95% paired-bootstrap intervals (2,000 resamples, seed 20260929, items sorted by id).

| Class | Chance | H5 Jev: with − without context | H5 Opus: with − without context | H6: memory alone − best no-context model (Opus in all three) | H7: Opus+context − Jev+context (descriptive) |
|---|---|---|---|---|---|
| S2a | 0.050 | +0.250 [+0.104, +0.396] **context helps** | +0.500 [+0.354, +0.646] **context helps** | +0.021 [−0.146, +0.188] **matches** | +0.354 [+0.208, +0.500] |
| S2b | 0.091 | +0.104 [−0.021, +0.229] no clear effect | +0.042 [−0.062, +0.167] no clear effect | −0.354 [−0.542, −0.167] **trails** | +0.125 [0.000, +0.250] |
| S2c | 0.111 | +0.042 [−0.125, +0.208] no clear effect | +0.229 [+0.063, +0.396] **context helps** | −0.083 [−0.250, +0.083] no clear difference | +0.229 [+0.104, +0.375] |

**The words are fixed by the plan:**
- *context helps*: the interval is entirely above 0 **and** the gain is at least 10 points.
- *beats* / *trails*: the interval is entirely above / below 0.
- *matches*: the interval includes 0 and the point estimate is within 3 points.
- *no clear difference*: the interval includes 0 and the point estimate is more than 3 points.

"Entirely above / below 0" is read strictly, so an interval whose bound is exactly 0 does not qualify.

## 3. Secondary reads

Against part 1's draw-1 records for the same items:

| Class | Jev + context vs part 1 Jev | Opus + context vs part 1 Opus | Local + context vs part 1 local (only read available) | Memory alone vs part 1 Opus |
|---|---|---|---|---|
| S2a | +0.271 [+0.125, +0.417] context helps | +0.521 [+0.375, +0.667] context helps | +0.271 [+0.125, +0.417] context helps | +0.042 [−0.146, +0.229] no clear difference |
| S2b | +0.125 [0.000, +0.250] no clear effect | +0.062 [−0.042, +0.167] no clear effect | −0.104 [−0.271, +0.083] no clear effect | −0.333 [−0.501, −0.146] trails |
| S2c | +0.042 [−0.125, +0.208] no clear effect | +0.146 [−0.021, +0.333] no clear effect | +0.125 [−0.042, +0.271] no clear effect | −0.167 [−0.333, 0.000] no clear difference |

- **Two secondary intervals have a bound of exactly 0** (Jev S2b; memory-alone vs part 1 S2c). Under the strict reading they stay *no clear effect* and *no clear difference*. A non-strict reading would change them, but neither is a primary result.
- **Opus on S2c is *context helps* against the concurrent comparator and *no clear effect* against part 1.** The difference comes from the comparator. Opus-without-context answered 17 of 48 correctly in this run and 21 of 48 in part 1, on identical items. Part 1 measured Opus changing its answer across re-asks on 21% of items. Re-ask noise of this size is why the plan made the concurrent arms primary (A1.1).
- **The local model has no concurrent no-context arm.** Its only read is against part 1, so it is secondary-only.

## 4. Accuracy and calibration per arm

Accuracy (correct out of 48) and ECE (equal-mass, 15 bins, the part 1 scorer):

| Arm | S2a | S2b | S2c | Pooled, 144 items |
|---|---|---|---|---|
| JEV+CTX | 0.375 (18), ECE 0.288 | 0.521 (25), ECE 0.176 | 0.354 (17), ECE 0.335 | 0.417 |
| JEV | 0.125 (6), ECE 0.263 | 0.417 (20), ECE 0.305 | 0.312 (15), ECE 0.171 | 0.285 |
| FRONTIER+CTX | 0.729 (35), ECE 0.168 | 0.646 (31), ECE 0.186 | 0.583 (28), ECE 0.124 | 0.653 |
| FRONTIER | 0.229 (11), ECE 0.145 | 0.604 (29), ECE 0.207 | 0.354 (17), ECE 0.181 | 0.396 |
| RETRIEVAL-ONLY | 0.250 (12), ECE 0.246 | 0.250 (12), ECE 0.392 | 0.271 (13), ECE 0.188 | 0.257 |
| LOCAL-P+CTX | 0.333 (16), ECE 0.634 | 0.354 (17), ECE 0.617 | 0.333 (16), ECE 0.645 | 0.340 |

- **Memory alone.** Its confidence is a fixed rule (A2.3): votes/5 on the winning label, with the remainder spread evenly over the other options. Its ECE describes that rule.
- **The local model** puts a top-1 probability of 0.99 or more on most items in every class, and gets about a third right. Its confidence carries no information here.

**Speed and cost.** These are from the certified run and are not recomputable from the released file. Over the 144 items:

| Arm | Median latency | List-price cost, 144 calls |
|---|---|---|
| JEV+CTX | 0.15 s | $0.0090 |
| FRONTIER+CTX | 6.9 s | $4.22 |

Neither cost was billed. Jev ran on an early-access allocation and Opus on a subscription. These figures are not comparable to part 1's per-decision ratio, which was measured on a different population.

## 5. Why S2b showed no effect (post-hoc; not pre-registered)

We counted how often the correct label appeared among the five neighbours' labels:

| Class | Items |
|---|---|
| S2a | 33 of 48 |
| S2b | 13 of 48 |
| S2c | 32 of 48 |

- **S2b.** On S2b the retrieved past records rarely pointed at the right program, so there was little for a model to use. Opus without context already scored 0.60 there.
- **Not a ceiling.** On S2a, Opus with context got 35 right, more than the 33 items whose label was in its context.
- These counts come from the private context file and are not recomputable from the released data.

## 6. What this is not

- **Not a test of the whole memory store.** It tests vector retrieval through the store's own embedding model, over a frozen file (A1.5). The store's relational and graph parts were not used. A result here is a claim about *retrieved context*.
- **Not individually certified comparators.** Certification gates covered the context arms. The concurrent no-context arms are comparators. They were held to the same model id and sampling settings by a pre-call check, and any paired read with a missing item is refused (A2.3).
- **Not more than one draw.** Part 1 measured how often answers change across re-asks: 5.3% of items for Jev, 21% for Opus. That is why every primary comparison here is concurrent and paired.

## 7. Reproduce

```bash
pip install numpy
python3 scripts/reproduce_s2_context.py              # every accuracy, ECE, interval and word above
python3 scripts/reproduce_s2_context.py --json s2.json
```

The released file is `results/jevctx-p2-20261002T000132Z-S2/per-item.csv`. It has one row per run, arm and item (1,296 rows: the part 2 arms plus part 1's draw-1 Jev, Opus and local records for the same 144 items). Its columns are an opaque item id, the class, the number of options, correct (0/1) and the top-1 probability. We checked the script's output against the private certified readout: 60 of 60 values are identical.
