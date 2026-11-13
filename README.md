# costcut

What do three common inference-cost levers really save on a small multi-hop QA pipeline,
when a quality gate fixed in advance has to hold?

The levers are a context budget, model routing and prompt caching. Each one runs on its own,
then all three run together. Cost comes from the API's own `usage` fields times the published
price table. Quality is checked against a gate written down below **before any lever ran**.
Whatever comes out is the result. Nothing was tuned to reach a target saving.

> Results: see [Results](#results) (filled in from `results.json` after the real run) and `report.txt`.

## Pipeline under test

For each question:

1. **Intent classification** (LLM): `bridge` or `comparison`. HotpotQA's own `type` field is the gold label.
2. **Query generation** (LLM): a structured retrieval request, `{"queries": [1-3 keyword queries], "entities": [...]}`.
3. **Retrieval** (local, $0): BM25 (k1=1.5, b=0.75) per query, top 50 each, fused with reciprocal-rank
   fusion (k=60), cut to 20 candidates.
4. **Grounded synthesis** (LLM): a short answer plus citations to numbered passages, from the passages only.

The prompts are in [`costcut/prompts.py`](costcut/prompts.py). Each system prompt is the stable prefix
(instructions, output schema, few-shot examples written for this project, none from HotpotQA). Everything
that varies per question goes in the user message.

## Data

HotpotQA, distractor setting, validation split ([hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa),
Yang et al., EMNLP 2018), **CC BY-SA 4.0**. `data/hotpotqa_distractor_validation_head.jsonl` holds the first
100 validation rows in published order, copied byte for byte from the sibling project
[sabot](https://github.com/theskipper007/sabot). HotpotQA was kept because it ships gold supporting
documents per question, which is what a retrieval metric needs.

- Questions: rows 0..n-1, in order, no filtering.
- Corpus: every context paragraph of all 100 rows (gold and distractors), one document per (row, paragraph),
  not de-duplicated: 992 documents. Each question is retrieved against all of them, not just its own 10.

## Configurations

| config | intent | query gen | synthesis | context bundle | prompt caching |
|---|---|---|---|---|---|
| `baseline` | large | large | large | all 20 candidates, whole (~2.9k est. tokens) | off |
| `context_2000` / `_1000` / `_500` | large | large | large | hard budget of 2000 / 1000 / 500 est. tokens, 200-token per-source cap, content-hash dedup | off |
| `routing_upstream` | **cheap** | **cheap** | large | as baseline | off |
| `routing_all` | **cheap** | **cheap** | **cheap** | as baseline | off |
| `caching` | large | large | large | as baseline | **on** |
| `combined` | **cheap** | **cheap** | large | 1000-token budget, cap, dedup | **on** |

- Large tier: `claude-sonnet-5-5`. Cheap tier: `claude-haiku-5-5`.
- Thinking is off on both tiers (`{"type": "between_tools"}` on Sonnet 5.5, which rejects `disabled`;
  `{"type": "disabled"}` on Haiku 5.5). Effort and sampling are left at the API defaults. The prompts are
  the same on both tiers.
- Context budget: walk the 20 ranked candidates; drop a document whose content hash was already added;
  cut each paragraph to at most 200 estimated tokens at a sentence boundary; add it if it fits the budget,
  otherwise stop. Tokens for the budget are estimated as ceil(chars/4). The cost uses the API's real counts.
- Prompt caching: one `cache_control: {"type": "ephemeral"}` (5-minute TTL) breakpoint at the end of each
  step's system prompt. Cache reads and writes are the API's `cache_read_input_tokens` and
  `cache_creation_input_tokens`, never estimates.
- `combined` uses the middle budget (1000). That choice was made here, before the sweep ran.

## Cost

```
cost = ( input_tokens * p_input + cache_write_5m * p_write_5m + cache_write_1h * p_write_1h
       + cache_read_input_tokens * p_cache_read + output_tokens * p_output ) / 1e6
```

All token counts come from the API's `usage` on each response (`input_tokens` excludes cached tokens).
Prices, USD per 1M tokens, from <https://platform.claude.com/docs/en/about-claude/pricing>, fetched 2026-11-13:

| model | input | 5m cache write | 1h cache write | cache read | output |
|---|---|---|---|---|---|
| claude-sonnet-5-5 | 2.00 | 2.50 | 4.00 | 0.10 | 10.00 |
| claude-haiku-5-5 (prompt <= 100k) | 0.10 | 0.125 | 0.20 | 0.01 | 0.50 |

Cost per question = the sum of the three LLM calls' costs, averaged over the n questions (and, for the
baseline, over its k reruns). Retrieval is local and costs $0 here.

## Pre-registration (written 2026-11-13, committed before any lever ran)

This section was committed to git before the first real API call. Its commit hash is in `results.json`
(`meta.git`), and `git log` shows it predates the results commit.

**Sample.** n = 30 questions (HotpotQA validation rows 0-29). n was chosen from the dry-run spend
projection so the whole experiment fits the $3 budget (projection $2.99 at n=30, which errs high; $3.98 at n=40).

**Noise floor.** k = 3 baseline reruns with independent samples (replicate tags b1, b2, b3). Each lever runs
once (r = 1, tag L1).

**Gated metrics** (both must hold):

1. `answer_f1`: mean HotpotQA token-F1 of the answer against the gold answer (official normalisation;
   yes/no must match exactly).
2. `bundle_recall`: mean fraction of the question's gold supporting titles present in the context bundle the
   synthesis model actually received.

Reported but not gated: exact match, NDCG@10 of the retrieval ranking, citation recall of gold titles,
intent accuracy.

**Rule.** For each gated metric, with s_1..s_k the baseline run means:

```
sigma_run = SD(s_1..s_k), ddof = 1                 (run-to-run noise, same questions)
SE        = sigma_run * sqrt(1/k + 1/r)
MDE       = max( (t_{0.95, k-1} + t_{0.80, k-1}) * SE ,  1/n )     one-sided alpha 0.05, power 0.80
drop      = mean(s) - lever mean
holds     iff drop <= MDE
PASS      iff it holds on both gated metrics
```

With k = 3 the t multiplier is large ((2.920 + 1.061) x sqrt(4/3) = 4.6 sigma), so this gate is coarse.
"Holds" means "no drop larger than this experiment can detect", not "proven equivalent". A paired
bootstrap 95% CI of each drop is reported next to every verdict for that reason.

**Fixed before the run, not tuned after:** the prompts, the three budgets (2000/1000/500), the 200-token
per-source cap, the 20 candidates, the combined configuration, the models, thinking off, n, k, the gate.

## Run it

```bash
pip install -e .[dev]
python -m pytest                     # cost math, gate logic, bundle, metrics
python -m costcut run --dry-run      # offline fake backend: plumbing check + projected spend
python -m costcut run                # the real run (reads ANTHROPIC_API_KEY from the environment)
```

`python -m costcut run` is also the one-command rerun. Every API response is stored in `runs/cache/`, keyed
by the full request plus a replicate tag, so with the committed cache a rerun reproduces `results.json`
exactly at $0. `--no-cache` makes a fresh, independent run (and real spend). The spend cap
(`--max-usd`, default 2.80) is checked against `runs/ledger.jsonl`, which records every real API call ever
made by this experiment, so aborted attempts count too.

## Limits

- Small n (30) and k = 3. The gate can only see large drops. Read the MDEs next to the verdicts.
- One dataset (short-answer multi-hop Wikipedia QA), one retriever (BM25), one prompt set, one provider.
  Savings depend on the shape of the traffic: how big the stable prefix is relative to the retrieved
  context, and how much output each step writes.
- Caching savings depend on traffic. Here 30 questions run with 4 workers, so the first concurrent calls of
  each step all pay the cache-write premium. At steady production volume the write is amortised further.
  The `combined` run follows `caching` and may read cache entries it wrote, as warm traffic would.
- Lever runs share one replicate tag, so the three context configs share the same intent and query calls
  (made once, charged to each). Their differences from each other are paired; each is compared with the
  baseline's k-run mean.
- Thinking is off. With adaptive thinking on, output tokens, and so routing savings, would differ.
- The token budget uses a chars/4 estimate, not the model's tokenizer.
