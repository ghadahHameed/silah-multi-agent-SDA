# Agent 2 — Qualification & Prioritization Agent

Agent 2 takes a list of candidate companies — already filtered by city by **Agent 1**
— computes a fit score for each one, and returns the top-N ranked highest to lowest.
**Agent 3** (Outreach) consumes this ranked output to write personalized emails.

Only the **product description** is required as input. The target industries and a
one-sentence buyer profile are inferred from it automatically.

## Pipeline position

```
Product Input + Prospect Data
        ↓
Agent 1: Product Understanding + city filter
        ↓
Agent 2: Qualification & Prioritization (Scoring)   ← this component
        ↓
Agent 3: Outreach & Next Action
        ↓
Human-in-the-Loop when needed
```

## Hybrid scoring

> **Changed from the original design.** Agent 2 previously ran with no LLM at all. It is now
> a hybrid: `industry_score` and `revenue_score` stay rule-based and deterministic, exactly
> as originally built; `description_score` comes from an LLM — but only for a shortlist, so
> the API bill stays flat. One further LLM call per run classifies the product's target
> industries and writes a one-sentence buyer profile, so nobody has to type either.

```
fit_score = 0.30 × industry_score
          + 0.30 × revenue_score
          + 0.40 × description_score
```

```
product description
      ↓  Step 0 — infer target industries + buyer profile, 1 LLM call per run
500 companies
      ↓  Stage 1 — sequential hard gates, 0 API calls:
         location (case-insensitive) → industry match (exact or adjacent) → revenue tier mid/top
survivors, ranked by industry_score then revenue_sar — top 20 only if more than 20 survive
      ↓  Stage 2 — 1 LLM call each → description_score → fit_score
top N → Agent 3
```

### Step 0 — target industries and buyer profile (inside `rank_prospects`)

One `OPENAI_MODEL` call **per run, not per company**, with a strict JSON schema, returns
two things:

- **`industries`** — which of the fixed known categories (`industry_lookup.KNOWN_INDUSTRIES`,
  the keys of the adjacency table) the product's buyers belong to. The schema's `enum` *is*
  that list, so the API itself cannot return an invented name; the result is filtered against
  the list again as a second guard. It is a list — a product can target more than one
  industry — and `industry_score` is the company's best match across it. A hand-typed
  industry turned out to be the weakest link (typing `technology` for a phone accessory
  matched `Telecommunications` and filled the shortlist with the largest telcos), which is
  why this is inferred.
- **`buyer_profile`** — one sentence describing the ideal buyer *as a business, in the
  register a company uses to describe itself*. Example output for a water bottle: *"A
  consumer goods brand or retailer selling reusable drinkware and other everyday lifestyle
  products."* This is what Stage 1 embeds — see the next section for why the raw product
  text cannot be used.

The result is cached per product description for the process, so an interactive loop or a
re-run makes no new call. If the call fails after retries, the run continues with **no**
target industries (industry term 0.0 for everyone) and embeds the raw product description
instead of a profile; both are logged plainly. Every evaluation-log record carries
`icp_industries`, `industry_inference_source`, `buyer_profile` and `buyer_profile_source`
(`"llm"`, `"provided"`, `"llm_empty"`, `"llm_failed"`), so a JSONL line is self-describing.

If the call **succeeds but returns no industry** — the text does not say who buys the product,
e.g. *"A subscription service for busy people."* — the run does not continue. That is a
different situation from an outage and is kept apart from it: `industry_inference_source` is
`"llm_empty"`, not `"llm_failed"`. `rank_prospects` raises `ClarificationNeeded` before any
gate runs, `qualify_prospects` returns `status: "needs_clarification"` with a `reason` and a
`prompt_for_user` (*"…Could you clarify who the typical buyer is?…"*), and every company is
logged as `stage: "not-evaluated"` with every score `null`. Running the gates on an empty list
would silently turn Step 2 into a pass-through — the whole city reaches the revenue gate and
the twenty largest companies get scored on size alone — which is exactly what this prevents.
The caller appends the answer to the description (the demo uses *"…. Typical buyer: …"*) and
calls again; the demo asks once and, if the list is still empty, stops with a clear message
rather than looping. An empty `icp_industries` argument counts as not provided.

`rank_prospects(..., icp_industries=[...], buyer_profile="...")` pins either or both — for
Agent 1 to pass a known answer, or for tests. Pinning only the industries still infers the
profile, deliberately: reverting to the raw product text as the embedding query would
reintroduce the problem below.

### Why Stage 1 embeds the buyer profile, not the product

`diagnose_embeddings.py` (re-runnable) established this. The local model,
`all-MiniLM-L6-v2`, is a *sentence-similarity* model: it scores whether two texts are about
the same thing. A product description and a company description never are, so it returns
~0 for every company regardless of fit — except where a stray word overlaps ("water" →
a desalination utility, "phone" → telcos), which is worse than zero. The pipeline itself was
verified correct: real function, real model, L2-normalised 384-d vectors, cosine matching
`sentence_transformers.util.cos_sim` to 3.9e-08, and a near-paraphrase pair scoring 0.745.

The fix is to compare like with like. Measured on the same model, code and dataset — only
the query text changed:

| query text | vs Panda Retail | vs Al Othaim | vs Elm (IT) | top-20 industries |
|---|---|---|---|---|
| `"a stainless steel water bottle"` | -0.013 | 0.006 | 0.028 | Fintech ×7, Banking ×4, Retail ×3 |
| LLM buyer profile | **0.39** | **0.36** | 0.30 | **Retail ×16** |

### Stage 1 — pre-filter gates (`rank_prospects`, components from `calculate_pre_filter_score`)

Runs on every company, locally, with no network calls. It is a **sequential hard filter**,
not a blended score — a company that fails a step never reaches the LLM, however strong
its description similarity:

| Step | Gate | Passes when |
|---|---|---|
| 1 | Location (optional, `target_location`) | `location` equals the target, case-insensitive |
| 2 | Industry | `industry_score > 0` — exact (1.0), adjacent (0.5) or fuzzy match against any target industry (`industry_lookup.industry_similarity`, see below) |
| 3 | Revenue tier | tier is `mid` or `top` (`scoring_config.REVENUE_TIERS`); `low`, missing, non-numeric or negative revenue is cut |
| 4 | Cap | only if more than `LLM_PREFILTER_TOP_N` (default **20**) survive: rank by `industry_score` descending (exact 1.0 above adjacent 0.5 above weaker relations), then `revenue_sar` descending as the tie-breaker only, then `embedding_score`, then `company_id`; the top 20 go to the LLM. If 20 or fewer survive, all of them go. |

Step 4 follows the pre-filter priority order — location → industry → revenue → description — so
a smaller exact-industry match is shortlisted ahead of a giant company with only a distant
relation (0.25); revenue never outranks a stronger industry match. The gate thresholds
live in `scoring_config.PREFILTER_MIN_INDUSTRY_SCORE` / `PREFILTER_REVENUE_CUT_TIERS`. If Step 0's
call *failed* (`"llm_failed"`), Step 2 is skipped rather than cutting every company — the log and
each reasoning trace say so — and Steps 3–4 still apply. If the call succeeded but inferred
nothing (`"llm_empty"`), the gates do not run at all: see *needs_clarification* under Step 0.

`industry_score`, `revenue_score`, `revenue_tier`, `revenue_sar`, `embedding_similarity` (raw
cosine of the **buyer profile** against `business_description` — local Sentence-Transformers,
no API call), `embedding_score` (that cosine on the fixed scale `sim ≤ 0.20 → 0.0`,
`sim ≥ 0.60 → 1.0`, linear between) and `pre_filter_score`
(`0.30 × industry + 0.30 × revenue + 0.40 × embedding`) are still recorded for every company,
including the ones cut. `pre_filter_score` is **audit-only** — it does not decide the
shortlist. `embedding_score` is the last Step 4 tie-breaker (after industry and revenue) and the
Stage 2 fallback if the LLM call fails. `pre_filter_cut` names the gate that cut a company: `industry`, `revenue`, `rank`
(passed both gates, outside the top 20) or `null` (shortlisted). The reasoning trace says
which text the embedding was compared against.

### Fuzzy industry matching

The company's `industry` comes from a curated dataset, and the target industries come from
Step 0's fixed vocabulary — but the matcher is kept forgiving because Stage 1 is a recall
funnel: a company wrongly scored 0.0 here never reaches the LLM. `industry_similarity`
tries, in order:

| Evidence | Score | Example |
|---|---|---|
| Exact match after alias resolution | 1.0 | `IT` → `Information Technology` |
| Same ignoring spacing/punctuation | 1.0 | `Cyber Security` ≡ `Cybersecurity` |
| Listed in `INDUSTRY_ADJACENCY` | 0.5 | `Fintech` ~ `Banking` |
| One name's tokens contain the other's | 0.6 | `Software` ⊂ `Software Development` |
| Partial token overlap (Jaccard) | ≤ 0.5 | generic words like *technology*, *services* discounted |
| Connected via one intermediate industry | 0.25 | `Travel Technology` ~ `Food Technology` |
| Near-identical spelling (typo tolerance) | ≤ 0.85 | `Informaton Technlogy` |
| Nothing | 0.0 | `Mining` vs `Banking` |

Fully deterministic — stdlib string work (`difflib`) and dict lookups, no model and no API
call. Tune `FUZZY_MIN_RATIO` in `scoring_config.py`; add short forms to `INDUSTRY_ALIASES`
in `industry_lookup.py`.

### Stage 2 — LLM description score (`score_description_llm` → `calculate_fit_score`)

Each shortlisted company gets one call to `scoring_config.OPENAI_MODEL` (default
**`gpt-5.6-luna`**) comparing the **real product description** to that company's
`business_description`. The response is a **strict JSON schema** — the API guarantees
exactly these two fields, so nothing is parsed out of free text:

```json
{ "description_score": 0.85, "reasoning": "ICT provider with cybersecurity services; likely training needs." }
```

`description_score` (clamped to 0.0–1.0) goes into the formula above with the two
deterministic components to produce `fit_score`. The LLM's one-sentence `reasoning` is
appended to the record's `reasoning` trace.

- **Temperature.** `gpt-5.6-luna` rejects any value other than its default (the API returns
  400 for `temperature=0.0`, verified), so `LLM_TEMPERATURE` is `None` for this model — set
  it only if you switch `OPENAI_MODEL` to one that supports it. Run-to-run consistency comes
  from the strict schema, the anchored scale in the prompt, and the per-run cache.
- **Cache.** Responses are cached per `(product_description, company_id)` for the life of
  the process, so re-ranking the same dataset against the same product costs no API calls.
  Failures are not cached, so an outage can be retried.
- **Cost guard.** A 500-row dataset costs 20 + 1 API calls, not 500. Tune with
  `LLM_PREFILTER_TOP_N`, or per call via `rank_prospects(..., prefilter_top_n=N)`.

### Fallback (`description_score_source`)

If the LLM call fails after retries (3 attempts, exponential backoff via `tenacity`) for a
company, its `embedding_score` from Stage 1 — the buyer profile against its description —
is reused as `description_score`, tagged `description_score_source: "embedding_fallback"`
rather than `"llm"`, and `fit_score` is still computed with the same formula. An API
outage degrades one score; it never drops a company or aborts the run.

Location is **not** part of the equation — it is a hard filter, not a score. Agent 1 applies
it before companies reach Agent 2; when the input has not been city-filtered yet (the
interactive demo, or a raw CSV), pass `target_location` to `rank_prospects` /
`qualify_prospects` and it is applied as Step 1, before any scoring: only companies whose
`location` equals it (case-insensitive, whitespace-trimmed) are scored, the rest are logged with
`stage: "location-filtered"` and never reach the industry / revenue / embedding terms or
the LLM.

## Determinism

`industry_score`, `revenue_score`, `embedding_score`, `pre_filter_score` and the shortlist
itself are fully reproducible **given the ICP** (industries + buyer profile). Two things are
LLM judgments: the once-per-run ICP inference and the per-company `description_score`.
Both are cached for the process, so a repeat of the same product in the same session is
identical; across sessions, a differently-worded buyer profile can shift the pre-filter and
close `fit_score` values can reorder. `pre_filter_score` and its components are the stable
numbers to compare across runs for a given profile.

## Data schema (input to Agent 2)

```
company_id, company_name, industry, location, business_description, website, email, revenue_sar
```

`website` / `email` may be null — Agent 2 doesn't use them for scoring.

## Files

| File | Responsibility |
|---|---|
| `qualification_agent.py` | Orchestration (`qualify_prospects`), CLI demo entry point |
| `scoring.py` | Step 0 (ICP inference, one call per run), Stage 1 (`score_industry`, `score_revenue`, `calculate_pre_filter_score`), Stage 2 (`score_description_llm`, `calculate_fit_score`), prompt sanitiser, LLM caches, `rank_prospects`, `build_handoff` / `match_level` (the Agent-3 contract), `select_prospect`, `summarize_funnel`, input validation, debug trace |
| `industry_lookup.py` | Industry adjacency table (`KNOWN_INDUSTRIES`), alias table, and the graded fuzzy matcher (`industry_similarity`) |
| `scoring_config.py` | Tunable weights / thresholds / model, prefilter size, concurrency (edit here, not in `scoring.py`) |
| `prospect_loader.py` | CSV loading into plain dicts, malformed-row handling |
| `silah_data.csv` | Prospect dataset (default source of truth) |
| `demo_agent2_run.py` | Interactive demo — pick a target city (Step 1 hard filter), type a product, choose how many scored companies to hand to Agent 3 (1–5, default 3; separate from the 20-company scoring cap), see the inferred industries, the buyer profile, a per-stage funnel count and the real selected / rejected / pre-filtered output, or a plain no-match message when nothing survives the gates; the TOP-N section prints the exact JSON Agent 3 receives |
| `diagnose_embeddings.py` | Standalone check that the embedding path is real, normalised and numerically correct, with labelled pairs and the real-dataset spread |
| `test_agent2_scoring.py` | Tests (no network, no model download, no API key — every external call is faked) |

## Usage

```python
from qualification_agent import qualify_prospects

result = qualify_prospects(
    icp_description="A B2B platform that helps organizations train employees on cybersecurity awareness.",
    csv_path="silah_data.csv",
    top_n=7,
)
# or pin the ICP yourself and skip inference entirely:
# qualify_prospects(
#     icp_description=...,
#     icp_industries=["Information Technology", "Cybersecurity"],
#     buyer_profile="Organizations that operate digital systems and train staff on security awareness.",
# )
```

Returns two separate channels — the Agent-3 handoff (`companies`, the contract below) and
the full scoring audit trail for developers (`evaluation_log`):

```json
{
  "status": "ok",
  "reason": null,
  "funnel_summary": "500 loaded -> 213 in Riyadh -> 40 with industry match -> 31 passing revenue tier -> 20 scored (capped at 20)",
  "companies": [
    {
      "company_name": "solutions by stc",
      "fit_score": 0.94,
      "match_level": "high match",
      "reason": "ICT provider with cybersecurity services; likely training needs.",
      "email": "needs human review — high fit but no contact info on file, search required"
    }
  ],
  "evaluation_log": [
    {
      "company_id": "SA-003",
      "company_name": "solutions by stc",
      "icp_industries": ["Cybersecurity", "Information Technology", "Software", "Technology"],
      "industry_inference_source": "llm",
      "buyer_profile": "Organizations across industries that operate digital systems and need to train their employees on cybersecurity awareness...",
      "buyer_profile_source": "llm",
      "industry_score": 1.0,
      "revenue_score": 1.0,
      "embedding_similarity": 0.43,
      "embedding_score": 0.57,
      "pre_filter_score": 0.83,
      "pre_filter_cut": null,
      "description_score": 0.85,
      "description_score_source": "llm",
      "description_reasoning": "ICT provider with cybersecurity services; likely training needs.",
      "fit_score": 0.94,
      "stage": "scored",
      "reasoning": ["industry: ...", "revenue: ...", "embedding: cosine sim 0.43 vs buyer profile -> normalized 0.57", "pre_filter_score = ...", "description (llm): ...", "fit_score = ..."],
      "selected": true
    },
    {
      "company_id": "SA-067",
      "company_name": "Saudi Tourism Company",
      "icp_industries": ["Cybersecurity", "Information Technology", "Software", "Technology"],
      "industry_inference_source": "llm",
      "buyer_profile": "Organizations across industries that ...",
      "buyer_profile_source": "llm",
      "industry_score": 0.0,
      "revenue_score": 0.3,
      "embedding_similarity": 0.14,
      "embedding_score": 0.0,
      "pre_filter_score": 0.09,
      "pre_filter_cut": "industry",
      "description_score": null,
      "description_score_source": null,
      "description_reasoning": null,
      "fit_score": null,
      "stage": "pre-filtered",
      "reasoning": ["industry: ...", "revenue: ...", "embedding: ...", "pre_filter_score = ...", "pre-filtered: Step 2 industry gate failed - industry_score 0.0 is no match (exact or adjacent) for ['Information Technology']; no LLM call"],
      "selected": false
    }
  ],
  "icp_industries": ["Cybersecurity", "Information Technology", "Software", "Technology"],
  "industry_inference_source": "llm",
  "buyer_profile": "Organizations across industries that ...",
  "buyer_profile_source": "llm",
  "selected": { "...": "top evaluation_log entry with stage == \"scored\"" },
  "needs_human_review": false,
  "review_reason": null
}
```

### What Agent 3 receives — the output contract

`companies` is the authoritative handoff (`rank_prospects` returns exactly this list; `qualify_prospects`
wraps it as shown above). Each entry has **exactly these five fields** and nothing else:

| Field | Value |
|---|---|
| `company_name` | as loaded from the dataset |
| `fit_score` | the blended score above, rounded to 2 decimals |
| `match_level` | `"high match"` if `fit_score` ≥ 0.7, `"medium match"` if 0.4 ≤ `fit_score` < 0.7, `"low match"` below 0.4 (`scoring_config.MATCH_LEVELS`) |
| `reason` | the LLM's own one-sentence verdict on the company as a buyer — or, if that call failed, a sentence saying the local-similarity fallback was used |
| `email` | the address on file, used as-is; when there is none, the string `"needs human review — {high\|medium\|low} fit but no contact info on file, search required"`, e.g. `"needs human review — medium fit but no contact info on file, search required"` |

```json
{
  "company_name": "Alinma Bank",
  "fit_score": 0.66,
  "match_level": "medium match",
  "reason": "Bank operates commercial buildings where smart, energy-efficient lighting could support facility management, though lighting is not central to its business.",
  "email": "needs human review — medium fit but no contact info on file, search required"
}
```

`company_id`, `industry_score`, `revenue_score`, `embedding_score`, `description_score`,
`pre_filter_score` and the full `reasoning` trace are deliberately **not** in this list: they
stay in `evaluation_log` and the JSONL file for developers. `scoring.build_handoff` builds each
entry; `scoring.match_level` does the banding.

`evaluation_log` covers every company: `stage` is `"scored"` (full
pipeline), `"pre-filtered"` (cut before the LLM step — the components and audit-only
`pre_filter_score` recorded, `pre_filter_cut` names the gate (`industry`, `revenue` or `rank`),
`description_score` / `fit_score` are `null`), `"location-filtered"`
(excluded by the optional `target_location` hard filter before any scoring — every score is
`null`), `"not-evaluated"` (the run stopped at Step 0 with no target industries — every score
is `null`, see *needs_clarification*) or `"skipped"` (unusable record). Nothing is silently dropped.

`status` is `"ok"`, `"no_matches"`, `"needs_clarification"` or `"empty_dataset"`; `prompt_for_user`
is set only for `"needs_clarification"`. **No matches is a valid outcome, not an
error.** When nothing survives the three hard gates (location → industry → revenue) — a real
product can genuinely have no fit in a city — Stage 2 is skipped (no LLM scoring call),
`companies` is empty, `reason` says why in one sentence (e.g. *"No companies found in
Riyadh matching Gaming with a sufficient revenue tier."*) and the JSONL log is still written
like any other run, with every company and the gate that cut it. `funnel_summary` is the
per-stage count on one line. `scoring.summarize_funnel` derives all three from the evaluation
log; the interactive demo reads its funnel from the same helper.

## Explainability / debug trace

Set `verbose=True` on `qualify_prospects`/`rank_prospects`, or the `DEBUG` environment
variable, to print a Reason → Act → Observe trace per company:

```
[SA-003] industry: exact match ('Information Technology' == ICP industry) -> 1.0
[SA-003] revenue: SAR 5.4B -> top tier -> 1.0
[SA-003] embedding: cosine sim 0.43 vs buyer profile -> normalized 0.57
[SA-003] pre_filter_score = 0.30*1.0 + 0.30*1.0 + 0.40*0.57 = 0.83
[SA-003] description (llm): ICT provider with cybersecurity services; likely training needs. -> 0.85
[SA-003] fit_score = 0.30*1.0 + 0.30*1.0 + 0.40*0.85 = 0.94
```

The last two lines only appear for shortlisted companies. Everyone else ends with
`pre-filtered: Step 2 industry gate failed ...`, `pre-filtered: Step 3 revenue gate failed ...` or
`pre-filtered: passed industry + revenue gates but ranked #k of N survivors by industry_score, then revenue_sar, outside the top 20; no LLM call`. When no
buyer profile was available, the embedding line reads `vs product description` instead.

## Robustness

- Every OpenAI call — the once-per-run ICP inference and each per-company description
  score — and the local embedding call are wrapped with `tenacity` retry/backoff (3
  attempts, exponential jitter). A missing API key is a config error, not a transient one —
  it is reported plainly and not retried.
- If ICP inference fails after retries, the run continues with no target industries
  (industry term 0.0 for everyone) and the raw product description as the embedding query,
  both tagged `"llm_failed"`, rather than aborting.
- If ICP inference succeeds but infers no industry, the run stops before any gate with
  `status: "needs_clarification"` and a `prompt_for_user`, instead of silently letting every
  company through the industry gate. Kept distinct from `"llm_failed"` in the log.
- Industry/revenue lookups never raise on missing or malformed fields — they return a
  defined fallback score with a logged reason.
- An LLM description-score failure falls back to that company's local `embedding_score`
  (`description_score_source: "embedding_fallback"`); the run continues with one degraded
  score rather than failing.
- Both LLM results are cached for the process — the ICP per product description, scores
  per `(product_description, company_id)` — so a re-run makes no new API calls.
- Zero survivors is handled explicitly: if no company passes location → industry → revenue,
  Stage 2 is skipped (no LLM scoring call), `qualify_prospects` returns `status: "no_matches"`
  with a `reason` and `funnel_summary`, and the log is written as usual. An expected outcome,
  not an error.
- `rank_prospects` keeps scoring the rest of the batch even if individual companies fail
  validation or scoring. Failures land in `evaluation_log` with `stage: "skipped"` and a
  single-line reason (`skipped: ...` / `scoring error: ...`) instead of aborting the run.

## Performance

Two design choices keep a 500-row run fast. Both were measured, not assumed.

**Stage 1 is batched and cached.** Every description goes through the embedding model in
one batched call, and each vector is cached by its text. Company descriptions don't change
between products, so a new product costs one profile embedding plus cached lookups — not
500 fresh encodes. Batched vectors match single-encode vectors to 1.5e-7, and 0 of 500
normalised scores differ at 4 dp, so this changes no results.

**Stage 2 runs concurrently.** The shortlist's API calls are independent network waits, so
they run on a bounded thread pool (`LLM_MAX_CONCURRENCY`, default 8). Twenty calls become
~3 rounds instead of 20 sequential waits. The bound keeps a burst from tripping the API
rate limit; a 429 still retries with backoff.

Same harness, 500 companies, 20 Stage 2 calls at a simulated 1.0 s each:

| | before | after |
|---|---|---|
| Stage 1, 500 descriptions, warm process | ~8.0 s | ~1.7 s |
| Stage 1, every later product in the same process | 7.3 s | 0.03 s |
| Stage 2, 20 calls | 20.0 s | ~3.0 s |
| End-to-end per query | 27.1 s | ~3.0 s |

The very first encode in a fresh process can run slower than steady state. That is the OS
paging the model weights in from disk — paid once, and not something the code controls.
The demo takes it during its "Loading…" step by calling
`embed_company_descriptions(companies)` before the prompt loop, so your first query is
already warm; call the same function ahead of any loop of your own.

`EMBEDDING_BATCH_SIZE` is not a tuning lever: 64 and 256 measured within noise.

## Security

- All inputs are validated before scoring (`validate_company_fields`): `revenue_sar` must be
  numeric and non-negative; `industry`/`location` must be non-empty; `business_description`
  must be non-empty and is truncated to `MAX_DESCRIPTION_LENGTH` before embedding or being
  sent to the API.
- No `eval()` or dynamic code execution anywhere in this module (asserted by a test).
- `SILAH_OPENAI_API_KEY` is read from the environment (`.env`, which is gitignored). It is
  never hardcoded (asserted by a test) and never logged, and the project-scoped name keeps
  it from colliding with a global `OPENAI_API_KEY`. Stage 1 needs no key at all.
- **Prompt hygiene.** The product description (both calls) and each company's
  `business_description` pass through `_sanitize_for_prompt` before any API call: control
  characters dropped, whitespace collapsed, and a short high-precision list of
  instruction-override phrases (*ignore previous instructions*, *you are now*, role markers,
  chat special tokens, the prompt's own fence) blanked to `[removed]`. The buyer profile the
  model returns goes through the same sanitiser before it is embedded or logged. Untrusted
  text is additionally fenced in the prompt and the system prompt instructs the model to
  treat it as data, not instructions. The description score is clamped to 0.0–1.0 on our
  side, and the industry classification can only return names from the fixed list, so a
  manipulated response cannot push a company above a legitimate one or invent a category.
- Logging records `company_id`, the resulting scores, `description_score_source`, the
  inferred industries and the buyer profile's length only — never the full prompt, the full
  API response, the `business_description` text or any contact info (`email`/`website`).
  The LLM's one-sentence `reasoning` and the buyer profile are kept in the evaluation
  record because they are the audit trail; the raw responses are not.

## Run

Set your key first. Either export it, or create a `.env` in the project root:

```
SILAH_OPENAI_API_KEY=sk-...
```

Without it, Step 0 and Stage 2 raise `SILAH_OPENAI_API_KEY is not set` on the first call.
Stage 1 and the test suite need no key.

```bash
pip install -r requirements.txt
python qualification_agent.py        # demo run against silah_data.csv
python demo_agent2_run.py            # interactive: type a product, ICP is inferred
python diagnose_embeddings.py        # verify the embedding path if scores look wrong
pytest test_agent2_scoring.py -v     # tests (offline, no key needed)
```

## Scope

Agent 2 does not research the product (Agent 1) and does not write outreach
messages or contact people (Agent 3).
