"""Tunable constants for Agent 2 scoring.

Everything that affects a score lives here so thresholds can be tuned without
touching the scoring logic itself, and so the same input always produces the
same output (fully deterministic and reproducible).
"""

# --------------------------------------------------------------------------- #
# Fit-score weights: fit = 0.30*industry + 0.30*revenue + 0.40*description
# --------------------------------------------------------------------------- #
INDUSTRY_WEIGHT = 0.30
REVENUE_WEIGHT = 0.30
DESCRIPTION_WEIGHT = 0.40

# --------------------------------------------------------------------------- #
# Industry score (rule-based lookup, see industry_lookup.py). The matcher
# reads these — they are the single source for the exact / adjacent scores.
# --------------------------------------------------------------------------- #
INDUSTRY_EXACT_MATCH_SCORE = 1.0
INDUSTRY_ADJACENT_SCORE = 0.5
INDUSTRY_NO_MATCH_SCORE = 0.0

# Fuzzy fallback for industry names that don't match the table outright.
# Two strings must be at least this similar (difflib ratio, 0.0-1.0) before
# typo tolerance kicks in. Lower = more generous = more false matches.
FUZZY_MIN_RATIO = 0.80

# --------------------------------------------------------------------------- #
# Revenue score — fixed tier thresholds (SAR), highest threshold first.
# Each entry is (minimum_revenue_sar, score, label). The first tier whose
# threshold the revenue meets or exceeds wins.
# --------------------------------------------------------------------------- #
REVENUE_TIERS: list[tuple[float, float, str]] = [
    (1_000_000_000, 1.0, "top"),
    (100_000_000, 0.6, "mid"),
    (0, 0.3, "low"),
]
REVENUE_MISSING_SCORE = 0.0

# --------------------------------------------------------------------------- #
# Stage 1 pre-filter gates — sequential HARD CUTS, applied in this order:
#   1. location == target_location (case-insensitive; optional, see rank_prospects)
#   2. industry_score > 0 and >= PREFILTER_MIN_INDUSTRY_SCORE (exact, adjacent or fuzzy match)
#   3. revenue tier not in PREFILTER_REVENUE_CUT_TIERS
#   4. if more than LLM_PREFILTER_TOP_N survive, the top N by industry_score
#      (exact match first), then revenue_sar as the tie-breaker only
# None of these are weights: a company failing a gate never reaches the LLM,
# however strong its description similarity. pre_filter_score is still
# recorded for the audit trail but does not decide the shortlist.
# --------------------------------------------------------------------------- #
# Step 2 passes when industry_score > INDUSTRY_NO_MATCH_SCORE (any exact, adjacent
# or fuzzy match) AND industry_score >= PREFILTER_MIN_INDUSTRY_SCORE. The default
# lets every positive match through; set it to INDUSTRY_ADJACENT_SCORE to cut
# fuzzy partial matches (scores between 0 and 0.5) and keep exact/adjacent only.
PREFILTER_MIN_INDUSTRY_SCORE = INDUSTRY_NO_MATCH_SCORE
PREFILTER_REVENUE_CUT_TIERS = ("low", "missing")  # "missing" = absent, non-numeric or negative

# --------------------------------------------------------------------------- #
# Embedding score — cosine similarity normalisation (fixed scale, not dynamic
# min-max across the batch, so scores stay reproducible run to run).
#
# The query embedded in Stage 1 is the LLM-written buyer profile, not the raw
# product text (see diagnose_embeddings.py for why). Calibrated against the
# real silah_data.csv distribution for three real profiles (water bottle,
# phone cover, cyber-training platform): medians 0.18-0.23, p95 0.33-0.38,
# maxima 0.55-0.59. Clearly irrelevant pairs (a telco for a water bottle, a
# bank for a phone cover, a supermarket for cyber training) sit at 0.13-0.22;
# clearly relevant ones at 0.36-0.59.
#
# LOW = 0.20 puts the floor at that noise level, so the irrelevant half of the
# dataset scores 0.0 instead of being rewarded for stray word overlap. HIGH =
# 0.60 sits just above the observed maximum, so nothing saturates and the
# ordering among the top candidates is preserved. Re-measure if you change
# EMBEDDING_MODEL_NAME or the buyer-profile prompt — these are specific to
# this model's similarity scale on this kind of query.
# --------------------------------------------------------------------------- #
DESCRIPTION_SIMILARITY_HIGH = 0.60
DESCRIPTION_SIMILARITY_LOW = 0.20
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Descriptions per forward pass when batch-embedding a dataset. Measured on
# the 500-row dataset: 64 and 256 land within noise (~1.7s), so this is not a
# tuning lever — the win is batching at all versus one encode() per row. A
# slow *first* encode in a fresh process is the OS paging the model weights
# in from disk, not batch size; don't tune this off a single cold run.
EMBEDDING_BATCH_SIZE = 64

# --------------------------------------------------------------------------- #
# Stage 2 — LLM description comparison (OpenAI)
#
# The shortlist is decided by the sequential gates above (PREFILTER_*), not
# by a blended score. For the companies that survive them, the final
# fit_score blends industry + revenue with an LLM comparison of the product
# description against each company's business_description.
# --------------------------------------------------------------------------- #
OPENAI_MODEL = "gpt-5.6-luna"

# Step 4 cap: if more than this many companies survive the gates, only the
# top N by industry_score (exact match first), then revenue_sar, are sent
# to the LLM. This is the cost guard: a 500-row dataset costs at most 20
# API calls, not 500.
LLM_PREFILTER_TOP_N = 20

LLM_MAX_OUTPUT_TOKENS = 200
LLM_TIMEOUT_SECONDS = 30
LLM_MAX_ATTEMPTS = 3

# Stage 2 calls are independent network waits, so this many run at once.
# Bounded so a burst doesn't trip the API rate limit; 429s retry with backoff.
LLM_MAX_CONCURRENCY = 8

# Sampling temperature for the description call. Lower is more repeatable,
# but gpt-5.6-luna rejects any value other than its default (verified: the
# API returns 400 for temperature=0.0), so this is None for that model.
# Set it (e.g. 0.0) only if OPENAI_MODEL is changed to one that supports it.
LLM_TEMPERATURE = None

# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
MAX_DESCRIPTION_LENGTH = 2000

# --------------------------------------------------------------------------- #
# Agent-3 handoff — match_level bands on fit_score, highest threshold first.
# The first band whose threshold the score meets or exceeds wins.
# --------------------------------------------------------------------------- #
MATCH_LEVELS: list[tuple[float, str]] = [
    (0.7, "high match"),
    (0.4, "medium match"),
    (0.0, "low match"),
]

# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
# select_prospect() flags the top pick for human review when its fit_score
# falls below this confidence threshold: the "medium match" boundary above,
# so a "low match" top pick is always flagged. One constant, not two 0.4s.
HUMAN_REVIEW_MIN_FIT_SCORE = MATCH_LEVELS[1][0]
