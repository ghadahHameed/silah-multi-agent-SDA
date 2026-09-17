"""Manual, end-to-end demo of Agent 2 (Qualification & Prioritization).

Not a test — a developer inspection tool. Loads the full prospect CSV, asks
for the target city FIRST (Step 1 of the sequential pre-filter: a
case-insensitive hard cut before any scoring), then runs the real
``rank_prospects``/``load_prospects`` pipeline against an ICP you type in
interactively, printing the per-step funnel and the actual selected +
rejected companies with their full score breakdown and reasoning trace, so
you can eyeball real output instead of just green test results. The TOP-N
section prints the exact JSON Agent 3 receives.

The top-N prompt (how many of the scored companies go to Agent 3) accepts
1-5 and defaults to 3. It is separate from ``LLM_PREFILTER_TOP_N``, the cap
on how many gate survivors get *scored* (at most 20, fewer if fewer survive).
A run where nothing survives the three gates prints a plain "no matches"
message instead of empty sections; that is a valid outcome, not an error.

If Step 0 cannot infer any target industry from the product text, the demo
prints the clarification question, asks once for "Additional detail", appends
it as "Typical buyer: ..." and re-runs. If the list is still empty it stops
with a clear message: one retry, never an open loop, and never the old silent
skip of the industry gate.

Run with:  ``python demo_agent2_run.py``
"""

import json
import os
from collections import Counter

from prospect_loader import load_prospects
from qualification_agent import DEFAULT_ICP_DESCRIPTION, DEFAULT_PROSPECTS_CSV
from scoring import (
    STATUS_NO_MATCHES,
    ClarificationNeeded,
    embed_company_descriptions,
    format_sar,
    rank_prospects,
    shortlist_key,
    summarize_funnel,
)
from scoring_config import LLM_PREFILTER_TOP_N, OPENAI_MODEL

# DEFAULT_ICP_DESCRIPTION (used when you just press Enter at the prompt; the
# target industries are inferred from it, not typed) and the CSV path come
# from qualification_agent so the two entry points share one definition.
DEFAULT_TOP_N = 3
# Bounds for the top-N prompt: how many of the LLM-scored companies are
# handed to Agent 3. Unrelated to LLM_PREFILTER_TOP_N (20), which caps how
# many gate survivors get *scored*; this picks how many of those get *selected*.
TOP_N_MIN = 1
TOP_N_MAX = 5


def _fmt(value) -> str:
    return f"{value:.2f}" if value is not None else " --  "


def _print_breakdown(entry: dict) -> None:
    tag = entry.get("description_score_source") or entry["stage"]
    print(f"  {entry['company_name']} ({entry['company_id']})  [{tag}]")
    print(f"    fit_score:         {_fmt(entry['fit_score'])}")
    print(f"    pre_filter_score:  {_fmt(entry['pre_filter_score'])}   (audit only; the shortlist is decided by the gates)")
    print(f"    industry_score:    {_fmt(entry['industry_score'])}")
    print(
        f"    revenue_score:     {_fmt(entry['revenue_score'])}   "
        f"({entry.get('revenue_tier') or '--'} tier, {format_sar(entry.get('revenue_sar'))})"
    )
    print(f"    embedding_score:   {_fmt(entry['embedding_score'])}   (local, no API)")
    print(f"    description_score: {_fmt(entry['description_score'])}   (LLM)")


def _print_reasoning(entry: dict) -> None:
    for line in entry["reasoning"]:
        print(f"    - {line}")


def _cities(companies: list) -> Counter:
    return Counter((company.get("location") or "").strip() for company in companies)


def _prompt_location(companies: list) -> str:
    """Ask for the target city first: location is Step 1 of the pre-filter.

    Matching is case-insensitive on the CSV's ``location`` column, and the
    input is checked against the cities actually present so a typo re-prompts
    instead of silently scoring nothing. Returns the CSV's own spelling.
    """
    available = _cities(companies)
    canonical = {city.casefold(): city for city in available if city}
    choices = ", ".join(f"{city} ({count})" for city, count in available.most_common() if city)
    while True:
        raw = input("Target city (Riyadh, Jeddah, or Dammam): ").strip()
        city = canonical.get(raw.casefold())
        if city:
            return city
        print(f"  No companies with location == {raw!r} (case-insensitive). Available: {choices}")


def _prompt_top_n() -> int:
    """Ask how many of the LLM-scored companies to hand to Agent 3 (1-5, default 3).

    Blank keeps the default. Anything non-numeric or outside 1-5 prints why
    and re-prompts rather than being silently clamped.
    """
    while True:
        raw = input(f"How many top companies to show? ({TOP_N_MIN}-{TOP_N_MAX}) [{DEFAULT_TOP_N}]: ").strip()
        if not raw:
            return DEFAULT_TOP_N
        try:
            top_n = int(raw)
        except ValueError:
            print(
                f"  {raw!r} is not a whole number. Enter a number between {TOP_N_MIN} and {TOP_N_MAX}, "
                f"or press Enter for {DEFAULT_TOP_N}."
            )
            continue
        if TOP_N_MIN <= top_n <= TOP_N_MAX:
            return top_n
        print(
            f"  {top_n} is out of range. Enter a number between {TOP_N_MIN} and {TOP_N_MAX}, "
            f"or press Enter for {DEFAULT_TOP_N}."
        )


def _print_pre_filtered(
    pre_filtered: list, industry_cut: list, revenue_cut: list, rank_cut: list, survivors: int
) -> None:
    cap = LLM_PREFILTER_TOP_N
    print(
        f"\n{len(pre_filtered)} PRE-FILTERED before the LLM step "
        f"({len(industry_cut)} failed the industry gate, {len(revenue_cut)} failed the revenue gate, "
        f"{len(rank_cut)} passed both but missed the top {cap} on industry_score, then revenue_sar)"
    )
    nearest = sorted(rank_cut, key=shortlist_key)[:3]  # the same Step 4 order scoring.py used
    if nearest:
        print(f"  {len(nearest)} nearest misses (best industry_score, then revenue_sar, among those cut at Step 4):")
        for entry in nearest:
            _print_breakdown(entry)
            _print_reasoning(entry)
    elif survivors:
        print("  (nobody was cut at Step 4: every company passing the gates went to the LLM)")
    else:
        print("  (nobody reached Step 4: no company passed both the industry and revenue gates)")


def _print_skipped(skipped: list) -> None:
    if skipped:
        print(f"\n{len(skipped)} COMPANIES SKIPPED DUE TO BAD DATA")
        for entry in skipped:
            print(f"  {entry['company_id']}: {entry['reasoning'][0]}")


def _rank_with_clarification(
    companies: list, product_description: str, top_n: int, target_location: str, evaluation_log: list
) -> tuple:
    """Run the pipeline; if Step 0 infers no industry, ask once and re-run.

    Returns ``(ranked_prospects, product_description_used)``. When the one
    clarification does not help either, prints a final message and returns
    ``(None, description)``: one retry, never an open loop, and never the old
    silent skip of the industry gate.
    """

    def _rank(description: str) -> list:
        return rank_prospects(  # target industries are inferred from the description
            companies,
            description,
            top_n=top_n,
            evaluation_log=evaluation_log,
            target_location=target_location,  # Step 1 of the pre-filter: case-insensitive hard cut
        )

    try:
        return _rank(product_description), product_description
    except ClarificationNeeded as need:
        evaluation_log.clear()  # every company was logged as not-evaluated; the retry starts clean
        print(f"\n{need.prompt_for_user}")
        clarification = input("Additional detail: ").strip()
        if not clarification:
            print("No additional detail given -- try a more specific product description.")
            return None, product_description
        product_description = f"{product_description.rstrip('. ')}. Typical buyer: {clarification}"
        print(f"Re-running with: {product_description!r}")

    try:
        return _rank(product_description), product_description
    except ClarificationNeeded as need:
        evaluation_log.clear()
        print(f"\n{need.reason}")
        print("Still unable to determine target industries -- try a more specific product description.")
        return None, product_description


def _run_once(companies: list) -> None:
    target_location = _prompt_location(companies)
    product_description = input("Describe your product: ").strip() or DEFAULT_ICP_DESCRIPTION
    top_n = _prompt_top_n()

    evaluation_log: list = []
    ranked_prospects, product_description = _rank_with_clarification(
        companies, product_description, top_n, target_location, evaluation_log
    )
    if ranked_prospects is None:
        return  # still no target industries after one clarification; the message is already printed

    scored = [entry for entry in evaluation_log if entry["stage"] == "scored"]
    pre_filtered = [entry for entry in evaluation_log if entry["stage"] == "pre-filtered"]
    skipped = [entry for entry in evaluation_log if entry["stage"] == "skipped"]
    fallbacks = sum(1 for entry in scored if entry["description_score_source"] == "embedding_fallback")

    # Steps 2-4 are sequential hard cuts; pre_filter_cut on each entry says
    # which gate cut it (industry / revenue / rank) or None if it reached the LLM.
    industry_cut = [entry for entry in pre_filtered if entry.get("pre_filter_cut") == "industry"]
    revenue_cut = [entry for entry in pre_filtered if entry.get("pre_filter_cut") == "revenue"]
    rank_cut = [entry for entry in pre_filtered if entry.get("pre_filter_cut") == "rank"]
    # The per-stage counts come from the same helper qualify_prospects uses
    # for its funnel_summary, so the demo and the agent can never disagree.
    summary = summarize_funnel(evaluation_log, target_location=target_location)
    industry_pass, survivors = summary["industry_pass"], summary["survivors"]
    cap = LLM_PREFILTER_TOP_N
    step4 = (
        f"top {cap} by industry_score (exact match first), then revenue_sar as tie-breaker ({survivors} > {cap})"
        if survivors > cap
        else f"all {survivors} go to the LLM ({survivors} <= {cap}, no ranking needed)"
    )

    if evaluation_log:
        first = evaluation_log[0]
        print(
            f"\nTarget industries ({first['industry_inference_source']}): "
            f"{', '.join(first['icp_industries']) or 'none'}"
        )
        print(
            f"Buyer profile ({first['buyer_profile_source']}): "
            f"{first['buyer_profile'] or '(none -- embedding the raw product description)'}"
        )
    funnel = (
        f"Funnel: {len(companies)} loaded -> {summary['in_location']} in {target_location} "
        f"-> {industry_pass} with industry match (exact or adjacent) -> {survivors} passing revenue tier "
        f"-> {step4} -> {len(scored)} scored by {OPENAI_MODEL}"
    )
    if fallbacks:
        funnel += f" ({fallbacks} via embedding fallback)"
    print(funnel)
    print(
        f"  Each step is a hard cut: {summary['location_filtered']} outside {target_location}, "
        f"{len(industry_cut)} with no industry match, {len(revenue_cut)} in the low/missing revenue tier, "
        f"{len(rank_cut)} passed both gates but ranked outside the top {cap} by industry_score, then revenue_sar."
    )

    if summary["status"] == STATUS_NO_MATCHES:
        # Nothing survived location -> industry -> revenue, so there was
        # nothing to score and the LLM was not called. A valid outcome (a
        # product can genuinely have no fit in a city), not an error: the
        # funnel above and the cut counts below are the audit trail.
        print()
        print("No companies matched all three filters (location, industry, revenue) for this product.")
        print("Try a different city, or the product description may need broader target industries.")
        print(f"  {summary['reason']}")
        print("  No LLM scoring calls were made: there was nothing to score.")
        _print_pre_filtered(pre_filtered, industry_cut, revenue_cut, rank_cut, survivors)
        _print_skipped(skipped)
        return

    # Complete breakdown of every company that reached Stage 2, in ranked
    # order, so the top-N and 3-lowest summaries below can be verified
    # against the full shortlist. Same evaluation_log entries, no re-scoring.
    shortlist = sorted(
        scored,
        key=lambda e: e["fit_score"] if e["fit_score"] is not None else -1.0,
        reverse=True,
    )
    print()
    print(
        f"=== ALL {len(shortlist)} COMPANIES EVALUATED BY THE LLM "
        f"({OPENAI_MODEL}; sorted by fit_score, highest first) ==="
    )
    for entry in shortlist:
        _print_breakdown(entry)
        _print_reasoning(entry)
        print("    => SELECTED (sent to Agent 3)" if entry["selected"] else "    => NOT SELECTED")

    print(f"\nTOP {top_n} SELECTED -- exactly what Agent 3 receives (the rank_prospects return value):")
    print(json.dumps(ranked_prospects, indent=2, ensure_ascii=False))

    lowest_scored = sorted((e for e in scored if not e["selected"]), key=lambda e: e["fit_score"])[:3]
    print(f"\n{len(lowest_scored)} LOWEST OF THE {len(scored)} LLM-EVALUATED (NOT selected)")
    for entry in lowest_scored:
        _print_breakdown(entry)
        _print_reasoning(entry)

    _print_pre_filtered(pre_filtered, industry_cut, revenue_cut, rank_cut, survivors)
    _print_skipped(skipped)


def main() -> None:
    companies = load_prospects(DEFAULT_PROSPECTS_CSV)
    by_city = ", ".join(f"{city} {count}" for city, count in _cities(companies).most_common() if city)
    print(f"Loaded {len(companies)} companies from {os.path.basename(DEFAULT_PROSPECTS_CSV)} ({by_city})")

    # Load the model and embed every description once, up front. scoring.py
    # caches both, so each product you type below costs one ICP embedding
    # plus cached lookups — not 500 fresh encodes.
    print("Loading embedding model and indexing company descriptions...")
    embed_company_descriptions(companies)

    while True:
        print()
        _run_once(companies)

        again = input("\nTest another product? (y/n): ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()
