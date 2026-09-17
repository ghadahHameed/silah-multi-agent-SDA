"""Agent 2 — Qualification & Prioritization for SILAH.

Agent 2 takes a list of candidate companies — already filtered by city by
Agent 1 — computes a fit score for each one, then returns the top-N ranked
highest to lowest for Agent 3 (Outreach).

Only the product description is required. The target industries are
inferred from it with one LLM call per run (Step 0 of ``scoring.rank_prospects``);
pass ``icp_industries`` explicitly to pin them instead.

Scoring then runs in two stages (see ``scoring.py``): a description-aware
pre-filter (industry + revenue + local embedding similarity, no API calls)
narrows the batch to the strongest ``LLM_PREFILTER_TOP_N`` candidates, then an
LLM compares the product description against each shortlisted company's
business description to produce the description score that feeds the
0.30/0.30/0.40 fit formula. Location is not re-scored or re-filtered here —
that hard filter already happened in Agent 1.

Requires ``SILAH_OPENAI_API_KEY`` in the environment (loaded from ``.env``).

Two output channels, kept separate:

* ``companies`` — the Agent-3 contract: exactly ``company_name``,
  ``fit_score``, ``match_level``, ``reason`` and ``email`` per company.
* ``evaluation_log``   — the full per-company scoring audit trail (also
  written to ``agent2_scores_log.jsonl``), for developers.

Run directly:  ``python qualification_agent.py``
"""

import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv

from prospect_loader import load_prospects
from scoring import STATUS_NO_MATCHES, ClarificationNeeded, rank_prospects, select_prospect, summarize_funnel

load_dotenv()

logger = logging.getLogger("agent2.qualification_agent")

DEFAULT_PROSPECTS_CSV = os.path.join(os.path.dirname(__file__), "silah_data.csv")
DEFAULT_LOG_PATH = os.path.join(os.path.dirname(__file__), "agent2_scores_log.jsonl")
DEFAULT_TOP_N = 7

# Sample product for the CLI runs (this module's __main__, and the interactive
# demo's Enter-for-default). Normally supplied by Agent 1 / product config.
DEFAULT_ICP_DESCRIPTION = (
    "A B2B platform that helps organizations train employees on "
    "cybersecurity awareness, phishing risks, and security best practices."
)


def qualify_prospects(
    icp_description: str,
    icp_industries: Optional[list] = None,
    buyer_profile: Optional[str] = None,
    csv_path: str = DEFAULT_PROSPECTS_CSV,
    top_n: int = DEFAULT_TOP_N,
    verbose: bool = False,
    log_path: str = DEFAULT_LOG_PATH,
    target_location: Optional[str] = None,
) -> dict:
    """Run Agent 2 end to end: load -> infer industries -> score -> rank -> select.

    ``target_location`` (optional) is the Step 1 hard filter: only companies
    whose ``location`` equals it exactly are scored; the rest are logged with
    ``stage == "location-filtered"``. Leave it None when Agent 1 has already
    filtered the CSV by city.

    Returns a dict with:
      - ``status``: "ok", or "no_matches" when nothing survived the hard gates
        (location -> industry -> revenue). That is a valid outcome, not an
        error: no LLM scoring call is made, ``companies`` is empty,
        ``reason`` says why in one sentence and the log is still written like
        any other run. "needs_clarification" when Step 0 could not infer any
        target industry from the description (too vague to say who buys it):
        the gates never run, every company is logged as "not-evaluated", and
        ``prompt_for_user`` is the question to put to the user - call again
        with their answer appended to ``icp_description``. "empty_dataset"
        when the CSV loaded no companies.
      - ``reason``: None when ``status`` is "ok", else the explanation.
      - ``prompt_for_user``: the clarification question when ``status`` is
        "needs_clarification", else None.
      - ``funnel_summary``: the per-stage counts as one line, e.g.
        "500 loaded -> 213 in Riyadh -> 40 with industry match -> 31 passing
        revenue tier -> 20 scored (capped at 20)".
      - ``companies``: the top-N for Agent 3, each exactly ``{company_name,
        fit_score, match_level, reason, email}`` (``scoring.build_handoff``).
      - ``evaluation_log``: full per-company audit trail (scores + reasoning
        + selected flag), also persisted to ``log_path`` as JSONL.
      - ``icp_industries`` / ``industry_inference_source``: the target
        industries used and where they came from ("llm", "provided",
        "llm_failed").
      - ``buyer_profile`` / ``buyer_profile_source``: the one-sentence buyer
        profile the pre-filter embedded, and where it came from. When it is
        None the raw product description was embedded instead.
      - ``selected``: the top scored record (with fit_score) or None.
      - ``needs_human_review`` / ``review_reason``.
    """
    companies = load_prospects(csv_path)

    if not companies:
        return {
            "status": "empty_dataset",
            "reason": "Prospect dataset is empty; nothing to qualify.",
            "prompt_for_user": None,
            "funnel_summary": "0 loaded",
            "companies": [],
            "evaluation_log": [],
            "icp_industries": list(icp_industries or []),
            "industry_inference_source": "not_run",
            "buyer_profile": buyer_profile,
            "buyer_profile_source": "not_run",
            "selected": None,
            "needs_human_review": True,
            "review_reason": "Prospect dataset is empty; nothing to qualify.",
        }

    evaluation_log: list = []
    try:
        ranked_prospects = rank_prospects(
            companies,
            icp_description,
            icp_industries=icp_industries,
            buyer_profile=buyer_profile,
            top_n=top_n,
            verbose=verbose,
            evaluation_log=evaluation_log,
            log_path=log_path,
            target_location=target_location,
        )
    except ClarificationNeeded as need:
        # Step 0 inferred no industry. Not an outage (that is "llm_failed" and
        # continues): the description does not say who buys the product, so
        # hand the question back rather than scoring the city on size alone.
        summary = summarize_funnel(evaluation_log, target_location=target_location)
        logger.info("needs clarification: %s (%s)", need.reason, summary["funnel_summary"])
        return {
            **need.as_dict(),
            "funnel_summary": summary["funnel_summary"],
            "companies": [],
            "evaluation_log": evaluation_log,
            "icp_industries": [],
            "selected": None,
            "needs_human_review": True,
            "review_reason": need.reason,
        }

    # Pre-filtered and skipped records have no fit_score; only fully scored
    # companies are candidates for selection.
    scored_sorted = sorted(
        (entry for entry in evaluation_log if entry["stage"] == "scored"),
        key=lambda entry: entry["fit_score"],
        reverse=True,
    )
    selection = select_prospect(scored_sorted)

    # Every log entry carries the run's target industries; read them off the first.
    first = evaluation_log[0] if evaluation_log else {}

    summary = summarize_funnel(evaluation_log, target_location=target_location)
    if summary["status"] == STATUS_NO_MATCHES:
        # Expected outcome, recorded like any other run: the JSONL written by
        # rank_prospects holds every company with the gate that cut it.
        logger.info("no matches: %s (%s)", summary["reason"], summary["funnel_summary"])

    return {
        "status": summary["status"],
        "reason": summary["reason"],
        "prompt_for_user": summary["prompt_for_user"],
        "funnel_summary": summary["funnel_summary"],
        "companies": ranked_prospects,
        "evaluation_log": evaluation_log,
        "icp_industries": first.get("icp_industries", list(icp_industries or [])),
        "industry_inference_source": first.get("industry_inference_source", "not_run"),
        "buyer_profile": first.get("buyer_profile", buyer_profile),
        "buyer_profile_source": first.get("buyer_profile_source", "not_run"),
        **selection,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # The target industries are inferred from the description, not typed by hand.
    result = qualify_prospects(
        DEFAULT_ICP_DESCRIPTION,
        verbose=os.getenv("DEBUG", "").strip().lower() in ("1", "true", "yes"),
    )
    print(json.dumps(result, indent=2, default=str))
