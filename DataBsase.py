import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_URL and SUPABASE_KEY must be set in the environment."
    )

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# -------------------------
# PRODUCTS
# -------------------------

def save_product(
    product_name: str,
    product_description: str,
    sender_name: str,
    sender_company_name: str,
    sender_email: str | None = None,
    sender_phone: str | None = None
):
    result = (
        supabase
        .table("products")
        .insert({
            "product_name": product_name,
            "product_description": product_description,
            "sender_name": sender_name,
            "sender_company_name": sender_company_name,
            "sender_email": sender_email,
            "sender_phone": sender_phone
        })
        .execute()
    )

    return result.data[0]


def get_products():
    result = (
        supabase
        .table("products")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )

    return result.data


# -------------------------
# RUNS
# -------------------------

def create_run(
    product_id: int,
    preferred_city: str | None = None,
    requested_company_count: int = 5
):
    result = (
        supabase
        .table("runs")
        .insert({
            "product_id": product_id,
            "preferred_city": preferred_city,
            "requested_company_count": requested_company_count
        })
        .execute()
    )

    return result.data[0]


def get_runs_for_product(product_id: int):
    result = (
        supabase
        .table("runs")
        .select("*")
        .eq("product_id", product_id)
        .order("created_at", desc=True)
        .execute()
    )

    return result.data


# -------------------------
# PROSPECTS
# -------------------------

def save_prospect(
    run_id: int,
    company_name: str,
    fit_score: float | None = None,
    match_level: str | None = None,
    reason: str | None = None,
    company_email: str | None = None,
    source_company_id: str | None = None,
    research_status: str | None = None
):
    result = (
        supabase
        .table("prospects")
        .insert({
            "run_id": run_id,
            "source_company_id": source_company_id,
            "company_name": company_name,
            "fit_score": fit_score,
            "match_level": match_level,
            "reason": reason,
            "company_email": company_email,
            "research_status": research_status
        })
        .execute()
    )

    return result.data[0]


def get_prospects_for_run(run_id: int):
    result = (
        supabase
        .table("prospects")
        .select("*")
        .eq("run_id", run_id)
        .order("fit_score", desc=True)
        .execute()
    )

    return result.data


# -------------------------
# OUTREACH
# -------------------------

def save_outreach(
    prospect_id: int,
    status: str = "not_sent",
    email_subject: str | None = None,
    message: str | None = None,
    status_reason: str | None = None,
    reply_status: str | None = None,
    sent_at: str | None = None
):
    if status == "sent" and sent_at is None:
        sent_at = datetime.now(timezone.utc).isoformat()

    result = (
        supabase
        .table("outreach")
        .insert({
            "prospect_id": prospect_id,
            "status": status,
            "email_subject": email_subject,
            "message": message,
            "status_reason": status_reason,
            "reply_status": reply_status,
            "sent_at": sent_at
        })
        .execute()
    )

    return result.data[0]


def update_outreach_status(
    outreach_id: int,
    status: str,
    reply_status: str | None = None
):
    updates = {
        "status": status
    }

    if reply_status is not None:
        updates["reply_status"] = reply_status

    if status == "sent":
        updates["sent_at"] = datetime.now(timezone.utc).isoformat()

    result = (
        supabase
        .table("outreach")
        .update(updates)
        .eq("id", outreach_id)
        .execute()
    )

    return result.data


def get_outreach_for_prospect(prospect_id: int):
    result = (
        supabase
        .table("outreach")
        .select("*")
        .eq("prospect_id", prospect_id)
        .order("created_at", desc=True)
        .execute()
    )

    return result.data
