"""Prospect data loading for Agent 2 (Qualification & Prioritization).

Loads a CSV of candidate companies — already filtered by city/location by
Agent 1 — into plain dicts matching the schema Agent 2 expects:

    company_id, company_name, industry, location, business_description,
    website, email, revenue_sar

Rows that are blank or missing a company_id/company_name are skipped rather
than raising, so a few malformed rows never abort the whole batch. No missing
company information is ever invented.
"""

import csv
import logging

logger = logging.getLogger("agent2.prospect_loader")

FIELDS = [
    "company_id",
    "company_name",
    "industry",
    "location",
    "business_description",
    "website",
    "email",
    "revenue_sar",
]


def _clean(value) -> str:
    """Normalise a raw CSV cell to a trimmed string."""
    if value is None:
        return ""
    return str(value).strip()


def load_prospects(csv_path: str) -> list[dict]:
    """Load prospect companies from a CSV file into a list of plain dicts.

    ``revenue_sar`` is parsed to ``float`` when present and numeric, else
    ``None`` (never fabricated, never crashes the load).
    """
    companies: list[dict] = []

    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        # Sniff the delimiter so both comma- and semicolon-separated exports work.
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            reader = csv.DictReader(handle, dialect=dialect)
        except csv.Error:
            reader = csv.DictReader(handle)

        for line_number, raw_row in enumerate(reader, start=2):
            # ``None`` key holds surplus columns from a malformed row; drop it.
            row = {
                (key or "").strip().lower(): _clean(value)
                for key, value in raw_row.items()
                if key is not None
            }

            if not any(row.values()):
                continue  # completely blank line

            company_id = row.get("company_id", "")
            company_name = row.get("company_name", "")
            if not company_id or not company_name:
                logger.warning("skipping row %d: missing company_id or company_name", line_number)
                continue

            revenue_sar = None
            revenue_raw = row.get("revenue_sar", "")
            if revenue_raw:
                try:
                    revenue_sar = float(revenue_raw)
                except ValueError:
                    logger.warning("company_id=%s: revenue_sar is not numeric, treating as missing", company_id)

            companies.append(
                {
                    "company_id": company_id,
                    "company_name": company_name,
                    "industry": row.get("industry", ""),
                    "location": row.get("location", ""),
                    "business_description": row.get("business_description", ""),
                    "website": row.get("website") or None,
                    "email": row.get("email") or None,
                    "revenue_sar": revenue_sar,
                }
            )

    return companies
