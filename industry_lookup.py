"""Rule-based industry compatibility lookup for Agent 2.

Deterministic string and dict work only — NOT semantic/LLM matching. A
prospect's stated ``industry`` is compared against the product's ICP industry
by ``industry_similarity``, which returns a graded 0.0-1.0 score: exact match,
alias resolution ("IT" -> "Information Technology"), curated adjacency, token
overlap, two-hop adjacency, then typo tolerance.

Keep ``INDUSTRY_ADJACENCY`` in sync with the ``industry`` values that actually
appear in the prospect dataset (see ``silah_data.csv``). Adjacency is
symmetric: listing ``"Banking": {"Insurance"}`` also treats "Insurance"
prospects as adjacent to an ICP industry of "Banking".
"""

from difflib import SequenceMatcher
from functools import lru_cache

from scoring_config import FUZZY_MIN_RATIO, INDUSTRY_ADJACENT_SCORE, INDUSTRY_EXACT_MATCH_SCORE

INDUSTRY_ADJACENCY: dict[str, set[str]] = {
    "Banking": {"Financial Services", "Fintech", "Insurance", "Investment"},
    "Financial Services": {"Banking", "Fintech", "Insurance", "Investment"},
    "Fintech": {"Banking", "Financial Services", "Investment", "Technology"},
    "Insurance": {"Banking", "Financial Services"},
    "Investment": {"Banking", "Financial Services", "Fintech", "Economic Development", "Agriculture Investment"},
    "Economic Development": {"Investment", "Agriculture Investment"},
    "Agriculture Investment": {"Investment", "Economic Development", "Food & Beverage"},
    "Information Technology": {
        "Software", "Technology", "Cybersecurity", "Artificial Intelligence",
        "Data & AI", "Cloud Communications",
    },
    "Software": {"Information Technology", "Technology", "Artificial Intelligence", "Data & AI", "Gaming"},
    "Technology": {
        "Information Technology", "Software", "Artificial Intelligence",
        "Data & AI", "Cybersecurity", "Gaming", "Education",
    },
    "Cybersecurity": {"Information Technology", "Technology", "Software"},
    "Artificial Intelligence": {"Information Technology", "Software", "Technology", "Data & AI"},
    "Data & AI": {"Information Technology", "Software", "Technology", "Artificial Intelligence"},
    "Cloud Communications": {"Information Technology", "Telecommunications", "Technology"},
    "Telecommunications": {"Information Technology", "Cloud Communications", "Technology"},
    "Education": {"Technology", "Information Technology"},
    "Gaming": {"Entertainment & Tourism", "Technology", "Software"},
    "Healthcare": {"Healthcare & Consumer", "Healthcare Supply Chain", "Pharmaceuticals"},
    "Healthcare & Consumer": {"Healthcare", "Consumer Goods"},
    "Healthcare Supply Chain": {"Healthcare", "Logistics", "Logistics & Infrastructure"},
    "Pharmaceuticals": {"Healthcare", "Chemicals"},
    "Retail": {"E-commerce Technology", "Food & Retail", "Consumer Goods"},
    "E-commerce Technology": {"Retail", "Delivery Technology", "Food & Retail"},
    "Food & Retail": {"Retail", "Food & Beverage", "Food Technology"},
    "Food & Beverage": {"Food & Retail", "Food Technology", "Food Delivery", "Agriculture Investment"},
    "Food Technology": {"Food & Beverage", "Food Delivery", "E-commerce Technology"},
    "Food Delivery": {"Delivery Technology", "Food & Beverage", "Food Technology"},
    "Delivery Technology": {"Logistics", "E-commerce Technology", "Food Delivery"},
    "Logistics": {"Logistics & Infrastructure", "Delivery Technology", "Aviation Services", "Healthcare Supply Chain"},
    "Logistics & Infrastructure": {"Logistics", "Construction", "Aviation Services"},
    "Construction": {"Real Estate", "Logistics & Infrastructure", "Manufacturing"},
    "Real Estate": {"Real Estate & Tourism", "Construction", "Hospitality"},
    "Real Estate & Tourism": {"Real Estate", "Tourism", "Tourism & Hospitality", "Hospitality"},
    "Tourism": {"Tourism & Hospitality", "Hospitality", "Travel", "Entertainment & Tourism", "Real Estate & Tourism"},
    "Tourism & Hospitality": {"Tourism", "Hospitality", "Travel", "Entertainment & Tourism"},
    "Hospitality": {"Tourism", "Tourism & Hospitality", "Real Estate", "Entertainment & Tourism"},
    "Travel": {"Tourism", "Travel Technology", "Airlines", "Aviation Services"},
    "Travel Technology": {"Travel", "Tourism", "E-commerce Technology"},
    "Airlines": {"Aviation Services", "Travel", "Logistics"},
    "Aviation Services": {"Airlines", "Travel", "Logistics", "Logistics & Infrastructure"},
    "Entertainment & Tourism": {"Tourism", "Tourism & Hospitality", "Gaming", "Hospitality"},
    "Manufacturing": {"Construction", "Chemicals", "Mining", "Consumer Goods"},
    "Chemicals": {"Manufacturing", "Pharmaceuticals", "Mining", "Energy"},
    "Mining": {"Manufacturing", "Chemicals", "Energy", "Utilities"},
    "Energy": {"Utilities", "Chemicals", "Mining"},
    "Utilities": {"Energy", "Mining"},
    "Consumer Goods": {"Retail", "Manufacturing", "Healthcare & Consumer", "Food & Beverage"},
}

# The fixed vocabulary of industry categories. Target-industry inference is
# constrained to exactly these names, so the model can never invent one.
KNOWN_INDUSTRIES: list[str] = list(INDUSTRY_ADJACENCY.keys())


def normalize(value: str) -> str:
    """Lowercase + trim for case-insensitive comparisons."""
    return (value or "").strip().lower()


_NORMALIZED_ADJACENCY: dict[str, set[str]] = {
    normalize(key): {normalize(v) for v in values} for key, values in INDUSTRY_ADJACENCY.items()
}


def industries_are_adjacent(industry_a: str, industry_b: str) -> bool:
    """True when the two industries are listed as adjacent in either direction.

    This is the adjacency test the industry gate relies on: ``industry_similarity``
    calls it for its 0.5 "adjacent industry" step (after alias resolution).
    """
    a, b = normalize(industry_a), normalize(industry_b)
    if not a or not b:
        return False
    return b in _NORMALIZED_ADJACENCY.get(a, set()) or a in _NORMALIZED_ADJACENCY.get(b, set())


# --------------------------------------------------------------------------- #
# Fuzzy matching
#
# The prospect's ``industry`` comes from a curated dataset, but the ICP
# industry is typed by a human at the prompt. "IT", "Cyber Security" and
# "InfoSec" all used to score 0.0 against the table, silently zeroing 30% of
# the prefilter weight. Everything below is still deterministic — stdlib
# string work and dict lookups, no model, no API.
# --------------------------------------------------------------------------- #

# Short forms and synonyms no string metric can bridge ("IT" is not
# character-similar to "Information Technology"). Keys are normalized.
INDUSTRY_ALIASES: dict[str, str] = {
    "it": "information technology",
    "i.t.": "information technology",
    "ict": "information technology",
    "information tech": "information technology",
    "info tech": "information technology",
    "tech": "technology",
    "infosec": "cybersecurity",
    "info sec": "cybersecurity",
    "cyber": "cybersecurity",
    "cyber security": "cybersecurity",
    "network security": "cybersecurity",
    "ai": "artificial intelligence",
    "ml": "artificial intelligence",
    "machine learning": "artificial intelligence",
    "saas": "software",
    "software development": "software",
    "dev": "software",
    "telecom": "telecommunications",
    "telco": "telecommunications",
    "finance": "financial services",
    "fintech": "fintech",
    "banking and finance": "banking",
    "banking & finance": "banking",
    "financial": "financial services",
    "ecommerce": "e-commerce technology",
    "e-commerce": "e-commerce technology",
    "e commerce": "e-commerce technology",
    "pharma": "pharmaceuticals",
    "f&b": "food & beverage",
    "food and beverage": "food & beverage",
    "oil & gas": "energy",
    "oil and gas": "energy",
    "property": "real estate",
    "hotels": "hospitality",
    "medical": "healthcare",
    "health": "healthcare",
}

# Tokens too generic to count as evidence on their own — "Travel Technology"
# and "Food Technology" share "technology" but are not the same market.
_GENERIC_TOKENS = {"services", "service", "technology", "tech", "group", "solutions", "and", "the", "co"}

_PUNCTUATION = str.maketrans({character: " " for character in "&/,-_().'\""})


def canonical(value: str) -> str:
    """Normalize, then resolve through the alias table."""
    normalized = normalize(value)
    collapsed = " ".join(normalized.split())
    return INDUSTRY_ALIASES.get(collapsed, collapsed)


def _compact(value: str) -> str:
    """Strip everything but letters and digits: 'Cyber Security' -> 'cybersecurity'."""
    return "".join(character for character in value if character.isalnum())


def _tokens(value: str) -> set[str]:
    return {token for token in value.translate(_PUNCTUATION).split() if token}


def _two_hop(a: str, b: str) -> bool:
    """True when a and b are connected through one intermediate industry."""
    for neighbour in _NORMALIZED_ADJACENCY.get(a, set()):
        if b in _NORMALIZED_ADJACENCY.get(neighbour, set()):
            return True
    return False


# The ICP industry is fixed for a whole batch and the dataset has ~47 distinct
# industry values, so 500 rows collapse to ~47 real computations. The
# SequenceMatcher fallback is the only step that isn't trivially cheap.
@lru_cache(maxsize=4096)
def industry_similarity(company_industry: str, icp_industry: str) -> tuple[float, str]:
    """Graded 0.0-1.0 industry match with a human-readable label.

    Tried in order, most reliable evidence first:

    1. exact match (after alias resolution)            -> 1.0
    2. same once spacing/punctuation is stripped       -> 1.0
    3. listed as adjacent in INDUSTRY_ADJACENCY        -> 0.5
    4. one name's tokens contain the other's           -> 0.6
    5. partial token overlap (Jaccard, generic tokens
       discounted)                                     -> up to 0.5
    6. connected via one intermediate industry         -> 0.25
    7. near-identical spelling (typo tolerance)        -> up to 0.85
    8. nothing                                         -> 0.0

    Deliberately generous: this feeds a recall funnel, and the LLM stage
    downstream does the precise judging. A missed company never reaches it.
    """
    if not company_industry or not icp_industry:
        return 0.0, "missing industry field"

    a, b = canonical(company_industry), canonical(icp_industry)

    if a == b:
        return INDUSTRY_EXACT_MATCH_SCORE, f"exact match ('{company_industry}' == ICP industry)"

    if _compact(a) == _compact(b):
        return INDUSTRY_EXACT_MATCH_SCORE, f"exact match ignoring spacing ('{company_industry}' ~ '{icp_industry}')"

    if industries_are_adjacent(a, b):
        return INDUSTRY_ADJACENT_SCORE, f"adjacent industry ('{company_industry}' ~ '{icp_industry}')"

    tokens_a, tokens_b = _tokens(a), _tokens(b)
    if tokens_a and tokens_b:
        if tokens_a <= tokens_b or tokens_b <= tokens_a:
            return 0.6, f"one name contains the other ('{company_industry}' ~ '{icp_industry}')"

        shared = tokens_a & tokens_b
        meaningful = shared - _GENERIC_TOKENS
        if meaningful:
            jaccard = len(shared) / len(tokens_a | tokens_b)
            # Generic-only overlap already excluded, so this is real evidence.
            score = round(min(0.5, jaccard), 4)
            if score > 0:
                return score, (
                    f"partial name overlap {sorted(meaningful)} "
                    f"('{company_industry}' ~ '{icp_industry}')"
                )

    if _two_hop(a, b):
        return 0.25, f"related via one step ('{company_industry}' ~ '{icp_industry}')"

    ratio = SequenceMatcher(None, _compact(a), _compact(b)).ratio()
    if ratio >= FUZZY_MIN_RATIO:
        score = round(min(0.85, ratio), 4)
        return score, f"near-identical spelling {ratio:.2f} ('{company_industry}' ~ '{icp_industry}')"

    return 0.0, f"no match ('{company_industry}' vs '{icp_industry}')"
