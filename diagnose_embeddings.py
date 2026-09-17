"""Diagnostic: is the embedding similarity in scoring.py actually working?

Run:  python diagnose_embeddings.py

Uses the REAL embedding path from scoring.py (no test fakes). Prints which
function and model are really in use, the vector shape / dtype / L2 norm, and
raw cosine similarity (no threshold normalisation) for a few labelled pairs.
Then cross-checks the same pairs through sentence_transformers.util.cos_sim
straight from the model, bypassing scoring.py's maths entirely, and finally
runs one product against every real company description so the actual
top / bottom separation is visible.

Any failure prints a loud banner with the traceback and exits non-zero.
Nothing is swallowed or silently substituted.
"""

import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRODUCT = "a stainless steel water bottle"

PAIRS = [
    ("similar pair", PRODUCT, "a reusable metal drink bottle", "expect HIGH: near-paraphrase"),
    ("unrelated pair", PRODUCT, "a corporate tax audit and compliance firm", "expect LOW"),
    ("retail buyer", PRODUCT, "a Saudi supermarket and hypermarket operator", "expect moderate-to-high?"),
    ("IT consultancy", PRODUCT, "an enterprise IT and digital transformation consultancy", "expect LOW"),
]


def banner(title: str) -> None:
    print("\n" + "=" * 76 + f"\n{title}\n" + "=" * 76)


def fail(message: str) -> None:
    print("\n" + "!" * 76 + f"\nFAILURE: {message}\n" + "!" * 76)
    traceback.print_exc()
    sys.exit(1)


def main() -> None:
    banner("1. Which embedding function and model is scoring.py REALLY using?")
    try:
        import sentence_transformers
        import torch

        import scoring
        from scoring_config import EMBEDDING_MODEL_NAME
    except Exception:
        fail("could not import scoring.py / its dependencies")

    print(f"python {sys.version.split()[0]}   sentence-transformers {sentence_transformers.__version__}   torch {torch.__version__}")
    embed = scoring._embed_text
    print(f"scoring._embed_text -> {embed.__module__}.{embed.__qualname__}")
    print(f"                       defined in {sys.modules[embed.__module__].__file__}")
    print(f"scoring._encode     -> {scoring._encode.__module__}.{scoring._encode.__qualname__}")
    print(f"configured model    -> EMBEDDING_MODEL_NAME = {EMBEDDING_MODEL_NAME!r}")
    if embed.__module__ != "scoring" or scoring._encode.__module__ != "scoring":
        fail("the embedding function has been REPLACED by something outside scoring.py (leftover monkey-patch?)")

    try:
        model = scoring._get_embedding_model()
    except Exception:
        fail("the Sentence-Transformers model failed to load")
    print(f"loaded model object -> {type(model).__module__}.{type(model).__name__}")
    try:
        print(f"underlying HF model -> {model[0].auto_model.config._name_or_path}")
    except Exception as error:
        print(f"underlying HF model -> (could not read: {error.__class__.__name__})")
    print(f"max_seq_length      -> {getattr(model, 'max_seq_length', 'n/a')}")
    dimension = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    print(f"embedding dimension -> {dimension()}")

    banner("2. Vector shape, dtype and L2 norm (norm should be ~1.0 if normalised)")
    try:
        vector = np.asarray(embed(PRODUCT))
    except Exception:
        fail("scoring._embed_text raised")
    print(f"type={type(vector).__name__}  shape={vector.shape}  dtype={vector.dtype}  ndim={vector.ndim}")
    print(f"L2 norm = {float(np.linalg.norm(vector)):.6f}")
    print(f"first 5 values = {np.round(vector[:5], 4).tolist()}")
    if vector.ndim != 1:
        fail(f"expected a 1-D vector, got shape {vector.shape} (batch-shaped output compared as one vector?)")

    banner("3. Raw cosine similarity via scoring._cosine_similarity (the real pipeline path)")
    ours = {}
    for label, a, b, expectation in PAIRS:
        sim = scoring._cosine_similarity(np.asarray(embed(a)), np.asarray(embed(b)))
        ours[label] = sim
        print(f"   [{label:14}] {a:31} vs {b[:46]:47} -> {sim:6.3f}   ({expectation})")

    banner("4. Cross-check: same pairs via sentence_transformers.util.cos_sim, bypassing scoring.py")
    from sentence_transformers import util

    max_gap = 0.0
    for label, a, b, _ in PAIRS:
        lib = float(util.cos_sim(model.encode(a, convert_to_tensor=True), model.encode(b, convert_to_tensor=True))[0][0])
        gap = abs(lib - ours[label])
        max_gap = max(max_gap, gap)
        print(f"   [{label:14}] library -> {lib:6.3f}   scoring.py -> {ours[label]:6.3f}   |diff| = {gap:.2e}")

    banner(f"5. Against the REAL dataset: {PRODUCT!r} vs every company description")
    from prospect_loader import load_prospects

    companies = load_prospects(os.path.join(HERE, "silah_data.csv"))
    scoring.embed_company_descriptions(companies)
    product_vector = np.asarray(embed(PRODUCT))
    scored = []
    for company in companies:
        description = (company["business_description"] or "")[:2000]
        if description.strip():
            scored.append((scoring._cosine_similarity(np.asarray(embed(description)), product_vector), company))
    scored.sort(key=lambda item: -item[0])
    sims = [s for s, _ in scored]
    print(f"   n={len(sims)}   max {sims[0]:.3f}   p95 {sims[len(sims) // 20]:.3f}   median {sims[len(sims) // 2]:.3f}   min {sims[-1]:.3f}")
    print("   top 5 by raw cosine:")
    for sim, company in scored[:5]:
        print(f"      {sim:6.3f}  {company['company_name'][:28]:29} {company['industry'][:22]:23} | {company['business_description'][:58]}")
    print("   bottom 3:")
    for sim, company in scored[-3:]:
        print(f"      {sim:6.3f}  {company['company_name'][:28]:29} {company['industry'][:22]:23} | {company['business_description'][:58]}")

    banner("6. Verdict")
    similar, unrelated = ours["similar pair"], ours["unrelated pair"]
    if max_gap > 1e-4:
        print(f"scoring.py's cosine DISAGREES with the library by up to {max_gap:.2e} -> the maths in scoring.py is wrong.")
    else:
        print(f"scoring.py's cosine matches the library to {max_gap:.2e} -> the maths in scoring.py is correct.")
    if similar < 0.4:
        print(f"near-paraphrase pair scored only {similar:.3f} -> the MODEL is not producing usable similarities (bad load / wrong model).")
    else:
        print(f"near-paraphrase pair {similar:.3f} vs unrelated pair {unrelated:.3f} -> the model separates TEXT similarity fine.")
        print("Low scores for 'product vs business description' pairs are the model measuring text closeness, not buyer fit.")


if __name__ == "__main__":
    main()
