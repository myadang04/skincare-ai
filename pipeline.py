"""
pipeline.py
───────────
Connects the ML model + ingredient dataset into one end-to-end function.

Usage:
    from pipeline import run
    results = run(ingredients, skin_type, concerns)
"""

import os
import pandas as pd
from model.predict import predict_ingredients_batch, load_ingredients

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "ingredients_final.csv")

LABEL_COLORS = {
    "good fit":            "#2e7d32",   # green
    "possible irritation": "#e65100",   # orange
    "poor fit":            "#c62828",   # red
    "unknown":             "#757575",   # grey
}

LABEL_ICONS = {
    "good fit":            "✅",
    "possible irritation": "⚠️",
    "poor fit":            "❌",
    "unknown":             "❓",
}


def run(ingredients: list, skin_type: str, concerns: list) -> dict:
    """
    Full analysis pipeline.

    Parameters
    ----------
    ingredients : list of ingredient name strings (from OCR)
    skin_type   : e.g. "Sensitive"
    concerns    : e.g. ["Acne", "Redness"]

    Returns
    -------
    dict with keys:
        overall_label    str   — worst label across all found ingredients
        overall_color    str   — hex colour for the overall label
        overall_icon     str   — emoji for overall label
        results          list  — per-ingredient dicts (see predict_ingredient)
        found_count      int   — ingredients matched in dataset
        flag_count       int   — ingredients flagged (possible irritation / poor fit)
        top_suggestions  list  — top 3 good-fit ingredients for this profile
        summary_stats    dict  — counts per label
    """
    # ── 1. Run ML model on every extracted ingredient ──────────────────────
    raw_results = predict_ingredients_batch(ingredients, skin_type, concerns)

    # ── 2. Separate found vs not-found ────────────────────────────────────
    found   = [r for r in raw_results if r["found"]]
    missing = [r for r in raw_results if not r["found"]]

    # ── 3. Overall label = worst label among found ingredients ────────────
    priority = {"poor fit": 2, "possible irritation": 1, "good fit": 0, "unknown": -1}
    if found:
        worst = max(found, key=lambda r: priority.get(r["label_name"], -1))
        overall_label = worst["label_name"]
    else:
        overall_label = "unknown"

    # ── 4. Summary counts ─────────────────────────────────────────────────
    counts = {"good fit": 0, "possible irritation": 0, "poor fit": 0, "unknown": 0}
    for r in raw_results:
        counts[r["label_name"]] += 1

    # ── 5. Top 3 ingredient suggestions ──────────────────────────────────
    top_suggestions = _get_suggestions(skin_type, concerns, exclude=ingredients)

    return {
        "overall_label":   overall_label,
        "overall_color":   LABEL_COLORS.get(overall_label, "#757575"),
        "overall_icon":    LABEL_ICONS.get(overall_label, "❓"),
        "results":         raw_results,
        "found_count":     len(found),
        "flag_count":      counts["possible irritation"] + counts["poor fit"],
        "top_suggestions": top_suggestions,
        "summary_stats":   counts,
    }


def _get_suggestions(skin_type: str, concerns: list[str], exclude: list[str], top_n: int = 3) -> list[dict]:
    """
    Return top_n ingredients from the dataset that are a 'good fit'
    for this skin profile and not already in the product.
    Ranked by number of matched concerns + breadth_score.
    """
    from model.predict import predict_ingredient, load_ingredients
    import ast

    df = load_ingredients()
    exclude_lower = {e.lower().strip() for e in exclude}

    candidates = []
    for _, row in df.iterrows():
        if row["name_lower"] in exclude_lower:
            continue
        result = predict_ingredient(row["name"], skin_type, concerns)
        if result["label_name"] == "good fit" and result["found"]:
            score = len(result["matched_concerns"]) * 2 + (result["breadth_score"] or 0)
            candidates.append((score, result))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in candidates[:top_n]]
