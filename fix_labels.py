"""
fix_labels.py
─────────────
Fixes training labels for ingredients that are incorrectly labeled as
"poor fit" for ALL skin types. Many high-sensitivity ingredients like
Ascorbic Acid, Retinol, and Lactic Acid should be "poor fit" for
Sensitive skin but "good fit" for Normal/Oily skin.

Run this ONCE from your project root:
    python fix_labels.py

Then delete the saved model and restart Streamlit to retrain.
"""

import os
import ast
import csv
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELED_CSV = os.path.join(BASE_DIR, "data", "processed", "labeled_pairs.csv")
INGREDIENTS_CSV = os.path.join(BASE_DIR, "data", "processed", "ingredients_final.csv")
MODEL_PKL = os.path.join(BASE_DIR, "model", "saved", "ingredient_classifier.pkl")

# Maps skin type → the avoid tag that makes an ingredient problematic
SKIN_TYPE_TO_AVOID = {
    "Sensitive": "Sensitive Skin",
    "Oily": "Oily Skin",
    "Dry": "Dry Skin",
    "Combination": "Combination Skin",
    "Normal": None,
}

# Maps user concerns → good_for tags
CONCERN_TO_GOODFOR = {
    "Acne": ["Acne"],
    "Blackheads": ["Blackheads"],
    "Redness": ["Redness"],
    "Dryness": ["Dry Skin"],
    "Hyperpigmentation": ["Pigmentation", "Post Blemish Marks"],
    "Sensitivity": ["Redness", "Impaired Skin Barrier"],
    "Dullness": ["Radiance"],
    "Wrinkles": ["Wrinkles", "Fine Lines"],
    "Fine Lines": ["Fine Lines"],
    "Enlarged Pores": ["Enlarged Pores"],
    "Dark Circles": ["Dark Circles"],
    "Texture": ["Texture"],
}


def parse_list(val):
    if pd.isna(val) or val == "":
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return [v.strip() for v in str(val).split(",") if v.strip()]


def compute_label(avoid_list, sensitivity, skin_type, concerns, good_for):
    """
    Improved labeling logic that properly accounts for skin type.
    
    Key difference from original: an ingredient is only "poor fit" if
    the user's SPECIFIC skin type is in the avoid list. High sensitivity
    alone doesn't make it poor fit for tolerant skin types.
    """
    skin_avoid_tag = SKIN_TYPE_TO_AVOID.get(skin_type)
    
    # Check if this skin type is specifically flagged
    skin_flagged = skin_avoid_tag and skin_avoid_tag in avoid_list
    
    # Check concern match
    matched_concerns = []
    for c in concerns:
        goodfor_tags = CONCERN_TO_GOODFOR.get(c, [c])
        if any(g in good_for for g in goodfor_tags):
            matched_concerns.append(c)
    
    # ── Decision logic ──
    
    # POOR FIT: skin type is explicitly in avoid list AND sensitivity >= 2
    if skin_flagged and sensitivity >= 2:
        return ("poor fit", 2, 5,
                f"Directly flagged in avoid list for {skin_type} skin and high irritancy ({sensitivity})")
    
    # POOR FIT: skin type is flagged AND sensitivity >= 1 AND Sensitive skin
    if skin_flagged and sensitivity >= 1 and skin_type == "Sensitive":
        return ("poor fit", 2, 4,
                f"Flagged for {skin_type} skin with moderate irritancy ({sensitivity})")
    
    # POSSIBLE IRRITATION: skin type flagged but lower severity
    if skin_flagged:
        return ("possible irritation", 1, 3,
                f"Flagged in avoid list for {skin_type} skin")
    
    # POSSIBLE IRRITATION: high sensitivity even if skin type not flagged
    # (e.g., Retinol for Normal skin — still somewhat irritating)
    if sensitivity >= 3 and skin_type in ("Combination",):
        return ("possible irritation", 1, 3,
                f"High irritancy ({sensitivity}) — caution for {skin_type} skin")
    
    # GOOD FIT: skin type not flagged, addresses concerns
    if len(matched_concerns) >= 2:
        return ("good fit", 0, 4,
                f"Addresses {len(matched_concerns)} concern(s); not flagged for {skin_type} skin")
    
    if len(matched_concerns) == 1:
        return ("good fit", 0, 3,
                f"Addresses 1 concern; not flagged for {skin_type} skin")
    
    # GOOD FIT: not flagged, no concerns matched but gentle
    if sensitivity <= 1:
        return ("good fit", 0, 2,
                f"Gentle ingredient; not flagged for {skin_type} skin")
    
    # POSSIBLE IRRITATION: moderate sensitivity, no concern match, not flagged
    return ("possible irritation", 1, 2,
            f"Moderate irritancy ({sensitivity}) with no direct concern match")


def main():
    if not os.path.exists(LABELED_CSV):
        print(f"ERROR: {LABELED_CSV} not found!")
        return
    if not os.path.exists(INGREDIENTS_CSV):
        print(f"ERROR: {INGREDIENTS_CSV} not found!")
        return

    # Load ingredients for their avoid lists and sensitivity scores
    ingredients_df = pd.read_csv(INGREDIENTS_CSV)
    ing_lookup = {}
    for _, row in ingredients_df.iterrows():
        ing_lookup[row["name"]] = {
            "avoid": parse_list(row["avoid"]),
            "good_for": parse_list(row["good_for"]),
            "sensitivity_score": int(row.get("sensitivity_score", 0)),
        }

    # Load and fix labeled pairs
    pairs_df = pd.read_csv(LABELED_CSV)
    original_labels = pairs_df["label"].copy()

    fixed_count = 0
    for idx, row in pairs_df.iterrows():
        name = row["ingredient_name"]
        skin_type = row["skin_type"]
        concerns_str = row["concerns"]

        if name not in ing_lookup:
            continue

        ing = ing_lookup[name]
        try:
            concerns = ast.literal_eval(concerns_str)
        except:
            concerns = [c.strip() for c in concerns_str.split(",")]

        new_label, new_label_int, new_confidence, new_reason = compute_label(
            avoid_list=ing["avoid"],
            sensitivity=ing["sensitivity_score"],
            skin_type=skin_type,
            concerns=concerns,
            good_for=ing["good_for"],
        )

        if new_label != row["label"]:
            pairs_df.at[idx, "label"] = new_label
            pairs_df.at[idx, "label_int"] = new_label_int
            pairs_df.at[idx, "confidence"] = new_confidence
            pairs_df.at[idx, "reason"] = new_reason
            fixed_count += 1

    print(f"Fixed {fixed_count} labels out of {len(pairs_df)} total rows")

    # Show some examples of what changed
    changed = pairs_df["label"] != original_labels
    if changed.any():
        sample = pairs_df[changed].head(20)
        print("\nSample of changed labels:")
        print(f"{'Ingredient':<35} {'Skin Type':<15} {'Old':<22} {'New':<22}")
        print("-" * 94)
        for idx, row in sample.iterrows():
            old = original_labels[idx]
            new = row["label"]
            print(f"{row['ingredient_name']:<35} {row['skin_type']:<15} {old:<22} {new:<22}")

    # Save
    pairs_df.to_csv(LABELED_CSV, index=False)
    print(f"\nSaved updated {LABELED_CSV}")

    # Delete model to force retrain
    if os.path.exists(MODEL_PKL):
        os.remove(MODEL_PKL)
        print(f"Deleted {MODEL_PKL} — model will retrain on next launch")

    print("\nDone! Restart Streamlit to retrain with corrected labels.")


if __name__ == "__main__":
    main()
