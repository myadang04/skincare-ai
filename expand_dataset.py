"""
expand_dataset.py
─────────────────
Run this script ONCE to:
  1. Append new common ingredients to data/processed/ingredients_final.csv
  2. Generate labeled training pairs for the new ingredients across all
     skin-type + concern combos (matching the existing labeled_pairs format)
  3. Append those new rows to data/processed/labeled_pairs.csv
  4. Delete the saved model so it retrains on next app launch

Usage:
    python expand_dataset.py

After running, restart your Streamlit app — the model will retrain automatically.
"""

import os
import ast
import csv
import pandas as pd

# ── Paths (adjust if your project structure differs) ──────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INGREDIENTS_CSV = os.path.join(BASE_DIR, "data", "processed", "ingredients_final.csv")
LABELED_CSV     = os.path.join(BASE_DIR, "data", "processed", "labeled_pairs.csv")
NEW_INGREDIENTS = os.path.join(BASE_DIR, "new_ingredients.csv")
MODEL_PKL       = os.path.join(BASE_DIR, "model", "saved", "ingredient_classifier.pkl")

# ── The same skin-type + concern combos used in the original training data ─
SKIN_CONCERN_COMBOS = [
    ("Sensitive", ["Acne", "Redness"]),
    ("Sensitive", ["Acne", "Enlarged Pores", "Redness"]),
    ("Sensitive", ["Dark Circles", "Redness"]),
    ("Sensitive", ["Hyperpigmentation", "Redness"]),
    ("Sensitive", ["Dryness", "Redness"]),
    ("Oily", ["Acne", "Blackheads"]),
    ("Oily", ["Acne", "Enlarged Pores"]),
    ("Oily", ["Enlarged Pores", "Hyperpigmentation"]),
    ("Oily", ["Dullness", "Hyperpigmentation"]),
    ("Dry", ["Dryness", "Hyperpigmentation"]),
    ("Dry", ["Dryness", "Fine Lines"]),
    ("Dry", ["Dryness", "Wrinkles"]),
    ("Normal", ["Hyperpigmentation", "Texture"]),
    ("Normal", ["Fine Lines", "Wrinkles"]),
    ("Normal", ["Wrinkles", "Hyperpigmentation"]),
    ("Combination", ["Acne", "Dryness"]),
]

# Maps skin type → the "avoid" tag that flags an ingredient as problematic
SKIN_TYPE_TO_AVOID = {
    "Sensitive": "Sensitive Skin",
    "Oily": "Oily Skin",
    "Dry": "Dry Skin",
    "Combination": "Combination Skin",
    "Normal": None,
}

# Maps user concerns → the good_for tags that would satisfy them
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


def generate_label(ingredient_row, skin_type, concerns):
    """
    Rule-based labeling that matches the logic used to create the original
    labeled_pairs.csv. Returns (label, label_int, confidence, reason).
    """
    good_for = parse_list(ingredient_row.get("good_for", "[]"))
    avoid = parse_list(ingredient_row.get("avoid", "[]"))
    sensitivity = int(ingredient_row.get("sensitivity_score", 0))
    category = ingredient_row.get("category", "Other")

    # Check if avoid list flags this skin type
    skin_avoid_tag = SKIN_TYPE_TO_AVOID.get(skin_type)
    avoid_triggered = skin_avoid_tag and skin_avoid_tag in avoid

    # Check how many concerns are addressed
    matched_concerns = []
    for c in concerns:
        goodfor_tags = CONCERN_TO_GOODFOR.get(c, [c])
        if any(g in good_for for g in goodfor_tags):
            matched_concerns.append(c)

    # ── Labeling rules ──
    # 1. Directly flagged in avoid list + high sensitivity → poor fit
    if avoid_triggered and sensitivity >= 2:
        return ("poor fit", 2, 5,
                f"Directly flagged in avoid list for {skin_type} skin and high irritancy ({sensitivity})")

    # 2. High sensitivity without direct avoid flag → poor fit for Sensitive
    if skin_type == "Sensitive" and sensitivity >= 3 and avoid_triggered:
        return ("poor fit", 2, 5,
                f"Directly flagged in avoid list for Sensitive skin and high irritancy ({sensitivity})")

    # 3. Avoid triggered but lower sensitivity → possible irritation
    if avoid_triggered:
        return ("possible irritation", 1, 3,
                f"Flagged in avoid list for {skin_type} skin")

    # 4. Moderate sensitivity for sensitive skin → possible irritation
    if skin_type == "Sensitive" and sensitivity >= 2:
        return ("possible irritation", 1, 3,
                f"Moderate irritancy ({sensitivity}) — possible irritation for Sensitive skin")

    # 5. Some sensitivity → possible irritation
    if sensitivity >= 2:
        return ("possible irritation", 1, 3,
                f"Moderate irritancy ({sensitivity})")

    # 6. Addresses concerns directly → good fit with high confidence
    if len(matched_concerns) >= 2:
        return ("good fit", 0, 4,
                f"Directly addresses {len(matched_concerns)} concern(s) and description confirms soothing profile")

    if len(matched_concerns) == 1:
        return ("good fit", 0, 3,
                f"Addresses 1 concern(s); low irritancy")

    # 7. Gentle with no concern match → good fit (neutral)
    if sensitivity == 0:
        return ("good fit", 0, 2,
                "Gentle ingredient with no concern match; neutral fit")

    # 8. Mild sensitivity, no concern match → possible irritation
    return ("possible irritation", 1, 2,
            "Mild irritancy with no direct concern match")


def main():
    # ── 1. Load existing ingredients ──
    if not os.path.exists(INGREDIENTS_CSV):
        print(f"ERROR: {INGREDIENTS_CSV} not found!")
        return

    existing = pd.read_csv(INGREDIENTS_CSV)
    existing_names = set(existing["name_lower"].str.strip())
    print(f"Existing ingredients: {len(existing_names)}")

    # ── 2. Load new ingredients ──
    if not os.path.exists(NEW_INGREDIENTS):
        print(f"ERROR: {NEW_INGREDIENTS} not found!")
        return

    new_df = pd.read_csv(NEW_INGREDIENTS)
    # Only add ingredients not already in the dataset
    new_df = new_df[~new_df["name_lower"].str.strip().isin(existing_names)]
    print(f"New ingredients to add: {len(new_df)}")

    if new_df.empty:
        print("No new ingredients to add — all already exist.")
    else:
        # Append to ingredients CSV
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(INGREDIENTS_CSV, index=False)
        print(f"Updated {INGREDIENTS_CSV} — now {len(combined)} ingredients")

    # ── 3. Generate labeled pairs for new ingredients ──
    new_rows = []
    for _, ing in new_df.iterrows():
        for skin_type, concerns in SKIN_CONCERN_COMBOS:
            label, label_int, confidence, reason = generate_label(ing, skin_type, concerns)
            new_rows.append({
                "ingredient_name": ing["name"],
                "skin_type": skin_type,
                "concerns": str(concerns),
                "label": label,
                "label_int": label_int,
                "confidence": confidence,
                "reason": reason,
                "sensitivity_score": ing["sensitivity_score"],
                "breadth_score": ing["breadth_score"],
                "category": ing["category"],
            })

    print(f"Generated {len(new_rows)} new labeled pairs")

    if new_rows and os.path.exists(LABELED_CSV):
        new_pairs_df = pd.DataFrame(new_rows)
        existing_pairs = pd.read_csv(LABELED_CSV)
        combined_pairs = pd.concat([existing_pairs, new_pairs_df], ignore_index=True)
        combined_pairs.to_csv(LABELED_CSV, index=False)
        print(f"Updated {LABELED_CSV} — now {len(combined_pairs)} rows")
    elif new_rows:
        new_pairs_df = pd.DataFrame(new_rows)
        new_pairs_df.to_csv(LABELED_CSV, index=False)
        print(f"Created {LABELED_CSV} with {len(new_pairs_df)} rows")

    # ── 4. Delete saved model so it retrains ──
    if os.path.exists(MODEL_PKL):
        os.remove(MODEL_PKL)
        print(f"Deleted {MODEL_PKL} — model will retrain on next app launch")
    else:
        print("No saved model found — will train fresh on next launch")

    print("\nDone! Restart your Streamlit app to retrain the model.")


if __name__ == "__main__":
    main()
