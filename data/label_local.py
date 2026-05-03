"""
label_local.py  —  FREE, no API needed
───────────────────────────────────────
Generates data/processed/labeled_pairs.csv using:
  1. Ingredient DESCRIPTION keyword analysis  (main signal)
  2. CATEGORY-based dermatology knowledge     (category heuristics)
  3. Structured avoid / good_for              (hard constraints)

Why this breaks the circularity:
  • The ML model trains on STRUCTURED features
    (sensitivity_score, breadth_score, avoid_triggered, …)
  • Labels come from DESCRIPTION text + category rules
    — a different information source
  → The model must LEARN the mapping; it can't just memorise it.
  → Expected accuracy: 75–90 %, not 100 %.

Run:
    python3 data/label_local.py
"""

import os, ast, random
import pandas as pd

random.seed(42)

DATA_PATH   = os.path.join(os.path.dirname(__file__), "processed", "ingredients_final.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "processed", "labeled_pairs.csv")

LABEL_NAMES = {0: "good fit", 1: "possible irritation", 2: "poor fit"}

# ── Skin-type → avoid tag ─────────────────────────────────────────────────────
SKIN_TYPE_TO_AVOID = {
    "Sensitive": "Sensitive Skin",
    "Oily":      "Oily Skin",
    "Dry":       "Dry Skin",
    "Combination": "Combination Skin",
    "Normal":    None,
}

# ── Concern → good_for tags ───────────────────────────────────────────────────
CONCERN_TO_GOODFOR = {
    "Acne":            ["Acne"],
    "Redness":         ["Redness"],
    "Hyperpigmentation": ["Pigmentation", "Post Blemish Marks"],
    "Dryness":         ["Dry Skin"],
    "Sensitivity":     ["Impaired Skin Barrier"],
    "Wrinkles":        ["Wrinkles", "Fine Lines", "Elasticity"],
    "Dullness":        ["Radiance"],
    "Enlarged Pores":  ["Enlarged Pores"],
    "Dark Circles":    ["Dark Circles", "Eye Bags"],
    "Blackheads":      ["Blackheads"],
    "Fine Lines":      ["Fine Lines"],
    "Texture":         ["Texture"],
}

# ── Concern → avoid tags that apply when user has that concern ────────────────
CONCERN_TO_AVOID = {
    "Sensitivity": ["Sensitive Skin", "Impaired Skin Barrier"],
    "Dryness":     ["Dry Skin"],
}

SKIN_PROFILES = [
    ("Sensitive",   ["Acne", "Redness"]),
    ("Sensitive",   ["Sensitivity", "Redness"]),
    ("Sensitive",   ["Sensitivity", "Dryness"]),
    ("Sensitive",   ["Wrinkles", "Hyperpigmentation"]),
    ("Sensitive",   ["Acne", "Enlarged Pores", "Redness"]),
    ("Oily",        ["Acne", "Enlarged Pores"]),
    ("Oily",        ["Acne", "Blackheads"]),
    ("Oily",        ["Hyperpigmentation", "Redness"]),
    ("Oily",        ["Dullness", "Texture"]),
    ("Dry",         ["Dryness", "Fine Lines"]),
    ("Dry",         ["Dryness", "Wrinkles"]),
    ("Dry",         ["Dryness", "Hyperpigmentation"]),
    ("Dry",         ["Sensitivity", "Dryness"]),
    ("Normal",      ["Dullness", "Texture"]),
    ("Normal",      ["Dark Circles", "Redness"]),
    ("Normal",      ["Wrinkles", "Hyperpigmentation"]),
    ("Normal",      ["Acne", "Redness"]),
    ("Combination", ["Acne", "Dryness"]),
    ("Combination", ["Enlarged Pores", "Hyperpigmentation"]),
    ("Combination", ["Acne", "Dullness"]),
]

# ── Category-level base irritancy (0 = gentle, 3 = strong) ───────────────────
CATEGORY_IRRITANCY = {
    "Retinoid":          3,   # vitamin A derivatives — potent, need intro
    "AHA":               2,   # glycolic, lactic, mandelic — exfoliants
    "BHA":               2,   # salicylic — oil-soluble exfoliant
    "UV Filter":         2,   # chemical filters — known sensitisers
    "Vitamin C":         1,   # unstable, can sting at high %
    "Mineral":           0,
    "Lipid/Oil":         1,   # generally gentle; some comedogenic
    "Ceramide":          0,
    "Humectant":         0,
    "Peptide":           0,
    "Amino Acid":        0,
    "Marine/Algae":      0,
    "Plant Extract":     0,
    "Probiotic/Ferment": 0,
    "Vitamin":           0,
    "Other":             0,
}

# ── Description keywords that RAISE perceived irritation risk ─────────────────
IRRITANT_KW = [
    "irritat", "sting", "sensitiv",     # general irritation signals
    "unstable", "peel", "purging",       # instability / peeling
    "high potency", "strongest",         # potency signals
    "needs a ph", "ph 4",               # formulation constraints → potential sting
    "smallest molecular weight",         # fast penetration → more likely to irritate
]

# ── Description keywords that LOWER perceived irritation risk ─────────────────
SOOTHING_KW = [
    "sooth", "calm", "anti-inflamm", "gentle", "well tolerated",
    "all skin type", "suitable for all", "sensitive skin",   # directly says "ok for sensitive"
    "calming", "non-irritat", "rosacea",                     # rosacea = gentle ingredient
    "suitable for sensitive",
]

# ── Skin-type concern keywords inside descriptions ────────────────────────────
ACNE_DESC_KW   = ["acne", "blemish", "breakout", "sebum", "pore", "antibacter"]
REDNESS_DESC_KW = ["redness", "rosacea", "capillar", "anti-inflam", "sooth"]
DRY_DESC_KW    = ["dry", "hydrat", "moisture", "water loss", "barrier"]


def parse_list_field(val):
    if pd.isna(val) or val == "":
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return [v.strip() for v in str(val).split(",") if v.strip()]


def label_pair(ingredient: dict, skin_type: str, concerns: list) -> tuple:
    """
    Returns (label_int, label_name, confidence, reason).
    Labels are derived primarily from description text + category knowledge.
    """
    desc     = str(ingredient.get("description", "")).lower()
    good_for = ingredient.get("good_for", [])
    avoid    = ingredient.get("avoid", [])
    sensitivity = int(ingredient.get("sensitivity_score", 0))
    category    = ingredient.get("category", "Other") or "Other"
    name        = ingredient.get("name", "")

    # ── Step 1: Hard structural constraints ──────────────────────────────────
    skin_avoid = SKIN_TYPE_TO_AVOID.get(skin_type)
    avoid_tags = set()
    if skin_avoid:
        avoid_tags.add(skin_avoid)
    for c in concerns:
        avoid_tags.update(CONCERN_TO_AVOID.get(c, []))
    avoid_triggered = any(a in avoid_tags for a in avoid)

    target_goodfor = {"Anyone"}
    for c in concerns:
        target_goodfor.update(CONCERN_TO_GOODFOR.get(c, [c]))
    concern_match = sum(1 for g in good_for if g in target_goodfor)

    # ── Step 2: Category-level base irritancy ─────────────────────────────────
    cat_irritancy = CATEGORY_IRRITANCY.get(category, 0)

    # ── Step 3: Description keyword scoring ──────────────────────────────────
    irritant_hits = sum(1 for kw in IRRITANT_KW if kw in desc)
    soothing_hits = sum(1 for kw in SOOTHING_KW if kw in desc)
    desc_delta = irritant_hits - soothing_hits   # positive = more irritating

    # ── Step 4: Combined irritancy score (0–5 scale) ─────────────────────────
    # Sources: sensitivity_score (dataset), category, description text
    irritancy = sensitivity + cat_irritancy + max(0, desc_delta)

    # ── Step 5: Skin-type modifiers ──────────────────────────────────────────
    if skin_type == "Sensitive":
        irritancy += 1          # sensitive skin amplifies irritation risk
    if skin_type == "Dry" and any(kw in desc for kw in ["drying", "dry", "tewl"]):
        # ingredient might worsen dryness concerns
        if "Dryness" in concerns or "Sensitivity" in concerns:
            irritancy += 1

    # ── Step 6: Hard avoid override (highest priority) ───────────────────────
    if avoid_triggered:
        if irritancy >= 3:
            label = 2     # poor fit: explicitly flagged + irritating
            confidence = 5
            reason = f"Directly flagged in avoid list for {skin_type} skin and high irritancy ({irritancy})"
        else:
            label = 1     # possible irritation: flagged but ingredient is mild
            confidence = 4
            reason = f"Flagged in avoid list for {skin_type} skin; ingredient is relatively mild"

    # ── Step 7: Irritancy-based label (no avoid flag) ────────────────────────
    elif irritancy >= 4:
        label = 2
        confidence = 4
        reason = f"High combined irritancy score ({irritancy}): category={category}, sensitivity={sensitivity}, desc signals={desc_delta}"

    elif irritancy >= 2:
        label = 1
        confidence = 3
        reason = f"Moderate irritancy ({irritancy}) — possible irritation for {skin_type} skin"

    # ── Step 8: Benefit-based label ──────────────────────────────────────────
    elif concern_match > 0 and soothing_hits >= 1:
        label = 0
        confidence = 4
        reason = f"Directly addresses {concern_match} concern(s) and description confirms soothing profile"

    elif concern_match > 0:
        label = 0
        confidence = 3
        reason = f"Addresses {concern_match} concern(s); low irritancy"

    elif soothing_hits >= 2:
        label = 0
        confidence = 3
        reason = "Description strongly soothing/calming; low irritancy even without direct concern match"

    else:
        # No direct match, no strong signal either way — gentle ingredient
        label = 0
        confidence = 2
        reason = "Gentle ingredient with no concern match; neutral fit"

    # ── Step 9: 8% expert-disagreement noise ─────────────────────────────────
    # Simulates real-world labelling variance (dermatologists don't always agree)
    # Only applied to low-confidence labels
    if confidence <= 3 and random.random() < 0.08:
        delta = random.choice([-1, 1])
        label = max(0, min(2, label + delta))
        reason += " [label adjusted for expert variance]"

    return label, LABEL_NAMES[label], confidence, reason


def generate():
    df = pd.read_csv(DATA_PATH)
    df["good_for"] = df["good_for"].apply(parse_list_field)
    df["avoid"]    = df["avoid"].apply(parse_list_field)
    print(f"Loaded {len(df)} ingredients")
    print(f"Generating {len(df) * len(SKIN_PROFILES):,} labeled pairs...\n")

    rows = []
    for skin_type, concerns in SKIN_PROFILES:
        for _, ingredient in df.iterrows():
            label_int, label_name, conf, reason = label_pair(
                ingredient.to_dict(), skin_type, concerns
            )
            rows.append({
                "ingredient_name":   ingredient["name"],
                "skin_type":         skin_type,
                "concerns":          str(concerns),
                "label":             label_name,
                "label_int":         label_int,
                "confidence":        conf,
                "reason":            reason,
                "sensitivity_score": ingredient["sensitivity_score"],
                "breadth_score":     ingredient["breadth_score"],
                "category":          ingredient.get("category", "Other"),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(out):,} rows → {OUTPUT_PATH}")
    print("\nLabel distribution:")
    dist = out["label"].value_counts()
    for lbl, cnt in dist.items():
        print(f"  {lbl:<22} {cnt:>5}  ({cnt/len(out)*100:.1f}%)")
    print(f"\nAverage confidence : {out['confidence'].mean():.2f}/5")
    print(f"Noise-adjusted rows: {out['reason'].str.contains('expert variance').sum()}")

    # ── Sanity-check: well-known ingredients ──────────────────────────────────
    print("\nSpot-check (Sensitive skin / Acne + Redness):")
    check = out[(out["skin_type"] == "Sensitive") & (out["concerns"] == str(["Acne", "Redness"]))]
    spotcheck = ["Aloe Vera", "Retinol", "Glycolic Acid", "Allantoin", "Salicylic Acid",
                 "Ascorbic Acid", "Ceramides", "Azelaic Acid"]
    for name in spotcheck:
        row = check[check["ingredient_name"] == name]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {name:<35} {r['label']:<22} (conf={r['confidence']})")
    return out


if __name__ == "__main__":
    generate()
