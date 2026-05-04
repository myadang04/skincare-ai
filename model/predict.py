import os
import ast
import re
import pickle
from typing import Optional
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved", "ingredient_classifier.pkl")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "ingredients_final.csv")

_artifacts: Optional[dict] = None
_ingredients_df: Optional[pd.DataFrame] = None

# ── Alias map: maps common label names → dataset name_lower ──────────────
_ALIAS_MAP = {
    # Ceramide variants → generic Ceramides entry
    "ceramide np": "ceramides",
    "ceramide ap": "ceramides",
    "ceramide ng": "ceramides",
    "ceramide ns": "ceramides",
    "ceramide eop": "ceramides",
    "ceramide eos": "ceramides",
    "ceramide 3": "ceramides",
    "ceramide 6-ii": "ceramides",
    # Beta-Glucan hyphen variant
    "beta-glucan": "beta glucan",
    # Vitamin E variants
    "tocopherol": "vitamin e",
    "alpha tocopherol": "vitamin e",
    "tocopherol acetate": "vitamin e",
    "tocopheryl acetate": "vitamin e",
    # Vitamin B5 variants
    "panthenol": "vitamin b5",
    "d-panthenol": "vitamin b5",
    "pantothenic acid": "vitamin b5",
    # Centella Asiatica variants
    "asiaticoside": "centella asiatica",
    "madecassoside": "centella asiatica",
    "madecassic acid": "centella asiatica",
    "asiatic acid": "centella asiatica",
    "centella asiatica leaf extract": "centella asiatica",
    # Apple stem cells variant
    "malus domestica fruit cell culture extract": "apple stem cells",
    # Hydrogenated lecithin → lecithin
    "hydrogenated lecithin": "lecithin",
    # 3-O-Ethyl Ascorbic Acid → Ethyl Ascorbic Acid
    "3-o-ethyl ascorbic acid": "ethyl ascorbic acid",
    # Sugar cane extract variants
    "saccharum officinarum extract": "sugar cane extract",
    "saccharum officinarum (sugarcane) extract": "sugar cane extract",
    "saccharum officinarum (sugar-cane) extract": "sugar cane extract",
    "sugarcane extract": "sugar cane extract",
    # 1,2-Hexanediol variants
    "1,2-hexanediol": "1 2-hexanediol",
    "2-hexanediol": "1 2-hexanediol",
    # Polyacrylate variant
    "polyacrylate crosspolymer-6": "polyacrylate crosspolymer-6",
    # Common OCR misreads
    "disodium eota": "disodium edta",
    "disodium edt a": "disodium edta",
}

# ── Common OCR misreads to fix before lookup ──────────────────────────────
_OCR_FIXES = {
    "eota": "edta",
    "lron": "iron",
    "0xide": "oxide",
    "giycerin": "glycerin",
    "nlacinamide": "niacinamide",
}

# ── Skin type → avoid tag mapping (used for hard override) ────────────────
_SKIN_TYPE_TO_AVOID_TAG = {
    "Sensitive": "Sensitive Skin",
    "Oily": "Oily Skin",
    "Dry": "Dry Skin",
    "Combination": "Combination Skin",
    "Normal": None,
}


def load_model() -> dict:
    global _artifacts
    if _artifacts is None:
        try:
            with open(MODEL_PATH, "rb") as f:
                _artifacts = pickle.load(f)
        except Exception:
            import os as _os
            if _os.path.exists(MODEL_PATH):
                _os.remove(MODEL_PATH)
            from model.train import train_and_evaluate
            _artifacts, _ = train_and_evaluate()
    return _artifacts


def load_ingredients() -> pd.DataFrame:
    global _ingredients_df
    if _ingredients_df is None:
        df = pd.read_csv(DATA_PATH)
        df["good_for"] = df["good_for"].apply(_parse_list)
        df["avoid"]    = df["avoid"].apply(_parse_list)
        _ingredients_df = df
    return _ingredients_df


def _parse_list(val):
    if pd.isna(val) or val == "":
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return [v.strip() for v in str(val).split(",") if v.strip()]


def _normalize_query(name: str) -> str:
    """Normalize an ingredient name for matching."""
    q = name.lower().strip()
    # Fix common OCR misreads
    for wrong, right in _OCR_FIXES.items():
        q = q.replace(wrong, right)
    # Remove parenthetical common names like "(Aqua)", "(Corn)"
    q = re.sub(r"\s*\([^)]*\)\s*", " ", q).strip()
    return q


def lookup_ingredient(name: str) -> Optional[dict]:
    """
    Find an ingredient row by name (case-insensitive).

    Matching priority:
      1. Exact match on name_lower
      2. Alias map (handles common variants like Ceramide NP → Ceramides)
      3. Normalized match (strip parentheticals, try again)
      4. Dataset name contained in query (e.g. query has extra words)
      5. Query substring of dataset name
      6. Fuzzy match (>= 85% similarity)
    """
    df = load_ingredients()
    query = name.lower().strip()

    # 1. Exact match on name_lower
    match = df[df["name_lower"] == query]
    if not match.empty:
        return match.iloc[0].to_dict()

    # 2. Alias map
    if query in _ALIAS_MAP:
        alias_target = _ALIAS_MAP[query]
        match = df[df["name_lower"] == alias_target]
        if not match.empty:
            return match.iloc[0].to_dict()

    # 3. Normalize and retry exact match
    normalized = _normalize_query(name)
    if normalized != query:
        match = df[df["name_lower"] == normalized]
        if not match.empty:
            return match.iloc[0].to_dict()
        # Also check alias for normalized form
        if normalized in _ALIAS_MAP:
            alias_target = _ALIAS_MAP[normalized]
            match = df[df["name_lower"] == alias_target]
            if not match.empty:
                return match.iloc[0].to_dict()

    # 4. Dataset name contained in query (e.g. query has extra words)
    match = df[df["name_lower"].apply(lambda n: n in query)]
    if not match.empty:
        return match.loc[match["name_lower"].str.len().idxmax()].to_dict()

    # 5. Query substring of dataset name
    match = df[df["name_lower"].str.contains(query, na=False, regex=False)]
    if not match.empty:
        return match.iloc[0].to_dict()

    # 6. Fuzzy match as last resort
    try:
        from thefuzz import fuzz
        best_score = 0
        best_idx = None
        for idx, row in df.iterrows():
            score = fuzz.ratio(query, row["name_lower"])
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_score >= 85:
            return df.loc[best_idx].to_dict()
    except ImportError:
        pass

    return None


def _build_feature_row(ingredient: dict, skin_type: str, concerns: list) -> pd.DataFrame:
    arts = load_model()
    le = arts["category_encoder"]
    feature_cols = arts["feature_cols"]
    concern_to_goodfor = arts["concern_to_goodfor"]
    concern_to_avoid   = arts["concern_to_avoid"]
    skin_type_to_avoid = arts["skin_type_to_avoid"]

    good_for = ingredient.get("good_for", [])
    avoid    = ingredient.get("avoid", [])

    target_goodfor = set()
    for c in concerns:
        target_goodfor.update(concern_to_goodfor.get(c, [c]))
    target_goodfor.add("Anyone")

    concern_match_count = sum(1 for g in good_for if g in target_goodfor)
    concern_match_ratio = concern_match_count / max(len(concerns), 1)

    avoid_tags = set()
    skin_avoid = skin_type_to_avoid.get(skin_type)
    if skin_avoid:
        avoid_tags.add(skin_avoid)
    for c in concerns:
        avoid_tags.update(concern_to_avoid.get(c, []))
    avoid_triggered = int(any(a in avoid_tags for a in avoid))

    cat = ingredient.get("category") or "Other"
    known = list(le.classes_)
    cat_enc = le.transform([cat])[0] if cat in known else (le.transform(["Other"])[0] if "Other" in known else 0)

    row = {
        "sensitivity_score":  ingredient.get("sensitivity_score", 0),
        "breadth_score":      ingredient.get("breadth_score", 0),
        "pregnancy_safe":     int(bool(ingredient.get("pregnancy_safe", True))),
        "concern_match_count": concern_match_count,
        "concern_match_ratio": concern_match_ratio,
        "avoid_triggered":    avoid_triggered,
        "is_sensitive":       int(skin_type == "Sensitive"),
        "is_oily":            int(skin_type == "Oily"),
        "is_dry":             int(skin_type == "Dry"),
        "num_concerns":       len(concerns),
        "good_for_count":     len(good_for),
        "category_enc":       cat_enc,
    }
    return pd.DataFrame([row])[feature_cols]


def predict_ingredient(name: str, skin_type: str, concerns: list) -> dict:
    """
    Predict fit classification for a single ingredient + user profile.

    Uses the ML model's prediction as a starting point, then applies
    hard overrides based on the ingredient's avoid list to ensure
    skin-type-specific accuracy.

    Returns a dict with keys:
      name, found, label, label_name, probability,
      good_for, avoid, matched_concerns,
      sensitivity_score, breadth_score, category
    """
    arts = load_model()
    model       = arts["model"]
    scaler      = arts["scaler"]
    label_map   = arts["label_map"]
    needs_scale = arts["needs_scale"]

    ingredient = lookup_ingredient(name)
    if ingredient is None:
        return {
            "name": name,
            "found": False,
            "label": None,
            "label_name": "unknown",
            "probability": None,
            "good_for": [],
            "avoid": [],
            "matched_concerns": [],
            "sensitivity_score": None,
            "breadth_score": None,
            "category": None,
        }

    X = _build_feature_row(ingredient, skin_type, concerns)
    X_input = scaler.transform(X) if needs_scale else X.values
    label = int(model.predict(X_input)[0])

    # ── Hard override: enforce avoid-list rules on top of model ──────
    # The ML model may not perfectly learn avoid-list interactions,
    # so we enforce them directly. This is a standard pattern in
    # production ML systems (business rules on top of model output).
    avoid_list = ingredient.get("avoid", [])
    sens = ingredient.get("sensitivity_score", 0)
    skin_avoid_tag = _SKIN_TYPE_TO_AVOID_TAG.get(skin_type)

    # Build the FULL set of avoid tags for this user (skin type + concerns)
    concern_to_avoid = arts.get("concern_to_avoid", {})
    all_user_avoid_tags = set()
    if skin_avoid_tag:
        all_user_avoid_tags.add(skin_avoid_tag)
    for c in concerns:
        all_user_avoid_tags.update(concern_to_avoid.get(c, []))

    # Check if ANY of the user's avoid tags match the ingredient's avoid list
    avoid_triggered = any(tag in avoid_list for tag in all_user_avoid_tags)

    if avoid_triggered:
        # User's skin type or concerns ARE flagged in avoid list → override
        if sens >= 2:
            label = 2  # poor fit
        elif sens >= 1:
            label = 1  # possible irritation
        else:
            label = 1  # possible irritation (flagged but gentle)
    elif skin_avoid_tag is None:
        # Normal skin (no avoid tag) — model prediction stands unless
        # the model incorrectly said poor fit for an unflagged ingredient
        if label == 2:
            # Check if ingredient addresses any concerns
            c2g = arts["concern_to_goodfor"]
            good_for_set = set(ingredient.get("good_for", []))
            matched = [c for c in concerns if any(g in good_for_set for g in c2g.get(c, [c]))]
            if len(matched) >= 1:
                label = 0  # good fit — addresses concerns, not flagged
            else:
                label = 1  # possible irritation at most
    else:
        # Skin type has an avoid tag but ingredient is NOT flagged for it
        if label == 2 and not avoid_triggered:
            # Model said poor fit but ingredient doesn't flag this skin type
            c2g = arts["concern_to_goodfor"]
            good_for_set = set(ingredient.get("good_for", []))
            matched = [c for c in concerns if any(g in good_for_set for g in c2g.get(c, [c]))]
            if len(matched) >= 1:
                label = 0  # good fit
            elif sens <= 1:
                label = 0  # gentle enough
            else:
                label = 1  # possible irritation

    probability = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[0]
        classes = list(model.classes_)
        probability = {label_map[c]: round(float(p), 4) for c, p in zip(classes, proba)}

    # Which of the user's concerns does this ingredient actually address?
    c2g = arts["concern_to_goodfor"]
    good_for_set = set(ingredient.get("good_for", []))
    matched = [c for c in concerns if any(g in good_for_set for g in c2g.get(c, [c]))]

    return {
        "name": ingredient.get("name", name),
        "found": True,
        "label": label,
        "label_name": label_map[label],
        "probability": probability,
        "good_for": ingredient.get("good_for", []),
        "avoid": ingredient.get("avoid", []),
        "matched_concerns": matched,
        "sensitivity_score": ingredient.get("sensitivity_score"),
        "breadth_score": ingredient.get("breadth_score"),
        "category": ingredient.get("category"),
    }


def predict_ingredients_batch(names: list, skin_type: str, concerns: list) -> list:
    """Predict fit for every ingredient in a list. Returns results in the same order."""
    return [predict_ingredient(n, skin_type, concerns) for n in names]


if __name__ == "__main__":
    test_cases = ["Aloe Vera", "Retinol", "Salicylic Acid", "Niacinamide", "Glycolic Acid",
                  "Beta-Glucan", "Ceramide NP", "Tocopherol", "Madecassoside"]
    print("\n=== SENSITIVE SKIN ===")
    results = predict_ingredients_batch(test_cases, skin_type="Sensitive", concerns=["Acne", "Redness"])
    print(f"{'Ingredient':<30} {'Found':>5}  {'Label':<22} {'Matched Concerns'}")
    print("-" * 80)
    for r in results:
        mc = ", ".join(r["matched_concerns"]) if r["matched_concerns"] else "—"
        print(f"{r['name']:<30} {str(r['found']):>5}  {r['label_name']:<22} {mc}")

    print("\n=== NORMAL SKIN ===")
    results = predict_ingredients_batch(test_cases, skin_type="Normal", concerns=["Acne", "Redness"])
    print(f"{'Ingredient':<30} {'Found':>5}  {'Label':<22} {'Matched Concerns'}")
    print("-" * 80)
    for r in results:
        mc = ", ".join(r["matched_concerns"]) if r["matched_concerns"] else "—"
        print(f"{r['name']:<30} {str(r['found']):>5}  {r['label_name']:<22} {mc}")

    print("\n=== DRY SKIN ===")
    results = predict_ingredients_batch(test_cases, skin_type="Dry", concerns=["Dryness", "Redness"])
    print(f"{'Ingredient':<30} {'Found':>5}  {'Label':<22} {'Matched Concerns'}")
    print("-" * 80)
    for r in results:
        mc = ", ".join(r["matched_concerns"]) if r["matched_concerns"] else "—"
        print(f"{r['name']:<30} {str(r['found']):>5}  {r['label_name']:<22} {mc}")