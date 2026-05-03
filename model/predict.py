import os
import ast
import pickle
from typing import Optional
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved", "ingredient_classifier.pkl")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "ingredients_final.csv")

_artifacts: Optional[dict] = None
_ingredients_df: Optional[pd.DataFrame] = None


def load_model() -> dict:
    global _artifacts
    if _artifacts is None:
        try:
            with open(MODEL_PATH, "rb") as f:
                _artifacts = pickle.load(f)
        except Exception:
            # Saved model is incompatible with the current sklearn version.
            # Delete it and retrain automatically.
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


def lookup_ingredient(name: str) -> Optional[dict]:
    """Find an ingredient row by name (case-insensitive; tries exact then partial match)."""
    df = load_ingredients()
    query = name.lower().strip()

    # 1. Exact match on name_lower
    match = df[df["name_lower"] == query]
    if not match.empty:
        return match.iloc[0].to_dict()

    # 2. Dataset name contained in query (e.g. query has extra words)
    match = df[df["name_lower"].apply(lambda n: n in query)]
    if not match.empty:
        # pick the longest matching name (most specific)
        return match.loc[match["name_lower"].str.len().idxmax()].to_dict()

    # 3. Query substring of dataset name
    match = df[df["name_lower"].str.contains(query, na=False, regex=False)]
    if not match.empty:
        return match.iloc[0].to_dict()

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
    test_cases = ["Aloe Vera", "Retinol", "Salicylic Acid", "Niacinamide", "Glycolic Acid"]
    results = predict_ingredients_batch(test_cases, skin_type="Sensitive", concerns=["Acne", "Redness"])
    print(f"\n{'Ingredient':<30} {'Found':>5}  {'Label':<22} {'Matched Concerns'}")
    print("-" * 80)
    for r in results:
        mc = ", ".join(r["matched_concerns"]) if r["matched_concerns"] else "—"
        print(f"{r['name']:<30} {str(r['found']):>5}  {r['label_name']:<22} {mc}")
