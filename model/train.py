import os
import ast
import re
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

DATA_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "ingredients_final.csv")
LABELS_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "labeled_pairs.csv")
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "saved")

LABEL_MAP = {0: "good fit", 1: "possible irritation", 2: "poor fit"}

# Maps UI skin types -> dataset avoid values
SKIN_TYPE_TO_AVOID = {
    "Sensitive": "Sensitive Skin",
    "Oily": "Oily Skin",
    "Dry": "Dry Skin",
    "Combination": "Combination Skin",
    "Normal": None,
}

# Maps UI concern names -> dataset good_for values
CONCERN_TO_GOODFOR = {
    "Acne": ["Acne"],
    "Redness": ["Redness"],
    "Hyperpigmentation": ["Pigmentation", "Post Blemish Marks"],
    "Dryness": ["Dry Skin"],
    "Sensitivity": ["Impaired Skin Barrier"],
    "Wrinkles": ["Wrinkles", "Fine Lines", "Elasticity"],
    "Dullness": ["Radiance"],
    "Enlarged Pores": ["Enlarged Pores"],
    "Dark Circles": ["Dark Circles", "Eye Bags"],
    "Blackheads": ["Blackheads"],
    "Fine Lines": ["Fine Lines"],
    "Texture": ["Texture"],
}

# Maps UI concerns -> avoid tags that apply when user has that concern
CONCERN_TO_AVOID = {
    "Sensitivity": ["Sensitive Skin", "Impaired Skin Barrier"],
    "Dryness": ["Dry Skin"],
}

# ── Description keyword groups ────────────────────────────────────────────
# These extract text-based signals from ingredient descriptions that the
# LLM used when generating labels but the original model couldn't see.
DESC_KEYWORD_GROUPS = {
    "desc_soothing": [
        "soothing", "calming", "calm", "gentle", "anti-irritant",
        "soothes", "soothe", "alleviates", "relieves",
    ],
    "desc_exfoliant": [
        "exfoliat", "peel", "dissolving", "keratin", "dead skin cells",
        "cell turnover", "skin renewal",
    ],
    "desc_anti_inflammatory": [
        "anti-inflammatory", "anti inflammatory", "reduces redness",
        "reduce redness", "inflammation", "inflammatory",
    ],
    "desc_antioxidant": [
        "antioxidant", "free radical", "oxidative", "neutralis",
        "neutraliz",
    ],
    "desc_brightening": [
        "brightening", "brighten", "pigmentation", "melanin",
        "tyrosinase", "dark spots", "hyperpigmentation", "lightening",
        "even skin tone", "evening out",
    ],
}


def extract_description_features(description: str) -> dict:
    """
    Extract keyword-based features from ingredient description text.
    Returns a dict of binary flags indicating presence of each keyword group.
    """
    desc_lower = str(description).lower() if pd.notna(description) else ""
    features = {}
    for feature_name, keywords in DESC_KEYWORD_GROUPS.items():
        features[feature_name] = int(any(kw in desc_lower for kw in keywords))
    return features


# Diverse user profiles used to generate training data
SAMPLE_PROFILES = [
    ("Sensitive", ["Acne", "Redness"]),
    ("Sensitive", ["Sensitivity", "Redness"]),
    ("Sensitive", ["Sensitivity", "Dryness"]),
    ("Sensitive", ["Wrinkles", "Hyperpigmentation"]),
    ("Sensitive", ["Acne", "Enlarged Pores", "Redness"]),
    ("Oily", ["Acne", "Enlarged Pores"]),
    ("Oily", ["Acne", "Blackheads"]),
    ("Oily", ["Hyperpigmentation", "Redness"]),
    ("Oily", ["Dullness", "Texture"]),
    ("Dry", ["Dryness", "Fine Lines"]),
    ("Dry", ["Dryness", "Wrinkles"]),
    ("Dry", ["Dryness", "Hyperpigmentation"]),
    ("Dry", ["Sensitivity", "Dryness"]),
    ("Normal", ["Dullness", "Texture"]),
    ("Normal", ["Dark Circles", "Redness"]),
    ("Normal", ["Wrinkles", "Hyperpigmentation"]),
    ("Normal", ["Acne", "Redness"]),
    ("Combination", ["Acne", "Dryness"]),
    ("Combination", ["Enlarged Pores", "Hyperpigmentation"]),
    ("Combination", ["Acne", "Dullness"]),
]


def parse_list_field(val):
    if pd.isna(val) or val == "":
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return [v.strip() for v in str(val).split(",") if v.strip()]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["good_for"] = df["good_for"].apply(parse_list_field)
    df["avoid"] = df["avoid"].apply(parse_list_field)
    return df


def compute_features(row, skin_type: str, concerns: list) -> dict:
    """Compute per-ingredient features for a given user profile."""
    good_for = row["good_for"]
    avoid = row["avoid"]

    # Expand user concerns into dataset good_for terms
    target_goodfor = set()
    for c in concerns:
        target_goodfor.update(CONCERN_TO_GOODFOR.get(c, [c]))
    target_goodfor.add("Anyone")

    concern_match_count = sum(1 for g in good_for if g in target_goodfor)
    concern_match_ratio = concern_match_count / max(len(concerns), 1)

    # Build the set of avoid tags that apply to this user profile
    avoid_tags = set()
    skin_avoid = SKIN_TYPE_TO_AVOID.get(skin_type)
    if skin_avoid:
        avoid_tags.add(skin_avoid)
    for c in concerns:
        avoid_tags.update(CONCERN_TO_AVOID.get(c, []))

    avoid_triggered = int(any(a in avoid_tags for a in avoid))

    # Base features
    feats = {
        "sensitivity_score": row["sensitivity_score"],
        "breadth_score": row["breadth_score"],
        "pregnancy_safe": int(bool(row["pregnancy_safe"])),
        "concern_match_count": concern_match_count,
        "concern_match_ratio": concern_match_ratio,
        "avoid_triggered": avoid_triggered,
        "is_sensitive": int(skin_type == "Sensitive"),
        "is_oily": int(skin_type == "Oily"),
        "is_dry": int(skin_type == "Dry"),
        "num_concerns": len(concerns),
        "good_for_count": len(good_for),
    }

    # Description-based text features
    desc_feats = extract_description_features(row.get("description", ""))
    feats.update(desc_feats)

    return feats


def derive_label(features: dict) -> int:
    """
    Rule-based label derivation.
    Returns 0 = good fit, 1 = possible irritation, 2 = poor fit
    """
    s = features["sensitivity_score"]
    avoid = features["avoid_triggered"]
    match = features["concern_match_count"]
    breadth = features["breadth_score"]
    is_sensitive = features["is_sensitive"]

    # ── POOR FIT ──
    if avoid and s >= 2:
        return 2
    if avoid and is_sensitive and s >= 1:
        return 2

    # ── POSSIBLE IRRITATION ──
    if avoid:
        return 1
    if s >= 3 and not (features.get("is_oily", 0) or features.get("is_dry", 0)):
        if not is_sensitive:
            return 1

    # ── GOOD FIT ──
    if match >= 2:
        return 0
    if match >= 1 and s <= 1:
        return 0
    if breadth >= 3 and s == 0:
        return 0
    if s <= 1:
        return 0

    return 1


FEATURE_COLS = [
    "sensitivity_score", "breadth_score", "pregnancy_safe",
    "concern_match_count", "concern_match_ratio", "avoid_triggered",
    "is_sensitive", "is_oily", "is_dry",
    "num_concerns", "good_for_count", "category_enc",
    # Description-based text features
    "desc_soothing", "desc_exfoliant", "desc_anti_inflammatory",
    "desc_antioxidant", "desc_brightening",
]


def generate_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the training set.

    Priority:
      1. If data/processed/labeled_pairs.csv exists (AI-generated ground truth),
         join those labels in.  Any pair NOT yet labeled falls back to derive_label().
      2. Otherwise every label is derived from the rule function (rule-based baseline).
    """
    # Load AI labels if available
    ai_labels: dict = {}
    if os.path.exists(LABELS_PATH):
        ldf = pd.read_csv(LABELS_PATH)
        for _, r in ldf.iterrows():
            label_val = r.get("label_int", r.get("label"))
            if isinstance(label_val, str):
                label_val = {"good fit": 0, "possible irritation": 1, "poor fit": 2}.get(label_val, 1)
            ai_labels[(r["ingredient_name"], r["skin_type"], r["concerns"])] = int(label_val)
        print(f"  AI labels loaded: {len(ai_labels)} pairs from {LABELS_PATH}")
    else:
        print("  No labeled_pairs.csv found — using rule-based labels")

    rows = []
    ai_count = 0
    rule_count = 0

    for skin_type, concerns in SAMPLE_PROFILES:
        concerns_key = str(concerns)
        for _, ingredient in df.iterrows():
            feats = compute_features(ingredient, skin_type, concerns)
            feats["category"] = ingredient.get("category", "Other") or "Other"

            key = (ingredient["name"], skin_type, concerns_key)
            if key in ai_labels:
                feats["label"] = ai_labels[key]
                feats["label_source"] = "ai"
                ai_count += 1
            else:
                feats["label"] = derive_label(feats)
                feats["label_source"] = "rule"
                rule_count += 1

            rows.append(feats)

    combined = pd.DataFrame(rows)
    print(f"  Label sources — AI: {ai_count}  |  Rule-based fallback: {rule_count}")
    return combined


def build_feature_matrix(combined: pd.DataFrame):
    le = LabelEncoder()
    combined["category_enc"] = le.fit_transform(combined["category"].fillna("Other"))
    X = combined[FEATURE_COLS].copy().astype(float)
    y = combined["label"]
    return X, y, le


def train_and_evaluate():
    # ------------------------------------------------------------------ #
    # STEP 1: Load dataset
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("STEP 1 — Loading dataset")
    print("=" * 60)
    df = load_data()
    print(f"  {len(df)} ingredients | columns: {df.columns.tolist()}")

    # ------------------------------------------------------------------ #
    # STEP 2: Generate training data
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 2 — Generating training data (ingredients x user profiles)")
    print("=" * 60)
    combined = generate_training_data(df)
    ai_pct = (combined["label_source"] == "ai").mean() * 100 if "label_source" in combined.columns else 0
    print(f"  {len(SAMPLE_PROFILES)} profiles x {len(df)} ingredients = {len(combined)} rows")
    print(f"  Labels from AI: {ai_pct:.1f}%  |  Rule-based fallback: {100 - ai_pct:.1f}%")
    for lbl, cnt in combined["label"].value_counts().sort_index().items():
        print(f"  Label {lbl} ({LABEL_MAP[lbl]}): {cnt} ({cnt / len(combined) * 100:.1f}%)")

    # ------------------------------------------------------------------ #
    # STEP 3: Feature engineering
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 3 — Feature engineering")
    print("=" * 60)
    X, y, category_encoder = build_feature_matrix(combined)
    print(f"  {len(FEATURE_COLS)} features: {FEATURE_COLS}")
    print(f"  Feature matrix shape: {X.shape}")

    # Show description feature coverage
    desc_cols = [c for c in FEATURE_COLS if c.startswith("desc_")]
    for col in desc_cols:
        pct = combined[col].mean() * 100
        print(f"    {col}: {pct:.1f}% of ingredients have this keyword")

    # ------------------------------------------------------------------ #
    # STEP 4: Train/test split
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 4 — Train/test split (80/20, stratified)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    X_all_sc = scaler.transform(X)

    # ------------------------------------------------------------------ #
    # STEP 5: Train and compare 5 models
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 5 — Training & comparing 5 models (5-fold CV + held-out test)")
    print("=" * 60)

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
        "Random Forest":       (RandomForestClassifier(n_estimators=200, random_state=42), False),
        "Gradient Boosting":   (GradientBoostingClassifier(n_estimators=200, random_state=42), False),
        "SVM (RBF kernel)":    (SVC(kernel="rbf", probability=True, random_state=42), True),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=7), True),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, (model, needs_scale) in models.items():
        Xtr = X_train_sc if needs_scale else X_train.values
        Xte = X_test_sc  if needs_scale else X_test.values
        Xcv = X_all_sc   if needs_scale else X.values

        cv_scores = cross_val_score(model, Xcv, y, cv=cv, scoring="accuracy")
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        test_acc = accuracy_score(y_test, y_pred)

        results[name] = {
            "model": model,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_accuracy": test_acc,
            "needs_scale": needs_scale,
            "y_pred": y_pred,
        }

        print(f"\n  [{name}]")
        print(f"    5-fold CV accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"    Test accuracy      : {test_acc:.4f}")
        labels_present = sorted(y_test.unique())
        target_names = [LABEL_MAP[l] for l in labels_present]
        report = classification_report(
            y_test, y_pred,
            labels=labels_present,
            target_names=target_names,
            zero_division=0,
        )
        for line in report.strip().split("\n"):
            print(f"      {line}")

    # ------------------------------------------------------------------ #
    # STEP 6: Summary table
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 6 — Model comparison summary")
    print("=" * 60)
    ranking = sorted(results.items(), key=lambda x: x[1]["test_accuracy"], reverse=True)
    print(f"\n  {'Model':<28} {'CV Acc (mean)':>14} {'Test Acc':>10}")
    print(f"  {'-'*28} {'-'*14} {'-'*10}")
    for name, res in ranking:
        tag = "  <- BEST" if name == ranking[0][0] else ""
        print(f"  {name:<28} {res['cv_mean']:.4f}+/-{res['cv_std']:.3f}   {res['test_accuracy']:.4f}{tag}")

    best_name, best_res = ranking[0]
    best_model = best_res["model"]
    print(f"\n  Winner: {best_name}  (test accuracy = {best_res['test_accuracy']:.4f})")

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
        print(f"\n  Feature importances ({best_name}):")
        for feat, imp in importances.sort_values(ascending=False).items():
            bar = "#" * int(imp * 40)
            print(f"    {feat:<26} {imp:.4f}  {bar}")

    # ------------------------------------------------------------------ #
    # STEP 7: Save best model
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 7 — Saving best model artifacts")
    print("=" * 60)
    os.makedirs(MODEL_DIR, exist_ok=True)
    artifacts = {
        "model": best_model,
        "scaler": scaler,
        "category_encoder": category_encoder,
        "feature_cols": FEATURE_COLS,
        "needs_scale": best_res["needs_scale"],
        "model_name": best_name,
        "label_map": LABEL_MAP,
        "concern_to_goodfor": CONCERN_TO_GOODFOR,
        "concern_to_avoid": CONCERN_TO_AVOID,
        "skin_type_to_avoid": SKIN_TYPE_TO_AVOID,
        "desc_keyword_groups": DESC_KEYWORD_GROUPS,
    }
    model_path = os.path.join(MODEL_DIR, "ingredient_classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"  Saved -> {model_path}")
    print("\nDone.")
    return artifacts, results


if __name__ == "__main__":
    train_and_evaluate()