"""
generate_labels.py
──────────────────
Uses Claude to label every (ingredient × skin profile) pair with:
  - label:      "good fit" | "possible irritation" | "poor fit"
  - confidence: 1–5
  - reason:     one-sentence explanation

Output → data/processed/labeled_pairs.csv

Run once:
    python3 data/generate_labels.py

Cost estimate: 247 ingredients × 20 profiles = 4,940 API calls
  Using claude-haiku-4-5 (~$0.25 per 1M input tokens)
  Each prompt ≈ 300 tokens  →  ~$0.37 total
"""

import os
import ast
import json
import time
import pandas as pd
from dotenv import load_dotenv
import anthropic

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "processed", "ingredients_final.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "processed", "labeled_pairs.csv")

# ── Skin profiles to generate labels for ──────────────────────────────────────
# (skin_type, concerns)  — same set used in model/train.py
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

LABEL_OPTIONS = ["good fit", "possible irritation", "poor fit"]


def parse_list_field(val):
    if pd.isna(val) or val == "":
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return [v.strip() for v in str(val).split(",") if v.strip()]


def build_prompt(ingredient: dict, skin_type: str, concerns: list) -> str:
    good_for = ingredient.get("good_for", [])
    avoid    = ingredient.get("avoid", [])
    desc     = str(ingredient.get("description", ""))[:600]   # keep prompt short
    sens     = ingredient.get("sensitivity_score", 0)
    breadth  = ingredient.get("breadth_score", 0)
    category = ingredient.get("category", "Other")

    return f"""You are a cosmetic dermatologist reviewing skincare ingredient safety.

INGREDIENT: {ingredient['name']}
Category: {category}
Description: {desc}
Known benefits (good_for): {good_for}
Who should avoid: {avoid}
Sensitivity score (0=gentle, 4=very irritating): {sens}
Breadth of benefit (0=narrow, 9=universal): {breadth}

USER PROFILE:
  Skin type : {skin_type}
  Concerns  : {", ".join(concerns)}

TASK: Based on your dermatology knowledge AND the ingredient data above, classify how well this ingredient fits this user.

Rules:
- "good fit"           → ingredient is beneficial and safe for this user's skin type and concerns
- "possible irritation"→ ingredient may help some concerns but carries meaningful risk for this profile
- "poor fit"           → ingredient is likely to cause problems or is not recommended for this profile

Respond with valid JSON only, no extra text:
{{"label": "<good fit|possible irritation|poor fit>", "confidence": <1-5>, "reason": "<one sentence>"}}"""


def call_claude(client: anthropic.Anthropic, prompt: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)

            # Validate label
            label = parsed.get("label", "").lower().strip()
            if label not in LABEL_OPTIONS:
                label = "possible irritation"   # safe fallback
            parsed["label"] = label

            return parsed

        except (json.JSONDecodeError, KeyError):
            if attempt < retries - 1:
                time.sleep(1)
            else:
                return {"label": "possible irritation", "confidence": 1, "reason": "parse error"}
        except anthropic.RateLimitError:
            wait = 2 ** attempt
            print(f"    Rate limit — waiting {wait}s...")
            time.sleep(wait)

    return {"label": "possible irritation", "confidence": 1, "reason": "max retries exceeded"}


def generate_labels(resume: bool = True):
    """
    Label all ingredient × profile pairs using Claude.

    Args:
        resume: If True and output file already exists, skip already-labeled rows.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Load ingredients
    df = pd.read_csv(DATA_PATH)
    df["good_for"] = df["good_for"].apply(parse_list_field)
    df["avoid"]    = df["avoid"].apply(parse_list_field)
    print(f"Loaded {len(df)} ingredients")

    # Load existing results if resuming
    existing = set()
    rows = []
    if resume and os.path.exists(OUTPUT_PATH):
        existing_df = pd.read_csv(OUTPUT_PATH)
        rows = existing_df.to_dict("records")
        for r in rows:
            existing.add((r["ingredient_name"], r["skin_type"], str(r["concerns"])))
        print(f"Resuming — {len(rows)} rows already done, skipping those")

    total  = len(df) * len(SKIN_PROFILES)
    done   = len(rows)
    skipped = 0

    print(f"Total pairs to label: {total}")
    print(f"Remaining           : {total - done}\n")

    for skin_type, concerns in SKIN_PROFILES:
        concerns_key = str(concerns)
        profile_label = f"{skin_type} / {', '.join(concerns)}"

        for _, ingredient in df.iterrows():
            key = (ingredient["name"], skin_type, concerns_key)
            if key in existing:
                skipped += 1
                continue

            prompt = build_prompt(ingredient.to_dict(), skin_type, concerns)
            result = call_claude(client, prompt)

            rows.append({
                "ingredient_name": ingredient["name"],
                "skin_type":       skin_type,
                "concerns":        concerns_key,
                "label":           result["label"],
                "label_int":       LABEL_OPTIONS.index(result["label"]),
                "confidence":      result.get("confidence", 3),
                "reason":          result.get("reason", ""),
                "sensitivity_score": ingredient["sensitivity_score"],
                "breadth_score":     ingredient["breadth_score"],
                "category":          ingredient.get("category", "Other"),
            })
            done += 1

            # Save checkpoint every 50 rows
            if done % 50 == 0:
                pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
                pct = done / total * 100
                print(f"  [{pct:5.1f}%] {done}/{total}  last: {ingredient['name']} | {profile_label} → {result['label']}")

            # Small delay to stay within rate limits
            time.sleep(0.05)

    # Final save
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone. {len(out_df)} labeled pairs saved to:\n  {OUTPUT_PATH}")
    print("\nLabel distribution:")
    print(out_df["label"].value_counts().to_string())
    print("\nAverage confidence:", round(out_df["confidence"].mean(), 2))
    return out_df


if __name__ == "__main__":
    generate_labels(resume=True)
