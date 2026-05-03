"""
ai/analyzer.py
──────────────
AI fallback for ingredients not found in the local dataset.

When an ingredient name doesn't match anything in ingredients_final.csv,
this module calls the Anthropic Claude API (claude-haiku-4-5, cheapest model)
and asks it to act as a cosmetic dermatologist.

The returned dict is intentionally identical to the dict returned by
model/predict.py::predict_ingredient() so the rest of the pipeline
(pipeline.py, app.py) works unchanged.

The only difference is:
  - result["found"]       = False   (not in local dataset)
  - result["ai_analyzed"] = True    (analysed by Claude)
  - result["ai_reason"]  = "..."   (Claude's plain-English explanation)
"""

import os
import json
import re
from typing import Optional

_client = None   # lazy-loaded Anthropic client


def _get_client():
    global _client
    if _client is None:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for AI ingredient analysis. "
                "Install it with:  pip install anthropic"
            )
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file or export it before running the app."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


_SYSTEM_PROMPT = """\
You are an expert cosmetic dermatologist and cosmetic chemist. \
When given an ingredient name and a user's skin profile, you evaluate \
whether that ingredient is a good fit, a possible irritant, or a poor fit.

You MUST respond with ONLY valid JSON — no prose, no markdown fences, no extra keys.

JSON schema (all fields required):
{
  "label_name":        "good fit" | "possible irritation" | "poor fit",
  "sensitivity_score": 0-4  (0 = very gentle, 4 = very irritating),
  "breadth_score":     0-5  (how broadly beneficial for skin in general),
  "category":          short category string e.g. "Humectant", "Retinoid", "Plant Extract",
  "good_for":          [list of skin concerns this ingredient helps with],
  "avoid":             [list of skin types / conditions that should avoid this ingredient],
  "matched_concerns":  [subset of user concerns that this ingredient addresses],
  "pregnancy_safe":    true | false,
  "reason":            "1-2 sentence plain-English explanation of your verdict"
}

Label definitions:
  good fit            — suitable for the user's skin type and addresses ≥1 concern without risk
  possible irritation — might cause irritation; patch-test recommended
  poor fit            — likely to irritate, worsen a concern, or is contraindicated for this skin type
"""


def _build_user_message(name: str, skin_type: str, concerns: list) -> str:
    concerns_str = ", ".join(concerns) if concerns else "none specified"
    return (
        f"Ingredient: {name}\n"
        f"User skin type: {skin_type}\n"
        f"User skin concerns: {concerns_str}\n\n"
        "Evaluate this ingredient for this user profile and return the JSON verdict."
    )


def _extract_json(text: str) -> dict:
    """Pull the first {...} block out of the model response."""
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    # Find first { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)   # let it raise if truly malformed


def analyze_unknown_ingredient(
    name: str,
    skin_type: str,
    concerns: list,
    *,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 512,
) -> dict:
    """
    Call Claude to analyse an ingredient that isn't in our local dataset.

    Returns a dict compatible with model/predict.py::predict_ingredient():

        name, found, ai_analyzed, label, label_name, probability,
        good_for, avoid, matched_concerns,
        sensitivity_score, breadth_score, category,
        ai_reason
    """
    base = {
        "name": name,
        "found": False,
        "ai_analyzed": True,
        "label": None,
        "label_name": "unknown",
        "probability": None,
        "good_for": [],
        "avoid": [],
        "matched_concerns": [],
        "sensitivity_score": None,
        "breadth_score": None,
        "category": None,
        "ai_reason": None,
    }

    try:
        client = _get_client()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": _build_user_message(name, skin_type, concerns)}
            ],
        )
        raw = response.content[0].text
        data = _extract_json(raw)

        # Map label_name → int
        label_name = data.get("label_name", "unknown").lower().strip()
        label_int_map = {"good fit": 0, "possible irritation": 1, "poor fit": 2}
        label_int = label_int_map.get(label_name)

        base.update({
            "label":             label_int,
            "label_name":        label_name if label_name in label_int_map else "unknown",
            "good_for":          data.get("good_for", []),
            "avoid":             data.get("avoid", []),
            "matched_concerns":  data.get("matched_concerns", []),
            "sensitivity_score": data.get("sensitivity_score"),
            "breadth_score":     data.get("breadth_score"),
            "category":          data.get("category"),
            "pregnancy_safe":    data.get("pregnancy_safe"),
            "ai_reason":         data.get("reason"),
        })

    except (EnvironmentError, ImportError):
        # No API key or package missing — return unknown gracefully
        base["ai_reason"] = (
            "AI analysis unavailable (ANTHROPIC_API_KEY not set or 'anthropic' package missing)."
        )
    except Exception as exc:
        # Any other error (network, JSON parse, etc.)
        base["ai_reason"] = f"AI analysis failed: {exc}"

    return base
