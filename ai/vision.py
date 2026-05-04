import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytesseract


def preprocess_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """Convert uploaded image bytes into a preprocessed OpenCV image for OCR."""
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode uploaded image.")

    # Resize up because skincare label text is usually small
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return processed


def preprocess_image_from_path(image_path: str) -> np.ndarray:
    """Load an image from disk and preprocess it for OCR."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(path, "rb") as f:
        return preprocess_image_from_bytes(f.read())


def run_ocr(image: np.ndarray) -> str:
    """Run Tesseract OCR on a preprocessed image."""
    config = "--oem 3 --psm 6"
    return pytesseract.image_to_string(image, lang="eng", config=config)


def _rejoin_hyphens(text: str) -> str:
    """
    Rejoin words that OCR split across lines with a hyphen.
    
    Examples:
        'Dipropy-\\nlene Glycol'  → 'Dipropylene Glycol'
        'Dipropy- lene Glycol'   → 'Dipropylene Glycol'
        'Cerami-\\nde NP'        → 'Ceramide NP'
        'Polyacry-\\nlate'       → 'Polyacrylate'
    
    BUT preserves intentional hyphens like:
        'Beta-Glucan'            → 'Beta-Glucan'   (no space/newline after hyphen)
        '1,2-Hexanediol'         → '1,2-Hexanediol'
    """
    # Pattern: hyphen followed by optional whitespace including newlines,
    # then a lowercase letter (indicates a word continuation, not a new word)
    text = re.sub(r"-\s*\n\s*", "-", text)        # collapse hyphen-newline first
    text = re.sub(r"-\s+([a-z])", r"\1", text)     # 'Dipropy- lene' → 'Dipropylene'
    return text


def extract_ingredients_block(text: str) -> Optional[str]:
    """Isolate the ingredients section from raw OCR text."""
    # --- NEW: rejoin hyphenated line breaks BEFORE collapsing whitespace ---
    text = _rejoin_hyphens(text)
    
    cleaned = re.sub(r"\s+", " ", text).strip()

    # --- IMPROVED: broader patterns that catch multilingual headers ---
    patterns = [
        # "Ingredients / Ingrédients :" or "Ingredients/Ingrédients:"
        r"ingredients?\s*/\s*ingr[eé]dients?\s*[:\-]\s*(.*)",
        # Standard English headers
        r"ingredients?\s*[:\-]\s*(.*)",
        r"ingredient\s*list\s*[:\-]\s*(.*)",
        r"active ingredients?\s*[:\-]\s*(.*)",
        # French-first labels
        r"ingr[eé]dients?\s*[:\-]\s*(.*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: if OCR gives a long comma-separated block, use it
    if cleaned.count(",") >= 5:
        return cleaned

    return None


def clean_ingredients_text(ingredients_text: str) -> str:
    """Clean up OCR noise and strip non-ingredient sections."""
    text = ingredients_text

    stop_words = [
        "warning", "directions", "how to use", "distributed by",
        "manufactured by", "caution", "uses", "net wt", "made in", "store at",
    ]
    for stop_word in stop_words:
        text = re.sub(rf"\b{re.escape(stop_word)}\b.*", "", text, flags=re.IGNORECASE)

    # Normalize separators
    for char in (";", "•", "·", "|"):
        text = text.replace(char, ",")

    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.")
    return text


def _normalize_ingredient_name(name: str) -> str:
    """
    Clean up a single ingredient name after splitting.
    
    Strips parenthetical-only entries, trailing/leading junk,
    and normalizes common OCR misreads.
    """
    # Remove leading/trailing whitespace and periods
    name = name.strip(" .")
    
    # Fix common OCR misreads
    name = name.replace("lron", "Iron")    # lowercase-L read as I
    name = name.replace("0xide", "Oxide")  # zero read as O
    
    return name


def split_ingredients(ingredients_text: str) -> List[str]:
    """Split a cleaned ingredient string into individual ingredient names."""
    parts = [_normalize_ingredient_name(part) for part in ingredients_text.split(",")]
    return [
        part for part in parts
        if (
            len(part) >= 2
            and not re.fullmatch(r"[\W_]+", part)
            # Filter out entries that are ONLY a parenthetical like "(Aqua)"
            # but keep entries that CONTAIN parentheticals like "Water (Aqua)"
            and not re.fullmatch(r"\(.*\)", part.strip())
        )
    ]


def extract_ingredients(
    file_bytes: bytes,
) -> Tuple[List[str], str, np.ndarray]:
    """
    Full OCR pipeline: bytes → preprocessed image → OCR → ingredient list.

    Args:
        file_bytes: Raw bytes of an uploaded image file.

    Returns:
        (ingredients, raw_ocr_text, processed_image)
        - ingredients: list of extracted ingredient name strings
        - raw_ocr_text: full OCR output before any parsing
        - processed_image: the thresholded image used for OCR
    """
    processed = preprocess_image_from_bytes(file_bytes)
    raw_text = run_ocr(processed)

    block = extract_ingredients_block(raw_text)
    if not block:
        return [], raw_text, processed

    cleaned = clean_ingredients_text(block)
    ingredients = split_ingredients(cleaned)
    return ingredients, raw_text, processed


def extract_ingredients_from_path(
    image_path: str,
) -> Tuple[List[str], str, np.ndarray]:
    """Convenience wrapper for loading from a file path instead of bytes."""
    with open(image_path, "rb") as f:
        return extract_ingredients(f.read())