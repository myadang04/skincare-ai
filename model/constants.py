"""
model/constants.py
------------------
Single source of truth for all vocabularies and feature definitions
shared between train.py and predict.py.

If you add a new skin type, concern, category, or keyword feature,
change it HERE only. Both train and predict will pick it up automatically.
Re-run train.py after any change here to regenerate the saved model.
"""

# ── Skin type vocabulary ───────────────────────────────────────────────────────
SKIN_TYPES = [
    "Oily Skin",
    "Dry Skin",
    "Combination Skin",
    "Sensitive Skin",
    "Normal Skin",
    "Impaired Skin Barrier",
]

# ── Concern vocabulary ─────────────────────────────────────────────────────────
ALL_CONCERNS = [
    "Acne",
    "Blackheads",
    "Dark Circles",
    "Dry Skin",
    "Elasticity",
    "Enlarged Pores",
    "Eye Bags",
    "Fine Lines",
    "Impaired Skin Barrier",
    "Oily Skin",
    "Pigmentation",
    "Post Blemish Marks",
    "Radiance",
    "Redness",
    "Texture",
    "UV Protection",
    "Wrinkles",
]

# ── Category vocabulary ────────────────────────────────────────────────────────
ALL_CATEGORIES = [
    "AHA",
    "Amino Acid",
    "BHA",
    "Ceramide",
    "Humectant",
    "Lipid/Oil",
    "Marine/Algae",
    "Mineral",
    "Other",
    "Peptide",
    "Plant Extract",
    "Probiotic/Ferment",
    "Retinoid",
    "UV Filter",
    "Vitamin",
    "Vitamin C",
]

# ── Keyword features ───────────────────────────────────────────────────────────
KEYWORD_FEATURES = {
    "kw_irritat":   ["irritat"],
    "kw_sensitiv":  ["sensitiv"],
    "kw_gentle":    ["gentle", "mild"],
    "kw_acid":      ["acid"],
    "kw_exfoliat":  ["exfoliat", "peel"],
    "kw_hydrat":    ["hydrat", "moistur"],
    "kw_acne":      ["acne", "blemish"],
    "kw_aging":     ["wrinkle", "collagen", "elastin", "anti-ag"],
    "kw_barrier":   ["barrier", "repair"],
    "kw_retinoid":  ["retinol", "retinoat", "retinyl"],
    "kw_vitamin_c": ["ascorb", "vitamin c"],
    "kw_peptide":   ["peptide"],
    "kw_pigment":   ["pigment", "bright", "melanin"],
    "kw_uv":        ["uv filter", "spf", "sunscreen"],
    "kw_sooth":     ["sooth", "calm"],
    "kw_antioxid":  ["antioxid"],
}

# ── Safe filler whitelist ──────────────────────────────────────────────────────
SAFE_FILLERS = frozenset({
    "aqua", "water", "eau",
    "glycerin", "glycerol", "propylene glycol", "butylene glycol",
    "pentylene glycol", "hexylene glycol",
    "carbomer", "xanthan gum", "hydroxyethylcellulose",
    "hydroxypropyl methylcellulose", "acrylates copolymer", "cellulose",
    "polysorbate 20", "polysorbate 60", "polysorbate 80",
    "cetearyl alcohol", "stearic acid", "palmitic acid",
    "glyceryl stearate", "peg-100 stearate", "sorbitan oleate",
    "sodium hydroxide", "triethanolamine", "potassium hydroxide",
    "phenoxyethanol", "ethylhexylglycerin", "caprylyl glycol",
    "sodium benzoate", "potassium sorbate", "benzyl alcohol",
    "dimethicone", "cyclopentasiloxane", "cyclohexasiloxane",
    "dimethiconol", "trimethylsiloxysilicate",
    "petrolatum", "mineral oil", "paraffin",
    "alcohol denat", "ethanol",
})

# ── Fragrance / sensitizer flags ───────────────────────────────────────────────
SENSITIZERS = frozenset({
    "parfum", "fragrance", "limonene", "linalool",
    "geraniol", "eugenol", "citronellol", "cinnamal",
    "coumarin", "benzyl salicylate", "benzyl benzoate",
    "farnesol", "isoeugenol",
})

SENSITIVE_SKIN_TYPES = frozenset({
    "Sensitive Skin",
    "Impaired Skin Barrier",
})


# ── Feature name normalizer ────────────────────────────────────────────────────
def _safe_col(s: str) -> str:
    """Normalize string to safe column name (replace / and spaces with _)."""
    return s.replace("/", "_").replace(" ", "_")


# ── Canonical feature column list ─────────────────────────────────────────────
def build_feature_columns() -> list:
    """
    Return the ordered feature column list exactly as produced during training.
    Import FEATURE_COLUMNS from this module in both train.py and predict.py.
    """
    cols = list(KEYWORD_FEATURES.keys()) + ["desc_len"]
    cols += [f"cat_{_safe_col(c)}" for c in ALL_CATEGORIES]
    cols += [f"skin_{_safe_col(s)}" for s in SKIN_TYPES]
    cols += [f"concern_{_safe_col(c)}" for c in ALL_CONCERNS]
    return cols


FEATURE_COLUMNS = build_feature_columns()