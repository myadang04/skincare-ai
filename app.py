import streamlit as st

from ai.vision import extract_ingredients

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Skincare Match Tool", layout="centered")
st.title("Skincare Match Tool")
st.write("Find out if a product is right for your skin.")

# ---------------------------------------------------------------------------
# Step 1 — User inputs
# ---------------------------------------------------------------------------
st.header("Step 1: Tell us about your skin")

skin_type = st.selectbox(
    "Skin type",
    ["Normal", "Oily", "Dry", "Combination", "Sensitive"],
)

skin_concerns = st.multiselect(
    "Skin concerns (select all that apply)",
    ["Acne", "Redness", "Hyperpigmentation", "Dryness", "Sensitivity",
     "Wrinkles", "Dullness", "Enlarged Pores", "Dark Circles"],
)

st.header("Step 2: Upload your product label")

uploaded_file = st.file_uploader(
    "Upload a clear photo of the ingredient list",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded label", use_column_width=True)

# ---------------------------------------------------------------------------
# Analyze button
# ---------------------------------------------------------------------------
st.divider()
run = st.button("Analyze Product", type="primary", disabled=uploaded_file is None)

if run:
    if not skin_concerns:
        st.warning("Please select at least one skin concern before analyzing.")
        st.stop()

    # --- Vision: extract ingredients ---
    with st.status("Extracting ingredients from label...", expanded=False) as status:
        file_bytes = uploaded_file.read() if uploaded_file.tell() == 0 else uploaded_file.getvalue()
        try:
            ingredients, raw_text, processed_img = extract_ingredients(file_bytes)
            status.update(label=f"Found {len(ingredients)} ingredient(s)", state="complete")
        except Exception as e:
            status.update(label="Extraction failed", state="error")
            st.error(f"Could not extract ingredients: {e}")
            st.stop()

    if not ingredients:
        st.warning("No ingredient list detected. Try a clearer photo of the label.")
        st.stop()

    with st.expander("Raw OCR text"):
        st.text_area("", raw_text, height=200)
    with st.expander("Processed image used for OCR"):
        st.image(processed_img, clamp=True)

    st.divider()

    # --- Dataset lookup + ML model (placeholder) ---
    with st.status("Checking ingredients against dataset & ML model...", expanded=False) as status:
        # TODO: implement in pipeline.py
        # results = pipeline.run(ingredients, skin_type, skin_concerns)
        status.update(label="Analysis complete (placeholder)", state="complete")

    # --- Overall fit classification ---
    st.header("Overall Fit")
    overall_fit = "good fit"          # TODO: replace with model output
    color = {"good fit": "green", "possible irritation": "orange", "poor fit": "red"}
    st.markdown(
        f"<span style='font-size:1.4rem; font-weight:bold; color:{color[overall_fit]};'>"
        f"{overall_fit.title()}</span>",
        unsafe_allow_html=True,
    )

    st.divider()

    # --- Per-ingredient breakdown ---
    st.header("Ingredient Breakdown")
    for ingredient in ingredients:
        verdict = "good fit"          # TODO: replace with per-ingredient model output
        icon = {"good fit": "✅", "possible irritation": "⚠️", "poor fit": "❌"}[verdict]
        with st.expander(f"{icon} {ingredient}"):
            st.markdown(f"**Verdict:** {verdict.title()}")
            st.markdown("_Analysis coming soon — dataset lookup and ML model not yet wired up._")

    st.divider()

    # --- Top 3 better-matched ingredient suggestions ---
    st.header("Top 3 Suggested Ingredients for Your Skin")
    st.info("Embedding similarity search not yet implemented.")
    # TODO: replace with recommender.py output
    # for suggestion in top_suggestions:
    #     st.markdown(f"- **{suggestion['name']}** — {suggestion['reason']}")

    st.divider()

    # --- Plain-English explanation ---
    st.header("What This Means for You")
    st.info("AI explanation not yet implemented.")
    # TODO: replace with analyzer.py output
