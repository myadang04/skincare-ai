import streamlit as st
from ai.vision import extract_ingredients
import pipeline
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Skincare Match Tool", layout="centered")
st.title("🧴 Skincare Match Tool")
st.write("Upload a product label to find out if it's right for your skin.")

# ── Step 1: User profile ───────────────────────────────────────────────────────
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

# ── Step 2: Upload ─────────────────────────────────────────────────────────────
st.header("Step 2: Upload your product label")

uploaded_file = st.file_uploader(
    "Upload a clear photo of the ingredient list",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded label", width=600)

# ── Analyze button ─────────────────────────────────────────────────────────────
st.divider()
run_btn = st.button("Analyze Product", type="primary", disabled=uploaded_file is None)

if run_btn:
    if not skin_concerns:
        st.warning("Please select at least one skin concern before analyzing.")
        st.stop()

    # ── Vision: OCR ──────────────────────────────────────────────────────────
    with st.status("Extracting ingredients from label...", expanded=False) as status:
        file_bytes = uploaded_file.getvalue()
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

    with st.expander("🔍 Raw OCR text"):
        st.text_area("", raw_text, height=150)
    with st.expander("🖼️ Processed image used for OCR"):
        st.image(processed_img, clamp=True)

    st.divider()

    # ── Pipeline: model analysis ──────────────────────────────────────────────
    with st.status("Analysing ingredients...", expanded=False) as status:
        try:
            analysis = pipeline.run(ingredients, skin_type, skin_concerns)
            status.update(
                label=f"Done — {analysis['found_count']} ingredients matched, "
                      f"{analysis['flag_count']} flagged",
                state="complete",
            )
        except Exception as e:
            status.update(label="Analysis failed", state="error")
            st.error(f"Pipeline error: {e}")
            st.stop()

    # ── Overall fit ───────────────────────────────────────────────────────────
    st.header("Overall Fit")

    overall = analysis["overall_label"]
    color   = analysis["overall_color"]
    icon    = analysis["overall_icon"]
    stats   = analysis["summary_stats"]

    st.markdown(
        f"<div style='padding:16px; border-radius:10px; background:{color}22; "
        f"border-left:5px solid {color};'>"
        f"<span style='font-size:1.5rem; font-weight:bold; color:{color};'>"
        f"{icon} {overall.title()}</span></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"✅ {stats['good fit']} good fit  •  "
        f"⚠️ {stats['possible irritation']} possible irritation  •  "
        f"❌ {stats['poor fit']} poor fit  •  "
        f"❓ {stats['unknown']} not in dataset"
    )

    st.divider()

    # ── Per-ingredient breakdown ──────────────────────────────────────────────
    st.header("Ingredient Breakdown")

    LABEL_COLOR = {
        "good fit":            "#2e7d32",
        "possible irritation": "#e65100",
        "poor fit":            "#c62828",
        "unknown":             "#757575",
    }
    LABEL_ICON = {
        "good fit":            "✅",
        "possible irritation": "⚠️",
        "poor fit":            "❌",
        "unknown":             "❓",
    }

    # Separate known vs unknown
    unknown_names = [r["name"] for r in analysis["results"] if not r["found"]]
    known_results = [r for r in analysis["results"] if r["found"]]

    # Show flagged first, then good fit
    order = {"poor fit": 0, "possible irritation": 1, "good fit": 2}
    sorted_results = sorted(known_results, key=lambda r: order.get(r["label_name"], 3))

    for r in sorted_results:
        lbl   = r["label_name"]
        icon  = LABEL_ICON[lbl]
        color = LABEL_COLOR[lbl]

        with st.expander(f"{icon} **{r['name']}**  —  {lbl.title()}"):

            # Verdict badge
            st.markdown(
                f"<span style='background:{color}22; color:{color}; padding:3px 10px; "
                f"border-radius:12px; font-weight:bold;'>{icon} {lbl.title()}</span>",
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Sensitivity", f"{r['sensitivity_score']} / 4")
            col2.metric("Breadth of benefit", f"{r['breadth_score']}")
            col3.metric("Category", r.get("category") or "—")

            if r["matched_concerns"]:
                st.success(f"**Addresses your concerns:** {', '.join(r['matched_concerns'])}")

            if r["good_for"]:
                st.markdown(f"**Good for:** {', '.join(r['good_for'])}")

            avoid_list = [a for a in r["avoid"] if a != "Related Allergy"]
            if avoid_list:
                st.warning(f"**Who should avoid:** {', '.join(avoid_list)}")

            if r.get("probability"):
                st.markdown("**Model confidence:**")
                for label_name, prob in sorted(r["probability"].items(),
                                               key=lambda x: x[1], reverse=True):
                    pct = int(prob * 100)
                    bar_color = LABEL_COLOR.get(label_name, "#999")
                    st.markdown(
                        f"<div style='display:flex; align-items:center; gap:8px; margin:2px 0'>"
                        f"<span style='width:160px; font-size:0.85rem;'>{label_name.title()}</span>"
                        f"<div style='flex:1; background:#eee; border-radius:4px; height:14px;'>"
                        f"<div style='width:{pct}%; background:{bar_color}; height:14px; "
                        f"border-radius:4px;'></div></div>"
                        f"<span style='font-size:0.85rem; width:36px;'>{pct}%</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    if unknown_names:
        st.caption(
            f"❓ **Not in our dataset yet:** {', '.join(unknown_names)}"
        )

    st.divider()

    # ── Top 3 suggestions ─────────────────────────────────────────────────────
    st.header("💡 Top 3 Ingredients That Would Work Better")
    st.caption(f"Good-fit ingredients for **{skin_type}** skin with **{', '.join(skin_concerns)}** — not already in this product.")

    suggestions = analysis["top_suggestions"]
    if suggestions:
        for i, s in enumerate(suggestions, 1):
            with st.expander(f"#{i} — {s['name']}  ({s.get('category', '')})", expanded=(i == 1)):
                col1, col2 = st.columns(2)
                col1.metric("Sensitivity", f"{s['sensitivity_score']} / 4")
                col2.metric("Breadth of benefit", f"{s['breadth_score']}")

                if s["matched_concerns"]:
                    st.success(f"**Addresses:** {', '.join(s['matched_concerns'])}")
                if s["good_for"]:
                    st.markdown(f"**Good for:** {', '.join(s['good_for'])}")
    else:
        st.info("No suggestions found — the product already covers your profile well!")

    st.divider()

    # ── AI-Powered Summary (Groq LLM) ────────────────────────────────────────
    st.header("📋 AI Summary")

    flagged = [r for r in analysis["results"] if r["label_name"] in ("possible irritation", "poor fit") and r["found"]]
    good    = [r for r in analysis["results"] if r["label_name"] == "good fit" and r["found"]]

    # Build context for the LLM
    good_names = [r["name"] for r in good]
    poor_items = []
    for r in flagged:
        item = f"{r['name']} ({r['label_name']}, sensitivity {r['sensitivity_score']}/4"
        if r.get("avoid"):
            avoid_clean = [a for a in r["avoid"] if a != "Related Allergy"]
            if avoid_clean:
                item += f", avoid: {', '.join(avoid_clean)}"
        if r.get("matched_concerns"):
            item += f", but addresses: {', '.join(r['matched_concerns'])}"
        item += ")"
        poor_items.append(item)

    suggestion_names = [s["name"] for s in suggestions[:3]] if suggestions else []

    prompt = f"""Based on this ingredient analysis, write a short product summary for someone with {skin_type} skin concerned about {', '.join(skin_concerns)}.

Overall verdict: {overall}
Good ingredients ({len(good_names)}): {', '.join(good_names[:8])}{'...' if len(good_names) > 8 else ''}
Problematic ingredients: {'; '.join(poor_items) if poor_items else 'None'}
Unknown ingredients: {len(unknown_names)}
Suggested alternatives: {', '.join(suggestion_names) if suggestion_names else 'None'}

Rules:
- Write 3-4 SHORT sentences. No run-on sentences.
- Be direct. Start with the verdict, then explain.
- If ingredients are flagged, say what they are and why they're bad for this skin type in plain terms.
- If it's a good fit, mention 2-3 key helpful ingredients and what they do.
- No filler phrases like "You'll be happy to know" or "With consistent use." Just say it.
- Sound like a friend who knows skincare, not a marketing brochure.
- Use "you/your" and keep it casual."""

    with st.status("Generating personalized summary...", expanded=False) as status:
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a straightforward skincare advisor. Be direct and concise. No fluff."},
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.4,
                max_tokens=200,
            )
            ai_summary = chat_completion.choices[0].message.content.strip()
            status.update(label="Summary generated", state="complete")
        except Exception as e:
            ai_summary = None
            status.update(label="AI summary unavailable", state="error")
            st.caption(f"Could not generate AI summary: {e}")

    # Display the AI summary with appropriate styling
    if ai_summary:
        if overall == "good fit":
            st.success(ai_summary)
        elif overall == "possible irritation":
            st.warning(ai_summary)
        elif overall == "poor fit":
            st.error(ai_summary)
        else:
            st.info(ai_summary)

        st.caption("*Summary generated by AI based on ingredient analysis results.*")
    else:
        # Fallback to template if Groq fails
        if overall == "good fit":
            st.success(
                f"This product looks like a **good fit** for your {skin_type} skin. "
                f"{len(good)} ingredient(s) directly address your concerns "
                f"({', '.join(skin_concerns)}) with no major red flags."
            )
        elif overall == "possible irritation":
            flagged_names = ", ".join(r["name"] for r in flagged)
            st.warning(
                f"This product may cause **some irritation** for your {skin_type} skin. "
                f"Watch out for: **{flagged_names}**. "
                f"Consider patch-testing before full use."
            )
        elif overall == "poor fit":
            poor = [r["name"] for r in flagged if r["label_name"] == "poor fit"]
            st.error(
                f"This product is likely a **poor fit** for your {skin_type} skin with "
                f"{', '.join(skin_concerns)}. "
                f"Problematic ingredient(s): **{', '.join(poor)}**. "
                f"We recommend looking for alternatives."
            )
        else:
            st.info("Could not determine overall fit — no ingredients matched our dataset.")