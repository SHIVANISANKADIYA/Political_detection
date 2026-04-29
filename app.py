# -*- coding: utf-8 -*-

import json
import re
from html import escape

import streamlit as st

from generate_description import (
    predict_entities,
    add_descriptions_to_predictions,
)

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Political Entity Detection",
    page_icon="🗳️",
    layout="wide",
)

st.title("🗳️ Multilingual Political Entity Detection")
st.caption("Enter political text in English, Hindi, Tamil, Telugu, or mixed language text.")

# =========================================================
# Sample texts
# =========================================================
SAMPLES = {
    "English - Leaders and parties": "PM Modi addressed a public rally in Varanasi. Rahul Gandhi met Congress leaders in Kerala. Amit Shah inaugurated a new BJP office in Lucknow.",
    "English - News paragraph": "The INLD leader Abhay Chautala held a rally in Rohtak. BJP and Congress both reacted to the statement.",
    "Hindi": "नरेंद्र मोदी ने दिल्ली में बीजेपी कार्यकर्ताओं को संबोधित किया। राहुल गांधी ने कांग्रेस नेताओं से मुलाकात की।",
    "Tamil": "நரேந்திர மோடி டெல்லியில் பேசினார். ராகுல் காந்தி காங்கிரஸ் தலைவர்களை சந்தித்தார்.",
    "Telugu": "నరేంద్ర మోదీ ఢిల్లీలో ప్రసంగించారు. రాహుల్ గాంధీ కాంగ్రెస్ నేతలను కలిశారు.",
}

# =========================================================
# Session state
# =========================================================
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = "English - Leaders and parties"

# =========================================================
# Helpers
# =========================================================
def highlight_entities(text, entities):
    """
    Highlight detected entities inside the original text.
    This uses simple regex replacement, so it works best when entity strings
    appear exactly in the input text.
    """
    if not text:
        return ""

    # unique entities only, longer first to reduce partial overlap
    unique_entities = []
    seen = set()
    for ent, _ in entities:
        ent_clean = ent.strip()
        if ent_clean and ent_clean not in seen:
            seen.add(ent_clean)
            unique_entities.append(ent_clean)

    unique_entities.sort(key=len, reverse=True)

    highlighted = escape(text)

    for ent in unique_entities:
        pattern = re.compile(re.escape(escape(ent)), flags=re.IGNORECASE)
        highlighted = pattern.sub(
            lambda m: f'<mark style="background-color:#fff3a0;padding:2px 4px;border-radius:4px;">{m.group(0)}</mark>',
            highlighted,
        )

    return highlighted


def show_result_cards(enriched_results):
    if not enriched_results:
        st.info("No entities detected.")
        return

    for item in enriched_results:
        with st.container(border=True):
            c1, c2 = st.columns([1, 4])
            with c1:
                st.markdown(f"### {item['entity']}")
                st.write(f"**Label:** `{item['label']}`")
            with c2:
                st.write(item["description"])


def build_download_json(enriched_results):
    return json.dumps(enriched_results, ensure_ascii=False, indent=2)


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("Controls")

    st.session_state.selected_sample = st.selectbox(
        "Choose a sample",
        list(SAMPLES.keys()),
        index=list(SAMPLES.keys()).index(st.session_state.selected_sample),
    )

    if st.button("Load sample"):
        st.session_state.text_input = SAMPLES[st.session_state.selected_sample]
        st.rerun()

    st.divider()
    st.write("Tips:")
    st.write("• Use full political sentences")
    st.write("• You can mix languages")
    st.write("• The description part uses Wikipedia/Wikidata")

# =========================================================
# Input area
# =========================================================
st.subheader("Input Text")
text = st.text_area(
    "Paste your political sentence or paragraph here",
    height=180,
    key="text_input",
    placeholder="Example: PM Modi met BJP leaders in Delhi.",
)

col_a, col_b = st.columns([1, 1])

run_clicked = col_a.button("Analyze Text", type="primary")
clear_clicked = col_b.button("Clear")

if clear_clicked:
    st.session_state.text_input = ""
    st.rerun()

# =========================================================
# Analysis
# =========================================================
if run_clicked:
    if not text.strip():
        st.warning("Please enter some text first.")
        st.stop()

    with st.spinner("Detecting entities and fetching descriptions..."):
        predicted_entities = predict_entities(text)
        enriched_results = add_descriptions_to_predictions(predicted_entities)

    # Summary stats
    total_entities = len(enriched_results)
    label_counts = {}
    for item in enriched_results:
        label_counts[item["label"]] = label_counts.get(item["label"], 0) + 1

    st.subheader("Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Entities", total_entities)
    m2.metric("Unique Labels", len(label_counts))
    m3.metric("Text Length", len(text))

    if label_counts:
        st.write("Label distribution:")
        st.json(label_counts)

    # Highlighted text
    st.subheader("Highlighted Text")
    highlighted_html = highlight_entities(text, predicted_entities)
    st.markdown(
        f"""
        <div style="
            line-height:1.8;
            font-size:1.05rem;
            padding:16px;
            border:1px solid #ddd;
            border-radius:10px;
            background:#fafafa;
            color:#111;
        ">
            {highlighted_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Table view
    st.subheader("Detected Entities")
    if enriched_results:
        table_data = [
            {
                "Entity": item["entity"],
                "Label": item["label"],
                "Description": item["description"],
            }
            for item in enriched_results
        ]
        st.dataframe(table_data, use_container_width=True, hide_index=True)

    # Card view
    st.subheader("Entity Descriptions")
    show_result_cards(enriched_results)

    # Raw JSON
    with st.expander("Show raw JSON output"):
        st.code(build_download_json(enriched_results), language="json")

    st.download_button(
        label="Download JSON result",
        data=build_download_json(enriched_results),
        file_name="political_entity_output.json",
        mime="application/json",
    )