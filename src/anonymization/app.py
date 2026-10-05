"""Streamlit demo: detect and anonymize sensitive entities in a piece of text."""

import html
from typing import cast

import pandas as pd
import streamlit as st
from presidio_analyzer import RecognizerResult

from anonymization.anonymizer import OPERATORS, Anonymizer, OperatorName, to_entities
from anonymization.config import LABELS, MODELS, ModelName

EXAMPLE_TEXT = (
    "John Smith works for Apple Inc. as a software engineer. "
    "His email address is john.smith@apple.com and his phone number is 015 29 58 58."
)

PALETTE = [
    "#fde68a", "#bfdbfe", "#bbf7d0", "#fecaca", "#ddd6fe", "#fbcfe8",
    "#a5f3fc", "#fed7aa", "#d9f99d", "#c7d2fe", "#e9d5ff", "#99f6e4",
]
LABEL_COLORS = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(LABELS)}


@st.cache_resource(show_spinner="Loading model...")
def load_anonymizer(model: ModelName, use_rules: bool) -> Anonymizer:
    return Anonymizer(model, use_rules=use_rules)


def _highlight(text: str, results: list[RecognizerResult]) -> str:
    """
    Wrap each detected entity in a coloured <mark> tag followed by its label.

    Args:
        text (str): The text to highlight.
        results (list[RecognizerResult]): The results of the NER detection.

    Returns:
        str: The highlighted text.
    """
    parts, cursor = [], 0
    for r in results:
        color = LABEL_COLORS.get(r.entity_type, "#e5e7eb")
        parts.append(html.escape(text[cursor:r.start]))
        parts.append(
            f'<mark style="background-color:{color}; padding:0.1em 0.3em; border-radius:0.3em;">'
            f"{html.escape(text[r.start:r.end])} "
            f'<b style="font-size:0.75em;">{r.entity_type}</b></mark>'
        )
        cursor = r.end
    parts.append(html.escape(text[cursor:]))
    return "".join(parts).replace("\n", "<br>")


def _entities_table(text: str, results: list[RecognizerResult]) -> pd.DataFrame:
    """
    Create a pandas DataFrame of the detected entities for display in a Streamlit table.

    Args:
        text (str): The text that was analyzed.
        results (list[RecognizerResult]): The results of the NER analysis.

    Returns:
        pd.DataFrame: A pandas DataFrame of the detected entities.
    """
    return pd.DataFrame([e.model_dump() for e in to_entities(results, text)])


st.set_page_config(page_title="Data anonymization", layout="wide")
st.title("Data anonymization")
st.write("Detect sensitive entities in a text with a pre-trained NER model and anonymize them.")

with st.sidebar:
    st.header("Settings")
    models = sorted(MODELS)
    model = cast(ModelName, st.selectbox("Model", models, index=models.index("spacy-lg")))
    use_rules = st.checkbox("Use rule-based recognizers", value=True)
    operators = sorted(OPERATORS)
    operator = cast(
        OperatorName, st.selectbox("Operator", operators, index=operators.index("replace"))
    )

text = st.text_area("Text to anonymize", value=EXAMPLE_TEXT, height=150)

if st.button("Anonymize", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        anonymizer = load_anonymizer(model, use_rules)
        results = anonymizer.detect(text)
        anonymized = anonymizer.anonymize(text, results, operator)

        st.subheader("Anonymized text")
        st.code(anonymized, language=None, wrap_lines=True)

        st.subheader("Detected entities")
        if results:
            st.dataframe(_entities_table(text, results), hide_index=True, width="stretch")
        else:
            st.info("No entities detected.")

        st.subheader("Original text with entities")
        st.markdown(_highlight(text, results), unsafe_allow_html=True)
