from typing import Dict, Any
from services.state import update_statement_at
import streamlit as st

def statement_editor(statement: dict, idx: int | None = None) -> dict:
    st.header("Edit Statement")

    orig_full_statement = statement.get("full_statement", "")
    orig_anchor = statement.get("anchor", "")
    orig_class_type = statement.get("class", "non-statement")

    orig_constitutive = statement.get("constitutive_components") or {"E": "", "F": "", "P": ""}
    orig_regulative = statement.get("regulative_components") or {"A": "", "D": "", "I": "", "B": "", "C": ""}

    anchor = st.text_area("Anchor", value=orig_anchor, height=100)
    full_statement = st.text_area("Full Statement", value=orig_full_statement, height=100)

    class_options = ["constitutive", "regulative", "non-statement"]
    class_type = st.selectbox(
        "Class",
        options=class_options,
        index=class_options.index(orig_class_type) if orig_class_type in class_options else 2,
    )

    constitutive_components = None
    regulative_components = None

    if class_type == "constitutive":
        st.subheader("Constitutive components")
        col1, col2 = st.columns(2)
        E = col1.text_input("E (Entity)", value=orig_constitutive.get("E", ""))
        F = col2.text_input("F (Function)", value=orig_constitutive.get("F", ""))
        P = st.text_area("P (Property)", value=orig_constitutive.get("P", ""))
        constitutive_components = {"E": E, "F": F, "P": P}
        regulative_components = None

    elif class_type == "regulative":
        st.subheader("Regulative components")
        col1, col2 = st.columns(2)
        A = col1.text_input("A (Agent)", value=orig_regulative.get("A", ""))
        D = col2.text_input("D (Deontic)", value=orig_regulative.get("D", ""))
        I = col1.text_area("I (aIm)", value=orig_regulative.get("I", ""))
        B = col2.text_area("B (oBject)", value=orig_regulative.get("B", ""))
        C = st.text_area("C (Conditions)", value=orig_regulative.get("C", ""))
        regulative_components = {"A": A, "D": D, "I": I, "B": B, "C": C}
        constitutive_components = None

    else:  # non-statement
        constitutive_components = None
        regulative_components = None

    # Detect changes
    changed = (
        full_statement != orig_full_statement
        or anchor != orig_anchor
        or class_type != orig_class_type
        or constitutive_components != (None if statement.get("constitutive_components") is None else orig_constitutive)
        or regulative_components != (None if statement.get("regulative_components") is None else orig_regulative)
    )

    if changed:
        new_statement = {
            **statement,
            "anchor": anchor,
            "full_statement": full_statement,
            "class": class_type,
            "constitutive_components": constitutive_components,
            "regulative_components": regulative_components,
        }
        update_statement_at(idx, new_statement)
        st.rerun()

    return {
        **statement,
        "anchor": anchor,
        "full_statement": full_statement,
        "class": class_type,
        "constitutive_components": constitutive_components,
        "regulative_components": regulative_components,
    }