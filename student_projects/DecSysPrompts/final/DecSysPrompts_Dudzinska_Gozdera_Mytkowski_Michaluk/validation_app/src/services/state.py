import streamlit as st
from typing import Any


def init_statements_state(json_data: list[dict], json_file_path: str, query_param_index: int = 0) -> None:
    """Initialize / reset session state for a new JSON file."""
    if (
        "last_json_path" not in st.session_state
        or st.session_state.last_json_path != json_file_path
    ):
        st.session_state.current_index = query_param_index
        st.session_state.statements = json_data
        st.session_state.last_json_path = json_file_path

    if "current_index" not in st.session_state:
        st.session_state.current_index = query_param_index
    if "statements" not in st.session_state:
        st.session_state.statements = json_data


def get_current_index() -> int:
    return int(st.session_state.current_index)


def get_statements() -> list[dict]:
    return st.session_state.statements

def set_statements(statements: list[dict]) -> None:
    st.session_state.statements = statements

def set_current_index(idx: int) -> None:
    st.session_state.current_index = idx

def insert_new_statement_after_current():
    """Insert a new empty statement after the current index."""
    statements = get_statements()
    idx = get_current_index()

    new_statement = {
        "anchor": "",
        "full_statement": "",
        "class": "non-statement",
        "constitutive_components": None,
        "regulative_components": None
    }

    # insert after current
    insert_pos = idx + 1
    statements.insert(insert_pos, new_statement)

    # persist changes
    set_statements(statements)

    # move selection to the new statement
    set_current_index(insert_pos)


def update_statement_at(idx: int, new_value: dict) -> None:
    st.session_state.statements[idx] = new_value


def delete_statement_at(idx: int) -> None:
    del st.session_state.statements[idx]