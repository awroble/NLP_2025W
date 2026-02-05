import json
import streamlit as st

from services.state import (
    get_current_index,
    get_statements,
    set_current_index,
    update_statement_at,
    delete_statement_at,
)


def make_prev_callback(edited_statement: dict, info_col) -> callable:
    def prev():
        idx = get_current_index()
        if idx > 0:
            update_statement_at(idx, edited_statement)
            set_current_index(idx - 1)
        else:
            info_col.info("You are at the first statement.")
    return prev


def make_next_callback(edited_statement: dict, info_col) -> callable:
    def next_():
        idx = get_current_index()
        statements = get_statements()
        total = len(statements)
        if idx < total - 1:
            update_statement_at(idx, edited_statement)
            set_current_index(idx + 1)
        else:
            info_col.info("You are at the last statement.")
    return next_


def make_delete_callback(edited_statement: dict) -> callable:
    def delete_current():
        statements = get_statements()
        if not statements:
            return

        idx = get_current_index()
        update_statement_at(idx, edited_statement)
        delete_statement_at(idx)

        statements = get_statements()
        if not statements:
            set_current_index(0)
            return
        if idx >= len(statements):
            set_current_index(len(statements) - 1)
    return delete_current


def make_save_all_callback(json_file_path: str) -> callable:
    def save_all():
        try:
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(
                    get_statements(),
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            st.success(f"All statements saved to {json_file_path}.")
        except Exception as e:
            st.error(f"Error saving file: {e}")
    return save_all