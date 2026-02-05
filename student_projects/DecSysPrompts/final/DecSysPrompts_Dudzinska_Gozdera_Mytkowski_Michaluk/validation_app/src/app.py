import streamlit as st
from pathlib import Path

from components.statement_editor import statement_editor
from components.context_viewer import context_viewer
from services.files import STATEMENTS_ROOT, list_statement_files, load_statements_and_context
from services.highlighting import highlight_in_context
from services.state import init_statements_state, get_current_index, get_statements, insert_new_statement_after_current, set_current_index
from services.navigation import (
    make_prev_callback,
    make_next_callback,
    make_delete_callback,
    make_save_all_callback,
)

def main():
    st.set_page_config(layout="wide")
    st.markdown(
            """
## **How to use this tool**

This app is for validation of institutional statements that were automatically extracted and classified using an LLM-based pipeline.

- The text on the left shows the original context with the current statement highlighted (if it can be easily matched, which is not always the case).
- You can edit the statement fields on the right.
- Changes are kept only in temporary memory until you click **Save all**.
- If you switch to a different JSON file *without* clicking **Save all**, all unsaved changes **will be lost**.
- You can add new statements or delete existing ones, but these actions are also temporary until you click **Save all**.
- The **Save all** button is crucial: use it to persist all current statements to the JSON file before changing files.
"""
        )
    json_files = list_statement_files(STATEMENTS_ROOT)
    if not json_files:
        st.error(f"No JSON files found under '{STATEMENTS_ROOT}'.")
        return

    options = [f.relative_to(STATEMENTS_ROOT) for f in json_files]

    query_params = st.query_params
    qp_file = query_params.get("file", [None])
    qp_file = qp_file[0] if isinstance(qp_file, list) else qp_file
    qp_statement_id = query_params.get("statement_id", 0)

    default_index = 0
    if qp_file is not None:
        try:
            default_index = [str(o) for o in options].index(qp_file)
        except ValueError:
            default_index = 0

    selected_rel = st.selectbox(
        "Select JSON file to validate",
        options,
        format_func=lambda x: str(x).replace(".json", ""),
        key="selected_json_file",
        index=default_index,
    )

    if str(selected_rel) != qp_file:
        # Keep URL in sync with current selection
        # This will update URL without needing a manual rerun
        st.query_params["file"] = str(selected_rel)
        qp_statement_id = 0
        set_current_index(0)

    json_file_path = str(Path(STATEMENTS_ROOT) / selected_rel)

    json_data, raw_context = load_statements_and_context(json_file_path)

    init_statements_state(json_data, json_file_path, qp_statement_id)

    statements = get_statements()
    total = len(statements)

    if total == 0:
        st.warning("No statements found in JSON file.")
        return

    idx = get_current_index()
    current_statement = statements[idx]
    anchor = current_statement.get("anchor", "")

    left_col, right_col = st.columns([1, 1])

    with right_col:
        st.subheader(f"Statement {idx + 1} / {total}")

        # row with navigation + add button
        edited_statement = statement_editor(current_statement, idx)

    with left_col:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.button("◀ Previous", on_click=make_prev_callback(edited_statement, col1))
        col2.button("Save all", on_click=make_save_all_callback(json_file_path))
        col3.button("🗑 Delete", on_click=make_delete_callback(edited_statement))
        col5.button("Next ▶", on_click=make_next_callback(edited_statement, col4))

        # Add new statement button
        if col4.button("➕ Add new"):
            insert_new_statement_after_current()
            st.rerun()

        highlighted_context = highlight_in_context(raw_context, anchor)
        context_viewer(highlighted_context)

    query_params["statement_id"] = get_current_index()
        
if __name__ == "__main__":
    main()