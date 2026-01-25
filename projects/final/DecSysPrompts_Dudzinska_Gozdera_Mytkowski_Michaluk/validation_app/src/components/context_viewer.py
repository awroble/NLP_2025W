import streamlit as st

def context_viewer(context: str) -> None:
    st.header("Context Viewer")
    with st.container(height=700):
        st.markdown(
            f"{context}",
            unsafe_allow_html=True
        )
    
