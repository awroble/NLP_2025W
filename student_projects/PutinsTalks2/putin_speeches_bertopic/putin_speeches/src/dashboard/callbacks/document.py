"""
Document detail page callbacks for the BERTopic Dashboard.
"""

from dash import Input, Output, State, no_update

from layouts.document import create_document_detail_content, create_error_content
from services import get_document_by_id


def register_document_callbacks(app):
    """
    Register document detail page callbacks.

    Args:
        app: The Dash application instance
    """

    @app.callback(
        Output("document-detail-content", "children"),
        Input("url", "pathname"),
        State("current-documents-store", "data")
    )
    def load_document_detail(pathname, docs_store):
        """
        Load document detail content when navigating to a document page.

        Args:
            pathname: Current URL path
            docs_store: List of document references from session

        Returns:
            Document detail content or error content
        """
        if not pathname or not pathname.startswith("/document/"):
            return no_update

        try:
            # Extract document index from URL
            doc_idx_str = pathname.split("/document/")[-1]
            doc_idx = int(doc_idx_str)

            # Check if we have session data
            if not docs_store:
                return create_error_content(
                    "Session Expired",
                    "Please go back to the home page and search again."
                )

            # Get the actual doc_id from the store
            if doc_idx < 0 or doc_idx >= len(docs_store):
                return create_error_content(
                    "Document Not Found",
                    f"Document index {doc_idx} is out of range."
                )

            doc_id = docs_store[doc_idx]["doc_id"]

            # Fetch document data
            doc = get_document_by_id(doc_id)

            # Create detail content
            return create_document_detail_content(doc)

        except (ValueError, IndexError, KeyError, TypeError) as e:
            return create_error_content(
                "Invalid Document",
                f"Could not load document: {str(e)}"
            )
        except Exception as e:
            return create_error_content(
                "Error",
                f"An unexpected error occurred: {str(e)}"
            )
