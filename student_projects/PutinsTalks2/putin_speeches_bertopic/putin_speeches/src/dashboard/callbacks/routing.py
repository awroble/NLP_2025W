"""
URL routing callbacks for the BERTopic Dashboard.
"""

from dash import callback, Input, Output, State

from layouts.home import create_home_layout, create_home_layout_with_results
from layouts.document import create_document_page_layout
from services import generate_plots_and_documents


def register_routing_callbacks(app):
    """
    Register URL routing callbacks.
    
    Args:
        app: The Dash application instance
    """
    
    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
        State("current-topics-store", "data"),
        State("last-search-store", "data"),
        State("current-documents-store", "data")
    )
    def display_page(pathname, topics_store, last_search, docs_store):
        """
        Route to the appropriate page based on URL pathname.
        
        Args:
            pathname: Current URL path
            topics_store: Stored topic data
            last_search: Last search query
            docs_store: Stored document data
        
        Returns:
            Page layout component
        """
        # Document detail page
        if pathname and pathname.startswith("/document/"):
            return create_document_page_layout()
        
        # Home page with existing results
        if topics_store and last_search:
            try:
                # Validate structure
                if all(
                    "similarity" in t and "topic_no" in t
                    for t in topics_store.values()
                ):
                    return _create_restored_home_layout(
                        topics_store, last_search
                    )
            except (TypeError, KeyError, AttributeError):
                pass  # Fall through to default layout
        
        # Default home page
        return create_home_layout()


def _create_restored_home_layout(topics_store, last_search):
    """
    Create home layout with restored results from session.
    
    Args:
        topics_store: Stored topic data
        last_search: Last search query
    
    Returns:
        Home layout with pre-populated results
    """
    topics_data = list(topics_store.values())
    topics_data.sort(key=lambda x: x["similarity"], reverse=True)
    
    selector_options = [
        {
            "label": f"Topic {t['topic_no']}: {t['name']} (similarity: {t['similarity']:.3f})",
            "value": t["topic_no"]
        }
        for t in topics_data
    ]
    
    selected_topic = topics_data[0]["topic_no"]
    plots_content, documents_content, _ = generate_plots_and_documents(
        selected_topic, topics_store
    )
    
    return create_home_layout_with_results(
        topics_store=topics_store,
        last_search=last_search,
        selector_options=selector_options,
        selected_topic=selected_topic,
        plots_content=plots_content,
        documents_content=documents_content
    )
