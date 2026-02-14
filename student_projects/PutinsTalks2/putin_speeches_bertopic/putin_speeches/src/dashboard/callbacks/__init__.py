"""Callback functions for the BERTopic Dashboard."""

from .routing import register_routing_callbacks
from .search import register_search_callbacks
from .document import register_document_callbacks


def register_all_callbacks(app):
    """
    Register all callbacks with the Dash app.
    
    Args:
        app: The Dash application instance
    """
    register_routing_callbacks(app)
    register_search_callbacks(app)
    register_document_callbacks(app)
