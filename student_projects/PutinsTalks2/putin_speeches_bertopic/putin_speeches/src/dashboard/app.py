"""
BERTopic Dashboard - Interactive topic exploration with Dash

Main application entry point. This file initializes the Dash app,
sets up the main layout with URL routing, and registers all callbacks.

Requirements:
    - dash
    - dash-bootstrap-components
    - plotly
    - pandas
    - bertopic

Usage:
    1. Update data_loader.py with your actual model and data
    2. Run: python app.py
    3. Open: http://localhost:8050
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from callbacks import register_all_callbacks


def create_app() -> dash.Dash:
    """
    Create and configure the Dash application.
    
    Returns:
        Configured Dash application instance
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True
    )
    
    # Main layout with URL routing and session stores
    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="current-documents-store", storage_type="memory"),
        dcc.Store(id="current-topics-store", storage_type="session"),
        dcc.Store(id="last-search-store", storage_type="session"),
        html.Div(id="page-content")
    ])
    
    # Register all callbacks
    register_all_callbacks(app)
    
    return app


# Create the application instance
app = create_app()
server = app.server  # For deployment with gunicorn/uwsgi


if __name__ == "__main__":
    # ==========================================================================
    # IMPORTANT: Before running, update data_loader.py with your actual:
    # - BERTopic model path
    # - Documents list
    # - Document metadata dictionary
    # - Topics over time DataFrame
    # ==========================================================================
    
    app.run(host='0.0.0.0', port=8050)
