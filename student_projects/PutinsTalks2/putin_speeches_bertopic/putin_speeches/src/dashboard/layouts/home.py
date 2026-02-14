"""
Home page layout for the BERTopic Dashboard.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from typing import Dict, Any, Optional

from config import CARD_STYLE, TOPIC_SECTION_GRADIENT
from .navbar import create_navbar


def _create_search_section(initial_value: str = "") -> dbc.Card:
    """
    Create the search input section.
    
    Args:
        initial_value: Pre-filled search value
    
    Returns:
        Card containing the search input
    """
    return dbc.Card([
        dbc.CardBody([
            html.H2("Search Topics", className="text-center mb-4 text-primary"),
            dbc.Row([
                dbc.Col([
                    dbc.InputGroup([
                        dbc.Input(
                            id="search-input",
                            placeholder="Enter search topic...",
                            type="text",
                            size="lg",
                            value=initial_value,
                            className="border-primary"
                        ),
                        dbc.Button(
                            "Search",
                            id="search-button",
                            color="primary",
                            size="lg"
                        )
                    ])
                ], width={"size": 8, "offset": 2})
            ])
        ])
    ], className="mb-4 shadow-sm border-0", style=CARD_STYLE)


def _create_hidden_elements() -> html.Div:
    """
    Create hidden placeholder elements required for callbacks.
    
    Returns:
        Div containing hidden elements
    """
    return html.Div([
        html.Div(id="plots-container"),
        html.Div(id="documents-container"),
        dcc.Dropdown(id="topic-selector", style={"display": "none"})
    ], id="topics-container")


def create_home_layout() -> html.Div:
    """
    Create the basic home page layout without results.
    
    Returns:
        Complete home page layout
    """
    return html.Div([
        create_navbar(),
        dbc.Container([
            _create_search_section(),
            _create_hidden_elements(),
            html.Div(id="document-detail-content", style={"display": "none"})
        ], fluid=True, className="px-4")
    ])


def create_home_layout_with_results(
    topics_store: Dict[str, Any],
    last_search: str,
    selector_options: list,
    selected_topic: int,
    plots_content: Any,
    documents_content: Any
) -> html.Div:
    """
    Create the home page layout with pre-populated search results.
    
    Args:
        topics_store: Dictionary of topic data
        last_search: Previous search query
        selector_options: Options for topic dropdown
        selected_topic: Currently selected topic number
        plots_content: Content for plots container
        documents_content: Content for documents container
    
    Returns:
        Complete home page layout with results
    """
    return html.Div([
        create_navbar(),
        dbc.Container([
            _create_search_section(initial_value=last_search),
            
            # Topic Analysis Section
            html.Div([
                html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(
                                "Topic Analysis",
                                className="text-center mb-4",
                                style={"color": "#4a4a4a", "fontWeight": "600"}
                            ),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label(
                                        "Select Topic:",
                                        className="fw-bold mb-2",
                                        style={"color": "#666"}
                                    ),
                                    dcc.Dropdown(
                                        id="topic-selector",
                                        options=selector_options,
                                        value=selected_topic,
                                        clearable=False,
                                        style={"fontSize": "14px"}
                                    )
                                ], width={"size": 10, "offset": 1})
                            ], className="mb-4"),
                            
                            html.Div(plots_content, id="plots-container")
                        ], style={"backgroundColor": "white"})
                    ], className="border-0", style={"borderRadius": "12px"})
                ], className="mb-4 shadow", style=TOPIC_SECTION_GRADIENT),
                
                html.Hr(className="my-4"),
                
                html.Div(documents_content, id="documents-container")
            ], id="topics-container"),
            
            html.Div(id="document-detail-content", style={"display": "none"})
        ], fluid=True, className="px-4")
    ])
