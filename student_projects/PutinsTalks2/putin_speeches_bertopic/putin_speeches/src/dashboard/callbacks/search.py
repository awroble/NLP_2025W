"""
Search-related callbacks for the BERTopic Dashboard.
"""

from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from config import CARD_STYLE, TOPIC_SECTION_GRADIENT
from layouts.navbar import create_navbar
from services import search_topics, generate_plots_and_documents


def register_search_callbacks(app):
    """
    Register search-related callbacks.
    
    Args:
        app: The Dash application instance
    """
    
    @app.callback(
        Output("page-content", "children", allow_duplicate=True),
        Output("current-topics-store", "data"),
        Output("last-search-store", "data"),
        Output("current-documents-store", "data", allow_duplicate=True),
        Input("search-button", "n_clicks"),
        State("search-input", "value"),
        prevent_initial_call=True
    )
    def handle_search(n_clicks, search_query):
        """
        Handle search button clicks.
        
        Args:
            n_clicks: Number of button clicks
            search_query: Search input value
        
        Returns:
            Tuple of (page_content, topics_store, search_query, docs_store)
        """
        if not search_query:
            return no_update, no_update, no_update, no_update
        
        # Search for topics
        topics_data = search_topics(search_query)
        
        if not topics_data:
            return no_update, {}, search_query, []
        
        # Create store data
        store_data = {str(t["topic_no"]): t for t in topics_data}
        
        # Generate initial content for first topic
        selected_topic = topics_data[0]["topic_no"]
        plots_content, documents_content, docs_store = generate_plots_and_documents(
            selected_topic, store_data
        )
        
        # Create selector options
        selector_options = [
            {
                "label": f"Topic {t['topic_no']}: {t['name']} (similarity: {t['similarity']:.3f})",
                "value": t["topic_no"]
            }
            for t in topics_data
        ]
        
        # Build page content
        page_content = _create_search_results_page(
            search_query=search_query,
            selector_options=selector_options,
            selected_topic=selected_topic,
            plots_content=plots_content,
            documents_content=documents_content
        )
        
        return page_content, store_data, search_query, docs_store
    
    @app.callback(
        Output("plots-container", "children"),
        Output("documents-container", "children"),
        Output("current-documents-store", "data", allow_duplicate=True),
        Input("topic-selector", "value"),
        State("current-topics-store", "data"),
        prevent_initial_call=True
    )
    def update_topic_content(selected_topic, topics_store):
        """
        Update plots and documents when topic selection changes.
        
        Args:
            selected_topic: Selected topic number
            topics_store: Stored topic data
        
        Returns:
            Tuple of (plots_content, documents_content, docs_store)
        """
        if selected_topic is None or not topics_store:
            return no_update, no_update, no_update
        
        plots_content, documents_content, docs_store = generate_plots_and_documents(
            selected_topic, topics_store
        )
        
        if plots_content is None:
            return no_update, no_update, no_update
        
        return plots_content, documents_content, docs_store


def _create_search_results_page(
    search_query: str,
    selector_options: list,
    selected_topic: int,
    plots_content,
    documents_content
) -> html.Div:
    """
    Create the search results page layout.
    
    Args:
        search_query: The search query
        selector_options: Options for topic dropdown
        selected_topic: Currently selected topic
        plots_content: Plots card content
        documents_content: Documents list content
    
    Returns:
        Complete page layout
    """
    return html.Div([
        create_navbar(),
        dbc.Container([
            # Search Input Section
            dbc.Card([
                dbc.CardBody([
                    html.H2(
                        "Search Topics",
                        className="text-center mb-4 text-primary"
                    ),
                    dbc.Row([
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.Input(
                                    id="search-input",
                                    placeholder="Enter search topic...",
                                    type="text",
                                    size="lg",
                                    value=search_query,
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
            ], className="mb-4 shadow-sm border-0", style=CARD_STYLE),
            
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
            
            # Hidden element for document page callback
            html.Div(id="document-detail-content", style={"display": "none"})
        ], fluid=True, className="px-4")
    ])
