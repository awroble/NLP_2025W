"""
Topic analysis section component for the BERTopic Dashboard.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from typing import List, Dict, Any, Optional

from config import TOPIC_SECTION_GRADIENT


def create_topic_analysis_section(
    selector_options: List[Dict[str, Any]],
    selected_topic: int,
    plots_content: Any
) -> html.Div:
    """
    Create the topic analysis section with selector and plots.
    
    Args:
        selector_options: List of options for the topic dropdown
        selected_topic: Currently selected topic number
        plots_content: Content to display in the plots container
    
    Returns:
        Div containing the topic analysis section
    """
    return html.Div([
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
        ], className="mb-4 shadow", style=TOPIC_SECTION_GRADIENT)
    ])


def create_plots_card(ctfidf_graph: dcc.Graph, time_graph: dcc.Graph) -> dbc.Card:
    """
    Create the plots card containing c-TF-IDF and time series graphs.
    
    Args:
        ctfidf_graph: The c-TF-IDF bar chart graph
        time_graph: The time series bar chart graph
    
    Returns:
        Card containing both graphs
    """
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([ctfidf_graph], width=6),
                dbc.Col([time_graph], width=6)
            ])
        ], className="p-2")
    ], className="border-0 shadow-sm", style={
        "backgroundColor": "white",
        "borderRadius": "10px"
    })
