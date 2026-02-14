"""
Card and selector UI components for the BERTopic Dashboard.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from typing import List, Dict, Any, Optional

from utils.helpers import format_date, truncate_text


def create_document_card(
    idx: int,
    title: str,
    date: Any,
    location: str,
    preview_text: str,
    probability: float,
    is_representative: bool
) -> dbc.Card:
    """
    Create a document preview card.
    
    Args:
        idx: Document index for the detail page link
        title: Document title
        date: Document date
        location: Document location
        preview_text: Truncated preview of document content
        probability: Relevancy probability score
        is_representative: Whether this is a representative document
    
    Returns:
        Dash Bootstrap Card component
    """
    date_str = format_date(date)
    
    representative_badge = None
    if is_representative:
        representative_badge = dbc.Badge(
            "★ Representative",
            color="success",
            className="mt-2"
        )
    
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5(title, className="card-title mb-1"),
                    html.Div([
                        dbc.Badge(date_str, color="secondary", className="me-2"),
                        dbc.Badge(
                            location,
                            color="light",
                            text_color="dark",
                            className="me-2"
                        ),
                    ], className="mb-2"),
                    html.P(
                        preview_text,
                        className="card-text text-muted small mb-2"
                    ),
                    dcc.Link(
                        dbc.Button("View Details →", color="primary", size="sm"),
                        href=f"/document/{idx}"
                    )
                ], width=9),
                dbc.Col([
                    html.Div([
                        html.Span("Relevancy", className="small text-muted d-block"),
                        html.H4(
                            f"{probability:.4f}",
                            className="text-primary mb-1"
                        )
                    ], className="text-end"),
                    representative_badge
                ], width=3, className="d-flex flex-column align-items-end justify-content-center")
            ])
        ])
    ], className="mb-3 shadow-sm")


def create_stat_card(
    title: str,
    value: str,
    color_class: str = "text-primary"
) -> dbc.Card:
    """
    Create a statistics card.
    
    Args:
        title: Card title/label
        value: Display value
        color_class: Bootstrap text color class
    
    Returns:
        Dash Bootstrap Card component
    """
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="text-muted mb-2"),
            html.H2(value, className=f"{color_class} mb-0")
        ])
    ], className="text-center h-100")


def create_topic_selector(
    topics_data: List[Dict[str, Any]],
    selected_value: Optional[int] = None
) -> dcc.Dropdown:
    """
    Create a topic selector dropdown.
    
    Args:
        topics_data: List of topic dictionaries with topic_no, name, similarity
        selected_value: Initially selected topic number
    
    Returns:
        Dash Dropdown component
    """
    options = [
        {
            "label": f"Topic {t['topic_no']}: {t['name']} (similarity: {t['similarity']:.3f})",
            "value": t["topic_no"]
        }
        for t in topics_data
    ]
    
    return dcc.Dropdown(
        id="topic-selector",
        options=options,
        value=selected_value,
        clearable=False,
        style={"fontSize": "14px"}
    )


def create_keyword_badges(
    words: List[str],
    max_words: int = 10,
    color: str = "info"
) -> List[dbc.Badge]:
    """
    Create a list of keyword badges.
    
    Args:
        words: List of words
        max_words: Maximum number of badges to create
        color: Badge color
    
    Returns:
        List of Badge components
    """
    return [
        dbc.Badge(word, color=color, className="me-1 mb-1")
        for word in words[:max_words]
    ]
