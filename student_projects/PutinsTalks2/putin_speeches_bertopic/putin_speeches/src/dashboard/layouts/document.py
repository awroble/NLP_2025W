"""
Document detail page layout for the BERTopic Dashboard.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from typing import Dict, Any, List, Optional

from config import DOCUMENT_TEXT_STYLE, TOP_WORDS_CHART_COUNT
from utils.helpers import (
    format_date,
    parse_topic_words,
    parse_representation,
    calculate_text_statistics
)
from components.figures import create_word_frequency_figure
from components.cards import create_stat_card, create_keyword_badges
from layouts.navbar import create_navbar


def create_document_page_layout() -> html.Div:
    """
    Create the document detail page layout structure.
    
    Returns:
        Document page layout with placeholders
    """
    return html.Div([
        create_navbar(),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [html.I(className="me-2"), "← Back to Documents"],
                        id="back-button",
                        color="outline-secondary",
                        href="/",
                        className="mb-4"
                    )
                ])
            ]),
            dbc.Spinner([
                html.Div(id="document-detail-content")
            ]),
            # Hidden elements for callbacks
            html.Div([
                html.Div(id="plots-container"),
                html.Div(id="documents-container"),
                html.Div(id="topics-container"),
                dbc.Input(id="search-input", style={"display": "none"}),
                dbc.Button(id="search-button", style={"display": "none"}),
                dcc.Dropdown(id="topic-selector", style={"display": "none"})
            ], style={"display": "none"})
        ], fluid=True, className="px-4")
    ])


def create_document_detail_content(doc: Dict[str, Any]) -> html.Div:
    """
    Create the detailed content for a document page.
    
    Args:
        doc: Dictionary containing document data and metadata
    
    Returns:
        Complete document detail content
    """
    # Parse data
    date_str = format_date(doc.get("date"))
    top_words = parse_topic_words(doc.get("Top_n_words"))
    representation = parse_representation(doc.get("Representation"))
    stats = calculate_text_statistics(doc["Document"])
    
    # Create word frequency chart
    word_chart = _create_word_chart(top_words)
    
    # Create representation badges
    rep_badges = create_keyword_badges(representation) if representation else [
        html.Span("N/A", className="text-muted")
    ]
    
    # Build the detail page
    return html.Div([
        _create_header_section(doc, date_str),
        _create_stats_row(doc, stats),
        _create_main_content_row(doc, date_str, rep_badges, word_chart),
        _create_additional_stats_row(stats)
    ])


def _create_word_chart(top_words: List) -> Any:
    """Create word frequency chart or placeholder."""
    if not top_words:
        return html.P("No word data available", className="text-muted")
    
    if isinstance(top_words[0], tuple):
        words = [w[0] for w in top_words[:TOP_WORDS_CHART_COUNT]]
        freqs = [w[1] for w in top_words[:TOP_WORDS_CHART_COUNT]]
    else:
        words = top_words[:TOP_WORDS_CHART_COUNT]
        freqs = list(range(len(words), 0, -1))
    
    fig = create_word_frequency_figure(words, freqs)
    return dcc.Graph(figure=fig)


def _create_header_section(doc: Dict[str, Any], date_str: str) -> dbc.Row:
    """Create the document header section."""
    title = doc.get("title", "Untitled")
    location = doc.get("location", "Unknown location")
    url = doc.get("url", "")
    
    source_link = None
    if url:
        source_link = html.A(
            [html.I(className="me-1"), "View Original Source →"],
            href=url,
            target="_blank",
            className="btn btn-outline-primary btn-sm"
        )
    
    return dbc.Row([
        dbc.Col([
            html.H2(title, className="mb-2"),
            html.Div([
                dbc.Badge(date_str, color="primary", className="me-2 fs-6"),
                dbc.Badge(location, color="secondary", className="me-2 fs-6"),
            ], className="mb-2"),
            html.P(f"Topic: {doc['Name']}", className="text-muted fs-5"),
            source_link
        ])
    ], className="mb-4")


def _create_stats_row(doc: Dict[str, Any], stats: Dict[str, Any]) -> dbc.Row:
    """Create the statistics cards row."""
    representative_text = "✓ Yes" if doc["Representative_document"] else "✗ No"
    representative_class = "text-success" if doc["Representative_document"] else "text-secondary"
    
    return dbc.Row([
        dbc.Col([
            create_stat_card("Relevancy Score", f"{doc['Probability']:.6f}", "text-primary")
        ], width=3),
        dbc.Col([
            create_stat_card("Topic Number", f"{doc['Topic']}", "text-info")
        ], width=3),
        dbc.Col([
            create_stat_card("Word Count", f"{stats['word_count']:,}", "text-success")
        ], width=3),
        dbc.Col([
            create_stat_card("Representative", representative_text, representative_class)
        ], width=3),
    ], className="mb-4")


def _create_main_content_row(
    doc: Dict[str, Any],
    date_str: str,
    rep_badges: List,
    word_chart: Any
) -> dbc.Row:
    """Create the main content row with document text and statistics."""
    location = doc.get("location", "Unknown location")
    url = doc.get("url", "")
    
    # URL display
    url_display = "N/A"
    if url:
        display_url = url[:50] + "..." if len(url) > 50 else url
        url_display = html.A(display_url, href=url, target="_blank")
    
    return dbc.Row([
        # Left: Document Text
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Document Text", className="mb-0")),
                dbc.CardBody([
                    html.Div(doc["Document"], style=DOCUMENT_TEXT_STYLE)
                ])
            ], className="h-100")
        ], width=7),
        
        # Right: Statistics
        dbc.Col([
            # Document Info Card
            dbc.Card([
                dbc.CardHeader(html.H6("Document Information", className="mb-0")),
                dbc.CardBody([
                    html.P([html.Strong("Date: "), date_str], className="mb-2"),
                    html.P([html.Strong("Location: "), location], className="mb-2"),
                    html.P([html.Strong("Source: "), url_display], className="mb-0"),
                ])
            ], className="mb-3"),
            
            # Topic Representation
            dbc.Card([
                dbc.CardHeader(html.H6("Topic Keywords", className="mb-0")),
                dbc.CardBody([
                    html.Div(rep_badges)
                ])
            ], className="mb-3"),
            
            # Word Frequency Chart
            dbc.Card([
                dbc.CardBody([word_chart])
            ]),
        ], width=5)
    ])


def _create_additional_stats_row(stats: Dict[str, Any]) -> dbc.Row:
    """Create the additional statistics row."""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("Additional Statistics", className="mb-0")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Strong("Character Count: "),
                            html.Span(f"{stats['character_count']:,}")
                        ], width=4),
                        dbc.Col([
                            html.Strong("Sentence Count (approx): "),
                            html.Span(f"{stats['sentence_count']}")
                        ], width=4),
                        dbc.Col([
                            html.Strong("Avg Word Length: "),
                            html.Span(f"{stats['avg_word_length']} chars")
                        ], width=4),
                    ])
                ])
            ])
        ])
    ], className="mt-4")


def create_error_content(
    title: str,
    message: str,
    show_home_button: bool = True
) -> html.Div:
    """
    Create error content for document page.
    
    Args:
        title: Error title
        message: Error message
        show_home_button: Whether to show a home button
    
    Returns:
        Error content div
    """
    children = [
        html.H4(title, className="text-center text-muted mt-5"),
        html.P(message, className="text-center")
    ]
    
    if show_home_button:
        children.append(
            dbc.Button(
                "Go to Home",
                href="/",
                color="primary",
                className="d-block mx-auto mt-3"
            )
        )
    
    return html.Div(children)
