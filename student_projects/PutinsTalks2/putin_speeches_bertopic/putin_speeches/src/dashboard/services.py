"""
Business logic services for generating content.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from config import (
    DOCUMENT_LIST_MAX_HEIGHT,
    TOP_WORDS_COUNT
)
from data_loader import (
    topic_model,
    list_of_speeches,
    get_document_metadata,
    get_topic_time_data,
    get_all_topic_time_data
)
from components.figures import create_ctfidf_figure, create_time_series_figure
from components.cards import create_document_card
from utils.helpers import truncate_text


def calculate_axis_limits(topics_store: Dict[str, Any]) -> Tuple[float, float, Any, Any]:
    """
    Calculate fixed axis limits for consistent plots across topics.
    
    Args:
        topics_store: Dictionary of topic data
    
    Returns:
        Tuple of (max_ctfidf, max_frequency, min_date, max_date)
    """
    # Calculate c-TF-IDF max
    all_scores = []
    for t in topics_store.values():
        all_scores.extend([w[1] for w in t.get("words", [])])
    max_ctfidf = max(all_scores) * 1.1 if all_scores else 1
    
    # Calculate time plot limits
    all_topic_nos = [int(t) for t in topics_store.keys()]
    time_data_all = get_all_topic_time_data(all_topic_nos)
    
    max_frequency = time_data_all["Frequency"].max() * 1.1 if not time_data_all.empty else 10
    min_date = time_data_all["Timestamp"].min() if not time_data_all.empty else None
    max_date = time_data_all["Timestamp"].max() if not time_data_all.empty else None
    
    return max_ctfidf, max_frequency, min_date, max_date


def generate_plots_content(
    topic_data: Dict[str, Any],
    selected_topic: int,
    max_ctfidf: float,
    max_frequency: float,
    date_range: Tuple[Any, Any]
) -> dbc.Card:
    """
    Generate the plots card content for a topic.
    
    Args:
        topic_data: Data for the selected topic
        selected_topic: The topic number
        max_ctfidf: Maximum c-TF-IDF score for y-axis
        max_frequency: Maximum frequency for y-axis
        date_range: (min_date, max_date) tuple
    
    Returns:
        Card containing both plots
    """
    # Create c-TF-IDF figure
    words = [w[0] for w in topic_data["words"]]
    scores = [w[1] for w in topic_data["words"]]
    ctfidf_fig = create_ctfidf_figure(words, scores, max_ctfidf)
    
    # Create time series figure
    topic_time_data = get_topic_time_data(selected_topic)
    time_fig = create_time_series_figure(
        topic_time_data["Timestamp"] if not topic_time_data.empty else pd.Series(),
        topic_time_data["Frequency"] if not topic_time_data.empty else pd.Series(),
        max_frequency,
        date_range
    )
    
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=ctfidf_fig,
                        id="topics-graph",
                        config={'displayModeBar': False}
                    )
                ], width=6),
                dbc.Col([
                    dcc.Graph(
                        figure=time_fig,
                        id="topics-time-graph",
                        config={'displayModeBar': False}
                    )
                ], width=6)
            ])
        ], className="p-2")
    ], className="border-0 shadow-sm", style={
        "backgroundColor": "white",
        "borderRadius": "10px"
    })


def generate_documents_content(
    selected_topic: int,
    topic_name: str
) -> Tuple[html.Div, List[Dict[str, Any]]]:
    """
    Generate the documents list content for a topic.
    
    Args:
        selected_topic: The topic number
        topic_name: The topic name/label
    
    Returns:
        Tuple of (documents_content, docs_store_data)
    """
    # Get documents for selected topic
    doc_info = topic_model.get_document_info(list_of_speeches)
    doc_info["doc_id"] = doc_info.index
    topic_docs = doc_info[doc_info["Topic"] == selected_topic].copy()
    
    # Add metadata
    topic_docs["title"] = topic_docs["doc_id"].apply(
        lambda x: get_document_metadata(x, "title", "Untitled")
    )
    topic_docs["location"] = topic_docs["doc_id"].apply(
        lambda x: get_document_metadata(x, "location", "Unknown")
    )
    topic_docs["url"] = topic_docs["doc_id"].apply(
        lambda x: get_document_metadata(x, "url", "")
    )
    topic_docs["date"] = topic_docs["doc_id"].apply(
        lambda x: get_document_metadata(x, "date", None)
    )
    
    # Extract year for sorting
    topic_docs["year"] = pd.to_datetime(
        topic_docs["date"], errors="coerce"
    ).dt.year
    
    # Sort by year (descending) then by relevancy/probability (descending)
    topic_docs = topic_docs.sort_values(
        by=["year", "Probability"],
        ascending=[False, False]
    )
    
    # Create document cards
    doc_cards = []
    for idx, (_, row) in enumerate(topic_docs.iterrows()):
        preview_text = truncate_text(row["Document"])
        
        card = create_document_card(
            idx=idx,
            title=row["title"],
            date=row["date"],
            location=row["location"],
            preview_text=preview_text,
            probability=row["Probability"],
            is_representative=row["Representative_document"]
        )
        doc_cards.append(card)
    
    # Store document IDs for detail page
    docs_store = topic_docs[["doc_id"]].reset_index(drop=True).to_dict("records")
    
    documents_content = html.Div([
        html.H4(
            f"Documents in Topic {selected_topic}: {topic_name}",
            className="mb-2"
        ),
        html.P(
            f"{len(topic_docs)} documents found, sorted by year and relevancy",
            className="text-muted mb-4"
        ),
        html.Div(
            doc_cards,
            style={
                "maxHeight": DOCUMENT_LIST_MAX_HEIGHT,
                "overflowY": "auto",
                "padding": "10px"
            }
        )
    ])
    
    return documents_content, docs_store


def generate_plots_and_documents(
    selected_topic: int,
    topics_store: Dict[str, Any]
) -> Tuple[Optional[dbc.Card], Optional[html.Div], Optional[List[Dict[str, Any]]]]:
    """
    Generate both plots and documents content for a selected topic.
    
    Args:
        selected_topic: The topic number
        topics_store: Dictionary of topic data
    
    Returns:
        Tuple of (plots_content, documents_content, docs_store)
    """
    # Get topic data
    topic_data = topics_store.get(str(selected_topic))
    if not topic_data:
        topic_data = topics_store.get(selected_topic)
        if not topic_data:
            return None, None, None
    
    # Calculate axis limits
    max_ctfidf, max_frequency, min_date, max_date = calculate_axis_limits(topics_store)
    
    # Generate plots
    plots_content = generate_plots_content(
        topic_data,
        selected_topic,
        max_ctfidf,
        max_frequency,
        (min_date, max_date)
    )
    
    # Generate documents
    topic_name = topic_data.get("name", f"Topic {selected_topic}")
    documents_content, docs_store = generate_documents_content(
        selected_topic,
        topic_name
    )
    
    return plots_content, documents_content, docs_store


def search_topics(search_query: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Search for topics matching a query.
    
    Args:
        search_query: The search term
        top_n: Number of top topics to return
    
    Returns:
        List of topic data dictionaries
    """
    topic_numbers, similarities = topic_model.find_topics(search_query, top_n=top_n)
    
    topics_data = []
    for topic_no, similarity in zip(topic_numbers, similarities):
        topic_words = topic_model.get_topic(topic_no)
        if topic_words:
            words_list = [(word, score) for word, score in topic_words[:TOP_WORDS_COUNT]]
            try:
                topic_name = topic_model.get_topic_info(topic_no)["Name"].values[0]
            except Exception:
                topic_name = f"Topic {topic_no}"
            
            topics_data.append({
                "topic_no": topic_no,
                "similarity": similarity,
                "words": words_list,
                "name": topic_name
            })
    
    return topics_data


def get_document_by_id(doc_id: int) -> Dict[str, Any]:
    """
    Fetch complete document data by ID.
    
    Args:
        doc_id: Document index
    
    Returns:
        Dictionary with all document fields
    """
    doc_info = topic_model.get_document_info(list_of_speeches)
    doc_row = doc_info.iloc[doc_id]
    
    return {
        "Document": doc_row["Document"],
        "Topic": doc_row["Topic"],
        "Name": doc_row["Name"],
        "Probability": doc_row["Probability"],
        "Representative_document": doc_row["Representative_document"],
        "Top_n_words": doc_row["Top_n_words"],
        "Representation": doc_row["Representation"],
        "title": get_document_metadata(doc_id, "title", "Untitled"),
        "location": get_document_metadata(doc_id, "location", "Unknown"),
        "url": get_document_metadata(doc_id, "url", ""),
        "date": get_document_metadata(doc_id, "date", None)
    }
