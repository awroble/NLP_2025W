"""
Plotly figure creation functions for the BERTopic Dashboard.
"""

import plotly.graph_objects as go
from typing import List, Tuple, Optional
import pandas as pd

from config import (
    PLOT_HEIGHT, PLOT_MARGIN, PLOT_COLORS, PLOT_TITLE_FONT
)


def create_ctfidf_figure(
    words: List[str],
    scores: List[float],
    max_score: float
) -> go.Figure:
    """
    Create a c-TF-IDF bar chart for topic keywords.
    
    Args:
        words: List of keyword strings
        scores: List of c-TF-IDF scores
        max_score: Maximum score for y-axis scaling
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Bar(
        x=words,
        y=scores,
        marker=dict(
            color=scores,
            colorscale=PLOT_COLORS["ctfidf_colorscale"],
            line=dict(color=PLOT_COLORS["border"], width=1)
        ),
        hovertemplate="<b>%{x}</b><br>c-TF-IDF: %{y:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Keywords</b>",
            font=PLOT_TITLE_FONT,
            x=0.5
        ),
        xaxis=dict(
            title="",
            tickangle=-45,
            tickfont=dict(size=11),
            gridcolor=PLOT_COLORS["grid_light"]
        ),
        yaxis=dict(
            title="c-TF-IDF Score",
            title_font=dict(size=12, color="#666"),
            range=[0, max_score],
            gridcolor=PLOT_COLORS["grid_medium"],
            tickfont=dict(size=10)
        ),
        height=PLOT_HEIGHT,
        margin=PLOT_MARGIN,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    
    return fig


def create_time_series_figure(
    timestamps: pd.Series,
    frequencies: pd.Series,
    max_frequency: float,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
) -> go.Figure:
    """
    Create a time series bar chart for topic frequency over time.
    
    Args:
        timestamps: Series of datetime values
        frequencies: Series of frequency values
        max_frequency: Maximum frequency for y-axis scaling
        date_range: Optional (min_date, max_date) tuple for x-axis range
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Bar(
        x=timestamps if not timestamps.empty else [],
        y=frequencies if not frequencies.empty else [],
        marker=dict(
            color=PLOT_COLORS["time_bar"],
            line=dict(color=PLOT_COLORS["border"], width=1)
        ),
        hovertemplate="<b>%{x|%B %Y}</b><br>Frequency: %{y}<extra></extra>"
    ))
    
    x_range = None
    if date_range and date_range[0] and date_range[1]:
        x_range = list(date_range)
    
    fig.update_layout(
        title=dict(
            text="<b>Frequency Over Time</b>",
            font=PLOT_TITLE_FONT,
            x=0.5
        ),
        xaxis=dict(
            title="",
            tickfont=dict(size=11),
            gridcolor=PLOT_COLORS["grid_light"],
            range=x_range
        ),
        yaxis=dict(
            title="Frequency",
            title_font=dict(size=12, color="#666"),
            range=[0, max_frequency],
            gridcolor=PLOT_COLORS["grid_medium"],
            tickfont=dict(size=10)
        ),
        height=PLOT_HEIGHT,
        margin=PLOT_MARGIN,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    
    return fig


def create_word_frequency_figure(
    words: List[str],
    frequencies: List[float]
) -> go.Figure:
    """
    Create a horizontal bar chart for word frequencies.
    
    Args:
        words: List of words
        frequencies: List of frequency/score values
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Bar(
        x=frequencies,
        y=words,
        orientation='h',
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        title="Top Words in Document",
        xaxis_title="Score/Frequency",
        yaxis_title="",
        height=350,
        margin=dict(l=120, r=20, t=50, b=40),
        yaxis=dict(autorange="reversed")
    )
    
    return fig
