"""
Configuration constants and styling for the BERTopic Dashboard.
"""

# =============================================================================
# STYLING CONSTANTS
# =============================================================================

NAVBAR_STYLE = {
    "background": "linear-gradient(90deg, #1a1a2e 0%, #16213e 100%)"
}

TOPIC_SECTION_GRADIENT = {
    "borderRadius": "15px",
    "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "padding": "4px"
}

CARD_STYLE = {
    "backgroundColor": "#f8f9fa"
}

# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

PLOT_HEIGHT = 380

PLOT_MARGIN = {
    "t": 50,
    "b": 120,
    "l": 60,
    "r": 20
}

PLOT_COLORS = {
    "ctfidf_colorscale": "Blues",
    "time_bar": "#ff7f0e",
    "border": "rgba(0,0,0,0.3)",
    "grid_light": "rgba(0,0,0,0.05)",
    "grid_medium": "rgba(0,0,0,0.1)"
}

PLOT_TITLE_FONT = {
    "size": 16,
    "color": "#333"
}

# =============================================================================
# DOCUMENT DISPLAY
# =============================================================================

DOCUMENT_PREVIEW_LENGTH = 250
DOCUMENT_LIST_MAX_HEIGHT = "600px"
DOCUMENT_TEXT_MAX_HEIGHT = "500px"

DOCUMENT_TEXT_STYLE = {
    "maxHeight": DOCUMENT_TEXT_MAX_HEIGHT,
    "overflowY": "auto",
    "whiteSpace": "pre-wrap",
    "fontFamily": "Georgia, serif",
    "fontSize": "1.05rem",
    "lineHeight": "1.8",
    "padding": "15px",
    "backgroundColor": "#fafafa",
    "borderRadius": "5px"
}

# =============================================================================
# TOP WORDS CONFIGURATION
# =============================================================================

TOP_WORDS_COUNT = 10
TOP_WORDS_CHART_COUNT = 15
