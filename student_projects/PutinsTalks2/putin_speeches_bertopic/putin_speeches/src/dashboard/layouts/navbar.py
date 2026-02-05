"""
Navbar component for the BERTopic Dashboard.
"""

from dash import html
import dash_bootstrap_components as dbc

from config import NAVBAR_STYLE


def create_navbar() -> dbc.Navbar:
    """
    Create the navigation bar component.
    
    Returns:
        Dash Bootstrap Navbar component
    """
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand([
                html.Span("🔍 ", style={"fontSize": "1.3rem"}),
                "BERTopic Explorer"
            ], href="/", className="fs-4 fw-bold"),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Home", href="/", className="text-white")),
            ])
        ], fluid=True),
        color="dark",
        dark=True,
        className="mb-4 shadow",
        style=NAVBAR_STYLE
    )
