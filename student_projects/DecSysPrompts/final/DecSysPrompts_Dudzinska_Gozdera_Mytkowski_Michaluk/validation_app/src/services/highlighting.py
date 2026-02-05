import html

def highlight_in_context(context: str, snippet: str) -> str:
    """Return context with the first occurrence of snippet highlighted, safely handling < and >."""
    if not snippet:
        return html.escape(context)

    snippet = snippet.replace("<", "\<")
    snippet = snippet.replace(">", "\>")
    idx = context.find(snippet)
    if idx == -1:
        return html.escape(context)

    before = html.escape(context[:idx])
    match = html.escape(context[idx:idx + len(snippet)])
    after = html.escape(context[idx + len(snippet):])

    highlighted = f"{before}<span style='background-color: #fff59d'>{match}</span>{after}"
    return highlighted
