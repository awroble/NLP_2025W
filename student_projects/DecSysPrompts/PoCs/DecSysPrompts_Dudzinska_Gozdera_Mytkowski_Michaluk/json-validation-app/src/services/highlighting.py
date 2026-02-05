def highlight_in_context(context: str, snippet: str) -> str:
    """Return context with the first occurrence of snippet highlighted."""
    if not snippet:
        return context
    idx = context.find(snippet)
    if idx == -1:
        return context

    before = context[:idx]
    match = context[idx:idx + len(snippet)]
    after = context[idx + len(snippet):]

    highlighted = f"{before}<span style='background-color: #fff59d'>{match}</span>{after}"
    return highlighted