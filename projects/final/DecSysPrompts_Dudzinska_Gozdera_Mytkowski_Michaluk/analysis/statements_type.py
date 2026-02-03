"""Functions to determine the type of statements in the prompt analysis."""

def get_statement_class(statement_json):
    """Retrieve the class of the statement: 'constitutive', 'regulative', or 'non-statement'."""
    return statement_json.get("class", "unknown")

def is_constitutive(statement_json):
    """Returns True if the statement is classified as a constitutive statement."""
    return get_statement_class(statement_json) == "constitutive"

def is_regulative(statement_json):
    """Returns True if the statement is classified as a regulative statement."""
    return get_statement_class(statement_json) == "regulative"

def is_non_statement(statement_json):
    """Returns True if the statement is classified as a non-statement."""
    return get_statement_class(statement_json) == "non-statement"

def get_deontic_value(statement_json):
    """Retrieve the deontic value from a regulative statement."""
    if not is_regulative(statement_json):
        return None
    return statement_json.get("regulative_components", {}).get("D","")

def get_aim_value(statement_json):
    """Retrieve the aim value from a regulative statement."""
    if not is_regulative(statement_json):
        return None
    components = statement_json.get("regulative_components", {})
    return components.get("I","")

def classify_regulative_statement(statement_json):
    """Classify a regulative statement into one of the following categories:
    - command (deontic: must, should, has to, will)
    - permission (deontic: may, can)
    - prohibition (deontic: must not, may not, cannot, not)
    - strategy (no deontic value)
    
    Returns the category as a string, or None if the statement is not regulative."""

    if not is_regulative(statement_json):
        return None

    deontic = get_deontic_value(statement_json)
    aim = get_aim_value(statement_json)
    deontic = (deontic or "").lower()
    aim = (aim or "").lower()
    # all deontics in analyzed subset:
    # {'can', 'cannot', 'has to', 'may', 'must', 'should', 'will'}
    if "not" in aim or "not" in deontic or deontic in ["cannot"]:
        return "prohibition"
    if deontic in ["must", "should", "has to", "will"]:
        return "command"
    if deontic in ["may", "can"]:
        return "permission"
    if deontic == "":
        return "strategy"
    return "unknown"

def is_command(statement_json):
    """Returns True if the regulative statement is classified as a command."""
    return classify_regulative_statement(statement_json) == "command"

def is_permission(statement_json):
    """Returns True if the regulative statement is classified as a permission."""
    return classify_regulative_statement(statement_json) == "permission"

def is_prohibition(statement_json):
    """Returns True if the regulative statement is classified as a prohibition."""
    return classify_regulative_statement(statement_json) == "prohibition"

def is_strategy(statement_json):
    """Returns True if the regulative statement is classified as a strategy."""
    return classify_regulative_statement(statement_json) == "strategy"
