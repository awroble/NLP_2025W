
"""Indicators calculations for prompts analysis."""

# pylint: disable=locally-disabled, invalid-name

from statements_type import is_command, is_constitutive, is_regulative, is_non_statement, is_permission, is_prohibition, is_strategy

def S(text):
    """The size of a prompt (S): number of characters in the prompt"""
    return len(text)

def NCtW(text, text_no_code):
    """The proportion of not-code related part of the prompt to the whole prompt (NCtW):
    ratio of non-code related instructions (in number of characters)
    to whole prompt (in number of characters)."""
    return len(text_no_code) / len(text) if len(text) > 0 else 0

def NStNC(statements_json):
    """The proportion of non-statement sentences to all not-code related sentences (NStNC):
    ratio of non-statement sentences (number of sentences)
    to all not-code related sentences (number of sentences)."""
    non_statement_sentences = sum(1 for stmt in statements_json if is_non_statement(stmt))
    total_non_code_sentences = len(statements_json)
    return non_statement_sentences / total_non_code_sentences if total_non_code_sentences > 0 else 0

def CStA(statements_json):
    """The proportion of constitutive statements to all statements (CStA):
    ratio of constitutive statements (number of sentences)
    to all statements (number of sentences)."""
    constitutive_statements = sum(1 for stmt in statements_json if is_constitutive(stmt))
    total_statements = len(statements_json)
    return constitutive_statements / total_statements if total_statements > 0 else 0

def StRS(statements_json):
    """The proportion of strategies to all regulative statements (StRS):
    ratio of strategies (number of sentences) to all regulative statements (number of sentences).
    Strategies are suggestions about the agent behavior. Usually they have no deontic value."""
    strategies = sum(1 for stmt in statements_json if is_strategy(stmt))
    total_regulative_statements = sum(1 for stmt in statements_json if is_regulative(stmt))
    return strategies / total_regulative_statements if total_regulative_statements > 0 else 0

def CtRS(statements_json):
    """The proportion of commands to all regulative statements (CtRS):
    ratio of commands (number of sentences) to all regulative statements
    (number of sentences). Commands are regulative statements that use
    deontic with value “must” or “should”."""
    commands = sum(1 for stmt in statements_json if is_command(stmt))
    total_regulative_statements = sum(1 for stmt in statements_json if is_regulative(stmt))
    return commands / total_regulative_statements if total_regulative_statements > 0 else 0

def PRtRS(statements_json):
    """ The proportion of permissions to all regulative statements (PRtRS):
    ratio of permissions (number of sentences) to all regulative
    statements (number of sentences). Permissions are regulative statements
    that use deontic with value “may” or “can”."""
    permissions = sum(1 for stmt in statements_json if is_permission(stmt))
    total_regulative_statements = sum(1 for stmt in statements_json if is_regulative(stmt))
    return permissions / total_regulative_statements if total_regulative_statements > 0 else 0

def PHtRS(statements_json):
    """The proportion of prohibitions to all regulative statements (PHtRS):
    ratio of prohibitions (number of sentences) to all regulative statements (number of sentences). 
    Prohibitions are regulative statements that use deontic with value “must not” or “may not”"""
    prohibitions = sum(1 for stmt in statements_json if is_prohibition(stmt))
    total_regulative_statements = sum(1 for stmt in statements_json if is_regulative(stmt))
    return prohibitions / total_regulative_statements if total_regulative_statements > 0 else 0

def get_regulative_components(statements_json):
    """Extract all regulative components from statements.
    Returns a list of regulative components (non-null values)."""
    regulative_components = [stmt['regulative_components'] for stmt in statements_json 
                            if stmt.get('regulative_components') is not None]
    return regulative_components