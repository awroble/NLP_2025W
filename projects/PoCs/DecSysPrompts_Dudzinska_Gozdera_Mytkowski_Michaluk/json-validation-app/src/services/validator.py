def validate_json(json_data):
    errors = []
    
    for statement in json_data:
        if 'full_statement' not in statement or not statement['full_statement']:
            errors.append("Full statement is required.")
        
        if 'class' not in statement or statement['class'] not in ['regulative', 'constitutive']:
            errors.append("Class must be either 'regulative' or 'constitutive'.")
        
        if 'deontic' not in statement:
            errors.append("Deontic must be present.")
        
        if 'action' not in statement or not statement['action']:
            errors.append("Action is required.")
        
        if 'conditions' not in statement:
            statement['conditions'] = ""  # Ensure conditions key exists
        
    return errors

def is_valid_json(json_data):
    errors = validate_json(json_data)
    return len(errors) == 0, errors