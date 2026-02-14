def load_json(file_path):
    import json
    with open(file_path, 'r') as file:
        return json.load(file)

def load_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def load_data(json_path, text_path):
    json_data = load_json(json_path)
    text_data = load_text(text_path)
    return json_data, text_data