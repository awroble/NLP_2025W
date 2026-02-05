from transformers import pipeline
import torch
import pycountry
import spacy

# A robust mapping for demonyms and common variations
DEMONYM_MAP = {
    "Russian": "Russia", "Soviets": "Russia", "Soviet": "Russia", "USSR": "Russia", "Russian Federation": "Russia",
    "American": "USA", "Americans": "USA", "U.S.": "USA", "US": "USA", "United States": "USA",
    "Ukrainian": "Ukraine", "Ukrainians": "Ukraine",
    "Chinese": "China",
    "French": "France",
    "German": "Germany", "Germans": "Germany",
    "British": "UK", "Britons": "UK", "United Kingdom": "UK",
    "Polish": "Poland", "Poles": "Poland",
    "Belarusian": "Belarus",
    "Syrian": "Syria",
    "Turkish": "Turkey",
    "Iranian": "Iran",
    "Korean": "South Korea",  # Context dependent, but usually safe default in this dataset
    "North Korean": "North Korea",
    "Japanese": "Japan",
}

# Includes formal names ("Iran, Islamic Republic of"), common names ("Iran"), and aliases
VALID_COUNTRIES = set()
for c in pycountry.countries:
    VALID_COUNTRIES.add(c.name)
    if hasattr(c, 'common_name'):
        VALID_COUNTRIES.add(c.common_name)
    if ',' in c.name:  # e.g. "Iran, Islamic Republic of" -> "Iran"
        VALID_COUNTRIES.add(c.name.split(',')[0])

# Add missing common aliases explicitly
VALID_COUNTRIES.update(["Russia", "USA", "UK", "Syria", "Iran", "Turkey", "Vietnam", "South Korea", "North Korea"])

VALID_COUNTRIES.remove('Russian Federation')  # Avoid confusion with demonym

VALID_LANGUAGES = set([l.name for l in pycountry.languages])


def load_ner_pipeline(model_name="dslim/bert-base-NER", device=None):
    """Loads the BERT NER pipeline."""
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    print(f"Loading NER model: {model_name} on device {device}...")
    return pipeline("ner", model=model_name, aggregation_strategy="simple", device=device)


def normalize_entity(text):
    return text.strip().title()


def extract_countries_bert(text, ner_pipeline):
    found_countries = []

    # Process in chunks to handle long text
    chunk_size = 512
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    try:
        results = ner_pipeline(chunks)
        if chunks and isinstance(results[0], dict):
            results = [results]

        # Allow ORG tags to catch things like "Russian Armed Forces"
        ALLOWED_TAGS = ['LOC', 'MISC', 'GPE', 'ORG']

        for chunk_res in results:
            for entity in chunk_res:
                if entity['entity_group'] in ALLOWED_TAGS:

                    raw_word = entity['word'].strip()
                    clean_word = raw_word.replace('.', '').replace(',', '')
                    norm_word = normalize_entity(clean_word)

                    # --- CHECK 1: Is it a Country? ---
                    if norm_word in VALID_COUNTRIES:
                        found_countries.append(norm_word)
                        continue

                    # --- CHECK 2: Is it in our Manual Map? ---
                    # Covers "Soviet", "British", "Americans"
                    if norm_word in DEMONYM_MAP:
                        found_countries.append(DEMONYM_MAP[norm_word])
                        continue

                    # --- CHECK 3: Is it a Language? ---
                    # If pycountry says it's a language (e.g., "German"), we treat it as valid.
                    # We might not know the country 100% (German -> Germany/Austria/Switzerland),
                    # so we default to looking it up in our DEMONYM_MAP.
                    if norm_word in VALID_LANGUAGES:
                        # If we have a specific mapping, use it (German -> Germany)
                        if norm_word in DEMONYM_MAP:
                            found_countries.append(DEMONYM_MAP[norm_word])

                    # --- CHECK 4: Split Multi-word Entities ---
                    # e.g., "Russian Armed Forces" -> checks "Russian"
                    sub_words = clean_word.split()
                    if len(sub_words) > 1:
                        for sub in sub_words:
                            norm_sub = normalize_entity(sub)
                            if norm_sub in DEMONYM_MAP:
                                found_countries.append(DEMONYM_MAP[norm_sub])
                            elif norm_sub in VALID_COUNTRIES:
                                found_countries.append(norm_sub)

    except Exception as e:
        print(f"Error: {e}")

    return found_countries


def load_zero_shot_pipeline(model_name="facebook/bart-large-mnli", device=None):
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    return pipeline("zero-shot-classification", model=model_name, device=device)


def get_hf_model_metadata(pipeline_obj):
    """
    Extracts versioning information from a Hugging Face pipeline for reporting.
    """
    model = pipeline_obj.model
    return {
        "model_name_or_path": getattr(model.config, "_name_or_path", "unknown"),
        "model_architecture": model.config.architectures[0] if model.config.architectures else "unknown",
        "transformers_version": getattr(model.config, "transformers_version", "unknown"),
        # specific commit hash if available (often requires offline=False or specific loading)
        "commit_hash": getattr(model.config, "_commit_hash", "local_files")
    }


def extract_context_sentences(texts, dates, target_terms, spacy_model_name="en_core_web_sm"):
    """
    Splits texts into sentences and filters for target terms, preserving dates.

    Args:
        texts (list): List of full transcript strings.
        dates (list): List of date strings/objects corresponding to texts.
        target_terms (list): List of terms to look for.
    """
    results = []

    try:
        if not spacy.util.is_package(spacy_model_name):
            spacy.cli.download(spacy_model_name)
        nlp = spacy.load(spacy_model_name)
        nlp.disable_pipes(["ner", "tagger", "attribute_ruler", "lemmatizer"])
    except Exception as e:
        print(f"Warning: Could not load spaCy model. Using fallback. {e}")
        nlp = None

    normalized_targets = [t.lower() for t in target_terms]

    # Ensure dates align with texts
    if len(dates) != len(texts):
        print("Warning: Date list length does not match text list length.")

    for i, text in enumerate(texts):
        if not isinstance(text, str):
            continue

        current_date = dates[i] if i < len(dates) else "Unknown"

        if nlp:
            nlp.max_length = len(text) + 1000
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            sentences = text.replace('!', '.').replace('?', '.').split('.')

        for sent in sentences:
            sent_lower = sent.lower()
            for target in normalized_targets:
                if target in sent_lower:
                    results.append({
                        "source_doc_id": i,
                        "date": current_date,  # NEW FIELD
                        "sentence": sent,
                        "found_term": target  # Acts as "Country" identifier
                    })
                    break

    return results


def classify_sentences_batch(pipeline_obj, sentences, candidate_labels, multi_label=False):
    """
    Runs zero-shot classification on a batch of sentences.
    """
    results = pipeline_obj(sentences, candidate_labels, multi_label=multi_label)

    # Normalize output to a list if single input
    if not isinstance(results, list):
        results = [results]

    simplified_results = []
    for res in results:
        simplified_results.append({
            "sequence": res['sequence'],
            "top_label": res['labels'][0],
            "top_score": res['scores'][0],
            "all_scores": dict(zip(res['labels'], res['scores']))
        })
    return simplified_results


def get_accompanying_terms(text, aliases, spacy_model_name="en_core_web_sm"):
    """
    Uses spaCy dependency parsing to find adjectives, compounds, and appositional
    modifiers for any token matching the target entity aliases.
    """
    try:
        if not spacy.util.is_package(spacy_model_name):
            spacy.cli.download(spacy_model_name)
        nlp = spacy.load(spacy_model_name)
        nlp.disable_pipes(["ner", "tagger", "attribute_ruler", "lemmatizer"])
    except Exception as e:
        print(f"Warning: Could not load spaCy model. Using fallback. {e}")
        nlp = None

    doc = nlp(text)
    accompanying_terms = []

    # Define dependency types that provide descriptive context
    DESCRIPTIVE_DEPS = ["amod", "compound", "appos", "poss"]

    for token in doc:
        # Check if the token or its lemma matches any of the target aliases
        if token.text in aliases or token.lemma_ in aliases:
            # Check the token's children for descriptive dependencies
            for child in token.children:
                if child.dep_ in DESCRIPTIVE_DEPS:
                    # Capture the term and normalize it
                    term = child.text.lower().replace("'", "").strip()
                    if term:
                        accompanying_terms.append(term)

    return accompanying_terms
