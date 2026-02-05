import spacy
import json
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

embedder = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_trf")

with open('claim_extraction/bbc_news_articles.json', 'r') as f:
    full_articles = json.load(f)

def extract_claim(doc):
    pairs = []

    SUBJECT_DEPS = {
        'nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'expl', 'agent'
    }

    OBJECT_DEPS = {
        'obj', 'iobj', 'attr', 'dobj', 'pobj', 'dative', 'oprd'
    }

    MODIFIER_DEPS = {
        'det', 'amod', 'compound', 'nummod', 'quantmod'
    }

    MODIFIER_DEEP_DEPS = {
        'appos', 'prep', 'poss', 'acl', 'relcl'
    }

    VERB_AUX = {'aux', 'auxpass', 'cop', 'advcl', 'prt'}


    for token in doc:
        if token.dep_ not in SUBJECT_DEPS:
            continue

        # =========================
        # ACTOR
        # =========================
        actor_tokens = {token}

        for child in token.children:
            if child.dep_ in MODIFIER_DEPS:
                actor_tokens.add(child)

            elif child.dep_ in MODIFIER_DEEP_DEPS:
                for t in child.subtree:
                    actor_tokens.add(t)

        actor_tokens = sorted(actor_tokens, key=lambda x: x.i)
        full_actor = " ".join(t.text for t in actor_tokens)

        # =========================
        # ACTION
        # =========================
        actions = []
        head = token.head

        # Verbal predicate
        if head.pos_ == "VERB":
            actions.append(head)

        # Copular / adjectival predicate
        elif head.pos_ in ("ADJ", "NOUN"):
            if any(c.dep_ == "cop" for c in head.children):
                actions.append(head)

        for action in actions:

            # -------------------------
            # VERB PHRASE
            # -------------------------
            verb_tokens = []
            verb_extra = []
            for child in action.children:
                if child.dep_ in VERB_AUX:
                    verb_tokens.append(child)
            for subverb in verb_tokens:
                for child in subverb.children:
                  if child.dep_ in VERB_AUX:
                      verb_tokens.append(child)
            
            verb_tokens.append(action)
            for subverb in verb_tokens:
              for child in subverb.children:
                if child.dep_ == 'prep':
                    verb_extra += [t for t in child.subtree]



            all_verb_tokens = sorted(verb_tokens + verb_extra, key=lambda x: x.i)
            full_verb = " ".join(t.text for t in all_verb_tokens)

            # -------------------------
            # OBJECT
            # -------------------------
            obj_tokens = set()
            objects = []
            for verb in verb_tokens:
                for child in verb.children:

                    if child.dep_ in OBJECT_DEPS:
                        for t in child.subtree:
                            obj_tokens.add(t)

                    elif child.dep_ in ('ccomp', 'xcomp'):
                        clause = sorted(child.subtree, key=lambda x: x.i)
                        objects.append(" ".join(t.text for t in clause))

            if obj_tokens:
                obj_tokens = sorted(obj_tokens, key=lambda x: x.i)
                objects.append(" ".join(t.text for t in obj_tokens))

            full_object = " ".join(objects).strip()

            pairs.append({
                'actor': full_actor,
                'action': full_verb,
                'object': full_object,
                # 'location': " | ".join(locations),
                # 'time': " | ".join(times)
            })

    return pairs


# ---------- Utility ----------

def to_text(x):
    if hasattr(x, "text"):   # spaCy Span / Token / Doc
        return x.text
    return str(x)

def tokenize(text):
    text = to_text(text).lower()
    return set(re.findall(r"\w+", text))

def token_overlap(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)

# ---------- 1. Well-formedness ----------

def well_formedness(claim):
    return all(claim.get(slot) and len(claim[slot].strip()) > 0
               for slot in ["actor", "action"])

# ---------- 2. Schema validity ----------

def has_noun_phrase(text):
    doc = nlp(text)
    return any(tok.pos_ in {"NOUN", "PROPN", "PRON"} for tok in doc)

def has_verb(text):
    doc = nlp(text)
    return any(tok.pos_ == "VERB" for tok in doc)

def schema_valid(claim):
    return (
        has_noun_phrase(claim["actor"]) and
        has_verb(claim["action"]) and
        len(claim["object"].split()) >= 3
    )

# ---------- 3. Faithfulness ----------

def faithfulness_score(claim, sentence):
    sent_tokens = tokenize(sentence)
    scores = []
    for slot in ["actor", "action", "object"]:
        slot_tokens = tokenize(claim[slot])
        scores.append(token_overlap(slot_tokens, sent_tokens))
    return np.mean(scores)

# ---------- 4. Coverage ----------

def coverage_ratio(claims, sentence):
    sent_tokens = tokenize(sentence)
    claim_tokens = set()
    for c in claims:
        for slot in ["actor", "action", "object"]:
            claim_tokens |= tokenize(c[slot])
    return len(claim_tokens & sent_tokens) / max(len(sent_tokens), 1)


# ---------- 5. Redundancy ----------

def redundancy_score(claims):
    if len(claims) < 2:
        return 0.0
    texts = [
        f"{c['actor']} {c['action']} {c['object']}"
        for c in claims
    ]
    embeddings = embedder.encode(texts)
    sims = cosine_similarity(embeddings)
    redundant = 0
    total = 0
    for i in range(len(sims)):
        for j in range(i + 1, len(sims)):
            total += 1
            if sims[i, j] > 0.9:
                redundant += 1
    return redundant / max(total, 1)

def evaluate_instance(instance):
    sentence = instance["sentence"]
    claims = instance["claims"]

    if not claims:
        return {}

    well_formed = [well_formedness(c) for c in claims]
    schema = [schema_valid(c) for c in claims]
    faithfulness = [faithfulness_score(c, sentence) for c in claims]

    return {
        "well_formedness_rate": np.mean(well_formed),
        "schema_validity_rate": np.mean(schema),
        "avg_faithfulness": np.mean(faithfulness),
        "coverage": coverage_ratio(claims, sentence),
        "redundancy": redundancy_score(claims),
        "claims_per_sentence": len(claims)
    }


#text = full_articles[1].get("text")
#doc = nlp(text)
data = [
    {'sentence': sent, 'claims': extract_claim(sent)}
    for article in full_articles
    for sent in nlp(article.get("text")).sents
]
results = [evaluate_instance(d) for d in data if d.get("claims")]
keys = results[0].keys()

results = {
    k: float(np.mean([r[k] for r in results]))
    for k in keys
}

print(results)