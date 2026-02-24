import os
from huggingface_hub import InferenceClient
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForCausalLM
from transformers import pipeline
import re
import torch
from prompts import PROPAGANDA_CLASSIFICATION, HISTORICAL_PROPAGANDA_CLASSIFICATION


def preprocessing():
    df = pd.read_csv('clean_data.csv', )
    df = df[df['word_count'] > 5]
    return df[['text', 'id']]


def split_sentences(text):
    max_len = 512
    if len(text) <= max_len:
        return [text]

    sentences = re.findall(r'[^.!?]+[.!?]?', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if len(current_chunk) + len(sentence) + 1 > max_len:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                splitted = sentence.split()
                half = len(splitted)//2
                chunks.append(" ".join(splitted[:half]))
                chunks.append(" ".join(splitted[half:]))
                current_chunk = " ".join(splitted[half:])
        else:
            current_chunk += " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def split_sentences_in_df(df):
    df["text"] = df["text"].apply(split_sentences)
    df = df.explode("text").reset_index(drop=True)
    return df.drop_duplicates()

def ner(df):
    client = InferenceClient(
        provider="hf-inference",
        api_key=os.environ["HF_TOKEN"],
    )

    results = []
    for i, row in df.iterrows():
        print(f"{i+1}/{df.shape[0]}")
        try:
            post_id = row['id']
            text = row['text']

            result = client.token_classification(
                text,
                model="dslim/bert-base-NER",
            )

            for entity in result:
                results.append({
                    'id': post_id,
                    'name': entity['word'],
                    'type': entity['entity_group']
                })
        except:
            result_df = pd.DataFrame(results)
            result_df.to_csv('ner_part.csv', index=False)
    result_df = pd.DataFrame(results)
    result_df.to_csv('ner.csv', index=False)


def manipulation_detection_hist(df):
    results = []
    df.reset_index(drop=True, inplace=True)
    tokenizer = AutoTokenizer.from_pretrained("propaganda-detection-historical-context")
    model = AutoModelForSequenceClassification.from_pretrained("propaganda-detection-historical-context")
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    for i, row in df.iterrows():
        print(f"{i + 1}/{df.shape[0]}")

        post_id = row['id']
        text = row['text']

        result = classifier(text)
        for score in result:
            results.append({
                'id': post_id,
                **score
            })
    result_df = pd.DataFrame(results)
    result_df.to_csv('propaganda1.csv', index=False)

def manipulation_detection(df):
    results = []
    df.reset_index(drop=True, inplace=True)

    tokenizer = AutoTokenizer.from_pretrained("PropagandaDetection")
    model = AutoModelForSequenceClassification.from_pretrained("PropagandaDetection")

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    for i, row in df.iterrows():
        print(f"{i + 1}/{df.shape[0]}")

        post_id = row['id']
        text = row['text']

        result = classifier(text)
        for score in result:
            results.append({
                'id': post_id,
                **score
            })


    result_df = pd.DataFrame(results)
    result_df.to_csv('propaganda2.csv', index=False)


def manipulation_detection_with_llm(df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results = []

    for i, row in df.iterrows():
        post_id = row['id']
        text = row['text']

        inputs = tokenizer(PROPAGANDA_CLASSIFICATION.format(text), return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=10)
        prop = tokenizer.decode(outputs[0], skip_special_tokens=True)

        inputs = tokenizer(HISTORICAL_PROPAGANDA_CLASSIFICATION.format(text), return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=10)
        hist_prop = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            'id': post_id,
            'propaganda': prop,
            'his_propaganda': hist_prop
        })

        print(f"{i + 1}/{df.shape[0]}, {prop=}, {hist_prop=}")

    result_df = pd.DataFrame(results)
    result_df.to_csv('propaganda_llm.csv', index=False)

if __name__=="__main__":
    df = preprocessing()
    ner(df)
    manipulation_detection_with_llm(df)
    df = split_sentences_in_df(df)
    manipulation_detection(df)
    manipulation_detection_hist(df)