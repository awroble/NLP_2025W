import pandas as pd
import numpy as np
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from textblob import TextBlob


def load_webis_corpus_2017(base_dir='data'):
    target_subdirs = [
        'clickbait17-train-170331',
        'clickbait17-train-170630'
    ]

    all_data_frames = []

    if not os.path.exists(base_dir):
        print(f"Base directory '{base_dir}' not found.")
        return

    for subdir in target_subdirs:
        dir_path = os.path.join(base_dir, subdir)
        instances_path = os.path.join(dir_path, 'instances.jsonl')
        truth_path = os.path.join(dir_path, 'truth.jsonl')

        if os.path.exists(instances_path) and os.path.exists(truth_path):
            df_instances = pd.read_json(instances_path, lines=True)
            df_truth = pd.read_json(truth_path, lines=True)
            merged_part = pd.merge(df_instances, df_truth, on='id')
            all_data_frames.append(merged_part)

    print(f"Successfully loaded data.")

    if not all_data_frames:
        print("No valid data found in the specified subdirectories.")
        return

    df = pd.concat(all_data_frames, ignore_index=True)

    df['headline'] = df['postText'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
    df['label'] = df['truthClass'].apply(lambda x: 1 if str(x).lower() == 'clickbait' else 0)

    final_df = df[['headline', 'label']].dropna()
    print(f"Dataset shape: {final_df.shape}")
    return final_df


def load_webis_corpus_2022(base_dir="data"):
    if not os.path.exists(base_dir):
        print(f"Base directory '{base_dir}' not found.")
        return

    data_dir = os.path.join(base_dir, 'webis-clickbait-22')
    train_path = os.path.join(data_dir, 'train.jsonl')
    val_path = os.path.join(data_dir, 'validation.jsonl')

    try:
        df_train = pd.read_json(train_path, lines=True)
        df_val = pd.read_json(val_path, lines=True)
    except ValueError as e:
        print(f"Error reading JSON: {e}")
        return
    except FileNotFoundError as e:
        print(f"File not found. Check your paths: {e}")
        return

    df_combined = pd.concat([df_train, df_val])
    df_combined["label"] = 1
    df_combined['headline'] = df_combined['postText'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
    final_df = df_combined[['headline', 'label']].dropna()

    print(f"Successfully loaded data.")
    print(f"Dataset shape: {final_df.shape}")
    return final_df


def prepare_english_data(base_dir="data", test_size=0.2):
    cb_2017 = load_webis_corpus_2017(base_dir)
    cb_2022 = load_webis_corpus_2022(base_dir)

    final_df = pd.concat([cb_2017, cb_2022])

    # Additional clean
    initial_count = len(final_df)
    clean_df = final_df.dropna(subset=['headline'])
    clean_df = clean_df[clean_df['headline'].astype(str).str.strip() != '']
    removed_count = initial_count - len(clean_df)

    print(f"Removed {removed_count} rows with missing or empty headlines.")

    train_df, val_df = train_test_split(clean_df, test_size=test_size, random_state=42, stratify=clean_df['label'])

    print(f"Total Samples: {len(clean_df)}")
    print(f"Training Set: {len(train_df)}")
    print(f"Validation Set: {len(val_df)}")
    print("\nClass Balance (Train):")
    print(train_df['label'].value_counts())

    return train_df, val_df


def perform_feature_engineering_manually(headlines):
    features = []
    for text in headlines:
        text = str(text)
        blob = TextBlob(text)
        row = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'all_caps': 1 if text.isupper() else 0,
            'sentiment_polarity': blob.sentiment.polarity,
            'sentiment_subjectivity': blob.sentiment.subjectivity,
            'starts_with_question': 1 if text.strip().lower().split()[0] in ['who', 'what', 'where', 'why',
                                                                             'how'] else 0
        }
        features.append(list(row.values()))
    return np.array(features)


class HandCraftedFeatures(BaseEstimator, TransformerMixin):
    """
    Class to perform feature engineering in a pipeline
    """

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        features = []
        for text in posts:
            text = str(text)
            blob = TextBlob(text)
            row = {
                'char_count': len(text),
                'word_count': len(text.split()),
                'exclamation_count': text.count('!'),
                'question_count': text.count('?'),
                'all_caps': 1 if text.isupper() else 0,
                'sentiment_polarity': blob.sentiment.polarity,
                'sentiment_subjectivity': blob.sentiment.subjectivity,
                'starts_with_question': 1 if text.strip().lower().split()[0] in ['who', 'what', 'where', 'why',
                                                                                 'how'] else 0
            }
            features.append(list(row.values()))
        return np.array(features)
