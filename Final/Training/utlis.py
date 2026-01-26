from sklearn.model_selection import train_test_split


def train_test_split_data(
    df,
    text_col,
    label_col,
    test_size=0.3,
    random_state=42
):
    X = df[text_col]
    y = df[label_col]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
