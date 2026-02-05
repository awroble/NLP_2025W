import torch

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_EVAL = 32
LR = 1e-3
WEIGHT_DECAY = 0.01
EPOCHS = 15

# Experiment Settings
OUTPUT_DIR = "ablation_results"
DATASET_NAME = "mkita/topic-discovery-for-news-articles-test"
DATA_SPLIT = "train"

# Feature Groups
BASE_FEATURE = 'embeddings'
OPTIONAL_FEATURES = ['sentiment_norm', 'bias_norm', 'subjectivity_norm', 'framing_score']