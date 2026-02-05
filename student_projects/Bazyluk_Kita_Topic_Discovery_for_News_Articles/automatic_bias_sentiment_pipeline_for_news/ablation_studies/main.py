import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets import load_dataset

from dataset import NewsTopicDataset
from engine import run_ablation_test
import config

def save_loss_plot(history, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Ablation: {title}')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 1. Setup Data
    raw_data = load_dataset(config.DATASET_NAME, split=config.DATA_SPLIT)
    train_test = raw_data.train_test_split(test_size=0.3, seed=42)
    val_test = train_test['test'].train_test_split(test_size=0.5, seed=42)
    
    loaders = {
        'train': DataLoader(NewsTopicDataset(train_test['train']), batch_size=config.BATCH_SIZE_TRAIN, shuffle=True),
        'val': DataLoader(NewsTopicDataset(val_test['train']), batch_size=config.BATCH_SIZE_EVAL),
        'test': DataLoader(NewsTopicDataset(val_test['test']), batch_size=config.BATCH_SIZE_EVAL)
    }

    num_classes = len(raw_data.unique("label"))

    # 2. Define Feature Combinations
    combinations = [[config.BASE_FEATURE] + list(c) 
                    for i in range(len(config.OPTIONAL_FEATURES) + 1) 
                    for c in itertools.combinations(config.OPTIONAL_FEATURES, i)]
    combinations.append(['full_feature_vector'])

    # 3. Run Experiments
    metrics_list = []
    for f_set in combinations:
        name = "+".join(f_set)
        print(f"Testing: {name}")
        
        acc, history = run_ablation_test(
            f_set, loaders, num_classes, config.DEVICE, 
            config.EPOCHS, config.LR, config.WEIGHT_DECAY
        )
        
        save_loss_plot(history, name, os.path.join(config.OUTPUT_DIR, f"plot_{name}.png"))
        metrics_list.append({'features': name, 'test_accuracy': acc})

    # 4. Save Summary
    pd.DataFrame(metrics_list).sort_values(by='test_accuracy', ascending=False).to_csv(
        os.path.join(config.OUTPUT_DIR, "ablation_summary.csv"), index=False
    )
    print(f"Results saved to {config.OUTPUT_DIR}")