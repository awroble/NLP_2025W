import torch
from torch.utils.data import Dataset

class NewsTopicDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Shifting labels to 0-indexed for CrossEntropyLoss
        shifted_label = int(item['label']) - 1
        
        return {
            'label': torch.tensor(shifted_label, dtype=torch.long),
            'embeddings': torch.tensor(item['embeddings'], dtype=torch.float32),
            'sentiment_norm': torch.tensor(item['sentiment_norm'], dtype=torch.float32),
            'bias_norm': torch.tensor(item['bias_norm'], dtype=torch.float32),
            'subjectivity_norm': torch.tensor(item['subjectivity_norm'], dtype=torch.float32),
            'framing_score': torch.tensor(item['framing_score'], dtype=torch.float32),
            'full_feature_vector': torch.tensor(item['full_feature_vector'], dtype=torch.float32)
        }