import torch.nn as nn

class FrozenFeatureClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FrozenFeatureClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)