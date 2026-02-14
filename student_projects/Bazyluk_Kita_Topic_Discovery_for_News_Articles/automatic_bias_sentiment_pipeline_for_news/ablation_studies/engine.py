import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import FrozenFeatureClassifier

def run_ablation_test(feature_keys, loaders, num_classes, device, epochs, lr, weight_decay):
    # Dynamic input dimension calculation
    sample_batch = next(iter(loaders['train']))
    input_dim = sum(sample_batch[k].view(sample_batch[k].size(0), -1).shape[1] for k in feature_keys)
    
    model = FrozenFeatureClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {'train_loss': [], 'val_loss': []}
    epoch_pbar = tqdm(range(epochs), desc=f"Testing {feature_keys}", leave=False)

    for epoch in epoch_pbar:
        # Training Phase
        model.train()
        running_train_loss = 0.0
        for batch in loaders['train']:
            x = torch.cat([batch[k].view(batch[k].size(0), -1).to(device) for k in feature_keys], dim=1)
            y = batch['label'].to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in loaders['val']:
                x = torch.cat([batch[k].view(batch[k].size(0), -1).to(device) for k in feature_keys], dim=1)
                y = batch['label'].to(device)
                running_val_loss += criterion(model(x), y).item()
        
        history['train_loss'].append(running_train_loss / len(loaders['train']))
        history['val_loss'].append(running_val_loss / len(loaders['val']))

    # Final Test Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loaders['test']:
            x = torch.cat([batch[k].view(batch[k].size(0), -1).to(device) for k in feature_keys], dim=1)
            y = batch['label'].to(device)
            predictions = torch.argmax(model(x), dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)
    
    return correct / total, history