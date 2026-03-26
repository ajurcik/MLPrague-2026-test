import torch
from sklearn.model_selection import train_test_split


def stratified_node_split(data, train_ratio=0.7, val_ratio=0.1):
    indices = torch.arange(data.num_nodes)
    labels = data.y.numpy()

    train_idx, remaining_idx = train_test_split(
        indices, 
        train_size=train_ratio, 
        stratify=labels, 
        random_state=42
    )

    remaining_labels = labels[remaining_idx]
    
    val_relative_ratio = val_ratio / (1 - train_ratio)
    
    val_idx, test_idx = train_test_split(
        remaining_idx, 
        train_size=val_relative_ratio, 
        stratify=remaining_labels, 
        random_state=42
    )

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data