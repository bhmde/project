import torch
from torch.utils.data import Dataset, DataLoader
import csv
import numpy as np

from game_dataset import GameStateUtilityDataset

def load_game_data(csv_file, use_current_player=True, batch_size=32, shuffle=True, precompute=True):
    """
    Load game data with binary vector representation.
    
    Args:
        csv_file (str): Path to the CSV file
        use_current_player (bool): Use current player's utility if True
        batch_size (int): Batch size for DataLoader
        shuffle (bool): Whether to shuffle the data
        precompute (bool): Whether to precompute all binary vectors
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = GameStateUtilityDataset(csv_file, use_current_player, precompute=precompute)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Example usage
def test_dataset(csv_file):
    """
    Test the dataset implementation with a sample CSV file.
    
    Args:
        csv_file (str): Path to the CSV file
    """
    # Create dataset with precomputation
    dataset = GameStateUtilityDataset(csv_file, use_current_player=True, precompute=False)
    print(f"Dataset size: {len(dataset)}")
    
    print(f"dataset[0]", dataset[0])
    print(f"dataset[1]", dataset[1])
    print(f"dataset[15]", dataset[15])
    
    # Test DataLoader
    dataloader = load_game_data(csv_file, batch_size=32, shuffle=False)
    for batch_idx, (features, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        # Print first sample in batch
        print(f"  First feature in batch: {features[0, :]}")
        print(f"  First label in batch: {labels[0].item()}")
        
        if batch_idx >= 1:  # Just show first two batches
            break


if __name__ == '__main__':
    test_dataset('data/zero_by_2_1000000_4_7_13.csv')
