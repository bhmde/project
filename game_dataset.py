import torch
from torch.utils.data import Dataset, DataLoader
import csv
import numpy as np

class GameStateUtilityDataset(Dataset):
    def __init__(self, csv_file, use_current_player=True, delimiter='|', precompute=True):
        """
        Custom dataset for game state data with binary vector representation.
        
        Args:
            csv_file (str): Path to the CSV file
            use_current_player (bool): If True, label is utility of current player,
                                      otherwise utility of other player
            delimiter (str): CSV delimiter character
            precompute (bool): Whether to precompute all binary vectors at init
        """
        # Read CSV file with specified delimiter
        self.data = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                # Convert all values to integers
                self.data.append([int(val) for val in row])
        
        self.use_current_player = use_current_player
        
        # Precompute binary vectors if enabled
        if precompute:
            print("Precomputing binary vectors for all states...")
            self.vectors = np.zeros((len(self.data), 64), dtype=np.float32)
            
            # Efficiently compute binary representation for all states
            for i, row in enumerate(self.data):
                state = row[0]  # First column is the state
                for bit_pos in range(64):
                    if state & (1 << bit_pos):
                        self.vectors[i, 63-bit_pos] = 1.0
                        
            self.vectors = torch.tensor(self.vectors, dtype=torch.float32)
        else:
            self.vectors = None
    
    def __len__(self):
        return len(self.data)
    
    def _state_to_vector(self, state):
        """
        Convert integer state to a 64-dimensional binary vector.
        
        Args:
            state (int): Integer state value
            
        Returns:
            torch.Tensor: 64-dimensional binary vector
        """
        # Create a tensor of zeros
        vec = torch.zeros(64, dtype=torch.float32)
        
        # Set bits in the vector
        for i in range(64):
            if state & (1 << i):
                vec[63-i] = 1.0
                
        return vec
    
    def __getitem__(self, idx):
        row = self.data[idx]
        state = row[0]          # State
        current_player = row[2]  # Current player
        utility_0 = row[3]      # Utility for player 0
        utility_1 = row[4]      # Utility for player 1
        
        # Get binary vector representation
        if self.vectors is not None:
            # Use precomputed vector
            x = self.vectors[idx]
        else:
            # Compute on-the-fly
            x = self._state_to_vector(state)
        
        # Get utility based on use_current_player flag
        if self.use_current_player:
            utility = utility_0 if current_player == 0 else utility_1
        else:
            utility = utility_1 if current_player == 0 else utility_0
        
        y = torch.tensor(utility, dtype=torch.float32)
        
        return x, y
