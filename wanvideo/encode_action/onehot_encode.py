import pandas as pd
import h5py
import torch
import numpy as np
from torch.nn import functional as F

class ActionDiscretizer:
    def __init__(self, action_min, action_max, action_bins=16):
        self.action_min = action_min
        self.action_max = action_max
        self.action_bins = action_bins
        
    def discretize_and_onehot_actions(self, actions):
        # Calculate the discretized index for each action
        bin_size = (self.action_max - self.action_min) / self.action_bins
        discretized = torch.floor((actions - self.action_min) / bin_size).long()
        discretized = torch.clamp(discretized, 0, self.action_bins-1)
        
        # Convert to one-hot encoding
        actions_onehot = F.one_hot(discretized, num_classes=self.action_bins)
        return actions_onehot.view(*actions.shape[:2], -1)  

def process_actions(json_path, output_path, action_bins=16):
    # Read file list
    df = pd.read_json(json_path)
    # df = pd.read_csv(json_path)
    file_paths = df['file_path'].tolist()
    
    # ===== Phase 1: Calculate global min/max =====
    # all_actions = []  # Initialize a list to store all actions
    
    # for path in file_paths:
    #     with h5py.File(path, 'r') as f:
    #         if 'action' not in f:
    #             continue
            
    #         actions = f['action'][:].flatten()  # Flatten all dimensions
    #         all_actions.append(actions)
    
    # # Combine all action data
    # combined_actions = np.concatenate(all_actions, axis=None)
    
    # # Calculate truncation range using 1% and 99% quantiles
    # lower = np.quantile(combined_actions, 0.01)
    # upper = np.quantile(combined_actions, 0.99)
    
    # # Add buffer to prevent boundary values from being truncated
    # buffer_size = (upper - lower) * 0.05
    # final_min = max(np.min(combined_actions), lower - buffer_size)
    # final_max = min(np.max(combined_actions), upper + buffer_size)
    
    # print(f"final_min: {final_min}, final_max: {final_max}")
    
    # ===== Phase 2: Process all files =====
    discretizer = ActionDiscretizer(
        action_min=-1.671177864074707,
        action_max=8.268219947814941,
        action_bins=action_bins
    )
    
    all_actions = []
    valid_paths = []
    
    for path in file_paths:
        with h5py.File(path, 'r') as f:
            if 'action' not in f:
                continue
            
            actions = torch.tensor(f['action'][:], dtype=torch.float32)
            
            # Discretization processing
            decoded = discretizer.discretize_and_onehot_actions(actions)
            
            all_actions.append(decoded)  # No need to stack
            valid_paths.append(path)
    
    print(f"all_actions shape: {all_actions[0].shape}")
    # Store as dict or list
    result = {
        'file_path': valid_paths,
        'encoded_action': all_actions,  # Store directly as list
        'action_min': -1.671177864074707,
        'action_max': 8.268219947814941
    }
    
    torch.save(result, output_path)
    print(f"Processed {len(valid_paths)} files.")


# 使用示例
process_actions(
    json_path="",
    output_path="",
    action_bins=16
)