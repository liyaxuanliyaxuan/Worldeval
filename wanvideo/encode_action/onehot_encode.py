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
        # 计算每个动作的离散化索引
        bin_size = (self.action_max - self.action_min) / self.action_bins
        discretized = torch.floor((actions - self.action_min) / bin_size).long()
        discretized = torch.clamp(discretized, 0, self.action_bins-1)
        
        # 转换为one-hot编码
        actions_onehot = F.one_hot(discretized, num_classes=self.action_bins)
        return actions_onehot.view(*actions.shape[:2], -1)  

def process_actions(json_path, output_path, action_bins=16):
    # 读取文件列表
    df = pd.read_json(json_path)
    # df = pd.read_csv(json_path)
    file_paths = df['file_path'].tolist()
    
    # ===== 第一阶段：计算全局极值 =====
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
    
    # ===== 第二阶段：处理所有文件 =====
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
            
            # 离散化处理
            decoded = discretizer.discretize_and_onehot_actions(actions)
            
            all_actions.append(decoded)  # 不用堆叠
            valid_paths.append(path)
    
    print(f"all_actions shape: {all_actions[0].shape}")
    # 存储为字典或列表
    result = {
        'file_path': valid_paths,
        'encoded_action': all_actions,  # 直接存储为列表
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