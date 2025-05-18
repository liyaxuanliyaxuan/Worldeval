import argparse
import os
import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from vqvae.vqvae import VqVae  # 假设你的VqVae类在此路径


class H5InferenceDataset(Dataset):
    def __init__(self, file_path, seq_length=16, action_dim=14):
        """
        Args:
            file_path (str): hdf5文件路径
            seq_length (int): 模型需要的序列长度
            action_dim (int): 动作维度
        """
        self.file_path = file_path
        self.seq_length = seq_length
        self.action_dim = action_dim
        
        with h5py.File(file_path, 'r') as f:
            self.actions = np.array(f['action'][:], dtype=np.float32)
            self.original_length = len(self.actions)
        
        # 使用滑动窗口和填充来分割序列
        self.sequences = []
        num_sequences = self.actions.shape[0]
        for i in range(num_sequences):
            # 计算序列的起始和结束索引
            end = i + 1
            start = max(0, end - self.seq_length)
            
            # 提取序列并进行填充
            seq = self.actions[start:end]
            if len(seq) < self.seq_length:
                padding = np.zeros((self.seq_length - len(seq), self.actions.shape[1]), dtype=np.float32)
                seq = np.vstack((padding, seq))
            
            self.sequences.append(seq)
        
        self.num_sequences = len(self.sequences)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx].astype(np.float32))

    def get_original_indices(self):
        """获取有效数据索引"""
        return slice(0, self.original_length)

def inference(args):
    # 设备配置
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    vqvae = VqVae(
        input_dim_h=args.seq_length,
        input_dim_w=args.action_dim,
        n_latent_dims=args.latent_dim,
        vqvae_n_embed=args.n_embed,
        vqvae_groups=args.n_groups,
        device=device,
        act_scale=args.act_scale,
        eval=True
    )
    vqvae.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # 处理单个文件
    def process_file(input_path):
        
        # 初始化数据集
        dataset = H5InferenceDataset(
            input_path,
            seq_length=args.seq_length,
            action_dim=args.action_dim
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        print(f"Number of batches: {len(dataloader)}")

        
        file_latents = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Processing {os.path.basename(input_path)}"):
                batch = batch.to(device)
                
                # 获取潜在变量 [batch_size, seq_length, latent_dim]
                latents, _ = vqvae.get_code(batch)
                print(f"latents shape: {latents.shape}")
                file_latents.append(latents.cpu().numpy())
        
        print(f"file_latents shape: {file_latents[0].shape}")
        latents = np.concatenate(file_latents, axis=0)  # [num_sequences, seq_length, latent_dim]
        # print(f"latents shape: {latents.shape}")
        latents = latents.reshape(-1, args.latent_dim)  # [total_steps, latent_dim]
        # print(f"latents shape after reshape: {latents.shape}")
        valid_latents = latents[:dataset.original_length]  # 去除填充部分
        valid_latents = torch.from_numpy(valid_latents)  # Convert to tensor
        print(f"valid_latents shape: {valid_latents.shape}")
        # valid_latents = valid_latents.squeeze(1)  # Remove the singleton dimension
        
        return valid_latents

    all_latents = []  # Initialize all_latents outside the process_file function

    # 处理所有文件
    with open(args.file_list_path, 'r') as f:
        if args.file_list_path.endswith('.json'):
            # 如果是JSON文件，读取每个item的file_path字段
            data = json.load(f)
            input_files = list(set(item['file_path'] for item in data))
        else:
            # 如果是文本文件，读取每行
            input_files = [line.strip() for line in f if line.strip().endswith('.hdf5')]
    
    for input_file in input_files:
        file_latents = process_file(input_file)  # Modify process_file to return latents
        all_latents.append(file_latents)  # Collect latents from each file
        print(f"Processed {input_file}")

    # Save latents and file paths in a .pt file
    torch.save({
        'encoded_action': all_latents,  # Save all latents as a list
        'file_path': input_files  # Save the list of hdf5 file paths
    }, args.save_path)

def parse_inference_args():
    parser = argparse.ArgumentParser(description="VQ-VAE Inference")
    
    # 必需参数
    parser.add_argument("--model_path", type=str, required=True,
                       help="训练好的模型路径")
    parser.add_argument("--file_list_path", type=str, required=True,
                       help="包含待处理hdf5文件路径列表的文本文件")
    
    # 模型参数 (必须与训练时一致)
    parser.add_argument("--seq_length", type=int, default=16,
                       help="必须与训练时使用的序列长度相同")
    parser.add_argument("--action_dim", type=int, default=14,
                       help="必须与训练时动作维度相同")
    parser.add_argument("--latent_dim", type=int, default=512,
                       help="必须与训练时潜在维度相同")
    parser.add_argument("--n_embed", type=int, default=32,
                       help="必须与训练时码本大小相同")
    parser.add_argument("--n_groups", type=int, default=4,
                       help="必须与训练时残差组数相同")
    parser.add_argument("--act_scale", type=float, default=5.0,
                       help="必须与训练时缩放因子相同")
    
    # 推理参数
    parser.add_argument("--batch_size", type=int, default=32,
                       help="推理批量大小")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="推理设备")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="数据加载线程数")
    
    # Add a new argument for the save path
    parser.add_argument("--save_path", type=str, default="actions_vqvae.pt",
                       help="保存潜在变量的路径")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_inference_args()
    inference(args)