import os
import argparse
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn as nn
from vqvae.vqvae import VqVae

# 新增自定义数据集类
class H5ActionDataset(Dataset):
    def __init__(self, file_list_path, seq_length=10, action_dim=14):
        """
        Args:
            file_list_path (str): 包含h5文件路径的文本文件路径
            seq_length (int): 动作序列的时间步长 (对应input_dim_h)
            action_dim (int): 动作维度 (对应input_dim_w)
        """
        # 从file_path.txt文件中读取所有h5文件的路径
        with open(file_list_path, 'r') as f:
            self.file_paths = [line.strip() for line in f if line.strip().endswith('.hdf5')]
        self.sequences = []
        
        # 预加载所有有效数据
        for path in self.file_paths:
            with h5py.File(path, 'r') as f:
                if 'action' not in f:
                    continue
                
                # 获取动作数据 [N, D]
                actions = np.array(f['action'][:], dtype=np.float32)
                
                # 使用滑动窗口和填充来分割序列
                num_sequences = actions.shape[0]
                for i in range(num_sequences):
                    # 计算序列的起始和结束索引
                    end = i + 1
                    start = max(0, end - seq_length)
                    
                    # 提取序列并进行填充
                    seq = actions[start:end]
                    if len(seq) < seq_length:
                        padding = np.zeros((seq_length - len(seq), actions.shape[1]), dtype=np.float32)
                        seq = np.vstack((padding, seq))
                    
                    self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # 返回形状为 [seq_length, action_dim]
        return torch.from_numpy(self.sequences[idx])

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 初始化数据集和数据加载器
    dataset = H5ActionDataset(
        args.file_list_path,
        seq_length=args.seq_length,
        action_dim=args.action_dim
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 初始化模型
    vqvae = VqVae(
        input_dim_h=args.seq_length,
        input_dim_w=args.action_dim,
        n_latent_dims=args.latent_dim,
        vqvae_n_embed=args.n_embed,
        vqvae_groups=args.n_groups,
        device=device,
        act_scale=args.act_scale,
        eval=False
    )
    
    # 训练循环
    for epoch in range(args.epochs):
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        
        for batch_idx, actions in enumerate(dataloader):
            actions = actions.to(device)

            encoder_loss, vq_loss, _, recon_loss = vqvae.vqvae_update(actions)

            total_recon_loss += recon_loss
            total_vq_loss += vq_loss.item()
                        
            if batch_idx % args.log_interval == 0:
                print(
                    f"Epoch: {epoch:03d} | Batch: {batch_idx:03d} | "
                    f"Recon Loss: {recon_loss:.4f} | VQ Loss: {vq_loss.item():.4f}"
                )
        
        # 保存模型
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"vqvae_epoch_{epoch+1}.pth")
            torch.save(vqvae.state_dict(), save_path)
            print(f"Model saved at {save_path}")
        
        # 打印epoch统计信息
        avg_recon = total_recon_loss / len(dataloader)
        avg_vq = total_vq_loss / len(dataloader)
        print(f"Epoch {epoch} Summary | Avg Recon: {avg_recon:.4f} | Avg VQ: {avg_vq:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="VQ-VAE Training")
    
    # Data parameters
    parser.add_argument("--file_list_path", type=str, required=True,
                       help="Path to the file containing h5 file paths")
    parser.add_argument("--seq_length", type=int, default=16,
                       help="Time steps of the action sequence")
    parser.add_argument("--action_dim", type=int, default=14,
                       help="Dimension of the action")
    
    # Model parameters
    parser.add_argument("--latent_dim", type=int, default=512,
                       help="Dimension of the latent space")
    parser.add_argument("--n_embed", type=int, default=32,
                       help="Size of the codebook")
    parser.add_argument("--n_groups", type=int, default=4,
                       help="Number of residual quantization groups")
    parser.add_argument("--act_scale", type=float, default=5.0,
                       help="Action scaling factor")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100,
                       help="Total number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Training device")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading threads")
    
    # Logging parameters
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Log printing interval (in batches)")
    parser.add_argument("--save_interval", type=int, default=10,
                       help="Model saving interval (in epochs)")
    parser.add_argument("--save_dir", type=str, default="./vqvae_checkpoints",
                       help="Directory to save the model")
    
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    train(args)

if __name__ == "__main__":
    main()