import os
import argparse
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn as nn
from vqvae.vqvae import VqVae

# Custom dataset class
class H5ActionDataset(Dataset):
    def __init__(self, file_list_path, seq_length=10, action_dim=14):
        """
        Args:
            file_list_path (str): Path to the text file containing h5 file paths
            seq_length (int): Time steps of the action sequence (corresponds to input_dim_h)
            action_dim (int): Dimension of the action (corresponds to input_dim_w)
        """
        # Read all h5 file paths from file_path.txt
        with open(file_list_path, 'r') as f:
            self.file_paths = [line.strip() for line in f if line.strip().endswith('.hdf5')]
        self.sequences = []
        
        # Preload all valid data
        for path in self.file_paths:
            with h5py.File(path, 'r') as f:
                if 'action' not in f:
                    continue
                
                # Get action data [N, D]
                actions = np.array(f['action'][:], dtype=np.float32)
                
                # Use sliding window and padding to split sequences
                num_sequences = actions.shape[0]
                for i in range(num_sequences):
                    # Calculate start and end indices of the sequence
                    end = i + 1
                    start = max(0, end - seq_length)
                    
                    # Extract sequence and pad if necessary
                    seq = actions[start:end]
                    if len(seq) < seq_length:
                        padding = np.zeros((seq_length - len(seq), actions.shape[1]), dtype=np.float32)
                        seq = np.vstack((padding, seq))
                    
                    self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Return shape [seq_length, action_dim]
        return torch.from_numpy(self.sequences[idx])

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Initialize dataset and dataloader
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
    
    # Initialize model
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
    
    # Training loop
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
        
        # Save model
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"vqvae_epoch_{epoch+1}.pth")
            torch.save(vqvae.state_dict(), save_path)
            print(f"Model saved at {save_path}")
        
        # Print epoch statistics
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