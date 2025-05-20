import argparse
import os
import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from vqvae.vqvae import VqVae  # Assume your VqVae class is in this path


class H5InferenceDataset(Dataset):
    def __init__(self, file_path, seq_length=16, action_dim=14):
        """
        Args:
            file_path (str): hdf5 file path
            seq_length (int): sequence length required by the model
            action_dim (int): action dimension
        """
        self.file_path = file_path
        self.seq_length = seq_length
        self.action_dim = action_dim
        
        with h5py.File(file_path, 'r') as f:
            self.actions = np.array(f['action'][:], dtype=np.float32)
            self.original_length = len(self.actions)
        
        # Use sliding window and padding to split sequences
        self.sequences = []
        num_sequences = self.actions.shape[0]
        for i in range(num_sequences):
            # Calculate start and end indices of the sequence
            end = i + 1
            start = max(0, end - self.seq_length)
            
            # Extract sequence and pad if necessary
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
        """Get valid data indices"""
        return slice(0, self.original_length)

def inference(args):
    # Device configuration
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model
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
    
    # Process a single file
    def process_file(input_path):
        
        # Initialize dataset
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
                
                # Get latents [batch_size, seq_length, latent_dim]
                latents, _ = vqvae.get_code(batch)
                print(f"latents shape: {latents.shape}")
                file_latents.append(latents.cpu().numpy())
        
        print(f"file_latents shape: {file_latents[0].shape}")
        latents = np.concatenate(file_latents, axis=0)  # [num_sequences, seq_length, latent_dim]
        # print(f"latents shape: {latents.shape}")
        latents = latents.reshape(-1, args.latent_dim)  # [total_steps, latent_dim]
        # print(f"latents shape after reshape: {latents.shape}")
        valid_latents = latents[:dataset.original_length]  # Remove padding part
        valid_latents = torch.from_numpy(valid_latents)  # Convert to tensor
        print(f"valid_latents shape: {valid_latents.shape}")
        # valid_latents = valid_latents.squeeze(1)  # Remove the singleton dimension
        
        return valid_latents

    all_latents = []  # Initialize all_latents outside the process_file function

    # Process all files
    with open(args.file_list_path, 'r') as f:
        if args.file_list_path.endswith('.json'):
            # If JSON file, read the file_path field of each item
            data = json.load(f)
            input_files = list(set(item['file_path'] for item in data))
        else:
            # If text file, read each line
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
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model")
    parser.add_argument("--file_list_path", type=str, required=True,
                       help="Text file containing the list of hdf5 file paths to process")
    
    # Model parameters (must be consistent with training)
    parser.add_argument("--seq_length", type=int, default=16,
                       help="Must be the same as the sequence length used during training")
    parser.add_argument("--action_dim", type=int, default=14,
                       help="Must be the same as the action dimension used during training")
    parser.add_argument("--latent_dim", type=int, default=512,
                       help="Must be the same as the latent dimension used during training")
    parser.add_argument("--n_embed", type=int, default=32,
                       help="Must be the same as the codebook size used during training")
    parser.add_argument("--n_groups", type=int, default=4,
                       help="Must be the same as the number of residual groups used during training")
    parser.add_argument("--act_scale", type=float, default=5.0,
                       help="Must be the same as the scaling factor used during training")
    
    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Inference batch size")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Inference device")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    
    # Add a new argument for the save path
    parser.add_argument("--save_path", type=str, default="actions_vqvae.pt",
                       help="Path to save the latent variables")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_inference_args()
    inference(args)