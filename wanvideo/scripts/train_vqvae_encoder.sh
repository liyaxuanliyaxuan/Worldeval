#! /bin/bash

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python encode_action/vqvae_train.py \
  --file_list_path all_files.txt \
  --seq_length 16 \
  --action_dim 14 \
  --latent_dim 512 \
  --n_embed 32 \
  --n_groups 4 \
  --epochs 100 \
  --batch_size 64 \
  --save_dir ./vqvae_checkpoints_seq16