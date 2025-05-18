#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python encode_action/vqvae_encode.py \
  --file_list_path wanvideo/data/pick_and_place/dexvla/pika/metadata.json \
  --model_path wanvideo/vqvae_checkpoints/vqvae_epoch_100.pth \
  --seq_length 1 \
  --action_dim 14 \
  --latent_dim 512 \
  --n_embed 32 \
  --n_groups 4 \
  --act_scale 5.0 \
  --batch_size 64 \
  --save_path wanvideo/data/pick_and_place/dexvla/pika/actions_vqvae_seq1.pt