#!/bin/bash

CUDA_VISIBLE_DEVICES="4,5,6,7" \
python train_wan_t2v_act_embed.py \
  --task train \
  --train_architecture lora \
  --dataset_path data/vqvae_example_dataset \
  --output_path ./models \
  --dit_path "Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors,Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors,Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors,Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors,Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors,Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors,Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors" \
  --image_encoder_path "Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --steps_per_epoch 500 \
  --max_epochs 40 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 16 \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2,action_alpha,action_proj.0,action_proj.2"\
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --action_alpha 0.3 \
  --action_dim 512 \
  --version "lora_act_alpha_0.3_vqvae" \
  --encode_mode "vqvae" \
  --action_encoded_path data/vqvae_example_dataset/train/actions_vqvae.pt