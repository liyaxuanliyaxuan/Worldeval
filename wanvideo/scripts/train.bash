#!/bin/bash

# 
# If training a model without action, you need to manually modify model_config.py
# dexvla: action_dim 1280
# pi0: action_dim 1024
# onehot: action_dim 224 
# vqvae: action_dim 512
# dp: action_dim 256  need： --lora_target_modules "q,k,v,o,ffn.0,ffn.2,action_alpha,action_proj.0,action_proj.2,action_proj.4"
#  

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
# python train_wan_t2v.py \
#   --task train \
#   --train_architecture lora \
#   --dataset_path data/example_dataset \
#   --output_path ./models \
#   --dit_path "Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors,Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors,Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors,Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors,Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors,Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors,Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors" \
#   --image_encoder_path "Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
#   --steps_per_epoch 500 \
#   --max_epochs 40 \
#   --learning_rate 1e-4 \
#   --lora_rank 16 \
#   --lora_alpha 16 \
#   --lora_target_modules "q,k,v,o,ffn.0,ffn.2,action_alpha,action_proj.0,action_proj.2"\
#   --accumulate_grad_batches 1 \
#   --use_gradient_checkpointing


CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
  python train_wan_t2v_act_embed.py \
  --task train \
  --train_architecture lora \
  --dataset_path data/dex_simpler_dataset/ \
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
  --action_dim 1280 \
  --encode_mode "dexvla" \
  --action_encoded_path "wanvideo/data/dex_simpler_dataset/train/all_actions.pt" \
  --version "lora_act_alpha_0.3_dex_sampled_simpleenv"
  
