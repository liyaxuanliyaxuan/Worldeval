#!/bin/bash

# 如果使用的是 不加入 11action 的模型，需要手动改 model_config.py
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python muti_gpu_infer.py \
# --lora_path "" \
# --meta_path "" \
# --output_subdir "lora_act_alpha_0.3_dex_ep30" \
# --action \
# --action_alpha 0.3 \
# --action_dim   1280 \
# --action_encoded_path ""

# pi0
  CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python muti_gpu_infer.py \
  --lora_path "" \
  --meta_path "" \
  --output_subdir "" \
  --action \
  --action_alpha 0.3 \
  --action_dim   1024 \
  --action_encoded_path ""

#onehot
#CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python muti_gpu_infer.py \
#--lora_path "" \
#--meta_path "" \
#--output_subdir "lora_act_alpha_0.3_onehot_ep30" \
#--action \
#--action_alpha 0.3 \
#--action_dim   224 \
#--action_encoded_path ""

#dp
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python muti_gpu_infer.py \
# --lora_path "" \
# --meta_path "" \
# --output_subdir "lora_act_alpha_0.3_dp_ep30" \
# --action \
# --action_alpha 0.3 \
# --action_dim   256 \
# --action_encoded_path ""