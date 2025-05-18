#!/bin/bash

# 如果使用的是 不加入 11action 的模型，需要手动改 model_config.py
CUDA_VISIBLE_DEVICES=0 python wan_14b_image_to_video_unified_act.py \
--lora_path "wanvideo/models/lightning_logs/lora_act_alpha_0.3_pi0_v2/checkpoints/epoch=29-step=1890.ckpt" \
--meta_path data/eval1/metadata.json \
--output_subdir "lora_act_alpha_0.3_pi0_ep30" \
--action \
--action_alpha 0.3 \
--action_dim   1024 \
--action_encoded_path data/eval1/actions_pi0.pt


# CUDA_VISIBLE_DEVICES=1 python wan_14b_image_to_video_unified_act.py \
# --lora_path "wanvideo/models/lightning_logs/lora_act_alpha_0.3_dp_ep40/checkpoints/epoch=29-step=2160.ckpt" \
# --meta_path data/ood_test5/metadata.json \
# --output_subdir "lora_act_alpha_0.3_dp_ep30" \
# --action \
# --action_alpha 0.3 \
# --action_dim   256 \
# --action_encoded_path data/ood_test5/actions_dp.pt