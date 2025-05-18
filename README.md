


## Setup

Wan-Video is a collection of video synthesis models open-sourced by Alibaba.

Before using this model, please install DiffSynth-Studio from **source code**.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

Wan-Video supports multiple Attention implementations. If you have installed any of the following Attention implementations, they will be enabled based on priority.

* [Flash Attention 3](https://github.com/Dao-AILab/flash-attention)
* [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
* [Sage Attention](https://github.com/thu-ml/SageAttention)
* [torch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) (default. `torch>=2.5.0` is recommended.)

## Train

Step 1: Install additional packages

```
pip install peft lightning pandas
```

Step 2: Prepare your dataset

You need to manage the training videos as follows:

```
data/example_dataset/
├── metadata.csv
└── train (empty dir)
```

`metadata.csv`:

```
file_path,file_name,text
/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/1_24_7z_extract/truncate_push_basket_to_left_1_24/episode_4.hdf5,episode_4.hdf5,""
```

Step 3: Data process

run scripts/data_process.sh

```shell
#! /bin/bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_wan_t2v_act_embed.py \
  --task data_process \
  --dataset_path data/dex2_example_dataset \
  --output_path ./models \
  --text_encoder_path "Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth" \
  --image_encoder_path "Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --vae_path "Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth" \
  --tiled \
  --num_frames 81 \
  --height 480 \
  --width 832 \
  --encode_mode dexvla \
  --action_encoded_path data/dex2_example_dataset/train/all_actions_dex.pt
```

After that, some cached files will be stored in the dataset folder.

```
data/example_dataset/
├── metadata.csv
└── train
    ├── file1.hdf5.tensors.pth
    └── file2.hdf5.tensors.pth
```

Step 4: Train

LoRA training:

run scripts/train.bash

 ```shell
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
python train_wan_t2v.py \
  --task train \
  --train_architecture lora \
  --dataset_path data/example_dataset \
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
  --use_gradient_checkpointing

```   
please separate the safetensor files with a comma. For example: `models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors,models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors,models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors,models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors,models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors,models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors`.

## Inference

Step1: 

Extract action embeddings in different vla policy and prepare encoded actions and save them in the pt file. 

Step2: 

Sample frames from hdf5 file to test using utils/sample_frames_from_dir_for_test, then it will generate metadata.json and the first frame for use.

Step3: 

Run scripts/muti_inference.bash

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python muti_gpu_infer.py \
--lora_path "" \
--meta_path "" \
--output_subdir "lora_act_alpha_0.3_dex_ep30" \
--action \
--action_alpha 0.3 \
--action_dim   1280 \
--action_encoded_path ""
```