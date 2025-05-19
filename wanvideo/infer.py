import os
import json
import argparse
import multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from diffsynth import ModelManager, WanVideoPipeline, WanVideoPipelineActEmbed, save_video, VideoData
from PIL import Image
import h5py
import numpy as np

def load_act_embed(file_path, start_index, end_index, action_encoded_path=None):
    if action_encoded_path is not None:
        data = torch.load(action_encoded_path, weights_only=False)
        file_index = data['file_path'].index(file_path)
        action_data = data['encoded_action'][file_index]
        action_data = action_data[start_index: end_index + 1]
        if not isinstance(action_data, np.ndarray):
            action_data = action_data.numpy()
        action_data = action_data.reshape(action_data.shape[0], -1)
    else:
        with h5py.File(file_path, 'r') as f:
            action_data = f['action_embed'][start_index: end_index + 1]
            if action_data.ndim == 3 and action_data.shape[1] == 1:
                action_data = action_data.squeeze(1)

    # if action_data.shape[0] < 81:
    #     padding = ((0, 81 - action_data.shape[0]), (0, 0))
    #     action_data = np.pad(action_data, padding, mode='constant')
    
    action_data = torch.tensor(action_data, dtype=torch.float32)
    return action_data

def process_chunk(meta_chunk, lora_path, output_subdir, action, action_encoded_path, action_alpha, action_dim, device_id, action_length):
    # Set current GPU device
    torch.cuda.set_device(device_id)
    
    # Initialize model manager
    model_manager = ModelManager(
        device="cuda",
        custom_params={"action_alpha": action_alpha, "action_dim": action_dim} if action else None
    )
    
    # Load base models
    model_manager.load_models(
        ["Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float32,
    )
    model_manager.load_models(
        [
            [
                "Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors",
                "Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors",
                "Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors",
                "Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors",
                "Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors",
                "Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors",
                "Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors",
            ],
            "Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16,
    )

    # Load LoRA
    if lora_path:
        model_manager.load_lora(lora_path, lora_alpha=1.0)
        output_subdir = output_subdir or "lora"
    else:
        output_subdir = output_subdir or "original"

    # Initialize pipeline
    if action:
        pipe = WanVideoPipelineActEmbed.from_model_manager(
            model_manager, 
            torch_dtype=torch.bfloat16,
            device="cuda"
        )
    else:
        pipe = WanVideoPipeline.from_model_manager(
            model_manager,
            torch_dtype=torch.bfloat16,
            device="cuda"
        )

    pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9)

    # Process each entry in the current chunk
    for entry in meta_chunk:
        try:
            start_frame_path = entry['image_path']
            prompt = entry['language']
            
            image = Image.open(start_frame_path)
            
            action_data = None
            if action:
                file_path = entry['file_path']
                start_index = entry.get('start_index', 0)
                action_data = load_act_embed(
                    file_path, start_index, start_index + action_length - 1, action_encoded_path
                ).to("cuda")

            # Video generation
            if action:
                video = pipe(
                    prompt=prompt,
                    negative_prompt="Vivid tones, overexposed, static, unclear details...",
                    input_image=image,
                    num_inference_steps=50,
                    seed=0, tiled=True,
                    action=action_data
                )
            else:
                video = pipe(
                    prompt=prompt,
                    negative_prompt="Vivid tones, overexposed, static, unclear details...",
                    input_image=image,
                    num_inference_steps=50, seed=0, tiled=True
                )

            # Save results
            output_dir = os.path.join(os.path.dirname(entry['image_path']), output_subdir)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f"{os.path.basename(start_frame_path).split('.')[0]}_video.mp4"
            )
            save_video(video, output_path, fps=15, quality=5)
            
            # Clear VRAM
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {entry['image_path']}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_path', type=str, required=True)
    parser.add_argument('--lora_path', type=str, default=None)
    parser.add_argument('--output_subdir', type=str, default=None)
    parser.add_argument('--action', action='store_true')
    parser.add_argument('--action_encoded_path', type=str, default=None)
    parser.add_argument('--action_alpha', type=float, default=0.1)
    parser.add_argument('--action_dim', type=int, default=1280)
    parser.add_argument('--action_length', type=int, default=81, help='Length of the action sequence')
    args = parser.parse_args()

    # Read metadata
    with open(args.meta_path, 'r') as f:
        meta_data = json.load(f)

    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    # Split data into chunks
    chunk_size = len(meta_data) // num_gpus + 1
    chunks = [meta_data[i::num_gpus] for i in range(num_gpus)]

    # Create and start processes
    processes = []
    for gpu_id in range(num_gpus):
        chunk = chunks[gpu_id]
        if not chunk:
            continue
            
        p = multiprocessing.Process(
            target=process_chunk,
            args=(
                chunk,
                args.lora_path,
                args.output_subdir,
                args.action,
                args.action_encoded_path,
                args.action_alpha,
                args.action_dim,
                gpu_id,  
                args.action_length
            )
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All tasks completed!")