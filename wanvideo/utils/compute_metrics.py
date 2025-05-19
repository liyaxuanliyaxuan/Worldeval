import h5py
import cv2
import numpy as np
import json
import os
import lpips
import torch
from DISTS_pytorch import DISTS
from torchmetrics.image.fid import FrechetInceptionDistance
import re

def read_frames_from_video(video_file_path, num_frames=81):
    cap = cv2.VideoCapture(video_file_path)
    frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def read_frames_from_hdf5(hdf5_file_path, num_frames=81, start_frame=0):
    print(hdf5_file_path, start_frame)
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        images = hdf5_file['observations/images/cam_high']
        frames = [np.array(images[i]) for i in range(start_frame, start_frame+num_frames)]
    return frames

def main():
    # Initialize evaluation models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips_model = lpips.LPIPS(net='alex').to(device)
    dists_model = DISTS().to(device)
    
    # Define paths and subdirectories
    metadata_path = ''
    subdirectory = 'lora_act_alpha_0.3_dex_ep30'

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Initialize metrics storage
    all_lpips = []
    all_dists = []
    real_images = []
    gen_images = []
    frame_results = []

    print(metadata)
    for item in metadata:
        hdf5_file_path = item['file_path']
        image_path = item['image_path']

        # Check if image_path ends with "v1"
        if image_path.endswith("v1"):
            subdirectory = "v1"
        
        # The first image and video correspond to each other
        directory, filename = os.path.split(image_path)
        video_file_path = os.path.join(directory, subdirectory,  filename.replace('.png', '_video.mp4') if subdirectory!="v1" else filename.replace('_v1.png', '_video.mp4'))
        
        # Extract index from filename and use it as start_frame
        match = re.search(r'_(\d+)\.png$', filename)
        start_frame = int(match.group(1)) if match else 0

        # print("***", start_frame)

        # Read frame data
        video_frames = read_frames_from_video(video_file_path)
        hdf5_frames = read_frames_from_hdf5(hdf5_file_path, num_frames=81, start_frame=start_frame)
        
        min_frames = min(len(video_frames), len(hdf5_frames))
        if min_frames == 0:
            continue

        for i in range(min_frames):
            video_frame = video_frames[i]
            hdf5_frame = hdf5_frames[i]
            
            # Preprocess video frames
            video_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            
            # Process HDF5 frame data type
            if hdf5_frame.dtype == np.float32 or hdf5_frame.dtype == np.float64:
                hdf5_frame = (hdf5_frame * 255).astype(np.uint8)
            
            # Resize to align dimensions
            target_h, target_w = hdf5_frame.shape[:2]
            video_resized = cv2.resize(video_rgb, (target_w, target_h))
            
            # Collect FID data (resize to 299x299)
            hdf5_fid = cv2.resize(hdf5_frame, (299, 299))
            video_fid = cv2.resize(video_resized, (299, 299))
            real_images.append(hdf5_fid)
            gen_images.append(video_fid)
            
            # 记录帧结果
            # frame_results.append({
            #     'frame_index': i,
            #     'lpips': lpips_val,
            #     'dists': dists_val
            # })
            
            # print(f"Processed frame {i+1}/{min_frames} | "
            #       f"LPIPS: {lpips_val:.4f} | "
            #       f"DISTS: {dists_val:.4f}")

    # Calculate FID
    fid_value = None
    if len(real_images) > 0 and len(gen_images) > 0:
        fid_metric = FrechetInceptionDistance(normalize=False).to(device)
        
        # Convert and batch process real images
        batch_size = 32
        for i in range(0, len(real_images), batch_size):
            batch = real_images[i:i+batch_size]
            batch_tensor = torch.stack([torch.from_numpy(img).permute(2,0,1) for img in batch]).to(device)
            fid_metric.update(batch_tensor.to(torch.uint8), real=True)
        
        # Convert and batch process generated images
        for i in range(0, len(gen_images), batch_size):
            batch = gen_images[i:i+batch_size]
            batch_tensor = torch.stack([torch.from_numpy(img).permute(2,0,1) for img in batch]).to(device)
            fid_metric.update(batch_tensor.to(torch.uint8), real=False)
        
        fid_value = fid_metric.compute().item()

    # Summarize results
    results = {
        # 'average_lpips': np.mean(all_lpips) if all_lpips else None,
        # 'average_dists': np.mean(all_dists) if all_dists else None,
        'fid': fid_value,
        # 'frame_results': frame_results
    }

    print(fid_value, metadata_path)

    # Save results
    output_subdir = os.path.join(os.path.dirname(metadata_path), subdirectory)
    os.makedirs(output_subdir, exist_ok=True)
    output_path = os.path.join(output_subdir, 'metrics_results.json')
    with open(output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

if __name__ == "__main__":
    main()