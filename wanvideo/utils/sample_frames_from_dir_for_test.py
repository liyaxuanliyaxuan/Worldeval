import h5py
import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

def get_text_from_hdf5(f, frame_id):
    """Extract text description from HDF5 file, return empty string if not found"""
    try:
        if 'substep_reasonings' in f and frame_id < f['substep_reasonings'].shape[0]:
            text = f['substep_reasonings'][frame_id].decode('utf-8').strip()
            if text == "" and 'language_raw' in f:
                text = f['language_raw'][0].decode('utf-8').strip()
        elif 'language_raw' in f:
            text = f['language_raw'][0].decode('utf-8').strip()
        else:
            text = "Clean the table."
    except:
        text = "Clean the table."
    return text

def process_hdf5_files(input_dir, output_dir):
    """Process all HDF5 files in the input directory"""
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize metadata list
    metadata = []
    
    # Get all HDF5 files in input directory with index >= 51
    input_dir = Path(input_dir)
    #  hdf5_files = list(input_dir.glob('*.hdf5'))
    hdf5_files = [file for file in input_dir.glob('*.hdf5') if int(file.stem) >= 0]
    
    print(f"Found {len(hdf5_files)} HDF5 files to process...")
    
    for file_path in hdf5_files:
        try:
            with h5py.File(file_path, 'r') as f:
                # Get frames data
                frames = f['observations/images/cam_high']
                total_frames = frames.shape[0]
                
                # Calculate indices for 5 frames
                # indices = np.linspace(0, total_frames - 1, 1, dtype=int)

                indices = [0]  # Start frame and middle frame
                
                for frame_idx in indices:
                    frame = frames[frame_idx]
                    
                    # Get text description
                    text = get_text_from_hdf5(f, frame_idx)
                    
                    # Create image filename
                    image_filename = f"{file_path.stem}_frame_{frame_idx}.png"
                    image_path = output_dir / image_filename
                    
                    # Save image
                    Image.fromarray(frame).save(image_path)
                    
                    # Add to metadata
                    metadata.append({
                        "image_path": str(image_path),
                        "file_path": str(file_path),
                        "frame_index": int(frame_idx),
                        "language": text
                    })
                    print(metadata)
                    
                    print(f"Processed {file_path.name} - frame {frame_idx}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Save metadata to JSON
    json_path = output_dir / "metadata.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\nCompleted! Saved {len(metadata)} images and metadata to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process HDF5 files and extract frames')
    parser.add_argument('--input', required=True, help='Input directory containing HDF5 files')
    parser.add_argument('--output', required=True, help='Output directory for images and metadata')
    
    args = parser.parse_args()
    
    process_hdf5_files(args.input, args.output)

    # Example usage:
    # python sample_frames_simple.py --input /home/jz08/lyx/hdf5/new_pick_and_place_source/lyx_brown_mug --output ./data/output_frames 