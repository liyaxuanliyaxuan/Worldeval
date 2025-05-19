import json
import os
import h5py
import numpy as np
from PIL import Image

# Read JSON file
with open('metadata.json', 'r') as f:
    data = json.load(f)

# Record already processed HDF5 file paths
processed_paths = set()

# Iterate through each item in JSON data
for item in data:
    hdf5_file_path = item['file_path']
    
    # Skip if this path has already been processed
    if hdf5_file_path in processed_paths:
        continue
    
    # Add to processed paths set
    processed_paths.add(hdf5_file_path)
    
    # Create folder to save images, using HDF5 filename as folder name
    output_folder = os.path.join('', os.path.basename(hdf5_file_path).replace('.hdf5', ''))
    os.makedirs(output_folder, exist_ok=True)
    
    # Read HDF5 file
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        # Assume data is stored in 'observations/images/cam_high'
        images = hdf5_file['observations/images/cam_high']
        
        # Iterate through image data and save
        for index, image_data in enumerate(images):
            # Convert image data to PIL image
            image = Image.fromarray(np.array(image_data))
            
            # Save image, named as index_frame.png
            image.save(os.path.join(output_folder, f'{index}_frame.png'))

    print(f"Images saved to {output_folder}")
