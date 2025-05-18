import json
import os
import h5py
import numpy as np
from PIL import Image

# 读取JSON文件
with open('metadata.json', 'r') as f:
    data = json.load(f)

# 记录已经处理过的HDF5文件路径
processed_paths = set()

# 遍历JSON数据中的每个item
for item in data:
    hdf5_file_path = item['file_path']
    
    # 如果该路径已经处理过，跳过
    if hdf5_file_path in processed_paths:
        continue
    
    # 添加到已处理路径集合
    processed_paths.add(hdf5_file_path)
    
    # 创建保存图像的文件夹，使用HDF5文件名作为文件夹名
    output_folder = os.path.join('', os.path.basename(hdf5_file_path).replace('.hdf5', ''))
    os.makedirs(output_folder, exist_ok=True)
    
    # 读取HDF5文件
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        # 假设数据存储在 'observations/images/cam_high'
        images = hdf5_file['observations/images/cam_high']
        
        # 遍历图像数据并保存
        for index, image_data in enumerate(images):
            # 将图像数据转换为PIL图像
            image = Image.fromarray(np.array(image_data))
            
            # 保存图像，命名为 index_frame.png
            image.save(os.path.join(output_folder, f'{index}_frame.png'))

    print(f"Images saved to {output_folder}")
