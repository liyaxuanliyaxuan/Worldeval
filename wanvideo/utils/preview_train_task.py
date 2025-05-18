import h5py
import numpy as np
import random
import os
from PIL import Image

# 读取文件路径
file_path = 'train_files.txt'
with open(file_path, 'r') as f:
    lines = f.readlines()

# 随机选择100个文件
selected_files = random.sample(lines, 100)

# 指定输出目录
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# 提取每个文件的第一帧
for file in selected_files:
    file = file.strip()  # 去掉换行符
    if os.path.exists(file):
        with h5py.File(file, 'r') as hdf:
            # 假设数据存储在 'observations/images/cam_high' 路径下
            if 'observations/images/cam_high' in hdf:
                first_frame = hdf['observations/images/cam_high'][0]
                # 将第一帧转换为图像并保存到指定目录
                image = Image.fromarray(first_frame)
                output_path = os.path.join(output_dir, f"{os.path.basename(file)}_first_frame.png")
                image.save(output_path)
                print(f"First frame from {file} saved to {output_path}")
            else:
                print(f"Path 'observations/images/cam_high' not found in {file}")
    else:
        print(f"File {file} does not exist.")