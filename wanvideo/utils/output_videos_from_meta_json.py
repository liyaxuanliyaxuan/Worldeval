import json
import os
import h5py
import numpy as np
import cv2
import imageio

def create_video_from_hdf5(hdf5_file_path, output_folder, fps=15):
    """
    从HDF5文件创建视频
    Args:
        hdf5_file_path: HDF5文件路径
        output_folder: 输出文件夹路径
        fps: 视频帧率
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 生成输出视频文件路径
    video_name_first = os.path.basename(hdf5_file_path).replace('.hdf5', '_frame_0.mp4')
    video_path_first = os.path.join(output_folder, video_name_first)
    video_name_last = os.path.basename(hdf5_file_path).replace('.hdf5', '_frame_100.mp4')
    video_path_last = os.path.join(output_folder, video_name_last)
    
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        # 获取图像数据
        images = hdf5_file['observations/images/cam_high']

        # 打印图像数据
        print(images.shape)
        
        # 使用imageio创建视频
        writer_first = imageio.get_writer(video_path_first, fps=fps)
        
        # 遍历前100帧并写入视频
        for image_data in images[:100]:
            frame = np.array(image_data)
            # # OpenCV使用BGR格式，如果输入是RGB需要转换
            # if frame.shape[-1] == 3:  # 确认是彩色图像
            #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer_first.append_data(frame)
        
        # 关闭writer
        writer_first.close()

        # 仅当图像数量大于100时生成后100帧视频
        if len(images) > 100:
            writer_last = imageio.get_writer(video_path_last, fps=fps)
            for image_data in images[-100:]:
                frame = np.array(image_data)
                writer_last.append_data(frame)
            writer_last.close()
        
    return video_path_first, video_path_last if len(images) > 100  else None

def main():
    # 设置输入输出路径
    json_path = ''
    output_base_dir = ''
    
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 记录已处理的HDF5文件
    processed_paths = set()
    
    # 遍历JSON数据
    for item in data:
        hdf5_file_path = item['file_path']
        
        # 跳过已处理的文件
        if hdf5_file_path in processed_paths:
            continue
        
        processed_paths.add(hdf5_file_path)
        
        try:
            # 创建视频
            video_path_first, video_path_last = create_video_from_hdf5(
                hdf5_file_path=hdf5_file_path,
                output_folder=output_base_dir
            )
            print(f"First 100 frames video created successfully: {video_path_first}")
            if video_path_last is not None:
                print(f"Last 100 frames video created successfully: {video_path_last}")
            
        except Exception as e:
            print(f"Error processing {hdf5_file_path}: {str(e)}")

if __name__ == "__main__":
    main() 
