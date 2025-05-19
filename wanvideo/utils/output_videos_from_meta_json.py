import json
import os
import h5py
import numpy as np
import cv2
import imageio

def create_video_from_hdf5(hdf5_file_path, output_folder, fps=15):
    """
    Create videos from HDF5 file
    Args:
        hdf5_file_path: Path to HDF5 file
        output_folder: Path to output folder
        fps: Video frame rate
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate output video file paths
    video_name_first = os.path.basename(hdf5_file_path).replace('.hdf5', '_frame_0.mp4')
    video_path_first = os.path.join(output_folder, video_name_first)
    video_name_last = os.path.basename(hdf5_file_path).replace('.hdf5', '_frame_100.mp4')
    video_path_last = os.path.join(output_folder, video_name_last)
    
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        # Get image data
        images = hdf5_file['observations/images/cam_high']

        # Print image data shape
        print(images.shape)
        
        # Use imageio to create video
        writer_first = imageio.get_writer(video_path_first, fps=fps)
        
        # Iterate through first 100 frames and write to video
        for image_data in images[:100]:
            frame = np.array(image_data)
            # # OpenCV uses BGR format, if input is RGB need to convert
            # if frame.shape[-1] == 3:  # Confirm it's a color image
            #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer_first.append_data(frame)
        
        # Close writer
        writer_first.close()

        # Only generate last 100 frames video when image count is greater than 100
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
