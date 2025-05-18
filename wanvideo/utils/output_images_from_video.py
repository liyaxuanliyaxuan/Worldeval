import cv2
import os
import argparse

def extract_frames(video_path, output_dir, interval=1, prefix='frame_', format='jpg'):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        # 如果没有读取到帧则退出循环
        if not ret:
            break
        
        # 按间隔保存帧
        if frame_count % interval == 0:
            # 生成文件名
            filename = f"{prefix}{saved_count:04d}.{format}"
            output_path = os.path.join(output_dir, filename)
            
            # 保存帧
            cv2.imwrite(output_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    # 释放资源
    cap.release()
    print(f"成功保存 {saved_count} 帧到 {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从视频中提取帧')
    parser.add_argument('--input', required=True, help='输入视频文件路径')
    parser.add_argument('--output', required=True, help='输出目录路径')
    parser.add_argument('--interval', type=int, default=1, 
                       help='帧间隔（每N帧保存一帧，默认1）')
    parser.add_argument('--prefix', default='frame_', 
                       help='文件名前缀（默认：frame_）')
    parser.add_argument('--format', default='jpg', 
                       help='图片格式（默认jpg，可选png等）')
    
    args = parser.parse_args()
    
    extract_frames(
        video_path=args.input,
        output_dir=args.output,
        interval=args.interval,
        prefix=args.prefix,
        format=args.format
    )