import cv2
import os
import argparse

def extract_frames(video_path, output_dir, interval=1, prefix='frame_', format='jpg'):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        # Exit loop if no frame is read
        if not ret:
            break
        
        # Save frames at specified interval
        if frame_count % interval == 0:
            # Generate filename
            filename = f"{prefix}{saved_count:04d}.{format}"
            output_path = os.path.join(output_dir, filename)
            
            # Save frame
            cv2.imwrite(output_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    # Release resources
    cap.release()
    print(f"Successfully saved {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from video')
    parser.add_argument('--input', required=True, help='Input video file path')
    parser.add_argument('--output', required=True, help='Output directory path')
    parser.add_argument('--interval', type=int, default=1, 
                       help='Frame interval (save every N frames, default: 1)')
    parser.add_argument('--prefix', default='frame_', 
                       help='Filename prefix (default: frame_)')
    parser.add_argument('--format', default='jpg', 
                       help='Image format (default: jpg, options: png, etc.)')
    
    args = parser.parse_args()
    
    extract_frames(
        video_path=args.input,
        output_dir=args.output,
        interval=args.interval,
        prefix=args.prefix,
        format=args.format
    )