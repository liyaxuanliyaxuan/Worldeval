import pandas as pd
import h5py
import os

def filter_hdf5_files(csv_path, output_csv_path, frame_threshold=700):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize a list to store rows with total_frames > frame_threshold
    filtered_rows = []
    
    print(f"Checking files for total_frames > {frame_threshold}...")
    
    for idx, row in df.iterrows():
        file_path = row['file_path']
        try:
            with h5py.File(file_path, 'r') as f:
                # Get frames data
                frames = f['observations/images/cam_high']
                total_frames = frames.shape[0]

                print(f"File {file_path} has {total_frames} frames.")
                
                # Check if total_frames is greater than the threshold
                if total_frames > frame_threshold:
                    filtered_rows.append(row)
                    print(f"File {file_path} has {total_frames} frames, added to filtered list.")
                    
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Create a new DataFrame from the filtered rows
    filtered_df = pd.DataFrame(filtered_rows)
    
    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv_path, index=False)
    
    print(f"\nCompleted! Saved filtered data to {output_csv_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Filter HDF5 files with total_frames > threshold')
    parser.add_argument('--csv', required=True, help='Path to the metadata CSV file')
    parser.add_argument('--output_csv', required=True, help='Output CSV file for filtered data')
    parser.add_argument('--frame_threshold', type=int, default=700, help='Frame threshold for filtering')
    
    args = parser.parse_args()
    
    filter_hdf5_files(args.csv, args.output_csv, args.frame_threshold)