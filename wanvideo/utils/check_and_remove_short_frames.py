import pandas as pd
import h5py
import os
from tqdm import tqdm

def check_frames(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Required minimum number of frames
    MIN_FRAMES = 81
    
    # Initialize list to store problematic files
    problematic_files = []
    
    print(f"Checking files for minimum {MIN_FRAMES} frames...")
    
    # Iterate through each file in the CSV
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Get file path
        file_path = row['file_path']
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Get the number of frames
                data = f['observations/images/cam_high']
                total_frames = data.shape[0]
                
                # Check if frames are less than required
                if total_frames < MIN_FRAMES:
                    problematic_files.append(idx)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Remove problematic files from DataFrame
    df_cleaned = df.drop(index=problematic_files)
    
    # Save the cleaned DataFrame back to the CSV
    df_cleaned.to_csv(csv_path, index=False)
    
    # Print results
    if problematic_files:
        print("\nFiles with insufficient frames removed from CSV:")
        print("--------------------------------")
        for idx in problematic_files:
            print(f"File: {df.loc[idx, 'file_path']}")
        print(f"\nTotal problematic files removed: {len(problematic_files)}")
    else:
        print("\nNo files found with insufficient frames.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Check HDF5 files for minimum frame count')
    parser.add_argument('--csv', required=True, help='Path to the metadata CSV file')
    
    args = parser.parse_args()
    
    check_frames(args.csv) 