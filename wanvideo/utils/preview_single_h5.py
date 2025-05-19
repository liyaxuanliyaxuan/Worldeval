import h5py
import argparse
import numpy as np

def preview_h5_file(file_path):
    """
    Preview the contents of an HDF5 file, displaying dataset structure and basic statistics
    """
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\n{'='*50}")
            print(f"File path: {file_path}")
            print(f"{'='*50}\n")
            # if 'reasoning' in f:
            #     print("reasoning: ", f"{f['reasoning'][:]}")
            # if 'raw_language' in f:
            #     print("raw_language: ", f"{f['raw_language'][:]}")
            # if 'language_raw' in f:
            #     print("language_raw: ", f"{f['language_raw'][:]}")
            # print(f"{f['reasoning'][:].shape}")

            def print_dataset_info(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"\nDataset name: {name}")
                    print(f"Shape: {obj.shape}")
                    print(f"Data type: {obj.dtype}")
                    
                    # If the dataset is not too large, display some basic statistics
                    if np.prod(obj.shape) < 1000000:  # Limit size to avoid memory issues
                        data = obj[()]
                        if np.issubdtype(obj.dtype, np.number):
                            print(f"Minimum: {np.min(data)}")
                            print(f"Maximum: {np.max(data)}")
                            print(f"Mean: {np.mean(data)}")
                            print(f"Standard deviation: {np.std(data)}")
                    
                    print("-" * 30)
            
            # Traverse all datasets in the file
            f.visititems(print_dataset_info)

    except Exception as e:
        print(f"Error: Unable to read file {file_path}")
        print(f"Error message: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Preview HDF5 file contents')
    parser.add_argument('file_path', type=str, help='Path to HDF5 file')
    args = parser.parse_args()
    
    preview_h5_file(args.file_path)

if __name__ == "__main__":
    main()
