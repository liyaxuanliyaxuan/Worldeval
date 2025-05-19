import torch
import argparse
from pathlib import Path
import numpy


def preview_pt_file(pt_path, max_items=10):
    """Preview list data saved in PT file
    
    Args:
        pt_path: Path to PT file
        max_items: Maximum number of list items to display, default 10 items
    """
    # Check if file exists
    if not Path(pt_path).exists():
        print(f"Error: File {pt_path} does not exist")
        return

    try:
        # Load PT file
        data = torch.load(pt_path, weights_only=False)
        
        # Ensure loaded data is list or dictionary type
        if not isinstance(data, (list, dict)):
            print(f"Warning: Loaded data is not a list or dictionary type, but {type(data)}")
            return
            
        # Get total length of list or dictionary
        total_length = len(data)
        print(f"\nFile: {pt_path}")
        print(f"List/Dictionary total length: {total_length}")
        
        # Display first few items
        print(f"\nFirst {min(max_items, total_length)} items:")
        if isinstance(data, list):
            for i, item in enumerate(data[:max_items]):
                print(f"\n[{i}] Type: {type(item)}")
                if isinstance(item, (torch.Tensor, list, tuple, dict)):
                    print(f"Shape/Length: {len(item) if isinstance(item, (list, tuple, dict)) else item.shape}")
                print(f"Content: {item}")
        elif isinstance(data, dict):
            for i, (key, value) in enumerate(data.items()):
                if i >= max_items:
                    break
                print(f"\nKey: {key} Type: {type(value)}")
                if isinstance(value, (torch.Tensor, list, tuple, dict)):
                    print(f"Shape/Length: {len(value) if isinstance(value, (list, tuple, dict)) else value.shape}")
                # if key == 'encoded_action':              
                #     if isinstance(value, list):
                #         print(f"encoded_action[0] Shape: {value[0].shape if isinstance(value[0], (torch.Tensor, numpy.ndarray)) else 'N/A'}")
                #     else:
                #         print(f"encoded_action Shape: {value.shape if isinstance(value, (torch.Tensor, numpy.ndarray)) else 'N/A'}")
                if isinstance(value, list) and len(value) > 0:
                    print(f"First item: {value[0]}")
                    print(f"First item shape: {value[0].shape if isinstance(value[0], (torch.Tensor, numpy.ndarray)) else 'N/A'}")
                # print(f"Content: {value}")
            
    except Exception as e:
        print(f"Error: Exception occurred while previewing file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Preview list data saved in PT file')
    parser.add_argument('pt_path', type=str, help='Path to PT file')
    parser.add_argument('--max-items', type=int, default=2, help='Maximum number of list items to display')
    
    args = parser.parse_args()
    preview_pt_file(args.pt_path, args.max_items)

if __name__ == '__main__':
    main() 