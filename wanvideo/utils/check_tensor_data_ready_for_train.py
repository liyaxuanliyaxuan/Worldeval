import torch

def test_action_data(pt_file_path, key='your_key'):
    try:
        # Load the .pt file
        data = torch.load(pt_file_path, map_location='cpu', weights_only=False)
        
        # Check if the data is a dictionary and contains the specified key
        if isinstance(data, dict):
            if key in data:
                action_data = data[key]
                print(f"Action data for key '{key}':")
                # print(action_data)
                # print(f"Shape of action data: {action_data.shape}")
                
                # Check and print action_min and action_max if they exist
                if 'action_min' in data:
                    print(f"Action min: {data['action_min']}")
                else:
                    print("Field 'action_min' not found in the .pt file.")
                
                if 'action_max' in data:
                    print(f"Action max: {data['action_max']}")
                else:
                    print("Field 'action_max' not found in the .pt file.")
            else:
                print(f"Key '{key}' not found in the .pt file.")
        else:
            print("The loaded data is not a dictionary. Please check the file format.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
pt_file_path = 'data/onehot_example_dataset/actions.pt'  # Replace with your .pt file path
test_action_data(pt_file_path, key='encoded_action')  # Replace 'your_key' with the actual key