import numpy as np

# Load and print the .npy file
def print_npy_file(file_path):
    try:
        # Load the .npy file
        data = np.load(file_path)
        
        # Print basic information
        print(f"File: {file_path}")
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Size: {data.size} elements")
        print("\nContents:")
        print(data)
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error loading file: {e}")

# Usage
file_path = "/home/kong/workspace/MOMOE/MOMoE/models/qmo/qmo_assistant_1303b/epoch_400_step_400/qtable.npy"  # Replace with your file path
print_npy_file(file_path)