import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_and_save_heatmap(tensor, save_path, title='Heatmap'):
    """
    Plots and saves a heatmap from a 2D tensor, with cell annotations and custom colors.

    Parameters:
    - tensor: A 2D tensor (PyTorch tensor or NumPy array)
    - save_path: Path to save the heatmap image.
    - title: Title of the heatmap.

    Returns:
    - None: Saves the heatmap as an image file.
    """
    # Convert tensor to numpy array if it's not already
    tensor = tensor.numpy()

    plt.figure(figsize=(8, 6))
    
    # Define custom colormap
    cmap = plt.get_cmap('coolwarm')
    
    # Create heatmap
    heatmap = plt.imshow(tensor, cmap=cmap, aspect='auto')
    
    # Annotate cells with the values
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            plt.text(j, i, f'{tensor[i, j]:.2f}', ha='center', va='center', color='black')

    plt.colorbar(heatmap)
    plt.savefig(save_path)
    plt.close()

def process_corr_files(base_path):
    """
    Iterates through the directory structure starting from base_path,
    finds all 'corr.pt' files, generates heatmaps, and saves them.

    Parameters:
    - base_path: The base directory to start searching from.

    Returns:
    - None
    """
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == 'corr.pt':
                corr_file_path = os.path.join(root, file)
                save_path = os.path.join(root, 'heatmap.png')
                
                # Load the tensor from the .pt file
                tensor = torch.load(corr_file_path)
                
                # Ensure the tensor is 2D
                if tensor.ndim == 2:
                    plot_and_save_heatmap(tensor, save_path)
                    print(f"Saved heatmap for {corr_file_path} at {save_path}")
                else:
                    print(f"Skipping {corr_file_path} as it is not a 2D tensor.")

# Set the base path
base_path = '/gpu-data4/dbek/thesis/scores'

# Run the process
process_corr_files(base_path)
