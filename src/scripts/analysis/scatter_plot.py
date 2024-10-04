import torch
import matplotlib.pyplot as plt

def plot_attention_vs_attribution(attention_scores: torch.Tensor, attributions: torch.Tensor):
    """
    Plots a scatter plot to visualize the relationship between attention scores and attributions.
    
    Parameters:
    - attention_scores: A 1D tensor containing attention scores.
    - attributions: A 1D tensor containing corresponding attributions.
    """
    # Ensure the tensors are 1-dimensional and have the same length
    assert attention_scores.dim() == 1, "Attention scores tensor should be 1-dimensional."
    assert attributions.dim() == 1, "Attributions tensor should be 1-dimensional."
    assert attention_scores.size(0) == attributions.size(0), "Attention scores and attributions tensors must have the same length."
    
    # Plot the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(attention_scores.numpy(), attributions.numpy(), alpha=0.6, c='blue', edgecolors='w', s=80)
    plt.title("Scatter Plot of Attention Scores vs. Attributions")
    plt.xlabel("Attention Scores")
    plt.ylabel("Attributions")
    plt.grid(True)
    plt.show()

# Example usage
# Creating dummy tensors for illustration (replace with your actual data)
attention_scores = torch.rand(100)  # Replace with actual attention scores tensor
attributions = torch.rand(100)  # Replace with actual attributions tensor

plot_attention_vs_attribution(attention_scores, attributions)
