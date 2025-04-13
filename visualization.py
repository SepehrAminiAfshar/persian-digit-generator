"""
Module for visualizing and saving generated samples from the Bayesian Network.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Dict, Tuple


def save_samples(samples: List[torch.Tensor], config_name: str) -> None:
    """
    Save generated samples as a grid image.
    
    Args:
        samples (List[torch.Tensor]): List of generated samples as tensors
        config_name (str): Name of the configuration used to generate samples
    """
    # Create output directory
    output_dir = 'output'
    generated_images_dir = os.path.join(output_dir, 'generated_images')
    os.makedirs(generated_images_dir, exist_ok=True)
    
    # Save grid of all samples
    grid_path = os.path.join(generated_images_dir, f"{config_name}_samples.png")
    plt.figure(figsize=(15, 3))
    for i, sample in enumerate(samples):
        plt.subplot(1, len(samples), i + 1)
        img = (sample.cpu().numpy() * 255).astype(np.uint8)
        plt.imshow(img, cmap='gray')
        plt.title(f"Sample {i+1}")
        plt.axis('off')
    plt.suptitle(f"Generated Samples - {config_name}")
    plt.tight_layout()
    plt.savefig(grid_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved samples grid to {grid_path}")


def plot_samples(samples: List[torch.Tensor], config_name: str) -> None:
    """
    Plot generated samples in a grid layout.
    
    Args:
        samples (List[torch.Tensor]): List of generated samples as tensors
        config_name (str): Name of the configuration used to generate samples
    """
    n_samples = len(samples)
    n_cols = min(5, n_samples)  # Maximum 5 columns
    n_rows = (n_samples + n_cols - 1) // n_cols  # Ceiling division
    
    plt.figure(figsize=(n_cols * 3, n_rows * 3))
    
    for i, sample in enumerate(samples):
        plt.subplot(n_rows, n_cols, i + 1)
        img = (sample.cpu().numpy() * 255).astype(np.uint8)
        plt.imshow(img, cmap='gray')
        plt.title(f"Sample {i+1}")
        plt.axis('off')
    
    plt.suptitle(f"Generated Samples - {config_name}")
    plt.tight_layout()
    plt.show()


def plot_training_curves(train_losses: List[float], val_losses: List[float], config_name: str) -> None:
    """
    Plot and save training and validation loss curves.
    
    Args:
        train_losses (List[float]): List of training losses per epoch
        val_losses (List[float]): List of validation losses per epoch
        config_name (str): Name of the configuration used for training
    """
    # Create output directory
    output_dir = 'output'
    train_valid_dir = os.path.join(output_dir, 'train_valid_loss_over_epoch')
    os.makedirs(train_valid_dir, exist_ok=True)
    
    # Create epochs list
    epochs = list(range(1, len(train_losses) + 1))
    
    # Plot the curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Losses - {config_name}')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_path = os.path.join(train_valid_dir, f"{config_name}_losses.png")
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved loss curves to {plot_path}")


def plot_experiment_results(results: List[List[float]], x_values: List[str], title: str, x_label: str, y_label: str = "Test Loss") -> None:
    """
    Plot experiment results with individual lines and mean line.
    
    Args:
        results (List[List[float]]): List of test loss lists for each configuration
        x_values (List[str]): List of x-axis values
        title (str): Plot title
        x_label (str): X-axis label
        y_label (str): Y-axis label
    """
    plt.figure(figsize=(10, 6))
    
    # Plot individual lines
    for i, result in enumerate(results):
        plt.plot(x_values, result, label=f'Configuration {i+1}', alpha=0.5)
    
    # Calculate and plot mean line
    mean_results = np.mean(results, axis=0)
    plt.plot(x_values, mean_results, label='Mean', linewidth=2, color='black', linestyle='--')
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    
    # Save plot
    output_dir = 'output/experiment_results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{title.lower().replace(' ', '_')}.png"))
    plt.close()
    print(f"Saved experiment results plot to {os.path.join(output_dir, f"{title.lower().replace(' ', '_')}.png")}") 