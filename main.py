import argparse
import yaml
from typing import Dict, Any, Optional
import os
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from models.config import NetworkConfig
from trainer import BayesianTrainer
from data_loader import prepare_data_for_training, get_resolution_str
from visualization import save_samples, plot_samples


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load a specific configuration from the YAML file.
    
    Args:
        config_name (str): Name of the configuration to load
        
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        ValueError: If config_name is not found in the YAML file
    """
    with open('configurations.yaml', 'r') as f:
        configs = yaml.safe_load(f)
    
    if config_name not in configs:
        raise ValueError(
            f"Configuration '{config_name}' not found. "
            f"Available configurations: {list(configs.keys())}"
        )
    
    return configs[config_name]

def create_network_config(config_dict: Dict[str, Any]) -> NetworkConfig:
    """
    Create a NetworkConfig object from a configuration dictionary.
    
    Args:
        config_dict (Dict[str, Any]): Configuration dictionary from YAML
        
    Returns:
        NetworkConfig: Initialized configuration object
    """
    return NetworkConfig(
        connectivity=config_dict['connectivity'],
        hidden_layers=config_dict['hidden_layers'],
        activation=config_dict['activation'],
        use_positional=config_dict['use_positional'],
        image_resolution=tuple(config_dict['image_resolution'])
    )

def run_training(config_name: str, batch_size: int = 32, num_epochs: int = 100, 
                learning_rate: float = 0.001, save_path: Optional[str] = None) -> float:
    """
    Run training with the specified configuration.
    
    Args:
        config_name (str): Name of the configuration to use
        batch_size (int): Batch size for training
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate for optimization
        save_path (Optional[str]): Path to save the trained model
        
    Returns:
        float: Test loss of the trained model
    """
    logger.info(f"Starting training with configuration: {config_name}")
    logger.info(f"Parameters: batch_size={batch_size}, num_epochs={num_epochs}, learning_rate={learning_rate}")
    
    # Create output directories
    output_dir = 'output'
    test_loss_dir = os.path.join(output_dir, 'test_loss')
    train_valid_dir = os.path.join(output_dir, 'train_valid_loss_over_epoch')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(test_loss_dir, exist_ok=True)
    os.makedirs(train_valid_dir, exist_ok=True)
    
    # Load configuration
    logger.info("Loading configuration...")
    config_dict = load_config(config_name)
    network_config = create_network_config(config_dict)
    
    # Get resolution string for data loading
    resolution = get_resolution_str(config_dict)
    logger.info(f"Image resolution: {resolution}")
    
    # Prepare data loaders
    logger.info("Preparing data loaders...")
    train_loader, val_loader, test_loader = prepare_data_for_training(
        resolution=resolution,
        batch_size=batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        num_workers=4
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = BayesianTrainer(
        image_size=network_config.get_image_resolution(),
        config=network_config,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    
    # Train the model
    logger.info(f"Starting training for {num_epochs} epochs...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        config_name=config_name
    )
    
    # Test the model
    logger.info("Evaluating on test set...")
    test_loss = trainer.test(test_loader, config_name)
    logger.info(f"Test Loss: {test_loss:.4f}")
    
    # Generate and visualize samples
    logger.info("Generating samples...")
    samples = trainer.generate_samples(num_samples=5)
    
    # Save samples to files
    logger.info("Saving generated samples...")
    save_samples(samples, config_name)
    
    # # Plot samples in a grid
    # logger.info("Plotting samples...")
    # plot_samples(samples, config_name)
    
    logger.info(f"Completed training for configuration: {config_name}")
    return test_loss

def main():
    """Command line interface for running training."""
    parser = argparse.ArgumentParser(description='Train Bayesian Network')
    parser.add_argument('--config_name', type=str, required=True,
                      help='Name of the configuration from configurations.yaml')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate for optimization')
    parser.add_argument('--save_path', type=str, default='model.pth',
                      help='Path to save the trained model')
    
    args = parser.parse_args()
    test_loss = run_training(
        config_name=args.config_name,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_path=args.save_path
    )
    print(f"Test Loss: {test_loss}")

if __name__ == "__main__":
    main()
