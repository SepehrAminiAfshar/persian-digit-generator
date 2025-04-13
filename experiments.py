"""
List of commands for running different experiments with main.py.
Each list contains sublists where one parameter is varied while others are fixed.
"""

import os
from typing import List
from visualization import plot_experiment_results
import argparse
from main import run_training
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Experiments varying activation functions
activation_experiments = [
    # Vary activation with 3 parents, 20x20
    [
        "medium_3parents_relu_pos_20",
        "medium_3parents_silu_pos_20",
    ],
    # Vary activation with 3 parents, 40x40
    [
        "medium_3parents_relu_pos_40",
        "medium_3parents_silu_pos_40",
    ],
    # Vary activation with 3 parents, 65x65
    [
        "medium_3parents_relu_pos_65",
        "medium_3parents_silu_pos_65",
    ]
]

# Experiments varying number of parents
parents_experiments = [
    # Vary parents with ReLU, 20x20
    [
        "medium_3parents_relu_pos_20",
        "medium_8parents_relu_pos_20",
        "medium_15parents_relu_pos_20",
    ],
    # Vary parents with ReLU, 40x40
    [
        "medium_3parents_relu_pos_40",
        "medium_8parents_relu_pos_40",
        "medium_15parents_relu_pos_40",
    ],
    # Vary parents with ReLU, 65x65
    [
        "medium_3parents_relu_pos_65",
        "medium_8parents_relu_pos_65",
        "medium_15parents_relu_pos_65",
    ]
]

# Experiments varying image resolution
resolution_experiments = [
    # Vary resolution with 3 parents, ReLU
    [
        "medium_3parents_relu_pos_20",
        "medium_3parents_relu_pos_40",
        "medium_3parents_relu_pos_65",
    ],
    # Vary resolution with 8 parents, ReLU
    [
        "small_8parents_relu_pos_20",
        "small_8parents_relu_pos_40",
        "small_8parents_relu_pos_65",
    ],
    # Vary resolution with 15 parents, ReLU
    [
        "small_15parents_relu_pos_20",
        "small_15parents_relu_pos_40",
        "small_15parents_relu_pos_65",
    ]
]

# Experiments varying positional encoding
positional_experiments = [
    # Vary positional encoding with 3 parents, 20x20
    [
        "medium_3parents_relu_pos_20",
        "medium_3parents_relu_20",
    ],
    # Vary positional encoding with 3 parents, 40x40
    [
        "medium_3parents_relu_pos_40",
        "medium_3parents_relu_40",
    ],
    # Vary positional encoding with 3 parents, 65x65
    [
        "medium_3parents_relu_pos_65",
        "medium_3parents_relu_65",
    ]
]

# Experiments varying model size (linear, small, medium)
size_experiments = [
    # Vary model size with 3 parents, 20x20
    [
        "linear_3parents_relu_pos_20",
        "small_3parents_relu_pos_20",
        "medium_3parents_relu_pos_20",
    ],
    # Vary model size with 3 parents, 40x40
    [
        "linear_3parents_relu_pos_40",
        "small_3parents_relu_pos_40",
        "medium_3parents_relu_pos_40",
    ],
    # Vary model size with 3 parents, 65x65
    [
        "linear_3parents_relu_pos_65",
        "small_3parents_relu_pos_65",
        "medium_3parents_relu_pos_65",
    ]
]

def run_experiment(config_names: List[str]) -> List[float]:
    """
    Run a list of configurations and collect test losses.
    
    Args:
        config_names (List[str]): List of configuration names to run
        
    Returns:
        List[float]: List of test losses from each configuration
    """
    test_losses = []
    total_configs = len(config_names)
    
    for i, config_name in enumerate(config_names, 1):
        logger.info(f"Running configuration {i}/{total_configs}: {config_name}")
        test_loss = run_training(
            config_name=config_name,
            batch_size=32,
            num_epochs=4,
            learning_rate=0.001
        )
        test_losses.append(test_loss)
        logger.info(f"Completed configuration {i}/{total_configs} with test loss: {test_loss:.4f}")
    
    return test_losses

def run_activation_experiments() -> None:
    """Run experiments varying activation functions and plot results."""
    logger.info("Starting activation function experiments...")
    results = []
    total_groups = len(activation_experiments)
    
    for i, config in enumerate(activation_experiments, 1):
        logger.info(f"Running activation experiment group {i}/{total_groups}")
        test_losses = run_experiment(config)
        results.append(test_losses)
        logger.info(f"Completed activation experiment group {i}/{total_groups}")
    
    logger.info("Plotting activation experiment results...")
    x_values = ['ReLU', 'SiLU']
    plot_experiment_results(
        results,
        x_values,
        "Test Loss vs Activation Function",
        "Activation Function"
    )
    logger.info("Completed activation function experiments")

def run_parents_experiments() -> None:
    """Run experiments varying number of parents and plot results."""
    logger.info("Starting parents experiments...")
    results = []
    total_groups = len(parents_experiments)
    
    for i, config in enumerate(parents_experiments, 1):
        logger.info(f"Running parents experiment group {i}/{total_groups}")
        test_losses = run_experiment(config)
        results.append(test_losses)
        logger.info(f"Completed parents experiment group {i}/{total_groups}")
    
    logger.info("Plotting parents experiment results...")
    x_values = ['3 Parents', '8 Parents', '15 Parents']
    plot_experiment_results(
        results,
        x_values,
        "Test Loss vs Number of Parents",
        "Number of Parents"
    )
    logger.info("Completed parents experiments")

def run_resolution_experiments() -> None:
    """Run experiments varying image resolution and plot results."""
    logger.info("Starting resolution experiments...")
    results = []
    total_groups = len(resolution_experiments)
    
    for i, config in enumerate(resolution_experiments, 1):
        logger.info(f"Running resolution experiment group {i}/{total_groups}")
        test_losses = run_experiment(config)
        results.append(test_losses)
        logger.info(f"Completed resolution experiment group {i}/{total_groups}")
    
    logger.info("Plotting resolution experiment results...")
    x_values = ['20x20', '40x40', '65x65']
    plot_experiment_results(
        results,
        x_values,
        "Test Loss vs Image Resolution",
        "Image Resolution"
    )
    logger.info("Completed resolution experiments")

def run_positional_experiments() -> None:
    """Run experiments varying positional encoding and plot results."""
    logger.info("Starting positional encoding experiments...")
    results = []
    total_groups = len(positional_experiments)
    
    for i, config in enumerate(positional_experiments, 1):
        logger.info(f"Running positional experiment group {i}/{total_groups}")
        test_losses = run_experiment(config)
        results.append(test_losses)
        logger.info(f"Completed positional experiment group {i}/{total_groups}")
    
    logger.info("Plotting positional encoding experiment results...")
    x_values = ['With Positional Encoding', 'Without Positional Encoding']
    plot_experiment_results(
        results,
        x_values,
        "Test Loss vs Positional Encoding",
        "Positional Encoding"
    )
    logger.info("Completed positional encoding experiments")

def run_size_experiments() -> None:
    """Run experiments varying model size and plot results."""
    logger.info("Starting model size experiments...")
    results = []
    total_groups = len(size_experiments)
    
    for i, config in enumerate(size_experiments, 1):
        logger.info(f"Running size experiment group {i}/{total_groups}")
        test_losses = run_experiment(config)
        results.append(test_losses)
        logger.info(f"Completed size experiment group {i}/{total_groups}")
    
    logger.info("Plotting model size experiment results...")
    x_values = ['Linear', 'Small', 'Medium']
    plot_experiment_results(
        results,
        x_values,
        "Test Loss vs Model Size",
        "Model Size"
    )
    logger.info("Completed model size experiments")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Bayesian Network experiments')
    parser.add_argument('--experiment', type=str, choices=['activation', 'parents', 'resolution', 'positional', 'size', 'all'],
                      default='all', help='Which experiment to run (default: all)')
    
    args = parser.parse_args()
    
    logger.info(f"Starting experiments with option: {args.experiment}")
    
    if args.experiment == 'activation' or args.experiment == 'all':
        run_activation_experiments()
    if args.experiment == 'parents' or args.experiment == 'all':
        run_parents_experiments()
    if args.experiment == 'resolution' or args.experiment == 'all':
        run_resolution_experiments()
    if args.experiment == 'positional' or args.experiment == 'all':
        run_positional_experiments()
    if args.experiment == 'size' or args.experiment == 'all':
        run_size_experiments()
    
    logger.info("All experiments completed") 