# Bayesian Network Image Generation

This project implements a Bayesian network for generating binary images. The network models dependencies between pixels using a configurable architecture that supports different connectivity patterns, activation functions, and positional encoding.

## Project Structure

```
.
├── data_loader.py      # Data loading and preprocessing
├── experiments.py      # Experiment configurations and execution
├── main.py            # Main training script
├── models/            # Model implementations
│   ├── bayesian_network.py
│   └── config.py
├── trainer.py         # Training logic
└── visualization.py   # Visualization utilities
```

## Features

- Configurable Bayesian network architecture
- Support for different connectivity patterns (3, 8, or 15 parents)
- Multiple activation functions (ReLU, SiLU)
- Positional encoding option
- Various model sizes (linear, small, medium)
- Multiple image resolutions (20x20, 40x40, 65x65)
- Comprehensive experiment framework
- Visualization of generated samples and training curves

## Experiments

The project includes several experiments to analyze different aspects of the model:

1. **Activation Functions**: Compare ReLU vs SiLU
2. **Number of Parents**: Test different connectivity patterns (3, 8, 15)
3. **Image Resolution**: Evaluate performance across resolutions
4. **Positional Encoding**: Test with and without positional features
5. **Model Size**: Compare linear, small, and medium architectures

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run experiments:
```bash
python experiments.py --experiment [activation|parents|resolution|positional|size|all]
```

3. View results in the `output` directory:
- Generated samples: `output/generated_images/`
- Training curves: `output/train_valid_loss_over_epoch/`
- Experiment results: `output/experiment_results/`

## Configuration

Model configurations are defined in `configurations.yaml`. Each configuration specifies:
- Connectivity pattern
- Number of hidden layers
- Activation function
- Positional encoding
- Image resolution

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- PyYAML
- tqdm 