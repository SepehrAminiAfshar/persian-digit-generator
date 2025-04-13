from dataclasses import dataclass, field
from typing import Literal, List, Tuple
import torch
import torch.nn as nn

CONNECTIVITY_TO_PARENT_OFFSETS = {
    3: [(0, -1), (-1, 0), (-1, -1)],
    8: [(0, -1), (-1, 0), (-1, -1), (0, -2), (-2, 0), (-2, -1), (-1, -2), (-2, -2)],
    15: [(0, -1), (-1, 0), (-1, -1), (0, -2), (-2, 0), (-2, -1), (-1, -2), (-2, -2), (0, -3), (-3, 0), (-3, -1), (-1, -3), (-3, -2), (-2, -3), (-3, -3)]
}

@dataclass(frozen=True)
class NetworkConfig:
    """
    Configuration class for Bayesian network parameters.
    
    This class stores all the configuration parameters needed to initialize
    and customize a Bayesian network. It uses type hints and literal types
    to ensure valid parameter values.
    
    Attributes:
        connectivity (Literal[3, 8, 15]): Number of parent nodes to consider.
            Must be one of: 3 (immediate neighbors), 8 (2-step neighbors),
            or 15 (3-step neighbors).
        hidden_layers (Literal[0, 1, 2]): Number of hidden layers in the MLP.
            Must be one of: 0 (linear model), 1 (one hidden layer),
            or 2 (two hidden layers).
        activation (Literal['relu', 'silu']): Activation function to use in the network.
            Must be one of: 'relu' (Rectified Linear Unit) or 'silu' (Sigmoid Linear Unit).
        use_positional (bool): Whether to include normalized x,y coordinates
            as additional features in the input.
        device (torch.device): Device to run the model on (CPU/GPU).
            Defaults to CUDA if available, else CPU.
        _mlp_model (nn.Module): Private field to store the cached MLP model.
            Initialized in __post_init__ and never modified afterward.
        image_resolution (Tuple[int, int]): Resolution of the images (height, width)
    """
    connectivity: Literal[3, 8, 15]
    hidden_layers: Literal[0, 1, 2]
    activation: Literal['relu', 'silu']
    use_positional: bool
    device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    _mlp_model: nn.Module = field(init=False)
    image_resolution: Tuple[int, int] = (28, 28)  # Default to MNIST size
    
    def __post_init__(self):
        """
        Validate the configuration parameters and initialize the MLP model.
        
        Raises:
            ValueError: If connectivity is not 3, 8, or 15
            ValueError: If hidden_layers is not 0, 1, or 2
            ValueError: If activation is not 'relu' or 'silu'
            ValueError: If device is not 'cuda' or 'cpu'
            ValueError: If image resolution contains non-positive values
        """
        if self.connectivity not in [3, 8, 15]:
            raise ValueError("Connectivity must be 3, 8, or 15")
        if self.hidden_layers not in [0, 1, 2]:
            raise ValueError("Hidden layers must be 0, 1, or 2")
        if self.activation not in ['relu', 'silu']:
            raise ValueError("Activation must be 'relu' or 'silu'")
        if any(dim <= 0 for dim in self.image_resolution):
            raise ValueError("image resolution dimensions must be positive")
        
        # Initialize the MLP model
        input_size = self.get_input_size()
        activation = self.get_activation()
        
        def _build_mlp_model(input_size: int, activation: nn.Module) -> nn.Module:
            """
            Build the MLP model based on number of hidden layers.
            
            Args:
                input_size (int): Size of the input features
                activation (nn.Module): Activation function to use
                
            Returns:
                nn.Module: The constructed MLP model
            """
            if self.hidden_layers == 0:
                # Simple linear model
                return nn.Linear(input_size, 1, bias=True)
            elif self.hidden_layers == 1:
                # MLP with one hidden layer
                return nn.Sequential(
                    nn.Linear(input_size, 64, bias=True),
                    activation,
                    nn.Linear(64, 1, bias=True)
                )
            else:  # hidden_layers == 2
                # MLP with two hidden layers
                return nn.Sequential(
                    nn.Linear(input_size, 64, bias=True),
                    activation,
                    nn.Linear(64, 64, bias=True),
                    activation,
                    nn.Linear(64, 1, bias=True)
                )
        
        model = _build_mlp_model(input_size, activation)
        
        # Use object.__setattr__ to bypass frozen=True
        object.__setattr__(self, '_mlp_model', model)
    
    def get_activation(self) -> nn.Module:
        """
        Get the PyTorch activation function module based on the configuration.
        
        Returns:
            nn.Module: The corresponding PyTorch activation function
        """
        if self.activation == 'relu':
            return nn.ReLU()
        else:  # silu
            return nn.SiLU()
    
    def get_parent_offsets(self) -> List[Tuple[int, int]]:
        """
        Get the list of parent offsets based on the connectivity configuration.
        
        Returns:
            List[Tuple[int, int]]: List of (row, column) offsets for parent positions
        """
        return CONNECTIVITY_TO_PARENT_OFFSETS[self.connectivity]
    
    def get_device(self) -> torch.device:
        """
        Get the device configuration.
        
        Returns:
            torch.device: The configured device (CPU/GPU)
        """
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Using CPU instead.")
            return torch.device('cpu')
        return self.device
    
    def get_use_positional(self) -> bool:
        """
        Get whether positional features should be used.
        
        Returns:
            bool: True if positional features should be included, False otherwise
        """
        return self.use_positional
    
    def get_input_size(self) -> int:
        """
        Get the input size for the network based on connectivity and positional features.
        
        The input size is:
        - connectivity if positional features are not used
        - connectivity + 2 if positional features are used (for x,y coordinates)
        
        Returns:
            int: The size of the input tensor
        """
        input_size = self.connectivity
        if self.use_positional:
            input_size += 2  # Add 2 for x,y coordinates
        return input_size
    
    def get_mlp_model(self) -> nn.Module:
        """
        Get the cached MLP model.
        
        The model is created once during initialization and never modified.
        This ensures consistent behavior and prevents accidental modifications.
        
        Returns:
            nn.Module: The cached MLP model
        """
        return self._mlp_model

    def get_image_resolution(self) -> Tuple[int, int]:
        """
        Get the configured image resolution.
        
        Returns:
            Tuple[int, int]: The image dimensions as (height, width)
        """
        return self.image_resolution
