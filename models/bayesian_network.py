import torch
import torch.nn as nn

from .config import NetworkConfig

class BayesianNetwork(nn.Module):
    """
    A Bayesian network implementation for modeling dependencies in binary images.
    
    This network models the probability distribution of binary pixels in an image
    based on their spatial relationships with neighboring pixels. The network uses
    a configuration object to determine its architecture and behavior.
    
    Attributes:
        config (NetworkConfig): Configuration object containing all network parameters
        image_size (tuple): Dimensions of the input image (height, width)
        model (nn.Module): MLP model for probability estimation
    """
    
    def __init__(self, image_size: tuple, config: NetworkConfig):
        """
        Initialize the Bayesian network with a configuration object.
        
        Args:
            image_size (tuple): Dimensions of the input image (height, width)
            config (NetworkConfig): Configuration object containing network parameters
        """
        super().__init__()
        self.config = config
        self.image_size = image_size
        self.model = config.get_mlp_model()
        
    def get_parent_values(self, image, i, j):
        """
        Extract parent values from the image for a given position.
        
        This method retrieves the values of parent pixels based on the configured
        connectivity pattern. For positions near the image boundaries, out-of-bounds
        parents are assigned a default value of 0.
        
        Args:
            image (torch.Tensor): Input image tensor
            i (int): Row index of the current pixel
            j (int): Column index of the current pixel
            
        Returns:
            torch.Tensor: Tensor containing parent values and optionally
                normalized positional coordinates
        """
        parent_offsets = self.config.get_parent_offsets()
        parents = []
        
        for di, dj in parent_offsets:
            parent_i, parent_j = i + di, j + dj
            if 0 <= parent_i < self.image_size[0] and 0 <= parent_j < self.image_size[1]:
                parents.append(image[parent_i, parent_j])
            else:
                parents.append(0)
        
        if self.config.get_use_positional():
            # Add normalized positional features
            parents.extend([j/self.image_size[1], i/self.image_size[0]])
            
        return torch.tensor([parents], dtype=torch.float32).to(self.config.get_device())
    
    def forward(self, x):
        """
        Forward pass of the network.
        
        This method processes the input tensor through the network to estimate
        the probability of each pixel being 1, given its parent values.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, height, width)
                containing binary values (0 or 1)
                
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, height, width)
                containing probability values between 0 and 1
        """
        return torch.sigmoid(self.model(x)).squeeze()
