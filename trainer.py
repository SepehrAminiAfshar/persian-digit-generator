import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List

from models.bayesian_network import BayesianNetwork
from models.config import NetworkConfig
from visualization import plot_training_curves

class BayesianTrainer:
    """
    Trainer class for the Bayesian network.
    
    This class handles the training process of the Bayesian network, including
    data loading, optimization, and evaluation.
    
    Attributes:
        network (BayesianNetwork): The Bayesian network model
        config (NetworkConfig): Configuration object for the network
        optimizer (optim.Optimizer): Optimizer for training
        device (torch.device): Device to run training on
    """
    
    def __init__(
        self,
        image_size: tuple,
        config: NetworkConfig,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        """
        Initialize the trainer with network configuration and training parameters.
        
        Args:
            image_size (tuple): Dimensions of the input image (height, width)
            config (NetworkConfig): Configuration object for the network
            learning_rate (float, optional): Learning rate for optimization.
                Defaults to 0.001.
            batch_size (int, optional): Batch size for training.
                Defaults to 32.
            num_workers (int, optional): Number of workers for data loading.
                Defaults to 4.
        """
        self.config = config
        self.device = config.get_device()
        self.image_size = image_size
        
        # Initialize network
        self.network = BayesianNetwork(image_size, config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Initialize learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5
        )
        
        # Training parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Store initial learning rate
        self.initial_lr = learning_rate
        self.current_lr = learning_rate
    
    def _loss(self, predictions: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Sum over image pixels (last two dimensions) and mean over batch (first dimension)
        return -torch.mean(torch.sum(batch * torch.log(predictions + 1e-10) + (1 - batch) * torch.log(1 - predictions + 1e-10), dim=(1, 2)))
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train the network for one epoch.
        
        Args:
            train_loader (DataLoader): DataLoader for training data
            
        Returns:
            float: Average loss for the epoch
        """
        self.network.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            parents = self.get_parents_for_batch(batch)
            predictions = self.network(parents)
            
            batch_loss = self._loss(batch=batch, predictions=predictions)
            
            # Backward pass
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            
            total_loss += batch_loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the network on the validation set.
        
        Args:
            val_loader (DataLoader): DataLoader for validation data
            
        Returns:
            float: Average loss on validation set
        """
        self.network.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Move batch to device and compute loss
                parents = self.get_parents_for_batch(batch)
                predictions = self.network(parents)
                batch_loss = self._loss(batch=batch, predictions=predictions)
                total_loss += batch_loss.item()
        
        return total_loss / len(val_loader)
    
    def get_current_learning_rate(self) -> float:
        """
        Get the current learning rate.
        
        Returns:
            float: Current learning rate
        """
        return self.optimizer.param_groups[0]['lr']
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, config_name: str) -> None:
        """
        Train the Bayesian Network model.
        
        Args:
            train_loader (DataLoader): DataLoader for training data
            val_loader (DataLoader): DataLoader for validation data
            num_epochs (int): Number of epochs to train
            config_name (str): Name of the configuration used for training
        """
        self.network.train()
        
        # Lists to store losses
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)
            
            # Get current learning rate
            current_lr = self.get_current_learning_rate()
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"LR: {current_lr:.6f}")
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Check if learning rate was reduced
            new_lr = self.get_current_learning_rate()
            if new_lr < current_lr:
                print(f"Learning rate reduced from {current_lr:.6f} to {new_lr:.6f}")
        
        # Plot and save training curves
        plot_training_curves(train_losses, val_losses, config_name)

    def get_parents_for_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Process a batch of images and return their parent values.
        
        Args:
            batch (torch.Tensor): Input batch of images of shape (batch_size, height, width)
            
        Returns:
            torch.Tensor: Tensor of shape (batch_size, height, width, num_parents)
                containing parent values for each pixel in the batch, maintaining
                spatial relationships
        """
        batch_size = batch.size(0)
        height, width = self.image_size
        num_parents = len(self.config.get_parent_offsets())
        if self.config.get_use_positional():
            num_parents += 2  # Add space for positional features
            
        # Initialize output tensor
        device = self.config.get_device()
        parents = torch.zeros((batch_size, height, width, num_parents), 
                            dtype=torch.float32, device=device)
        
        # Process each image in the batch
        for b in range(batch_size):
            for i in range(height):
                for j in range(width):
                    # Get parent values for this position
                    parents[b, i, j] = self.network.get_parent_values(batch[b], i, j).squeeze(0)
        
        return parents

    def test(self, test_loader: DataLoader, config_name: str) -> float:
        """
        Evaluate the model on test data and save the test loss.
        
        Args:
            test_loader (DataLoader): DataLoader for test data
            config_name (str): Name of the configuration used for testing
            
        Returns:
            float: Test loss
        """
        self.network.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                # Move batch to device and compute loss
                parents = self.get_parents_for_batch(batch)
                predictions = self.network(parents)
                batch_loss = self._loss(batch=batch, predictions=predictions)
                total_loss += batch_loss.item()
        
        test_loss = total_loss / len(test_loader)
        
        # Create output directory
        output_dir = 'output'
        test_loss_dir = os.path.join(output_dir, 'test_loss')
        os.makedirs(test_loss_dir, exist_ok=True)
        
        # Save test loss
        test_data = {
            'config_name': config_name,
            'test_loss': test_loss
        }
        
        loss_file = os.path.join(test_loss_dir, f'{config_name}_test_loss.json')
        with open(loss_file, 'w') as f:
            json.dump(test_data, f, indent=4)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Saved test loss to {loss_file}")
        
        return test_loss

    def generate_samples(self, num_samples: int = 1) -> List[torch.Tensor]:
        """
        Generate samples from the trained model.
        
        Args:
            num_samples (int, optional): Number of samples to generate.
                Defaults to 1.
                
        Returns:
            List[torch.Tensor]: List of generated samples
        """
        self.network.eval()
        samples = []
        
        for _ in range(num_samples):
            # Initialize with zeros
            image = torch.zeros(self.image_size, device=self.device)
            
            # Generate pixels sequentially
            for i in range(self.image_size[0]):
                for j in range(self.image_size[1]):
                    # Get parent values for this position
                    parents = self.network.get_parent_values(image, i, j)
                    
                    # Get probability from network
                    with torch.no_grad():
                        prob = self.network(parents).item()
                    
                    # Sample pixel value
                    image[i, j] = 1 if torch.rand(1, device=self.device).item() < prob else 0
            
            samples.append(image.cpu())
        
        return samples[0] if num_samples == 1 else samples
