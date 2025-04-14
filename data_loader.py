import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Dict, Any

class HandwrittenDigitsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        self.labels = []
        
        # Walk through the directory and collect all image files
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, file))
                    # Extract label from directory name (assuming directory structure like '0', '1', etc.)
                    label = int(os.path.basename(root))
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            # Convert to binary (0 or 1) by thresholding
            image = (image > 0.5).float()
            image: torch.Tensor = image.squeeze()
        
        return image, label

def get_train_loader(resolution='20x20', batch_size=32, num_workers=4):
    """
    Get a data loader for training data.
    
    Args:
        resolution (str): Resolution of the images ('20x20', '40x40', or '65x65')
        batch_size (int): Batch size for the data loader
        num_workers (int): Number of workers for data loading
    
    Returns:
        DataLoader: PyTorch DataLoader for training data
    """
    # Define the data directory based on resolution
    data_dir = os.path.join('data', f'kntu_handwritten_digits_binary_{resolution}')
    
    # Define transforms - only convert to tensor, no normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create dataset
    dataset = HandwrittenDigitsDataset(data_dir, transform=transform)
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    return train_loader

class BinaryImageDataset(Dataset):
    """
    Dataset class for binary images.
    
    This dataset handles loading and preprocessing of binary image data
    for training the Bayesian network.
    
    Attributes:
        data (torch.Tensor): Binary image data of shape (n_samples, height, width)
    """
    
    def __init__(self, data: torch.Tensor):
        """
        Initialize the dataset.
        
        Args:
            data (torch.Tensor): Binary image data of shape (n_samples, height, width)
                Values should be 0 or 1
        """
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

def create_data_loaders(
    data: torch.Tensor,
    batch_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Splits the data into train, validation, and test sets according to the
    provided ratios and creates DataLoader objects for each.
    
    Args:
        data (torch.Tensor): Binary image data of shape (n_samples, height, width)
        batch_size (int): Batch size for the data loaders
        train_ratio (float, optional): Proportion of data to use for training.
            Defaults to 0.7 (70%).
        val_ratio (float, optional): Proportion of data to use for validation.
            Defaults to 0.15 (15%). The remaining data will be used for testing.
        num_workers (int, optional): Number of workers for data loading.
            Defaults to 4.
        seed (int, optional): Random seed for reproducible data splitting.
            Defaults to None.
            
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test
            data loaders
            
    Raises:
        ValueError: If train_ratio + val_ratio >= 1
    """
    if train_ratio + val_ratio >= 1:
        raise ValueError(
            f"Train ({train_ratio}) + validation ({val_ratio}) ratios must sum to less than 1"
        )
    
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
    
    # Create dataset
    dataset = BinaryImageDataset(data)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

def load_binary_images(path: str) -> torch.Tensor:
    """
    Load binary images from a file.
    
    Args:
        path (str): Path to the file containing binary image data
            
    Returns:
        torch.Tensor: Binary image data of shape (n_samples, height, width)
    """
    # TODO: Implement actual loading logic based on your data format
    # This is a placeholder that should be modified based on your data format
    data = np.load(path)  # or any other loading method
    return torch.from_numpy(data).float() 

def get_resolution_str(config_dict: Dict[str, Any]) -> str:
    """
    Convert image resolution tuple to string format used by data loader.
    
    Args:
        config_dict (Dict[str, Any]): Configuration dictionary
        
    Returns:
        str: Resolution string in format 'HxW'
    """
    height, width = config_dict['image_resolution']
    return f"{height}x{width}"

def prepare_data_for_training(
    resolution: str,
    batch_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare data for training by loading the full dataset and splitting it into
    train, validation, and test sets.
    
    Args:
        resolution (str): Resolution of the images ('20x20', '40x40', or '65x65')
        batch_size (int): Batch size for the data loaders
        train_ratio (float, optional): Proportion of data to use for training.
            Defaults to 0.7 (70%).
        val_ratio (float, optional): Proportion of data to use for validation.
            Defaults to 0.15 (15%). The remaining data will be used for testing.
        num_workers (int, optional): Number of workers for data loading.
            Defaults to 4.
            
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test
            data loaders
    """
    # Get the full dataset
    train_loader = get_train_loader(
        resolution=resolution,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Extract all data from the loader
    all_data = []
    for batch in train_loader:
        images, _ = batch  # We only need the images, not the labels
        all_data.append(images)
    all_data = torch.cat(all_data, dim=0)
    
    # Create train/val/test splits
    return create_data_loaders(
        data=all_data,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        num_workers=num_workers
    ) 