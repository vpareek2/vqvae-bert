import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from pathlib import Path
from datasets.block import BlockDataset, LatentBlockDataset
import numpy as np
from typing import Tuple, Any


def get_transforms(is_training: bool) -> transforms.Compose:
    """
    Get transforms for training or validation.

    Args:
        is_training: Whether to include training augmentations
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

# def load_cifar() -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
#     """Load and normalize CIFAR10 dataset."""
#     train = datasets.CIFAR10(
#         root="data",
#         train=True,
#         download=True,
#         transform=get_transforms(is_training=True)
#     )

#     val = datasets.CIFAR10(
#         root="data",
#         train=False,
#         download=True,
#         transform=get_transforms(is_training=False)
#     )

#     return train, val


def load_block() -> Tuple[BlockDataset, BlockDataset]:
    """Load and normalize Block dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train = BlockDataset(
        data_dir="datasets",  # Point directly to your datasets folder
        train=True,
        transform=transform
    )

    val = BlockDataset(
        data_dir="datasets",  # Point directly to your datasets folder
        train=False,
        transform=transform
    )

    return train, val


def load_latent_block() -> Tuple[LatentBlockDataset, LatentBlockDataset]:
    """Load latent block dataset."""
    data_folder_path = Path.cwd()
    data_file_path = data_folder_path / 'data' / 'latent_e_indices.npy'

    train = LatentBlockDataset(
        str(data_file_path),
        train=True,
        transform=None
    )

    val = LatentBlockDataset(
        str(data_file_path),
        train=False,
        transform=None
    )

    return train, val


def data_loaders(
    train_data: Any,
    val_data: Any,
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and validation."""
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    return train_loader, val_loader


def load_data_and_data_loaders(
    dataset: str,
    batch_size: int
) -> Tuple[Any, Any, DataLoader, DataLoader, float]:
    """
    Load dataset and create data loaders.

    Args:
        dataset: Name of dataset ('CIFAR10', 'BLOCK', or 'LATENT_BLOCK')
        batch_size: Batch size for data loaders

    Returns:
        Tuple containing:
            - training_data: Training dataset
            - validation_data: Validation dataset
            - training_loader: Training data loader
            - validation_loader: Validation data loader
            - x_train_var: Variance of training data
    """
    if dataset == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(np.array(training_data.data) / 255.0)

    elif dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        # For BLOCK dataset, use a fixed variance since images are already normalized
        x_train_var = 0.5  # A reasonable default for normalized images

    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data)

    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10, BLOCK, and LATENT_BLOCK datasets are supported.'
        )

    return training_data, validation_data, training_loader, validation_loader, x_train_var

def readable_timestamp() -> str:
    """Generate a readable timestamp for file naming."""
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(
    model: torch.nn.Module,
    results: dict,
    hyperparameters: dict,
    timestamp: str
) -> None:
    """
    Save model state, results, and hyperparameters.

    Args:
        model: The model to save
        results: Dictionary of training results
        hyperparameters: Dictionary of hyperparameters
        timestamp: Timestamp string for file naming
    """
    save_path = Path.cwd() / 'results'

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }

    torch.save(
        results_to_save,
        save_path / f'vqvae_data_{timestamp}.pth'
    )

def save_reconstruction_grid(model, val_loader, device, save_path, n_samples=8):
    """
    Save a grid of original vs reconstructed images.

    Args:
        model: The VQ-VAE model
        val_loader: Validation data loader
        device: Current device (cuda/cpu)
        save_path: Path to save the grid image
        n_samples: Number of samples to show in grid
    """
    model.eval()

    # Get a batch of images
    x, _ = next(iter(val_loader))
    x = x[:n_samples].to(device)

    with torch.no_grad():
        # Get reconstructions
        _, x_hat, _ = model(x)

    # Denormalize images
    def denorm(tensor):
        return (tensor * 0.5 + 0.5).clamp(0, 1)

    x = denorm(x)
    x_hat = denorm(x_hat)

    # Create comparison grid
    comparison = torch.cat([x, x_hat], dim=0)
    grid = torchvision.utils.make_grid(comparison, nrow=n_samples, normalize=False)

    # Save image
    torchvision.utils.save_image(grid, save_path)

    model.train()

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed value
    """
    import random
    import numpy as np
    import torch

    # Python's random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
