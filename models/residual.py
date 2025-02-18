import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ResidualLayer(nn.Module):
    """
    One residual layer implementation.

    Args:
        in_dim (int): The input dimension
        h_dim (int): The hidden layer dimension
        res_h_dim (int): The hidden dimension of the residual block
    """
    def __init__(self, in_dim: int, h_dim: int, res_h_dim: int):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_dim,
                res_h_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(True),
            nn.Conv2d(
                res_h_dim,
                h_dim,
                kernel_size=1,
                stride=1,
                bias=False
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual layer.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output with residual connection
        """
        return x + self.res_block(x)


class ResidualStack(nn.Module):
    """
    A stack of residual layers.

    Args:
        in_dim (int): The input dimension
        h_dim (int): The hidden layer dimension
        res_h_dim (int): The hidden dimension of the residual block
        n_res_layers (int): Number of residual layers to stack
    """
    def __init__(self, in_dim: int, h_dim: int, res_h_dim: int, n_res_layers: int):
        super().__init__()
        self.n_res_layers = n_res_layers
        # Create separate instances of ResidualLayer instead of duplicating the same instance
        self.stack = nn.ModuleList([
            ResidualLayer(in_dim, h_dim, res_h_dim)
            for _ in range(n_res_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual stack.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output after passing through all residual layers
        """
        for layer in self.stack:
            x = layer(x)
        return F.relu(x)


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create random test data
    x = torch.randn(3, 40, 40, 200, device=device)

    # Test Residual Layer
    res = ResidualLayer(40, 40, 20).to(device)
    with torch.no_grad():
        res_out = res(x)
    print(f'Residual Layer output shape: {res_out.shape}')

    # Test Residual Stack
    res_stack = ResidualStack(40, 40, 20, 3).to(device)
    with torch.no_grad():
        res_stack_out = res_stack(x)
    print(f'Residual Stack output shape: {res_stack_out.shape}')
