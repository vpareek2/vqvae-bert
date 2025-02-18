import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from models.residual import ResidualStack


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Args:
        in_dim (int): The input dimension
        h_dim (int): The hidden layer dimension
        n_res_layers (int): Number of residual layers to stack
        res_h_dim (int): The hidden dimension of the residual block
    """
    def __init__(self, in_dim: int, h_dim: int, n_res_layers: int, res_h_dim: int):
        super().__init__()
        kernel = 4
        stride = 2

        self.conv_stack = nn.Sequential(
            nn.Conv2d(
                in_dim,
                h_dim // 2,
                kernel_size=kernel,
                stride=stride,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                h_dim // 2,
                h_dim,
                kernel_size=kernel,
                stride=stride,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                h_dim,
                h_dim,
                kernel_size=kernel-1,
                stride=stride-1,
                padding=1
            ),
            ResidualStack(
                h_dim,
                h_dim,
                res_h_dim,
                n_res_layers
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Encoded representation
        """
        return self.conv_stack(x)


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random data
    x = torch.randn(3, 40, 40, 200, device=device)

    # Test encoder
    encoder = Encoder(40, 128, 3, 64).to(device)

    with torch.no_grad():
        encoder_out = encoder(x)

    print(f'Encoder out shape: {encoder_out.shape}')
