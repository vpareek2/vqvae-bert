import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from models.residual import ResidualStack

class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi
    maps back to the original space z -> x.

    Args:
        in_dim (int): The input dimension
        h_dim (int): The hidden layer dimension
        res_h_dim (int): The hidden dimension of the residual block
        n_res_layers (int): Number of residual layers to stack
    """
    def __init__(self, in_dim: int, h_dim: int, n_res_layers: int, res_h_dim: int):
        super().__init__()  # Simplified parent class init
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim,
                kernel_size=kernel-1,
                stride=stride-1,
                padding=1
            ),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(
                h_dim,
                h_dim // 2,
                kernel_size=kernel,
                stride=stride,
                padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                h_dim//2,
                3,
                kernel_size=kernel,
                stride=stride,
                padding=1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Decoded output
        """
        return self.inverse_conv_stack(x)


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random data
    x = torch.randn(3, 40, 40, 200)

    # Test decoder
    decoder = Decoder(40, 128, 3, 64).to(device)
    x = x.to(device)

    with torch.no_grad():
        decoder_out = decoder(x)

    print(f'Decoder out shape: {decoder_out.shape}')
