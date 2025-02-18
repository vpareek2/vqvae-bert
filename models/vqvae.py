import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Union
from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder


class VQVAE(nn.Module):
    """
    Vector Quantized Variational AutoEncoder implementation.

    Args:
        h_dim (int): Hidden dimension
        res_h_dim (int): Residual hidden dimension
        n_res_layers (int): Number of residual layers
        n_embeddings (int): Number of embedding vectors
        embedding_dim (int): Dimension of embedding vectors
        beta (float): Commitment loss coefficient
        save_img_embedding_map (bool, optional): Whether to save image to embedding mappings.
            Defaults to False.
    """
    def __init__(
        self,
        h_dim: int,
        res_h_dim: int,
        n_res_layers: int,
        n_embeddings: int,
        embedding_dim: int,
        beta: float,
        save_img_embedding_map: bool = False
    ):
        super().__init__()

        # Encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)

        # Pre-quantization convolution
        self.pre_quantization_conv = nn.Conv2d(
            h_dim,
            embedding_dim,
            kernel_size=1,
            stride=1
        )

        # Pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings,
            embedding_dim,
            beta
        )

        # Decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        # Optional mapping from images to embeddings
        self.img_to_embedding_map = (
            {i: [] for i in range(n_embeddings)} if save_img_embedding_map else None
        )

    def forward(
        self,
        x: torch.Tensor,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VQVAE.

        Args:
            x (torch.Tensor): Input image tensor
            verbose (bool, optional): Whether to print debug information. Defaults to False.

        Returns:
            Tuple containing:
                - embedding_loss (torch.Tensor): Vector quantization loss
                - x_hat (torch.Tensor): Reconstructed image
                - perplexity (torch.Tensor): Perplexity metric
        """
        # Encode
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        # Quantize
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)

        # Decode
        x_hat = self.decoder(z_q)

        if verbose:
            print(f'Original data shape: {x.shape}')
            print(f'Encoded data shape: {z_e.shape}')
            print(f'Reconstructed data shape: {x_hat.shape}')
            raise AssertionError("Verbose mode - execution stopped")

        return embedding_loss, x_hat, perplexity


# if __name__ == "__main__":
#     # Test the VQVAE
#     batch_size, channels, height, width = 2, 3, 256, 256

#     # Model hyperparameters
#     h_dim = 128
#     res_h_dim = 32
#     n_res_layers = 2
#     n_embeddings = 512
#     embedding_dim = 64
#     beta = 0.25

#     # Create random input
#     x = torch.randn(batch_size, channels, height, width)

#     # Initialize model
#     vqvae = VQVAE(
#         h_dim=h_dim,
#         res_h_dim=res_h_dim,
#         n_res_layers=n_res_layers,
#         n_embeddings=n_embeddings,
#         embedding_dim=embedding_dim,
#         beta=beta
#     )

#     # Move to available device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = x.to(device)
#     vqvae = vqvae.to(device)

#     # Test forward pass
#     with torch.no_grad():
#         embedding_loss, x_hat, perplexity = vqvae(x)

#     print(f'Input shape: {x.shape}')
#     print(f'Reconstruction shape: {x_hat.shape}')
#     print(f'Perplexity: {perplexity:.2f}')
#     print(f'Embedding loss: {embedding_loss:.6f}')
