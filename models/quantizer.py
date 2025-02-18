import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Args:
        n_e (int): Number of embeddings
        e_dim (int): Dimension of embedding
        beta (float): Commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """
    def __init__(self, n_e: int, e_dim: int, beta: float):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Maps the encoder output z to a discrete one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)

        Args:
            z (torch.Tensor): Input tensor of shape (batch, channel, height, width)

        Returns:
            Tuple containing:
                - loss (torch.Tensor): Commitment loss
                - z_q (torch.Tensor): Quantized vectors
                - perplexity (torch.Tensor): Perplexity metric
                - min_encodings (torch.Tensor): One-hot encodings
                - min_encoding_indices (torch.Tensor): Indices of closest embeddings
        """
        # Move to same device as input
        device = z.device

        # Reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # Compute distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
             torch.sum(self.embedding.weight**2, dim=1) -
             2 * torch.matmul(z_flattened, self.embedding.weight.t()))

        # Find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e, device=device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # Compute commitment loss
        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # Preserve gradients
        z_q = z + (z_q - z).detach()

        # Calculate perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # Reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

    def get_codebook_usage(self, z: torch.Tensor) -> dict:
        """
        Calculate codebook usage statistics for a batch of encoded vectors.

        Args:
            z (torch.Tensor): Input tensor of shape (batch, channel, height, width)

        Returns:
            dict: Usage statistics including percent used and distribution
        """
        with torch.no_grad():
            # Reshape z -> (batch, height, width, channel) and flatten
            z = z.permute(0, 2, 3, 1).contiguous()
            z_flattened = z.view(-1, self.e_dim)

            # Calculate distances
            d = (torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
                 torch.sum(self.embedding.weight**2, dim=1) -
                 2 * torch.matmul(z_flattened, self.embedding.weight.t()))

            # Get encodings
            min_encoding_indices = torch.argmin(d, dim=1)

            # Calculate usage statistics
            unique_indices = torch.unique(min_encoding_indices)
            usage_percent = (len(unique_indices) / self.n_e) * 100

            # Count frequency of each codebook vector
            usage_counts = torch.bincount(min_encoding_indices, minlength=self.n_e)

            return {
                'percent_used': usage_percent,
                'usage_counts': usage_counts.cpu().numpy(),
                'n_active': len(unique_indices)
            }


if __name__ == "__main__":
    # Test the vector quantizer
    batch_size, channels, height, width = 4, 64, 32, 32
    n_embeddings, embedding_dim = 512, 64
    beta = 0.25

    # Create random input
    z = torch.randn(batch_size, channels, height, width)

    # Initialize and test quantizer
    vq = VectorQuantizer(n_embeddings, embedding_dim, beta)

    # Move to available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = z.to(device)
    vq = vq.to(device)

    with torch.no_grad():
        loss, z_q, perplexity, min_encodings, min_encoding_indices = vq(z)

    print(f'Input shape: {z.shape}')
    print(f'Quantized shape: {z_q.shape}')
    print(f'Perplexity: {perplexity:.2f}')
    print(f'Loss: {loss:.6f}')
