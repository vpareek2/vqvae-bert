import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal

@dataclass
class TransformerConfig:
    vocab_size: int
    embedding_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    max_seq_length: int = 32
    output_dim: int = 512
    output_type: Literal["classification", "embedding"] = "classification"

class TransformerEncoder(nn.Module):
    """
    A lightweight transformer encoder designed for mapping short text (3–5 tokens)
    to a latent space. The module prepends a learned [CLS] token to the input
    sequence, processes the tokens through multiple transformer layers, and then
    projects the pooled [CLS] representation to an output space.

    Args:
        vocab_size (int): Number of tokens in the vocabulary.
        embedding_dim (int, optional): Dimension of token embeddings. Default: 256.
        num_layers (int, optional): Number of transformer encoder layers. Default: 6.
        num_heads (int, optional): Number of attention heads. Default: 8.
        ffn_dim (int, optional): Dimension of the feed-forward network inside each layer. Default: 1024.
        dropout (float, optional): Dropout probability. Default: 0.1.
        max_seq_length (int, optional): Maximum sequence length (excluding [CLS]). Default: 32.
        output_dim (int, optional): Dimension of the output projection. For classification,
            this is typically the number of codebook entries (e.g., 512); for embedding output,
            it is the desired latent dimension. Default: 512.
        output_type (str, optional): Either "classification" (returns raw logits) or
            "embedding" (returns normalized latent embeddings). Default: "classification".
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.output_type = config.output_type
        self.max_seq_length = config.max_seq_length

        # Token embedding layer: maps token IDs to vectors.
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        # Positional embedding: supports positions 0 .. max_seq_length (position 0 will be for [CLS])
        self.pos_embedding = nn.Embedding(config.max_seq_length + 1, config.embedding_dim)
        # Learned [CLS] token, prepended to every input sequence.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))

        # Build transformer encoder layers.
        # Using batch_first=True so that input shape is (batch, seq_len, embedding_dim).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        # Output projection head.
        # For "classification", output_dim equals the number of classes (e.g., 512 codebook entries).
        # For "embedding", output_dim is the latent dimension you want (e.g., 64 or 128).
        self.output_head = nn.Linear(config.embedding_dim, config.output_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for embeddings and output head"""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.output_head.weight)
        if self.output_head.bias is not None:
            nn.init.zeros_(self.output_head.bias)

    def forward(self, input_ids):
        """
        Forward pass of the transformer encoder.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_length)
                containing token indices.

        Returns:
            torch.Tensor: If output_type="classification", returns logits of shape
                (batch_size, output_dim). If output_type="embedding", returns a normalized
                latent embedding of shape (batch_size, output_dim).
        """
        batch_size, seq_length = input_ids.size()
        # Enforce maximum sequence length.
        if seq_length > self.max_seq_length:
            input_ids = input_ids[:, :self.max_seq_length]
            seq_length = self.max_seq_length
        # Obtain token embeddings: (batch_size, seq_length, embedding_dim)
        token_embeds = self.token_embedding(input_ids)
        # Create [CLS] token for each example: (batch_size, 1, embedding_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate [CLS] token and token embeddings along sequence dimension.
        # Resulting shape: (batch_size, seq_length + 1, embedding_dim)
        x = torch.cat((cls_tokens, token_embeds), dim=1)
        # Create position indices (0 for [CLS], then 1, 2, ... for tokens)
        positions = torch.arange(0, seq_length + 1, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embedding(positions)
        # Add positional information.
        x = x + pos_embeds
        # Pass through the transformer encoder.
        x = self.transformer_encoder(x)
        # Use the [CLS] token's output as the pooled representation.
        cls_output = x[:, 0, :]  # (batch_size, embedding_dim)
        # Project the pooled representation.
        output = self.output_head(cls_output)
        if self.output_type == "classification":
            return output  # logits (e.g., for 512 codebook classes)
        elif self.output_type == "embedding":
            # Return normalized latent embeddings.
            return F.normalize(output, dim=-1)
        else:
            raise ValueError(f"Invalid output_type '{self.output_type}'. Choose 'classification' or 'embedding'.")

# if __name__ == "__main__":
#     # Example usage:
#     # Suppose we have a vocabulary of 3000 tokens.
#     vocab_size = 3000
#     # Create a random batch of token indices (batch_size=4, seq_length=5).
#     dummy_input = torch.randint(low=0, high=vocab_size, size=(4, 5))

#     # Instantiate the transformer encoder.
#     # Here, output_type "classification" returns logits for codebook prediction.
#     transformer_encoder = TransformerEncoder(
#         vocab_size=vocab_size,
#         embedding_dim=256,
#         num_layers=6,
#         num_heads=8,
#         ffn_dim=1024,
#         dropout=0.1,
#         max_seq_length=32,
#         output_dim=512,  # e.g., number of codebook entries
#         output_type="classification"
#     )

#     # Forward pass.
#     output_logits = transformer_encoder(dummy_input)
#     print("Output logits shape:", output_logits.shape)

#     # If you prefer a latent embedding output:
#     transformer_encoder.output_type = "embedding"
#     latent_embeddings = transformer_encoder(dummy_input)
#     print("Latent embeddings shape:", latent_embeddings.shape)
