import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from typing import Dict, Any
import utils
from models.vqvae import VQVAE
from models.quantizer import VectorQuantizer
from utils import save_reconstruction_grid
from pathlib import Path

# Set up argument parser
parser = argparse.ArgumentParser()

# Hyperparameters
timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_updates", type=int, default=7000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=48)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.5)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--dataset", type=str, default='CIFAR10')
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename", type=str, default=timestamp)
parser.add_argument("--checkpoint_dir", type=str, default='checkpoints')
parser.add_argument("--checkpoint_interval", type=int, default=500)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print(f'Results will be saved in ./results/vqvae_{args.filename}.pth')

# Load data and define batch data loaders
training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
    args.dataset, args.batch_size)

# Set up VQ-VAE model
model = VQVAE(
    args.n_hiddens,
    args.n_residual_hiddens,
    args.n_residual_layers,
    args.n_embeddings,
    args.embedding_dim,
    args.beta
).to(device)

# Set up optimizer
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}

def validate(model, validation_loader, device, x_val_var):
    """Run validation loop and return metrics."""
    model.eval()
    val_recon_errors = []
    val_losses = []
    val_perplexities = []

    with torch.no_grad():
        for (x, _) in validation_loader:
            x = x.to(device)

            # Get encoded vectors before quantization
            z_e = model.encoder(x)
            z_e = model.pre_quantization_conv(z_e)

            # Get codebook usage for this batch
            codebook_stats = model.vector_quantization.get_codebook_usage(z_e)

            # Regular forward pass
            embedding_loss, x_hat, perplexity = model(x)
            recon_loss = torch.mean((x_hat - x)**2) / x_val_var
            loss = recon_loss + embedding_loss

            val_recon_errors.append(recon_loss.item())
            val_losses.append(loss.item())
            val_perplexities.append(perplexity.item())

    model.train()
    return {
        'recon_error': np.mean(val_recon_errors),
        'loss': np.mean(val_losses),
        'perplexity': np.mean(val_perplexities),
        'codebook_usage': codebook_stats['percent_used'],
        'n_active_codes': codebook_stats['n_active']
    }

def train():
    best_val_loss = float('inf')
    best_codebook_usage = 0.0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    for i in range(args.n_updates):
        (x, _) = next(iter(training_loader))
        x = x.to(device)

        optimizer.zero_grad()
        embedding_loss, x_hat, perplexity = model(x)

        recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        # Record results
        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        if i % args.log_interval == 0:
            # Save model and print values
            if args.save:
                timestamp = utils.readable_timestamp()
                grid_path = Path('results') / f'recon_grid_{args.filename}_{i}.png'
                save_reconstruction_grid(
                    model,
                    validation_loader,
                    device,
                    grid_path
                )
                val_metrics = validate(model, validation_loader, device, x_train_var)

                # Save periodic checkpoint
                if i % args.checkpoint_interval == 0:
                    checkpoint_path = checkpoint_dir / f'checkpoint_{i:06d}.pt'
                    torch.save({
                        'iteration': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_metrics': val_metrics,
                        'args': args,
                    }, checkpoint_path)

                # Save best model based on validation loss
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    torch.save({
                        'iteration': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_metrics': val_metrics,
                        'args': args,
                    }, checkpoint_dir / 'best_model.pt')

                print(
                    f'Update #{i}',
                    f'Train Recon Error: {np.mean(results["recon_errors"][-args.log_interval:]):.6f}',
                    f'Val Recon Error: {val_metrics["recon_error"]:.6f}',
                    f'Train Loss: {np.mean(results["loss_vals"][-args.log_interval:]):.6f}',
                    f'Val Loss: {val_metrics["loss"]:.6f}',
                    f'Train Perplexity: {np.mean(results["perplexities"][-args.log_interval:]):.6f}',
                    f'Val Perplexity: {val_metrics["perplexity"]:.6f}',
                    f'Codebook Usage: {val_metrics["codebook_usage"]:.1f}%'
                )

if __name__ == "__main__":
    train()
