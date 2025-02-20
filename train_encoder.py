import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import wandb
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class TrainingConfig:
    # Training params
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 100
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # LR scheduler
    lr_scheduler_type: str = "cosine"
    min_lr_ratio: float = 0.1

    # Mixed precision
    mixed_precision: bool = True

    # Save/Load
    output_dir: str = "checkpoints"
    save_steps: int = 500
    eval_steps: int = 200

    # Early stopping
    patience: int = 10          # Increase from 5 to 10
    min_delta: float = 0.0001   # Decrease from 0.001 to 0.0001

    # Logging
    wandb_project: Optional[str] = None
    log_steps: int = 10

class TransformerMetrics:
    def __init__(self, n_embeddings: int, device: torch.device):
        """Initialize metrics calculator."""
        self.n_embeddings = n_embeddings
        self.device = device

    def calculate_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict:
        """
        Calculate comprehensive metrics for text-to-latent prediction.
        """
        batch_size = outputs.size(0)

        # If targets are scalar (or have only one value per sample), assume a 1x1 grid.
        if targets.dim() < 3 or (targets.dim() == 2 and targets.size(1) == 1):
            h, w = 1, 1
        else:
            h, w = targets.shape[1:3]

        # 1. Top-k accuracies (k=1,3,5)
        topk_acc = {}
        for k in [1, 3, 5]:
            _, pred_topk = outputs.topk(k, dim=-1)
            correct_topk = torch.any(pred_topk == targets.unsqueeze(-1), dim=-1)
            topk_acc[f'top{k}_acc'] = correct_topk.float().mean().item()

        # 2. Per-region accuracy (divide spatial dimensions into 3x3 grid)
        region_acc = torch.zeros(9, device=self.device)
        if h == 1 and w == 1:
            # If there is only one spatial element, region accuracy is just the overall accuracy.
            region_acc[:] = topk_acc['top1_acc']
        else:
            idx = 0
            h_splits = torch.linspace(0, h, 4, dtype=torch.long)
            w_splits = torch.linspace(0, w, 4, dtype=torch.long)
            for i in range(3):
                for j in range(3):
                    region_pred = outputs[:, h_splits[i]:h_splits[i+1], w_splits[j]:w_splits[j+1]]
                    region_target = targets[:, h_splits[i]:h_splits[i+1], w_splits[j]:w_splits[j+1]]
                    pred_indices = region_pred.argmax(dim=-1)
                    region_acc[idx] = (pred_indices == region_target).float().mean()
                    idx += 1

        # 3. Codebook usage distribution similarity
        pred_dist = torch.zeros(self.n_embeddings, device=self.device)
        target_dist = torch.zeros(self.n_embeddings, device=self.device)
        pred_indices = outputs.argmax(dim=-1)
        for i in range(self.n_embeddings):
            pred_dist[i] = (pred_indices == i).float().mean()
            target_dist[i] = (targets == i).float().mean()
        distribution_similarity = F.cosine_similarity(pred_dist.unsqueeze(0),
                                                      target_dist.unsqueeze(0))[0].item()

        # 4. Spatial coherence score
        if h == 1 and w == 1:
            # With a single element, spatial coherence is not defined; use overall accuracy.
            spatial_coherence = topk_acc['top1_acc']
        else:
            pred_matches = (pred_indices[:, :-1, :] == pred_indices[:, 1:, :])
            target_matches = (targets[:, :-1, :] == targets[:, 1:, :])
            vertical_coherence = (pred_matches == target_matches).float().mean().item()

            pred_matches = (pred_indices[:, :, :-1] == pred_indices[:, :, 1:])
            target_matches = (targets[:, :, :-1] == targets[:, :, 1:])
            horizontal_coherence = (pred_matches == target_matches).float().mean().item()
            spatial_coherence = (vertical_coherence + horizontal_coherence) / 2

        # 5. Average prediction confidence
        confidence = torch.softmax(outputs, dim=-1).amax(dim=-1).mean().item()

        return {
            'exact_match_acc': topk_acc['top1_acc'],
            'top3_acc': topk_acc['top3_acc'],
            'top5_acc': topk_acc['top5_acc'],
            'region_accuracies': region_acc.tolist(),
            'avg_region_acc': region_acc.mean().item(),
            'distribution_similarity': distribution_similarity,
            'spatial_coherence': spatial_coherence,
            'confidence': confidence
        }


def validate_training_setup(
    train_dataset,
    val_dataset,
    model,
    config: TrainingConfig
) -> None:
    """
    Validate training setup before starting.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        model: Model instance
        config: Training configuration

    Raises:
        ValueError: If validation fails
    """
    # Check dataset sizes
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty")

    # Get a sample batch
    sample_tokens, sample_codes = train_dataset[0]

    # Check shapes
    expected_output_shape = sample_codes.shape
    if not hasattr(model, 'output_dim'):
        raise ValueError("Model must have output_dim attribute")

    # Check device availability
    if config.mixed_precision and not torch.cuda.is_available():
        print("Warning: Mixed precision training enabled but CUDA is not available")

    # Check vocabulary coverage
    unique_tokens = set()
    for i in range(min(1000, len(train_dataset))):  # Check first 1000 samples
        tokens, _ = train_dataset[i]
        unique_tokens.update(tokens.tolist())
    if max(unique_tokens) >= train_dataset.vocab_size:
        raise ValueError("Dataset contains token IDs outside vocabulary range")

    with torch.no_grad():
        sample_output = model(sample_tokens.unsqueeze(0))
        # For classification, sample_output should have shape (batch, num_classes)
        # and sample_codes is a scalar target (or a 1D tensor for a batch).
        if sample_output.dim() != 2 or sample_output.size(1) != model.output_dim:
            raise ValueError(
                f"Model output shape {sample_output.shape} doesn't match expected "
                f"shape (batch, {model.output_dim}) for classification."
            )

    # Print training setup summary
    print("\nTraining Setup Summary:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Vocabulary size: {train_dataset.vocab_size}")
    print(f"Input sequence length: {sample_tokens.shape}")
    print(f"Target code shape: {sample_codes.shape}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Mixed precision: {config.mixed_precision}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Number of epochs: {config.epochs}\n")

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: TrainingConfig,
        resume_from: Optional[str] = None
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up model
        self.model = model.to(self.device)

        # Initialize metrics calculator
        self.metrics = TransformerMetrics(model.output_dim, self.device)

        # Data
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Set up LR scheduler
        self.total_steps = len(train_dataloader) * config.epochs
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps,
            eta_min=config.learning_rate * config.min_lr_ratio
        )

        # Mixed precision
        self.scaler = GradScaler(device='cuda', enabled=config.mixed_precision)

        # Metrics tracking
        self.best_val_metric = float('-inf')
        self.patience_counter = 0
        self.global_step = 0
        self.current_epoch = 0

        # Training state
        self.training_state = {
            'best_val_metric': self.best_val_metric,
            'patience_counter': self.patience_counter,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'rng_state': None,
        }

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb if specified and not resuming
        if config.wandb_project and not resume_from:
            wandb.init(project=config.wandb_project)

        # Resume from checkpoint if specified
        if resume_from:
            self.resume_from_checkpoint(resume_from)

    def save_checkpoint(self, is_best: bool = False):
        # Update training state with current values and RNG states
        import random
        import numpy as np
        self.training_state.update({
            'best_val_metric': self.best_val_metric,
            'patience_counter': self.patience_counter,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            }
        })

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'training_state': self.training_state,
            'config': self.config.__dict__
        }

        # Save checkpoint file
        if not is_best:
            checkpoint_path = self.output_dir / f"checkpoint_epoch{self.current_epoch}_step{self.global_step}.pt"
            # Remove older checkpoints if more than 3 exist
            existing_checkpoints = sorted(self.output_dir.glob("checkpoint_epoch*.pt"))
            if len(existing_checkpoints) >= 3:
                for checkpoint_to_remove in existing_checkpoints[:-2]:
                    checkpoint_to_remove.unlink()
        else:
            checkpoint_path = self.output_dir / "best_model.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def resume_from_checkpoint(self, checkpoint_path: str):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # Validate checkpoint structure
        required_keys = [
            'model_state_dict',
            'optimizer_state_dict',
            'scheduler_state_dict',
            'scaler_state_dict',
            'training_state',
            'config'
        ]
        if not all(key in checkpoint for key in required_keys):
            missing = set(required_keys) - set(checkpoint.keys())
            raise ValueError(f"Checkpoint is missing required keys: {missing}")

        # Load model state
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            raise ValueError(f"Failed to load model state: {str(e)}")

        # Load optimizer state and move tensors to correct device if necessary
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.device.type == 'cuda':
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
        except Exception as e:
            raise ValueError(f"Failed to load optimizer state: {str(e)}")

        # Load scheduler state
        try:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            raise ValueError(f"Failed to load scheduler state: {str(e)}")

        # Load scaler state
        try:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except Exception as e:
            raise ValueError(f"Failed to load scaler state: {str(e)}")

        # Restore training state
        training_state = checkpoint['training_state']
        self.best_val_metric = training_state['best_val_metric']
        self.patience_counter = training_state['patience_counter']
        self.global_step = training_state['global_step']
        self.current_epoch = training_state['current_epoch']

        # Restore RNG states
        if 'rng_state' in training_state and training_state['rng_state'] is not None:
            rng_state = training_state['rng_state']
            import random
            import numpy as np
            random.setstate(rng_state['python'])
            np.random.set_state(rng_state['numpy'])
            torch.set_rng_state(rng_state['torch'])
            if torch.cuda.is_available() and rng_state['cuda'] is not None:
                torch.cuda.set_rng_state_all(rng_state['cuda'])

        print(f"Resumed training from epoch {self.current_epoch}, global step {self.global_step}")
        print(f"Best validation metric so far: {self.best_val_metric:.4f}")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        all_metrics = []
        for batch in self.val_dataloader:
            tokens, codes = batch
            tokens = tokens.to(self.device)
            codes = codes.to(self.device)
            with autocast(device_type=self.device.type, enabled=self.config.mixed_precision):
                outputs = self.model(tokens)
                loss = self.criterion(outputs.view(-1, self.model.output_dim), codes.view(-1))
                metrics = self.metrics.calculate_metrics(outputs, codes)
                metrics['loss'] = loss.item()
                all_metrics.append(metrics)
            total_loss += loss.item() * tokens.size(0)
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key != 'region_accuracies':
                avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        self.model.train()
        return avg_metrics

    def train(self):
        print(f"Starting training on device: {self.device}")
        print(f"Starting from epoch {self.current_epoch}")
        self.model.train()
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            epoch_loss = 0
            epoch_metrics = []
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch}",
                initial=self.global_step % len(self.train_dataloader),
                total=len(self.train_dataloader)
            )
            for batch in progress_bar:
                tokens, codes = batch
                tokens = tokens.to(self.device)
                codes = codes.to(self.device)
                with autocast(device_type=self.device.type, enabled=self.config.mixed_precision):
                    outputs = self.model(tokens)
                    loss = self.criterion(outputs.view(-1, self.model.output_dim), codes.view(-1))
                with torch.no_grad():
                    batch_metrics = self.metrics.calculate_metrics(outputs, codes)
                    epoch_metrics.append(batch_metrics)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()
                epoch_loss += loss.item()
                self.global_step += 1
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'top5_acc': f"{batch_metrics['top5_acc']:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                if self.global_step % self.config.eval_steps == 0:
                    val_metrics = self.evaluate()
                    print(f"\nStep {self.global_step} validation:")
                    print(f"Loss: {val_metrics['loss']:.4f}")
                    print(f"Top-1 Accuracy: {val_metrics['exact_match_acc']:.4f}")
                    print(f"Top-5 Accuracy: {val_metrics['top5_acc']:.4f}")
                    print(f"Spatial Coherence: {val_metrics['spatial_coherence']:.4f}")
                    print(f"Distribution Similarity: {val_metrics['distribution_similarity']:.4f}")
                    if self.config.wandb_project:
                        wandb.log({
                            'val_loss': val_metrics['loss'],
                            'val_top1_acc': val_metrics['exact_match_acc'],
                            'val_top5_acc': val_metrics['top5_acc'],
                            'val_spatial_coherence': val_metrics['spatial_coherence'],
                            'val_distribution_similarity': val_metrics['distribution_similarity']
                        })
                        combined_metric = (
                            val_metrics['top5_acc'] * 0.4 +
                            val_metrics['distribution_similarity'] * 0.3 +
                            val_metrics['spatial_coherence'] * 0.3
                        )

                        if combined_metric > self.best_val_metric + self.config.min_delta:
                            self.best_val_metric = combined_metric
                            self.patience_counter = 0
                            self.save_checkpoint(is_best=True)
                        else:
                            self.patience_counter += 1
                    if self.patience_counter >= self.config.patience:
                        print(f"Early stopping triggered at step {self.global_step}")
                        return
                    self.model.train()
            self.save_checkpoint()
            avg_epoch_metrics = {}
            for key in epoch_metrics[0].keys():
                if key != 'region_accuracies':
                    avg_epoch_metrics[key] = sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
            print(f"\nEpoch {epoch} completed:")
            print(f"Average loss: {epoch_loss / len(self.train_dataloader):.4f}")
            print(f"Average top-5 accuracy: {avg_epoch_metrics['top5_acc']:.4f}")
            print(f"Average spatial coherence: {avg_epoch_metrics['spatial_coherence']:.4f}")
            print(f"Average distribution similarity: {avg_epoch_metrics['distribution_similarity']:.4f}")
            if self.config.wandb_project:
                wandb.log({
                    'epoch': epoch,
                    'epoch_loss': epoch_loss / len(self.train_dataloader),
                    **{f'epoch_{k}': v for k, v in avg_epoch_metrics.items()}
                })

def main():
    import argparse
    from torch.utils.data import DataLoader
    from encoder.model import TransformerConfig, TransformerEncoder
    from datasets.text_latent import TextToLatentDataset
    from utils import set_seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--vocab_path", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--seed", type=int, default=12, help="Random seed for reproducibility")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load datasets
    train_dataset = TextToLatentDataset(args.train_data, args.vocab_path)
    val_dataset = TextToLatentDataset(args.val_data, args.vocab_path)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model configuration
    model_config = TransformerConfig(
        vocab_size=train_dataset.vocab_size,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        ffn_dim=1024,
        dropout=0.1,
        max_seq_length=5,  # 5 adjectives
        output_dim=512,    # VQ-VAE codebook size
        output_type="classification"
    )

    # Initialize model
    model = TransformerEncoder(model_config)

    # Create training configuration
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project
    )

    # Validate setup
    validate_training_setup(train_dataset, val_dataset, model, training_config)

    # Initialize trainer and start training
    trainer = Trainer(model, train_dataloader, val_dataloader, training_config)
    trainer.train()

if __name__ == "__main__":
    main()

# To run code
# python train_encoder.py \
#     --train_data datasets/processed_train.npy \
#     --val_data datasets/processed_val.npy \
#     --vocab_path datasets/vocabulary.json \
#     --batch_size 32 \
#     --epochs 100 \
#     --learning_rate 3e-4 \
#     --output_dir checkpoints \
#     --wandb_project text-to-latent \
#     --seed 42

# python train_encoder.py --train_data datasets/processed_train.npy --val_data datasets/processed_val.npy --wandb_project VQVAE_transformer
