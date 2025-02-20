import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional

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
    eval_steps: int = 100

    # Early stopping
    patience: int = 5
    min_delta: float = 0.001

    # Logging
    wandb_project: Optional[str] = None
    log_steps: int = 10

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: TrainingConfig,
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up model
        self.model = model.to(self.device)

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
        self.scaler = GradScaler(enabled=config.mixed_precision)

        # Metrics tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0

        # Initialize wandb if specified
        if config.wandb_project:
            wandb.init(project=config.wandb_project)

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }

        # Save latest checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model if applicable
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch in self.val_dataloader:
            tokens, codes = batch
            tokens = tokens.to(self.device)
            codes = codes.to(self.device)

            with autocast(enabled=self.config.mixed_precision):
                outputs = self.model(tokens)
                loss = self.criterion(outputs.view(-1, self.model.output_dim), codes.view(-1))

                # Calculate accuracy
                pred = outputs.argmax(dim=-1)
                correct_predictions += (pred == codes).sum().item()
                total_predictions += codes.numel()

            total_loss += loss.item() * tokens.size(0)

        avg_loss = total_loss / len(self.val_dataloader.dataset)
        accuracy = correct_predictions / total_predictions

        self.model.train()
        return {'loss': avg_loss, 'accuracy': accuracy}

    def train(self):
        print(f"Starting training on device: {self.device}")
        print(f"Total epochs: {self.config.epochs}")

        self.model.train()

        for epoch in range(self.config.epochs):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")

            for batch in progress_bar:
                tokens, codes = batch
                tokens = tokens.to(self.device)
                codes = codes.to(self.device)

                # Forward pass with mixed precision
                with autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(tokens)
                    loss = self.criterion(outputs.view(-1, self.model.output_dim), codes.view(-1))

                # Calculate accuracy
                pred = outputs.argmax(dim=-1)
                correct = (pred == codes).sum().item()
                total = codes.numel()

                epoch_correct += correct
                epoch_total += total

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()

                epoch_loss += loss.item()
                self.global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{correct/total:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    metrics = {
                        'train_loss': loss.item(),
                        'train_accuracy': correct/total,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'step': self.global_step,
                    }

                    if self.config.wandb_project:
                        wandb.log(metrics)

                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    val_metrics = self.evaluate()

                    print(f"\nStep {self.global_step} validation:")
                    print(f"Loss: {val_metrics['loss']:.4f}")
                    print(f"Accuracy: {val_metrics['accuracy']:.4f}")

                    if self.config.wandb_project:
                        wandb.log({
                            'val_loss': val_metrics['loss'],
                            'val_accuracy': val_metrics['accuracy']
                        })

                    # Early stopping and model saving
                    if val_metrics['loss'] < self.best_val_loss - self.config.min_delta:
                        self.best_val_loss = val_metrics['loss']
                        self.patience_counter = 0
                        self.save_checkpoint(is_best=True)
                    else:
                        self.patience_counter += 1

                    if self.patience_counter >= self.config.patience:
                        print(f"Early stopping triggered at step {self.global_step}")
                        return

                    self.model.train()

                # Regular checkpoint saving
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

            # End of epoch statistics
            epoch_loss = epoch_loss / len(self.train_dataloader)
            epoch_accuracy = epoch_correct / epoch_total

            print(f"\nEpoch {epoch} completed:")
            print(f"Average loss: {epoch_loss:.4f}")
            print(f"Average accuracy: {epoch_accuracy:.4f}")

            if self.config.wandb_project:
                wandb.log({
                    'epoch': epoch,
                    'epoch_loss': epoch_loss,
                    'epoch_accuracy': epoch_accuracy
                })

def main():
    import argparse
    from torch.utils.data import DataLoader
    from encoder.model import TransformerConfig, TransformerEncoder
    from datasets.text_latent import TextToLatentDataset
    from utils import set_seed  # Import the set_seed function

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

    # Initialize trainer and start training
    trainer = Trainer(model, train_dataloader, val_dataloader, training_config)
    trainer.train()

if __name__ == "__main__":
    main()
# to run
# python train_encoder.py \
    # --train_data datasets/processed_train.npy \
    # --val_data datasets/processed_val.npy \
    # --vocab_path datasets/vocabulary.json \
    # --batch_size 32 \
    # --epochs 100 \
    # --learning_rate 3e-4 \
    # --output_dir checkpoints \
    # --wandb_project text-to-latent
