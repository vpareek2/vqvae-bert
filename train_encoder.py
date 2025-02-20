import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import os
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class TrainingConfig:
    # Model params set in model config

    # Training params
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
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
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up model
        torch.set_float32_matmul_precision('high')
        self.model = model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = DDP(self.model)
        self.model = torch.compile(self.model)

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
        self.lr_scheduler = self.get_lr_scheduler()

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

    def get_lr_scheduler(self):
        if self.config.lr_scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_steps,
                eta_min=self.config.learning_rate * self.config.min_lr_ratio
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.lr_scheduler_type}")

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

    def train_step(self, batch):
        tokens, codes = batch
        tokens = tokens.to(self.device)
        codes = codes.to(self.device)

        # Forward pass with mixed precision
        with autocast(enabled=self.config.mixed_precision):
            outputs = self.model(tokens)
            loss = self.criterion(outputs.view(-1, self.model.output_dim), codes.view(-1))
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()

        return loss.item() * self.config.gradient_accumulation_steps

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_samples = 0

        for batch in self.val_dataloader:
            tokens, codes = batch
            tokens = tokens.to(self.device)
            codes = codes.to(self.device)

            with autocast(enabled=self.config.mixed_precision):
                outputs = self.model(tokens)
                loss = self.criterion(outputs.view(-1, self.model.output_dim), codes.view(-1))

            total_loss += loss.item() * tokens.size(0)
            total_samples += tokens.size(0)

        avg_loss = total_loss / total_samples
        self.model.train()
        return avg_loss

    def train(self):
        print(f"Starting training on device: {self.device}")
        print(f"Total steps: {self.total_steps}")

        self.model.train()

        for epoch in range(self.config.epochs):
            epoch_loss = 0
            step_in_epoch = 0

            progress_bar = tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch}")

            for step, batch in enumerate(self.train_dataloader):
                # Training step
                loss = self.train_step(batch)
                epoch_loss += loss
                step_in_epoch += 1

                # Gradient accumulation handling
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                    # Optimizer step with gradient scaling
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.log_steps == 0:
                        avg_loss = epoch_loss / step_in_epoch
                        lr = self.optimizer.param_groups[0]['lr']

                        metrics = {
                            'loss': avg_loss,
                            'learning_rate': lr,
                            'epoch': epoch,
                            'step': self.global_step,
                        }

                        if self.config.wandb_project:
                            wandb.log(metrics)

                        progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0:
                        val_loss = self.evaluate()

                        if self.config.wandb_project:
                            wandb.log({'val_loss': val_loss})

                        # Early stopping and model saving
                        if val_loss < self.best_val_loss - self.config.min_delta:
                            self.best_val_loss = val_loss
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

                progress_bar.update(1)

            progress_bar.close()

            # End of epoch logging
            avg_epoch_loss = epoch_loss / step_in_epoch
            print(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")

            if self.config.wandb_project:
                wandb.log({'epoch': epoch, 'epoch_loss': avg_epoch_loss})

def main():
    # Example usage
    config = TrainingConfig(
        batch_size=32,
        learning_rate=3e-4,
        epochs=100,
        wandb_project="text-to-latent"
    )

    # Initialize model, dataloaders
    # model = TransformerEncoder(...)
    # train_dataloader = DataLoader(...)
    # val_dataloader = DataLoader(...)

    trainer = Trainer(model, train_dataloader, val_dataloader, config)
    trainer.train()

if __name__ == "__main__":
    main()
