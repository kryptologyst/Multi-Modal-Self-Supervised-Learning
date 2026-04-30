"""Training script for multi-modal self-supervised learning."""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import ToyMultimodalDataset, collate_fn, create_data_splits
from src.eval.metrics import compute_contrastive_metrics, format_metrics
from src.models.clip_model import ContrastiveCLIPModel, ContrastiveLoss
from src.utils.config import load_config, create_directories, print_config
from src.utils.device import setup_device_and_seed, move_to_device


class Trainer:
    """Trainer class for multi-modal self-supervised learning."""
    
    def __init__(self, config_path: str):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.device = setup_device_and_seed(
            device_type=self.config.device.type,
            deterministic=self.config.device.deterministic
        )
        
        # Create directories
        create_directories(self.config)
        
        # Initialize model and loss
        self.model = ContrastiveCLIPModel(
            model_name=self.config.model.vision_encoder.model_name,
            projection_dim=self.config.model.vision_encoder.projection_dim,
            temperature=self.config.model.temperature,
            freeze_backbone=self.config.model.vision_encoder.freeze_backbone,
        ).to(self.device)
        
        self.loss_fn = ContrastiveLoss(
            temperature=self.config.loss.temperature,
            label_smoothing=self.config.loss.label_smoothing,
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        
        # Initialize data loaders
        self._setup_data_loaders()
        
        # Initialize logging
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def _setup_data_loaders(self):
        """Setup data loaders for training, validation, and testing."""
        # Create data splits
        create_data_splits(
            data_dir=self.config.paths.data_dir,
            train_split=self.config.data.train_split,
            val_split=self.config.data.val_split,
            test_split=self.config.data.test_split,
            total_samples=1000  # Fixed for toy dataset
        )
        
        # Create datasets
        self.train_dataset = ToyMultimodalDataset(
            data_dir=self.config.paths.data_dir,
            split="train",
            image_size=self.config.data.image_size,
            max_text_length=self.config.data.max_text_length,
        )
        
        self.val_dataset = ToyMultimodalDataset(
            data_dir=self.config.paths.data_dir,
            split="val",
            image_size=self.config.data.image_size,
            max_text_length=self.config.data.max_text_length,
        )
        
        self.test_dataset = ToyMultimodalDataset(
            data_dir=self.config.paths.data_dir,
            split="test",
            image_size=self.config.data.image_size,
            max_text_length=self.config.data.max_text_length,
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        
        print(f"Data loaders created:")
        print(f"  Train: {len(self.train_loader)} batches")
        print(f"  Val: {len(self.val_loader)} batches")
        print(f"  Test: {len(self.test_loader)} batches")
    
    def _setup_logging(self):
        """Setup logging and tensorboard."""
        if self.config.logging.use_tensorboard:
            log_dir = Path(self.config.paths.log_dir) / "tensorboard"
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = move_to_device(batch, self.device)
            
            # Forward pass
            outputs = self.model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            
            # Compute loss
            loss_dict = self.loss_fn(
                logits_per_image=outputs["logits_per_image"],
                logits_per_text=outputs["logits_per_text"],
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Compute metrics
            metrics = compute_contrastive_metrics(
                logits_per_image=outputs["logits_per_image"],
                logits_per_text=outputs["logits_per_text"],
                image_embeddings=outputs["image_embeddings"],
                text_embeddings=outputs["text_embeddings"],
            )
            
            # Add loss to metrics
            metrics.update({
                "total_loss": loss_dict["total_loss"].item(),
                "loss_i2t": loss_dict["loss_i2t"].item(),
                "loss_t2i": loss_dict["loss_t2i"].item(),
            })
            
            epoch_metrics.append(metrics)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_dict['total_loss'].item():.4f}",
                "acc": f"{metrics['accuracy']:.4f}",
                "r@1": f"{metrics['recall_at_1']:.4f}",
            })
            
            # Log to tensorboard
            if self.writer and self.global_step % self.config.logging.log_every_n_steps == 0:
                for key, value in metrics.items():
                    self.writer.add_scalar(f"train/{key}", value, self.global_step)
            
            self.global_step += 1
        
        # Average metrics over epoch
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
        
        return avg_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        epoch_metrics = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                # Move batch to device
                batch = move_to_device(batch, self.device)
                
                # Forward pass
                outputs = self.model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                
                # Compute loss
                loss_dict = self.loss_fn(
                    logits_per_image=outputs["logits_per_image"],
                    logits_per_text=outputs["logits_per_text"],
                )
                
                # Compute metrics
                metrics = compute_contrastive_metrics(
                    logits_per_image=outputs["logits_per_image"],
                    logits_per_text=outputs["logits_per_text"],
                    image_embeddings=outputs["image_embeddings"],
                    text_embeddings=outputs["text_embeddings"],
                )
                
                # Add loss to metrics
                metrics.update({
                    "total_loss": loss_dict["total_loss"].item(),
                    "loss_i2t": loss_dict["loss_i2t"].item(),
                    "loss_t2i": loss_dict["loss_t2i"].item(),
                })
                
                epoch_metrics.append(metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss_dict['total_loss'].item():.4f}",
                    "acc": f"{metrics['accuracy']:.4f}",
                    "r@1": f"{metrics['recall_at_1']:.4f}",
                })
        
        # Average metrics over epoch
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
        
        return avg_metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.paths.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        print_config(self.config)
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log epoch results
            print(f"\nEpoch {epoch} Results:")
            print("Train Metrics:")
            print(format_metrics(train_metrics, "  "))
            print("Val Metrics:")
            print(format_metrics(val_metrics, "  "))
            
            # Log to tensorboard
            if self.writer:
                for key, value in train_metrics.items():
                    self.writer.add_scalar(f"epoch/train_{key}", value, epoch)
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f"epoch/val_{key}", value, epoch)
            
            # Save checkpoint
            is_best = val_metrics["total_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["total_loss"]
            
            if epoch % self.config.training.save_every_n_epochs == 0 or is_best:
                self.save_checkpoint(is_best)
        
        print("Training completed!")
        
        # Close tensorboard writer
        if self.writer:
            self.writer.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train multi-modal self-supervised learning model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Trainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.current_epoch = checkpoint["epoch"]
        trainer.global_step = checkpoint["global_step"]
        trainer.best_val_loss = checkpoint["best_val_loss"]
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
