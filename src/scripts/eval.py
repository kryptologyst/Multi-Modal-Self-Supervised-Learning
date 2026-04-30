"""Evaluation script for multi-modal self-supervised learning."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import ToyMultimodalDataset, collate_fn
from src.eval.metrics import compute_contrastive_metrics, format_metrics, compute_metrics_summary
from src.models.clip_model import ContrastiveCLIPModel
from src.utils.config import load_config
from src.utils.device import setup_device_and_seed, move_to_device


class Evaluator:
    """Evaluator class for multi-modal self-supervised learning."""
    
    def __init__(self, config_path: str, checkpoint_path: str):
        """
        Initialize evaluator.
        
        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to model checkpoint
        """
        self.config = load_config(config_path)
        self.device = setup_device_and_seed(
            device_type=self.config.device.type,
            deterministic=self.config.device.deterministic
        )
        
        # Load model
        self.model = ContrastiveCLIPModel(
            model_name=self.config.model.vision_encoder.model_name,
            projection_dim=self.config.model.vision_encoder.projection_dim,
            temperature=self.config.model.temperature,
            freeze_backbone=self.config.model.vision_encoder.freeze_backbone,
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        print(f"Model loaded from: {checkpoint_path}")
        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        
        # Setup data loaders
        self._setup_data_loaders()
    
    def _setup_data_loaders(self):
        """Setup data loaders for evaluation."""
        # Create datasets
        self.test_dataset = ToyMultimodalDataset(
            data_dir=self.config.paths.data_dir,
            split="test",
            image_size=self.config.data.image_size,
            max_text_length=self.config.data.max_text_length,
        )
        
        self.val_dataset = ToyMultimodalDataset(
            data_dir=self.config.paths.data_dir,
            split="val",
            image_size=self.config.data.image_size,
            max_text_length=self.config.data.max_text_length,
        )
        
        # Create data loaders
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
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
        
        print(f"Data loaders created:")
        print(f"  Test: {len(self.test_loader)} batches")
        print(f"  Val: {len(self.val_loader)} batches")
    
    def evaluate_split(self, data_loader: DataLoader, split_name: str) -> Dict[str, float]:
        """
        Evaluate model on a data split.
        
        Args:
            data_loader: Data loader for the split
            split_name: Name of the split
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        all_metrics = []
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f"Evaluating {split_name}")
            for batch in pbar:
                # Move batch to device
                batch = move_to_device(batch, self.device)
                
                # Forward pass
                outputs = self.model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                
                # Compute metrics
                metrics = compute_contrastive_metrics(
                    logits_per_image=outputs["logits_per_image"],
                    logits_per_text=outputs["logits_per_text"],
                    image_embeddings=outputs["image_embeddings"],
                    text_embeddings=outputs["text_embeddings"],
                )
                
                all_metrics.append(metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    "acc": f"{metrics['accuracy']:.4f}",
                    "r@1": f"{metrics['recall_at_1']:.4f}",
                    "r@5": f"{metrics['recall_at_5']:.4f}",
                })
        
        # Average metrics over all batches
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
        return avg_metrics
    
    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on all splits.
        
        Returns:
            Dictionary containing metrics for each split
        """
        print("Starting evaluation...")
        
        results = {}
        
        # Evaluate validation set
        val_metrics = self.evaluate_split(self.val_loader, "validation")
        results["validation"] = val_metrics
        
        # Evaluate test set
        test_metrics = self.evaluate_split(self.test_loader, "test")
        results["test"] = test_metrics
        
        return results
    
    def save_results(self, results: Dict[str, Dict[str, float]], output_path: str):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
            output_path: Path to save results
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    def print_results(self, results: Dict[str, Dict[str, float]]):
        """
        Print evaluation results.
        
        Args:
            results: Evaluation results
        """
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        for split_name, metrics in results.items():
            print(f"\n{split_name.upper()} SET:")
            print("-" * 40)
            print(format_metrics(metrics, "  "))
        
        # Summary
        print(f"\nSUMMARY:")
        print("-" * 40)
        for split_name, metrics in results.items():
            print(f"{split_name.capitalize()} - Accuracy: {metrics['accuracy']:.4f}, "
                  f"Recall@1: {metrics['recall_at_1']:.4f}, "
                  f"Recall@5: {metrics['recall_at_5']:.4f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate multi-modal self-supervised learning model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="outputs/evaluation_results.json", help="Path to save results")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = Evaluator(args.config, args.checkpoint)
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()
