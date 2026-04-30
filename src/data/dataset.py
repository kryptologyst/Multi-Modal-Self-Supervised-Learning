"""Data loading and preprocessing utilities."""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor


class ToyMultimodalDataset(Dataset):
    """
    Toy dataset for multi-modal self-supervised learning.
    
    Creates synthetic image-text pairs for demonstration purposes.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        split: str = "train",
        image_size: int = 224,
        max_text_length: int = 77,
        num_samples: int = 1000,
    ):
        """
        Initialize the toy dataset.
        
        Args:
            data_dir: Directory to store/load data
            split: Dataset split ("train", "val", "test")
            image_size: Size of images
            max_text_length: Maximum text length
            num_samples: Number of samples to generate
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.num_samples = num_samples
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CLIP processor
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Generate or load data
        self.data = self._generate_or_load_data()
        
    def _generate_or_load_data(self) -> List[Dict]:
        """Generate or load synthetic data."""
        data_file = self.data_dir / f"toy_data_{self.split}.json"
        
        if data_file.exists():
            with open(data_file, 'r') as f:
                return json.load(f)
        else:
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> List[Dict]:
        """Generate synthetic image-text pairs."""
        # Define categories and their descriptions
        categories = {
            "animals": [
                "A cute dog playing in the park",
                "A fluffy cat sitting on a windowsill",
                "A colorful bird flying in the sky",
                "A majestic horse running in a field",
                "A playful dolphin swimming in the ocean"
            ],
            "objects": [
                "A red car parked on the street",
                "A beautiful flower blooming in a garden",
                "A modern building with glass windows",
                "A delicious pizza on a wooden table",
                "A vintage bicycle leaning against a wall"
            ],
            "scenes": [
                "A peaceful sunset over the mountains",
                "A busy city street with people walking",
                "A serene lake surrounded by trees",
                "A cozy living room with warm lighting",
                "A bustling market with colorful stalls"
            ]
        }
        
        data = []
        random.seed(42)  # For reproducibility
        
        for i in range(self.num_samples):
            # Randomly select category and description
            category = random.choice(list(categories.keys()))
            text = random.choice(categories[category])
            
            # Generate synthetic image (random noise for demo)
            image = self._generate_synthetic_image(category)
            
            data.append({
                "id": f"{self.split}_{i:06d}",
                "text": text,
                "category": category,
                "image_path": f"synthetic_{i:06d}.jpg"
            })
        
        # Save generated data
        data_file = self.data_dir / f"toy_data_{self.split}.json"
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return data
    
    def _generate_synthetic_image(self, category: str) -> Image.Image:
        """Generate a synthetic image based on category."""
        # Create a simple synthetic image with different colors for different categories
        colors = {
            "animals": (100, 150, 200),  # Blue-ish
            "objects": (200, 100, 150),  # Pink-ish
            "scenes": (150, 200, 100),   # Green-ish
        }
        
        base_color = colors.get(category, (128, 128, 128))
        
        # Create image with some variation
        image_array = np.random.randint(
            max(0, base_color[0] - 50),
            min(255, base_color[0] + 50),
            size=(self.image_size, self.image_size, 3),
            dtype=np.uint8
        )
        
        # Add some structure (simple patterns)
        center = self.image_size // 2
        radius = self.image_size // 4
        
        y, x = np.ogrid[:self.image_size, :self.image_size]
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        
        # Create a circular pattern
        image_array[mask] = [
            min(255, base_color[0] + 50),
            min(255, base_color[1] + 50),
            min(255, base_color[2] + 50)
        ]
        
        return Image.fromarray(image_array)
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        item = self.data[idx]
        
        # Generate synthetic image
        image = self._generate_synthetic_image(item["category"])
        
        # Process with CLIP processor
        inputs = self.processor(
            text=item["text"],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "text": item["text"],
            "category": item["category"],
            "id": item["id"]
        }


def create_data_splits(
    data_dir: str = "data",
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    total_samples: int = 1000
) -> None:
    """
    Create train/val/test splits of the toy dataset.
    
    Args:
        data_dir: Directory to store data
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        total_samples: Total number of samples
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    # Calculate split sizes
    train_size = int(total_samples * train_split)
    val_size = int(total_samples * val_split)
    test_size = total_samples - train_size - val_size
    
    # Create datasets for each split
    train_dataset = ToyMultimodalDataset(data_dir, "train", num_samples=train_size)
    val_dataset = ToyMultimodalDataset(data_dir, "val", num_samples=val_size)
    test_dataset = ToyMultimodalDataset(data_dir, "test", num_samples=test_size)
    
    print(f"Created datasets:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of batch items
        
    Returns:
        Batched tensors
    """
    # Stack tensors
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    
    # Keep text and metadata as lists
    texts = [item["text"] for item in batch]
    categories = [item["category"] for item in batch]
    ids = [item["id"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "texts": texts,
        "categories": categories,
        "ids": ids
    }
