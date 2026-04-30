"""Tests for the multi-modal self-supervised learning package."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.clip_model import ContrastiveCLIPModel, ContrastiveLoss
from src.data.dataset import ToyMultimodalDataset, collate_fn
from src.eval.metrics import compute_retrieval_metrics, compute_contrastive_metrics
from src.utils.device import get_device, set_seed
from src.utils.config import load_config


class TestContrastiveCLIPModel:
    """Test cases for ContrastiveCLIPModel."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = ContrastiveCLIPModel()
        assert isinstance(model, ContrastiveCLIPModel)
        assert hasattr(model, 'clip_model')
        assert hasattr(model, 'vision_projection')
        assert hasattr(model, 'text_projection')
        assert hasattr(model, 'logit_scale')
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = ContrastiveCLIPModel()
        batch_size = 4
        
        # Create dummy inputs
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (batch_size, 77))
        attention_mask = torch.ones(batch_size, 77)
        
        # Forward pass
        outputs = model(pixel_values, input_ids, attention_mask)
        
        # Check outputs
        assert "image_embeddings" in outputs
        assert "text_embeddings" in outputs
        assert "logits_per_image" in outputs
        assert "logits_per_text" in outputs
        
        # Check shapes
        assert outputs["image_embeddings"].shape == (batch_size, 512)
        assert outputs["text_embeddings"].shape == (batch_size, 512)
        assert outputs["logits_per_image"].shape == (batch_size, batch_size)
        assert outputs["logits_per_text"].shape == (batch_size, batch_size)
    
    def test_embedding_normalization(self):
        """Test that embeddings are normalized."""
        model = ContrastiveCLIPModel()
        batch_size = 2
        
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (batch_size, 77))
        attention_mask = torch.ones(batch_size, 77)
        
        outputs = model(pixel_values, input_ids, attention_mask)
        
        # Check normalization
        image_norms = torch.norm(outputs["image_embeddings"], dim=-1)
        text_norms = torch.norm(outputs["text_embeddings"], dim=-1)
        
        assert torch.allclose(image_norms, torch.ones_like(image_norms), atol=1e-6)
        assert torch.allclose(text_norms, torch.ones_like(text_norms), atol=1e-6)


class TestContrastiveLoss:
    """Test cases for ContrastiveLoss."""
    
    def test_loss_initialization(self):
        """Test loss function initialization."""
        loss_fn = ContrastiveLoss()
        assert isinstance(loss_fn, ContrastiveLoss)
        assert hasattr(loss_fn, 'temperature')
        assert hasattr(loss_fn, 'cross_entropy')
    
    def test_loss_computation(self):
        """Test loss computation."""
        loss_fn = ContrastiveLoss()
        batch_size = 4
        
        # Create dummy logits
        logits_per_image = torch.randn(batch_size, batch_size)
        logits_per_text = torch.randn(batch_size, batch_size)
        
        # Compute loss
        loss_dict = loss_fn(logits_per_image, logits_per_text)
        
        # Check outputs
        assert "total_loss" in loss_dict
        assert "loss_i2t" in loss_dict
        assert "loss_t2i" in loss_dict
        
        # Check that losses are scalars
        assert loss_dict["total_loss"].dim() == 0
        assert loss_dict["loss_i2t"].dim() == 0
        assert loss_dict["loss_t2i"].dim() == 0
        
        # Check that losses are positive
        assert loss_dict["total_loss"].item() > 0
        assert loss_dict["loss_i2t"].item() > 0
        assert loss_dict["loss_t2i"].item() > 0


class TestToyMultimodalDataset:
    """Test cases for ToyMultimodalDataset."""
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = ToyMultimodalDataset(num_samples=10)
        assert isinstance(dataset, ToyMultimodalDataset)
        assert len(dataset) == 10
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        dataset = ToyMultimodalDataset(num_samples=5)
        
        item = dataset[0]
        
        # Check item structure
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "pixel_values" in item
        assert "text" in item
        assert "category" in item
        assert "id" in item
        
        # Check tensor shapes
        assert item["input_ids"].shape == (77,)
        assert item["attention_mask"].shape == (77,)
        assert item["pixel_values"].shape == (3, 224, 224)
    
    def test_collate_fn(self):
        """Test collate function."""
        dataset = ToyMultimodalDataset(num_samples=3)
        
        # Create batch
        batch = [dataset[i] for i in range(3)]
        collated = collate_fn(batch)
        
        # Check batch structure
        assert "input_ids" in collated
        assert "attention_mask" in collated
        assert "pixel_values" in collated
        assert "texts" in collated
        assert "categories" in collated
        assert "ids" in collated
        
        # Check batch shapes
        assert collated["input_ids"].shape == (3, 77)
        assert collated["attention_mask"].shape == (3, 77)
        assert collated["pixel_values"].shape == (3, 3, 224, 224)
        assert len(collated["texts"]) == 3
        assert len(collated["categories"]) == 3
        assert len(collated["ids"]) == 3


class TestMetrics:
    """Test cases for evaluation metrics."""
    
    def test_compute_retrieval_metrics(self):
        """Test retrieval metrics computation."""
        batch_size = 4
        embedding_dim = 512
        
        # Create dummy embeddings
        image_embeddings = torch.randn(batch_size, embedding_dim)
        text_embeddings = torch.randn(batch_size, embedding_dim)
        
        # Normalize embeddings
        image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)
        
        # Compute metrics
        metrics = compute_retrieval_metrics(image_embeddings, text_embeddings)
        
        # Check that metrics are computed
        assert "recall_at_1" in metrics
        assert "recall_at_5" in metrics
        assert "recall_at_10" in metrics
        assert "median_rank" in metrics
        assert "mean_rank" in metrics
        
        # Check that metrics are in valid ranges
        assert 0 <= metrics["recall_at_1"] <= 1
        assert 0 <= metrics["recall_at_5"] <= 1
        assert 0 <= metrics["recall_at_10"] <= 1
        assert metrics["median_rank"] >= 1
        assert metrics["mean_rank"] >= 1
    
    def test_compute_contrastive_metrics(self):
        """Test contrastive metrics computation."""
        batch_size = 4
        
        # Create dummy logits and embeddings
        logits_per_image = torch.randn(batch_size, batch_size)
        logits_per_text = torch.randn(batch_size, batch_size)
        image_embeddings = torch.randn(batch_size, 512)
        text_embeddings = torch.randn(batch_size, 512)
        
        # Compute metrics
        metrics = compute_contrastive_metrics(
            logits_per_image, logits_per_text, image_embeddings, text_embeddings
        )
        
        # Check that metrics are computed
        assert "accuracy" in metrics
        assert "recall_at_1" in metrics
        assert "median_rank" in metrics
        assert "image_embedding_norm" in metrics
        assert "text_embedding_norm" in metrics
        
        # Check that metrics are in valid ranges
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["recall_at_1"] <= 1
        assert metrics["median_rank"] >= 1


class TestUtils:
    """Test cases for utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("cpu")
        assert device.type == "cpu"
        
        device = get_device("auto")
        assert device.type in ["cpu", "cuda", "mps"]
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Check that random numbers are deterministic
        torch.manual_seed(42)
        rand1 = torch.randn(10)
        
        set_seed(42)
        torch.manual_seed(42)
        rand2 = torch.randn(10)
        
        assert torch.allclose(rand1, rand2)


if __name__ == "__main__":
    pytest.main([__file__])
