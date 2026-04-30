"""CLIP-style contrastive learning model implementation."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class ContrastiveCLIPModel(nn.Module):
    """
    CLIP-style model for multi-modal self-supervised learning.
    
    Implements dual encoders with contrastive learning for image-text matching.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        projection_dim: int = 512,
        temperature: float = 0.07,
        freeze_backbone: bool = False,
    ):
        """
        Initialize the contrastive CLIP model.
        
        Args:
            model_name: Name of the pre-trained CLIP model
            projection_dim: Dimension of the projection layer
            temperature: Temperature parameter for contrastive loss
            freeze_backbone: Whether to freeze the backbone encoders
        """
        super().__init__()
        
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        # Load pre-trained CLIP model
        self.clip_model = CLIPModel.from_pretrained(model_name)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Get original projection dimensions
        self.vision_projection_dim = self.clip_model.visual_projection.out_features
        self.text_projection_dim = self.clip_model.text_projection.out_features
        
        # Create new projection layers if dimensions don't match
        if self.vision_projection_dim != projection_dim:
            self.vision_projection = nn.Linear(self.vision_projection_dim, projection_dim)
        else:
            self.vision_projection = nn.Identity()
            
        if self.text_projection_dim != projection_dim:
            self.text_projection = nn.Linear(self.text_projection_dim, projection_dim)
        else:
            self.text_projection = nn.Identity()
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))
        
        # Initialize projections
        self._init_projections()
    
    def _init_projections(self) -> None:
        """Initialize projection layers."""
        if isinstance(self.vision_projection, nn.Linear):
            nn.init.normal_(self.vision_projection.weight, std=0.02)
            nn.init.zeros_(self.vision_projection.bias)
            
        if isinstance(self.text_projection, nn.Linear):
            nn.init.normal_(self.text_projection.weight, std=0.02)
            nn.init.zeros_(self.text_projection.bias)
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings.
        
        Args:
            pixel_values: Image pixel values
            
        Returns:
            Image embeddings
        """
        # Get image features from CLIP
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
        
        # Project to common space
        image_embeddings = self.vision_projection(image_features)
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        
        return image_embeddings
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            input_ids: Text token IDs
            attention_mask: Attention mask for text
            
        Returns:
            Text embeddings
        """
        # Get text features from CLIP
        text_features = self.clip_model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Project to common space
        text_embeddings = self.text_projection(text_features)
        
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        return text_embeddings
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            pixel_values: Image pixel values
            input_ids: Text token IDs
            attention_mask: Attention mask for text
            
        Returns:
            Dictionary containing embeddings and logits
        """
        # Encode images and text
        image_embeddings = self.encode_image(pixel_values)
        text_embeddings = self.encode_text(input_ids, attention_mask)
        
        # Compute similarity matrix
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.T
        logits_per_text = logits_per_image.T
        
        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "logit_scale": logit_scale,
        }
    
    def get_embeddings(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get embeddings for images and/or text.
        
        Args:
            pixel_values: Image pixel values (optional)
            input_ids: Text token IDs (optional)
            attention_mask: Attention mask for text (optional)
            
        Returns:
            Dictionary containing embeddings
        """
        embeddings = {}
        
        if pixel_values is not None:
            embeddings["image_embeddings"] = self.encode_image(pixel_values)
            
        if input_ids is not None and attention_mask is not None:
            embeddings["text_embeddings"] = self.encode_text(input_ids, attention_mask)
        
        return embeddings


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for multi-modal self-supervised learning.
    
    Implements InfoNCE loss for image-text matching.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for scaling logits
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(
        self,
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss.
        
        Args:
            logits_per_image: Logits for image-to-text similarity
            logits_per_text: Logits for text-to-image similarity
            
        Returns:
            Dictionary containing loss components
        """
        batch_size = logits_per_image.size(0)
        
        # Create labels (diagonal elements are positive pairs)
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        # Compute losses
        loss_i2t = self.cross_entropy(logits_per_image / self.temperature, labels)
        loss_t2i = self.cross_entropy(logits_per_text / self.temperature, labels)
        
        # Total loss is the average
        total_loss = (loss_i2t + loss_t2i) / 2
        
        return {
            "total_loss": total_loss,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
        }


def create_model(config: Dict) -> ContrastiveCLIPModel:
    """
    Create model from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized model
    """
    model_config = config.get("model", {})
    
    return ContrastiveCLIPModel(
        model_name=model_config.get("vision_encoder", {}).get("model_name", "openai/clip-vit-base-patch32"),
        projection_dim=model_config.get("vision_encoder", {}).get("projection_dim", 512),
        temperature=model_config.get("temperature", 0.07),
        freeze_backbone=model_config.get("vision_encoder", {}).get("freeze_backbone", False),
    )


def create_loss(config: Dict) -> ContrastiveLoss:
    """
    Create loss function from configuration.
    
    Args:
        config: Loss configuration
        
    Returns:
        Initialized loss function
    """
    loss_config = config.get("loss", {})
    
    return ContrastiveLoss(
        temperature=loss_config.get("temperature", 0.07),
        label_smoothing=loss_config.get("label_smoothing", 0.0),
    )
