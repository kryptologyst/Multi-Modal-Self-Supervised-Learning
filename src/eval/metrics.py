"""Evaluation metrics for multi-modal retrieval tasks."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple


def compute_retrieval_metrics(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute retrieval metrics for image-text matching.
    
    Args:
        image_embeddings: Image embeddings [batch_size, embedding_dim]
        text_embeddings: Text embeddings [batch_size, embedding_dim]
        k_values: List of k values for Recall@k
        
    Returns:
        Dictionary containing retrieval metrics
    """
    batch_size = image_embeddings.size(0)
    device = image_embeddings.device
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(image_embeddings, text_embeddings.T)
    
    # Get ranks for image-to-text retrieval
    i2t_ranks = []
    for i in range(batch_size):
        # Get similarities for image i with all texts
        similarities = similarity_matrix[i]
        # Sort in descending order and get ranks
        _, indices = torch.sort(similarities, descending=True)
        rank = (indices == i).nonzero(as_tuple=True)[0].item() + 1
        i2t_ranks.append(rank)
    
    # Get ranks for text-to-image retrieval
    t2i_ranks = []
    for i in range(batch_size):
        # Get similarities for text i with all images
        similarities = similarity_matrix[:, i]
        # Sort in descending order and get ranks
        _, indices = torch.sort(similarities, descending=True)
        rank = (indices == i).nonzero(as_tuple=True)[0].item() + 1
        t2i_ranks.append(rank)
    
    # Convert to tensors
    i2t_ranks = torch.tensor(i2t_ranks, device=device)
    t2i_ranks = torch.tensor(t2i_ranks, device=device)
    
    # Compute metrics
    metrics = {}
    
    # Recall@k for image-to-text
    for k in k_values:
        recall_i2t = (i2t_ranks <= k).float().mean().item()
        metrics[f"recall_i2t_at_{k}"] = recall_i2t
    
    # Recall@k for text-to-image
    for k in k_values:
        recall_t2i = (t2i_ranks <= k).float().mean().item()
        metrics[f"recall_t2i_at_{k}"] = recall_t2i
    
    # Average recall@k
    for k in k_values:
        avg_recall = (metrics[f"recall_i2t_at_{k}"] + metrics[f"recall_t2i_at_{k}"]) / 2
        metrics[f"recall_at_{k}"] = avg_recall
    
    # Median rank
    metrics["median_rank_i2t"] = torch.median(i2t_ranks.float()).item()
    metrics["median_rank_t2i"] = torch.median(t2i_ranks.float()).item()
    metrics["median_rank"] = (metrics["median_rank_i2t"] + metrics["median_rank_t2i"]) / 2
    
    # Mean rank
    metrics["mean_rank_i2t"] = i2t_ranks.float().mean().item()
    metrics["mean_rank_t2i"] = t2i_ranks.float().mean().item()
    metrics["mean_rank"] = (metrics["mean_rank_i2t"] + metrics["mean_rank_t2i"]) / 2
    
    return metrics


def compute_accuracy(logits: torch.Tensor) -> float:
    """
    Compute accuracy from logits.
    
    Args:
        logits: Logits tensor [batch_size, num_classes]
        
    Returns:
        Accuracy score
    """
    predictions = torch.argmax(logits, dim=-1)
    labels = torch.arange(logits.size(0), device=logits.device)
    correct = (predictions == labels).float().mean().item()
    return correct


def compute_contrastive_metrics(
    logits_per_image: torch.Tensor,
    logits_per_text: torch.Tensor,
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for contrastive learning.
    
    Args:
        logits_per_image: Image-to-text logits
        logits_per_text: Text-to-image logits
        image_embeddings: Image embeddings
        text_embeddings: Text embeddings
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    # Accuracy metrics
    metrics["accuracy_i2t"] = compute_accuracy(logits_per_image)
    metrics["accuracy_t2i"] = compute_accuracy(logits_per_text)
    metrics["accuracy"] = (metrics["accuracy_i2t"] + metrics["accuracy_t2i"]) / 2
    
    # Retrieval metrics
    retrieval_metrics = compute_retrieval_metrics(image_embeddings, text_embeddings)
    metrics.update(retrieval_metrics)
    
    # Embedding statistics
    metrics["image_embedding_norm"] = torch.norm(image_embeddings, dim=-1).mean().item()
    metrics["text_embedding_norm"] = torch.norm(text_embeddings, dim=-1).mean().item()
    
    # Similarity statistics
    batch_size = image_embeddings.size(0)
    labels = torch.arange(batch_size, device=image_embeddings.device)
    
    # Positive similarities (diagonal elements)
    positive_sim_i2t = logits_per_image[labels, labels].mean().item()
    positive_sim_t2i = logits_per_text[labels, labels].mean().item()
    
    metrics["positive_similarity_i2t"] = positive_sim_i2t
    metrics["positive_similarity_t2i"] = positive_sim_t2i
    metrics["positive_similarity"] = (positive_sim_i2t + positive_sim_t2i) / 2
    
    # Negative similarities (off-diagonal elements)
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=image_embeddings.device)
    negative_sim_i2t = logits_per_image[mask].mean().item()
    negative_sim_t2i = logits_per_text[mask].mean().item()
    
    metrics["negative_similarity_i2t"] = negative_sim_i2t
    metrics["negative_similarity_t2i"] = negative_sim_t2i
    metrics["negative_similarity"] = (negative_sim_i2t + negative_sim_t2i) / 2
    
    # Similarity gap
    metrics["similarity_gap_i2t"] = positive_sim_i2t - negative_sim_i2t
    metrics["similarity_gap_t2i"] = positive_sim_t2i - negative_sim_t2i
    metrics["similarity_gap"] = (metrics["similarity_gap_i2t"] + metrics["similarity_gap_t2i"]) / 2
    
    return metrics


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    Format metrics for logging.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for metric names
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{prefix}{key}: {value:.4f}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    
    return "\n".join(lines)


def compute_metrics_summary(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Compute summary statistics across multiple metric dictionaries.
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Dictionary with mean and std of each metric
    """
    if not metrics_list:
        return {}
    
    summary = {}
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    for key in all_keys:
        values = [m.get(key, 0.0) for m in metrics_list if key in m]
        if values:
            summary[f"{key}_mean"] = sum(values) / len(values)
            summary[f"{key}_std"] = (sum((v - summary[f"{key}_mean"]) ** 2 for v in values) / len(values)) ** 0.5
    
    return summary
