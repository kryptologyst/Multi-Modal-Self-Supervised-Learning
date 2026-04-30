# Multi-Modal Self-Supervised Learning

A production-ready implementation of multi-modal self-supervised learning using CLIP-style contrastive learning for image-text matching.

## Overview

This project implements a CLIP-style model that learns to associate images with their corresponding text descriptions through contrastive learning, without requiring explicit supervision. The model uses dual encoders (vision and text) with a contrastive loss function to learn meaningful cross-modal representations.

## Features

- **CLIP-style Architecture**: Dual encoders with contrastive learning
- **Self-Supervised Learning**: No labeled data required
- **Modern Tech Stack**: PyTorch 2.x, Transformers, OmegaConf
- **Device Support**: CUDA, MPS (Apple Silicon), CPU with automatic fallback
- **Comprehensive Evaluation**: Multiple retrieval metrics and analysis tools
- **Interactive Demo**: Streamlit-based web interface
- **Production Ready**: Type hints, tests, CI/CD, documentation

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- MPS (optional, for Apple Silicon)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Multi-Modal-Self-Supervised-Learning.git
cd Multi-Modal-Self-Supervised-Learning
```

2. Install dependencies:
```bash
pip install -e .
```

3. Install development dependencies (optional):
```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Training

Train the model with default configuration:

```bash
python -m src.scripts.train --config configs/config.yaml
```

### 2. Evaluation

Evaluate a trained model:

```bash
python -m src.scripts.eval --config configs/config.yaml --checkpoint checkpoints/best_model.pt
```

### 3. Demo

Launch the interactive demo:

```bash
streamlit run src/scripts/demo.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architectures
│   ├── eval/              # Evaluation metrics
│   ├── utils/             # Utility functions
│   └── scripts/           # Training, evaluation, and demo scripts
├── configs/               # Configuration files
├── data/                  # Data directory
├── checkpoints/          # Model checkpoints
├── logs/                  # Training logs
├── outputs/               # Evaluation outputs
├── assets/                # Generated assets
├── tests/                 # Unit tests
├── demo/                  # Demo assets
└── notebooks/             # Jupyter notebooks
```

## Configuration

The project uses YAML configuration files for easy customization. Key configuration options:

### Model Configuration
- `model_name`: Pre-trained CLIP model to use
- `projection_dim`: Dimension of the projection layer
- `temperature`: Temperature parameter for contrastive loss
- `freeze_backbone`: Whether to freeze the backbone encoders

### Training Configuration
- `epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimization
- `batch_size`: Batch size for training
- `mixed_precision`: Enable mixed precision training

### Data Configuration
- `image_size`: Size of input images
- `max_text_length`: Maximum text sequence length
- `train_split`: Fraction of data for training

## Dataset

The project includes a toy dataset generator that creates synthetic image-text pairs for demonstration purposes. The dataset includes:

- **Animals**: Dogs, cats, birds, horses, dolphins
- **Objects**: Cars, flowers, buildings, pizza, bicycles
- **Scenes**: Sunsets, city streets, lakes, living rooms, markets

### Custom Dataset

To use your own dataset, implement a custom dataset class following the `ToyMultimodalDataset` interface:

```python
class CustomDataset(Dataset):
    def __init__(self, ...):
        # Initialize your dataset
    
    def __len__(self):
        # Return dataset size
    
    def __getitem__(self, idx):
        # Return item with keys: input_ids, attention_mask, pixel_values, text, category, id
```

## Model Architecture

The model implements a CLIP-style architecture with:

1. **Vision Encoder**: Pre-trained ViT encoder for images
2. **Text Encoder**: Pre-trained transformer encoder for text
3. **Projection Layers**: Linear layers to project embeddings to common space
4. **Contrastive Loss**: InfoNCE loss for image-text matching

### Key Components

- **Dual Encoders**: Separate encoders for images and text
- **Contrastive Learning**: Learn by contrasting positive and negative pairs
- **Normalized Embeddings**: L2-normalized embeddings for stable training
- **Learnable Temperature**: Adaptive temperature parameter

## Evaluation Metrics

The model is evaluated using comprehensive retrieval metrics:

### Retrieval Metrics
- **Recall@K**: Fraction of queries where correct item is in top-K results
- **Median Rank**: Median rank of correct items
- **Mean Rank**: Average rank of correct items

### Accuracy Metrics
- **Image-to-Text Accuracy**: Fraction of correctly matched image-text pairs
- **Text-to-Image Accuracy**: Fraction of correctly matched text-image pairs

### Similarity Metrics
- **Positive Similarity**: Average similarity of matching pairs
- **Negative Similarity**: Average similarity of non-matching pairs
- **Similarity Gap**: Difference between positive and negative similarities

## Training

### Training Process

1. **Data Loading**: Load image-text pairs from dataset
2. **Forward Pass**: Encode images and text to embeddings
3. **Similarity Computation**: Compute similarity matrix
4. **Loss Calculation**: Apply contrastive loss
5. **Backward Pass**: Update model parameters
6. **Evaluation**: Compute metrics on validation set

### Training Features

- **Mixed Precision**: Automatic mixed precision for faster training
- **Gradient Clipping**: Prevent gradient explosion
- **Learning Rate Scheduling**: Warmup and decay
- **Checkpointing**: Save best model and regular checkpoints
- **TensorBoard Logging**: Real-time training visualization

## Demo Application

The Streamlit demo provides an interactive interface for:

### Image-Text Matching
- Upload images and enter text descriptions
- Visualize similarity matrix
- See best matches for each image

### Retrieval Demo
- Enter text queries to find similar images
- Browse through dataset with similarity scores
- Interactive search interface

### Model Analysis
- View model statistics and configuration
- Analyze learned representations
- Safety disclaimers and limitations

## API Reference

### Core Classes

#### `ContrastiveCLIPModel`
Main model class implementing CLIP-style contrastive learning.

```python
model = ContrastiveCLIPModel(
    model_name="openai/clip-vit-base-patch32",
    projection_dim=512,
    temperature=0.07,
    freeze_backbone=False
)
```

#### `ContrastiveLoss`
InfoNCE contrastive loss for image-text matching.

```python
loss_fn = ContrastiveLoss(temperature=0.07, label_smoothing=0.0)
```

#### `ToyMultimodalDataset`
Synthetic dataset for demonstration purposes.

```python
dataset = ToyMultimodalDataset(
    data_dir="data",
    split="train",
    image_size=224,
    max_text_length=77
)
```

### Utility Functions

#### Device Management
```python
device = get_device("auto")  # Auto-detect best device
set_seed(42, deterministic=True)  # Set random seeds
```

#### Configuration Management
```python
config = load_config("configs/config.yaml")
create_directories(config)  # Create necessary directories
```

## Development

### Code Style

The project uses:
- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **Type Hints**: Full type annotation
- **Docstrings**: Google-style docstrings

### Running Tests

```bash
pytest tests/ -v
```

### Pre-commit Hooks

Install pre-commit hooks:

```bash
pre-commit install
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Safety & Limitations

### Important Disclaimers

- **Research/Educational Use**: This project is intended for research and educational purposes
- **Not Production Ready**: Do not use for production applications without proper validation
- **Bias Awareness**: Models may exhibit biases present in training data
- **Synthetic Data**: Demo uses synthetic data for demonstration purposes
- **No Medical/Biometric Use**: Not suitable for medical diagnosis or biometric identification

### Limitations

- Limited to image-text pairs
- Requires significant computational resources for training
- Performance depends on quality of pre-trained encoders
- May not generalize well to domain-specific tasks

## Citation

If you use this project in your research, please cite:

```bibtex
@software{multimodal_ssl_2026,
  title={Multi-Modal Self-Supervised Learning},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Multi-Modal-Self-Supervised-Learning}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the CLIP model architecture
- Hugging Face for the Transformers library
- The PyTorch team for the deep learning framework
- The open-source community for various tools and libraries

## Contact

For questions, issues, or contributions, please visit:
- GitHub: [github.com/kryptologyst](https://github.com/kryptologyst)
- Issues: [GitHub Issues](https://github.com/kryptologyst/issues)

---

**Note**: This is a demonstration project showcasing multi-modal self-supervised learning techniques. For production use, please ensure proper validation, testing, and compliance with relevant regulations.
# Multi-Modal-Self-Supervised-Learning
