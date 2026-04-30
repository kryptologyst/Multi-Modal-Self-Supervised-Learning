"""Streamlit demo for multi-modal self-supervised learning."""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from src.models.clip_model import ContrastiveCLIPModel
from src.utils.config import load_config
from src.utils.device import setup_device_and_seed, move_to_device
from src.data.dataset import ToyMultimodalDataset


def load_model(config_path: str, checkpoint_path: str):
    """Load the trained model."""
    config = load_config(config_path)
    device = setup_device_and_seed(device_type=config.device.type)
    
    model = ContrastiveCLIPModel(
        model_name=config.model.vision_encoder.model_name,
        projection_dim=config.model.vision_encoder.projection_dim,
        temperature=config.model.temperature,
        freeze_backbone=config.model.vision_encoder.freeze_backbone,
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, device, config


def compute_similarity_matrix(model, device, texts, images):
    """Compute similarity matrix between texts and images."""
    from transformers import CLIPProcessor
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Process texts
    text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    text_inputs = move_to_device(text_inputs, device)
    
    # Process images
    image_inputs = processor(images=images, return_tensors="pt")
    image_inputs = move_to_device(image_inputs, device)
    
    # Get embeddings
    with torch.no_grad():
        text_embeddings = model.encode_text(
            text_inputs["input_ids"],
            text_inputs["attention_mask"]
        )
        image_embeddings = model.encode_image(image_inputs["pixel_values"])
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(image_embeddings, text_embeddings.T)
    
    return similarity_matrix.cpu().numpy()


def visualize_similarity_matrix(similarity_matrix, texts, images):
    """Visualize similarity matrix."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        xticklabels=[f"Text {i+1}" for i in range(len(texts))],
        yticklabels=[f"Image {i+1}" for i in range(len(images))],
        ax=ax
    )
    
    ax.set_title("Image-Text Similarity Matrix")
    ax.set_xlabel("Text Descriptions")
    ax.set_ylabel("Images")
    
    plt.tight_layout()
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Multi-Modal Self-Supervised Learning Demo",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Multi-Modal Self-Supervised Learning Demo")
    st.markdown("""
    This demo showcases a CLIP-style model trained with contrastive learning for image-text matching.
    The model learns to associate images with their corresponding text descriptions without explicit supervision.
    """)
    
    # Sidebar for model loading
    st.sidebar.header("Model Configuration")
    
    config_path = st.sidebar.text_input(
        "Config Path",
        value="configs/config.yaml",
        help="Path to the configuration file"
    )
    
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint Path",
        value="checkpoints/best_model.pt",
        help="Path to the model checkpoint"
    )
    
    if st.sidebar.button("Load Model"):
        try:
            with st.spinner("Loading model..."):
                model, device, config = load_model(config_path, checkpoint_path)
                st.session_state.model = model
                st.session_state.device = device
                st.session_state.config = config
            st.sidebar.success("Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
    
    # Main content
    if "model" not in st.session_state:
        st.warning("Please load a model first using the sidebar.")
        return
    
    model = st.session_state.model
    device = st.session_state.device
    config = st.session_state.config
    
    # Demo sections
    tab1, tab2, tab3 = st.tabs(["Image-Text Matching", "Retrieval Demo", "Model Analysis"])
    
    with tab1:
        st.header("Image-Text Matching")
        st.markdown("Upload images and enter text descriptions to see how well the model matches them.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Images")
            uploaded_images = st.file_uploader(
                "Upload images",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Upload multiple images to compare with text descriptions"
            )
            
            if uploaded_images:
                images = [Image.open(img) for img in uploaded_images]
                st.write(f"Uploaded {len(images)} images")
                
                # Display images
                for i, img in enumerate(images):
                    st.image(img, caption=f"Image {i+1}", use_column_width=True)
        
        with col2:
            st.subheader("Text Descriptions")
            text_input = st.text_area(
                "Enter text descriptions (one per line)",
                value="A cute dog playing in the park\nA fluffy cat sitting on a windowsill\nA red car parked on the street",
                help="Enter text descriptions, one per line"
            )
            
            texts = [line.strip() for line in text_input.split('\n') if line.strip()]
            
            if texts:
                st.write(f"Entered {len(texts)} text descriptions:")
                for i, text in enumerate(texts):
                    st.write(f"{i+1}. {text}")
        
        if uploaded_images and texts:
            if st.button("Compute Similarities"):
                with st.spinner("Computing similarities..."):
                    similarity_matrix = compute_similarity_matrix(model, device, texts, images)
                
                st.subheader("Similarity Matrix")
                fig = visualize_similarity_matrix(similarity_matrix, texts, images)
                st.pyplot(fig)
                
                # Show best matches
                st.subheader("Best Matches")
                for i, img in enumerate(images):
                    best_text_idx = np.argmax(similarity_matrix[i])
                    best_score = similarity_matrix[i, best_text_idx]
                    st.write(f"Image {i+1} best matches: \"{texts[best_text_idx]}\" (score: {best_score:.3f})")
    
    with tab2:
        st.header("Retrieval Demo")
        st.markdown("Test the model's ability to retrieve relevant images given text queries.")
        
        # Load toy dataset for retrieval demo
        try:
            dataset = ToyMultimodalDataset(
                data_dir=config.paths.data_dir,
                split="test",
                image_size=config.data.image_size,
                max_text_length=config.data.max_text_length,
            )
            
            st.subheader("Query Interface")
            query_text = st.text_input(
                "Enter a text query",
                value="A beautiful flower blooming in a garden",
                help="Enter a text description to find similar images"
            )
            
            if st.button("Search Images"):
                with st.spinner("Searching..."):
                    # Get query embedding
                    from transformers import CLIPProcessor
                    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    
                    query_inputs = processor(text=[query_text], return_tensors="pt", padding=True, truncation=True)
                    query_inputs = move_to_device(query_inputs, device)
                    
                    with torch.no_grad():
                        query_embedding = model.encode_text(
                            query_inputs["input_ids"],
                            query_inputs["attention_mask"]
                        )
                    
                    # Compute similarities with dataset images
                    similarities = []
                    for i in range(min(20, len(dataset))):  # Limit to first 20 for demo
                        item = dataset[i]
                        
                        # Generate synthetic image
                        image = dataset._generate_synthetic_image(item["category"])
                        
                        # Process image
                        image_inputs = processor(images=[image], return_tensors="pt")
                        image_inputs = move_to_device(image_inputs, device)
                        
                        with torch.no_grad():
                            image_embedding = model.encode_image(image_inputs["pixel_values"])
                        
                        # Compute similarity
                        similarity = torch.matmul(query_embedding, image_embedding.T).item()
                        similarities.append((i, similarity, item["text"], item["category"]))
                    
                    # Sort by similarity
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    # Display results
                    st.subheader("Search Results")
                    for rank, (idx, sim, text, category) in enumerate(similarities[:5]):
                        st.write(f"**Rank {rank+1}** (similarity: {sim:.3f})")
                        st.write(f"Text: {text}")
                        st.write(f"Category: {category}")
                        st.write("---")
        
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            st.info("Please make sure the dataset has been created by running the training script first.")
    
    with tab3:
        st.header("Model Analysis")
        st.markdown("Analyze the model's learned representations and performance.")
        
        # Model statistics
        st.subheader("Model Statistics")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Parameters", f"{total_params:,}")
        with col2:
            st.metric("Trainable Parameters", f"{trainable_params:,}")
        with col3:
            st.metric("Model Size", f"{total_params * 4 / 1024 / 1024:.1f} MB")
        
        # Configuration
        st.subheader("Model Configuration")
        config_dict = {
            "Model Name": config.model.vision_encoder.model_name,
            "Projection Dimension": config.model.vision_encoder.projection_dim,
            "Temperature": config.model.temperature,
            "Freeze Backbone": config.model.vision_encoder.freeze_backbone,
        }
        
        for key, value in config_dict.items():
            st.write(f"**{key}**: {value}")
        
        # Safety disclaimer
        st.subheader("Safety & Limitations")
        st.warning("""
        **Important Disclaimers:**
        
        - This is a research/educational demo and should not be used for production applications
        - The model may exhibit biases present in the training data
        - Results should be interpreted with caution and validated for specific use cases
        - This demo uses synthetic data for demonstration purposes
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Multi-Modal Self-Supervised Learning Demo | 
        <a href='https://github.com/kryptologyst' target='_blank'>github.com/kryptologyst</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
