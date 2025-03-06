#!/usr/bin/env python3
"""
Streamlit app for Vector Database Lightroom

This app provides a simple UI for:
1. Searching the vector database by text or image
2. Displaying search results with thumbnails
3. Viewing image metadata
"""

import os
import sys
import streamlit as st
import torch
import clip
from PIL import Image
import io
from qdrant_client import QdrantClient
import numpy as np
import rawpy
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.search import (
    load_clip_model, 
    embed_text, 
    embed_image, 
    load_and_preprocess_image,
    COLLECTION_NAME
)

# Set page config
st.set_page_config(
    page_title="Vector Database Lightroom",
    page_icon="🖼️",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None

@st.cache_resource
def get_clip_model():
    """Load and cache the CLIP model"""
    return load_clip_model()

def display_image(file_path):
    """Display an image (handles regular images and RAW files)"""
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext in [".dng", ".raw", ".cr2", ".nef", ".arw", ".orf", ".raf"]:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess()
                return Image.fromarray(rgb)
        else:
            return Image.open(file_path).convert("RGB")
    except Exception as e:
        st.error(f"Error loading image {file_path}: {str(e)}")
        return None

def search_qdrant(query_vector, top_k=10):
    """Search Qdrant for the top_k most similar vectors."""
    client = QdrantClient(url="http://localhost:6333")

    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    return search_result

def main():
    st.title("Vector Database Lightroom")
    
    # Sidebar for search options
    with st.sidebar:
        st.header("Search Options")
        search_method = st.radio(
            "Search Method",
            ["Text", "Image"]
        )
        
        query_vector = None
        
        if search_method == "Text":
            text_query = st.text_input("Enter text query")
            if st.button("Search by Text"):
                if text_query:
                    with st.spinner("Generating text embedding..."):
                        model, _ = get_clip_model()
                        query_vector = embed_text(model, text_query)
                else:
                    st.warning("Please enter a text query")
                    
        else:  # Image search
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "dng", "raw", "cr2", "nef"])
            if uploaded_file is not None:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Search by Image"):
                    with st.spinner("Generating image embedding..."):
                        model, preprocess = get_clip_model()
                        
                        # For DNG/RAW files, save temporarily and use rawpy
                        ext = os.path.splitext(uploaded_file.name)[1].lower()
                        if ext in [".dng", ".raw", ".cr2", ".nef", ".arw", ".orf", ".raf"]:
                            temp_path = f"temp_upload{ext}"
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            preprocessed_img = load_and_preprocess_image(temp_path, preprocess)
                            os.remove(temp_path)  # Clean up
                        else:
                            img = Image.open(uploaded_file).convert("RGB")
                            preprocessed_img = preprocess(img)
                            
                        query_vector = embed_image(model, preprocessed_img)
        
        # Number of results slider
        top_k = st.slider("Number of results", min_value=1, max_value=50, value=10)
        
        # Perform search if query_vector is available
        if query_vector:
            with st.spinner("Searching database..."):
                results = search_qdrant(query_vector, top_k=top_k)
                st.session_state.search_results = results
                st.success(f"Found {len(results)} results!")
    
    # Main area for displaying results
    if st.session_state.search_results:
        st.header("Search Results")
        
        # Display results in a grid
        cols = 3  # Number of columns in the grid
        rows = (len(st.session_state.search_results) + cols - 1) // cols  # Calculate number of rows needed
        
        for row in range(rows):
            with st.container():
                columns = st.columns(cols)
                for col in range(cols):
                    idx = row * cols + col
                    if idx < len(st.session_state.search_results):
                        result = st.session_state.search_results[idx]
                        file_path = result.payload.get("file_path", "Unknown")
                        
                        with columns[col]:
                            # Create a unique key for each button
                            button_key = f"img_button_{idx}"
                            
                            # Try to display the image thumbnail
                            thumbnail = display_image(file_path)
                            if thumbnail:
                                # Resize for thumbnail
                                thumbnail.thumbnail((300, 300))
                                st.image(thumbnail, caption=f"Score: {result.score:.4f}")
                            else:
                                st.write(f"Image: {Path(file_path).name}")
                                
                            if st.button(f"View Details", key=button_key):
                                st.session_state.selected_image = result
    
    # Detail view for selected image
    if st.session_state.selected_image:
        st.header("Image Details")
        
        result = st.session_state.selected_image
        file_path = result.payload.get("file_path", "Unknown")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            img = display_image(file_path)
            if img:
                st.image(img, caption=Path(file_path).name, use_column_width=True)
            else:
                st.error("Could not load image")
        
        with col2:
            st.subheader("Metadata")
            st.write(f"**Filename:** {Path(file_path).name}")
            st.write(f"**Path:** {file_path}")
            st.write(f"**Similarity Score:** {result.score:.4f}")
            
            # Display other metadata from payload
            st.json(result.payload)
            
        if st.button("Back to Results"):
            st.session_state.selected_image = None
            st.experimental_rerun()

if __name__ == "__main__":
    main() 