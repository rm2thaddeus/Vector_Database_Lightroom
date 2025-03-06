#!/usr/bin/env python3
"""
search.py

Use:
    python search.py --text "search query here"
    OR
    python search.py --image /path/to/query.jpg

Description:
    This script queries the Qdrant collection and retrieves the top N most similar images.
    It can also be imported and used by other Python scripts (like the Streamlit app).

Example:
    python search.py --text "a sunset over the mountains" --top 5
    python search.py --image "../images/test.dng" --top 3
"""

import sys
import os
import argparse
import clip
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, Range, PointStruct
from qdrant_client.http import models as rest
import rawpy

COLLECTION_NAME = "image_collection"
VECTOR_SIZE = 512

def load_clip_model():
    """Load the CLIP model (ViT-B/32) and return (model, preprocess)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model.to(device), preprocess

def embed_text(model, text):
    """Compute a normalized CLIP embedding for the input text."""
    device = next(model.parameters()).device
    with torch.no_grad():
        text_tokens = clip.tokenize([text]).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features[0].cpu().numpy().tolist()

def load_and_preprocess_image(image_path, preprocess):
    """
    Load and preprocess the image for CLIP.
    For RAW files, use rawpy to convert to RGB.
    Returns a PIL Image.
    """
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".dng", ".raw", ".cr2", ".nef", ".arw", ".orf", ".raf"]:
        with rawpy.imread(image_path) as raw:
            rgb = raw.postprocess()
        pil_image = Image.fromarray(rgb)
    else:
        pil_image = Image.open(image_path).convert("RGB")

    return preprocess(pil_image)

def embed_image(model, preprocessed_image):
    """Compute a normalized CLIP embedding for the input image."""
    device = next(model.parameters()).device
    with torch.no_grad():
        image_input = preprocessed_image.unsqueeze(0).to(device)
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features[0].cpu().numpy().tolist()

def search_qdrant(query_vector, top_k=5, collection_name=COLLECTION_NAME):
    """Search Qdrant for the top_k most similar vectors."""
    client = QdrantClient(url="http://localhost:6333")

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    return search_result

def process_query(text=None, image_path=None, top_k=5):
    """
    Process a query (either text or image) and return search results.
    This function is useful for external imports (like Streamlit).
    
    Args:
        text (str, optional): Text query. Defaults to None.
        image_path (str, optional): Path to image query. Defaults to None.
        top_k (int, optional): Number of results to return. Defaults to 5.
        
    Returns:
        list: Search results from Qdrant
    """
    model, preprocess = load_clip_model()
    
    # Prepare query vector
    if text:
        query_vector = embed_text(model, text)
    elif image_path:
        preprocessed_img = load_and_preprocess_image(image_path, preprocess)
        query_vector = embed_image(model, preprocessed_img)
    else:
        raise ValueError("Either text or image_path must be provided")
        
    # Perform similarity search
    results = search_qdrant(query_vector, top_k=top_k)
    return results

def main():
    parser = argparse.ArgumentParser(description="Search images by text or image query.")
    parser.add_argument("--text", type=str, help="Text query for CLIP-based search.")
    parser.add_argument("--image", type=str, help="Image path for CLIP-based search.")
    parser.add_argument("--top", type=int, default=5, help="Number of top results to retrieve.")
    args = parser.parse_args()

    # Validate input
    if not args.text and not args.image:
        print("Error: You must provide either --text or --image.")
        sys.exit(1)

    # Process query
    try:
        if args.text:
            print(f"Generating embedding for text query: '{args.text}'")
            results = process_query(text=args.text, top_k=args.top)
        else:
            if not os.path.exists(args.image):
                print(f"Error: Image file '{args.image}' does not exist.")
                sys.exit(1)
            print(f"Generating embedding for image query: '{args.image}'")
            results = process_query(image_path=args.image, top_k=args.top)
            
        # Display results
        print(f"\nTop {args.top} results:")
        for i, point in enumerate(results):
            file_path = point.payload.get("file_path", "Unknown")
            print(f"{i+1}. ID: {point.id}, Score: {point.score:.4f}, Path: {file_path}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
