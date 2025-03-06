#!/usr/bin/env python3
"""
batch_process.py

Use:
    python batch_process.py /path/to/images

This script processes all images (and subfolders) starting from the provided
folder path. For each image:
  1. Extracts metadata using PyExifTool.
  2. Loads and preprocesses the image for CLIP.
  3. Generates a 512-dimensional embedding.
  4. Upserts the embedding + metadata into the local Qdrant database.

Dependencies:
  - QdrantClient
  - CLIP
  - PyTorch
  - Pillow
  - pyexiftool
  - rawpy
"""

import sys
import os
import clip
import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from exiftool import ExifTool
from exiftool import ExifToolHelper



COLLECTION_NAME = "image_collection"
VECTOR_SIZE = 512

def load_clip_model():
    """Load the CLIP model (ViT-B/32) and return (model, preprocess)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model.to(device), preprocess

def get_image_paths(root_folder):
    """
    Recursively collect all image file paths from the specified folder.
    Returns a list of absolute paths.
    """
    supported_extensions = (".jpg", ".jpeg", ".png", ".dng", ".raw", 
                            ".cr2", ".nef", ".arw", ".orf", ".raf")
    image_paths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            # Check file extension in a case-insensitive manner
            if file.lower().endswith(supported_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


def extract_metadata(image_path):
    with ExifToolHelper() as et:
        metadata = et.get_metadata([image_path])  # Expects a list
    return metadata

def extract_metadata(image_path):
    with ExifToolHelper() as et:
        metadata = et.get_metadata([image_path])  # Expects a list
    return metadata
etadata using PyExifTool.
def extract_metadata(image_path):
    with ExifToolHelper() as et:
        metadata = et.get_metadata([image_path])  # Expects a list
    return metadata
onary of metadata.
def extract_metadata(image_path):
    with ExifToolHelper() as et:
        metadata = et.get_metadata([image_path])  # Expects a list
    return metadata

def extract_metadata(image_path):
    with ExifToolHelper() as et:
        metadata = et.get_metadata([image_path])  # Expects a list
    return metadata

def extract_metadata(image_path):
    with ExifToolHelper() as et:
        metadata = et.get_metadata([image_path])  # Expects a list
    return metadata
lper() as et:
def extract_metadata(image_path):
    with ExifToolHelper() as et:
        metadata = et.get_metadata([image_path])  # Expects a list
    return metadata
et_metadata(file_path)
def extract_metadata(image_path):
    with ExifToolHelper() as et:
        metadata = et.get_metadata([image_path])  # Expects a list
    return metadata
o a simplified dictionary
def extract_metadata(image_path):
    with ExifToolHelper() as et:
        metadata = et.get_metadata([image_path])  # Expects a list
    return metadata
{k: v for k, v in tags.items()}
def extract_metadata(image_path):
    with ExifToolHelper() as et:
        metadata = et.get_metadata([image_path])  # Expects a list
    return metadata


def load_and_preprocess_image(image_path, preprocess):
    """
    Load and preprocess the image for CLIP.
    For RAW files, use rawpy to convert to RGB.
    Returns a PIL Image.
    """
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".dng", ".raw", ".cr2", ".nef", ".arw", ".orf", ".raf"]:
        # Use rawpy to open RAW files
        with rawpy.imread(image_path) as raw:
            rgb = raw.postprocess()
        pil_image = Image.fromarray(rgb)
    else:
        # Open as normal image
        pil_image = Image.open(image_path).convert("RGB")

    return preprocess(pil_image)

def compute_clip_embedding(model, preprocessed_image):
    """
    Compute the embedding for the given preprocessed image using CLIP.
    Returns a list of floats (embedding).
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        # CLIP expects a batch of images
        image_input = preprocessed_image.unsqueeze(0).to(device)
        # Encode image to get embeddings
        image_features = model.encode_image(image_input)
        # Normalize the embeddings
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    # Convert to Python list
    return image_features[0].cpu().numpy().tolist()

def process_images(folder_path):
    """
    Main function to process images in the given folder and upsert to Qdrant.
    """
    # Initialize Qdrant client
    client = QdrantClient(path="database/vector_db")

    # Load CLIP model
    model, preprocess = load_clip_model()

    # Get all images
    image_paths = get_image_paths(folder_path)

    # Batch upserts
    points = []
    batch_size = 16
    counter = 0

    for image_path in image_paths:
        # Extract metadata
        metadata = extract_metadata(image_path)

        # Preprocess and embed
        preprocessed = load_and_preprocess_image(image_path, preprocess)
        embedding = compute_clip_embedding(model, preprocessed)

        # Use the image path as the unique ID (or any unique scheme you prefer)
        point_id = hash(image_path)

        # Prepare point for Qdrant
        points.append(
            rest.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "file_path": image_path,
                    "metadata": metadata
                }
            )
        )

        # Periodically upsert in batches for efficiency
        if len(points) >= batch_size:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            points = []
        counter += 1
        print(f"Processed: {image_path}")

    # Upsert any remaining points
    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    print(f"\nDone! {counter} images processed and upserted to the Qdrant collection.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_process.py /path/to/images")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Error: The specified path '{folder_path}' is not a directory.")
        sys.exit(1)

    process_images(folder_path)

if __name__ == "__main__":
    main()
