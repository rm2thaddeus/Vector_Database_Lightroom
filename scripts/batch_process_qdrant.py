#!/usr/bin/env python3
"""
batch_process_qdrant.py

Use:
    python batch_process_qdrant.py --image_dir /path/to/images

This script:
  1. Scans a directory for image/RAW files.
  2. Extracts metadata using pyexiftool.
  3. Generates CLIP embeddings for each file.
  4. Stores the metadata + embeddings in a local Qdrant collection.

Pre-requisites:
  - Run 'python install_dependencies.py' to install required Python packages.
  - Qdrant server must be running locally on port 6333.
  - ExifTool must be installed on the system (for pyexiftool).
"""

import os
import argparse
import exiftool
import torch
import clip
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    CollectionInfo,
    PointStruct
)

def extract_metadata(files):
    """
    Extract metadata from image files using pyexiftool.

    :param files: List of image file paths.
    :return: List of metadata dictionaries (in the same order as files).
    """
    metadata_list = []
    with exiftool.ExifTool() as et:
        # Extract all metadata for all files at once
        raw_metadata = et.get_metadata_batch(files)

    for meta in raw_metadata:
        metadata_list.append(meta)

    return metadata_list

def generate_clip_embedding(model, preprocess, file_path, device):
    """
    Generate a CLIP embedding for the given image file.

    :param model: The loaded CLIP model.
    :param preprocess: The CLIP preprocessing transform.
    :param file_path: Path to the image file.
    :param device: 'cpu' or 'cuda'.
    :return: A list (vector) representing the image embedding.
    """
    image = Image.open(file_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
    # Normalize the embedding and convert to CPU numpy array
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding[0].cpu().numpy().tolist()

def main(image_dir, collection_name="image_metadata"):
    """
    Main function:
      - Connects to Qdrant (localhost:6333).
      - Creates collection if not exists.
      - Walks through the directory to find images.
      - Extracts metadata and embeddings.
      - Inserts data into Qdrant.
    """
    # 1. Connect to Qdrant
    qdrant_client = QdrantClient(host="localhost", port=6333)
    
    # 2. Check if collection exists, else create it
    existing_collections = [col.name for col in qdrant_client.get_collections().collections]
    if collection_name not in existing_collections:
        print(f"Collection '{collection_name}' not found. Creating new collection...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )
    else:
        print(f"Collection '{collection_name}' already exists.")

    # 3. Prepare CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 4. Gather all image files from the directory
    valid_extensions = (".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif", ".dng", ".nef", ".cr2", ".arw")
    all_files = []
    for root, dirs, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(valid_extensions):
                full_path = os.path.join(root, f)
                all_files.append(full_path)

    if not all_files:
        print(f"No image/RAW files found in directory: {image_dir}")
        return

    print(f"Found {len(all_files)} files to process...")

    # 5. Extract metadata in one batch call
    print("Extracting metadata...")
    metadata_list = extract_metadata(all_files)

    # 6. For each file, generate an embedding and create a Qdrant PointStruct
    points_to_upsert = []
    for idx, file_path in enumerate(all_files):
        file_metadata = metadata_list[idx]

        # Clean or transform metadata if needed; here we just keep it as-is
        # Convert it to a minimal dict if desired; for example:
        # minimal_meta = {
        #   "SourceFile": file_metadata.get("SourceFile", ""),
        #   "EXIF:Model": file_metadata.get("EXIF:Model", ""),
        #   ...
        # }

        # Generate CLIP embedding
        try:
            embedding_vector = generate_clip_embedding(model, preprocess, file_path, device)
        except Exception as e:
            print(f"Error generating embedding for {file_path}: {e}")
            continue

        # Qdrant requires a unique ID. We'll just use an incremental index
        point_id = idx + 1  # or you could use a hash of the file name

        # Construct the Qdrant payload (metadata)
        payload = {
            "file_name": os.path.basename(file_path),
            "metadata": file_metadata  # Keep entire metadata dict
        }

        # Create the point structure with vector and payload
        point = PointStruct(
            id=point_id,
            vector=embedding_vector,  # CLIP embedding
            payload=payload
        )
        points_to_upsert.append(point)

    # 7. Upsert (insert/update) all points into Qdrant at once
    print(f"Upserting {len(points_to_upsert)} records into Qdrant...")
    qdrant_client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points_to_upsert
    )
    print("Upsert completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process images, extract metadata, generate CLIP embeddings, store in Qdrant."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to the directory containing images/RAW files."
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="image_metadata",
        help="Name of the Qdrant collection to use (will be created if it does not exist)."
    )
    args = parser.parse_args()

    main(args.image_dir, args.collection_name)
