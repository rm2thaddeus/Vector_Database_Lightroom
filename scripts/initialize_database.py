#!/usr/bin/env python3
"""
initialize_database.py

Use:
    python initialize_database.py

This script creates or verifies a Qdrant collection to store image embeddings.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Define the collection name and vector dimensionality.
COLLECTION_NAME = "image_collection"
VECTOR_SIZE = 512  # CLIP ViT-B/32 generates 512-dimensional embeddings

def main():
    # Initialize the Qdrant client with a local (file-based) storage path.
    client = QdrantClient(path="database/vector_db")

    # Create the collection if it does not exist.
    # If it already exists, this call will ensure the schema matches.
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

    print(f"Collection '{COLLECTION_NAME}' has been initialized/recreated successfully.")

if __name__ == "__main__":
    main()
