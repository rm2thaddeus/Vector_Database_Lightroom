#!/usr/bin/env python3
"""
install_dependencies.py

Use:
    python install_dependencies.py

This script installs all the required Python dependencies for the project.
Ensure that you have a working internet connection and that pip is available.
"""

import subprocess
import sys

def install(package):
    """Install a package via pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    dependencies = [
        # Qdrant client for managing a local vector database.
        "qdrant-client==1.2.0",
        # PyTorch (required by CLIP).
        "torch",
        # OpenAI's CLIP (install directly from GitHub).
        "git+https://github.com/openai/CLIP.git",
        # Pillow for image manipulation.
        "Pillow",
        # PyExifTool (wrapper around the system-installed ExifTool).
        "pyexiftool",
        # rawpy for reading RAW files (e.g., .CR2, .NEF).
        "rawpy"
    ]

    print("Installing required dependencies...")
    for dep in dependencies:
        print(f"Installing {dep}...")
        install(dep)

    print("\nAll dependencies have been installed successfully!")
