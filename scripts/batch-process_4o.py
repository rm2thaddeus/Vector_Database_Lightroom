import os
import sys
import json
from exiftool import ExifToolHelper

def process_images(image_folder, output_file="metadata.json"):
    """Extract metadata from images, modify metadata if needed, and save it to a JSON file."""
    if not os.path.exists(image_folder):
        print(f"Error: The folder '{image_folder}' does not exist.")
        sys.exit(1)

    # Get all image files (adjust extensions as needed)
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".dng"))]
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return

    print(f"Processing {len(image_files)} images...")
    
    with ExifToolHelper() as et:
        metadata = et.get_metadata(image_files)
    
    # Example: Modify metadata if needed
    for entry in metadata:
        if "EXIF:Artist" not in entry:
            entry["EXIF:Artist"] = "Unknown Artist"  # Add a default value
    
    # Save metadata to a JSON file
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(metadata, json_file, indent=4)
    
    print(f"Metadata saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        folder_path = os.path.join(os.getcwd(), "images")  # Default to "images/" if no argument is given
        print("No folder specified. Using default: images/")
    else:
        folder_path = sys.argv[1]

    process_images(folder_path)
