# Vector Database Lightroom Streamlit App

This Streamlit app provides a user-friendly interface for searching and exploring the vector database of images.

## Features

- Search by text description (e.g., "sunset over mountains", "people at the beach")
- Search by uploading an image (similarity search)
- View search results with thumbnails in a grid layout
- View detailed image information and metadata
- Adjust number of search results

## Prerequisites

- Python 3.8+
- Qdrant server running on localhost:6333
- CLIP model dependencies (PyTorch, etc.)

## How to Run

1. Make sure your Qdrant server is running:
   ```
   # If using Docker
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. Launch the Streamlit app:
   ```
   cd /path/to/Vector_Database_Lightroom
   streamlit run app/app.py
   ```

3. Open your browser and navigate to http://localhost:8501

## Usage Tips

- For best text search results, be descriptive and specific
- The app supports both regular image formats (JPG, PNG) and RAW formats (DNG, CR2, NEF, etc.)
- Use the slider in the sidebar to adjust how many results you want to see
- Click "View Details" on any search result to see full metadata information

## Troubleshooting

- If images don't load, ensure the file paths in the database are correct and accessible
- If search results are unexpected, try modifying your query or uploading a different image
- Check console output for any error messages if the app doesn't work as expected 