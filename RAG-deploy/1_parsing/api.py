from fastapi import FastAPI
from pathlib import Path
import logging
from main import convert_and_save, extract_nodes  # Import functions from main.py

app = FastAPI()

INPUT_DIR = Path(r"C:\\Users\\LENOVO\\Desktop\\DOCKER-TRY\\1_parsing\\input_pdfs")
OUTPUT_DIR = Path(r"C:\\Users\\LENOVO\\Desktop\\DOCKER-TRY\\1_parsing\\output_md")

@app.post("/parsing")
async def process_existing_files():
    """Processes all files in the input folder without manual upload."""
    
    # Run the parsing process
    convert_and_save()

    # Extract nodes from the generated markdown files
    processed_files = []
    for md_file in OUTPUT_DIR.glob("*.md"):
        extract_nodes(md_file)
        processed_files.append(md_file.name)

    if processed_files:
        return {"message": "Processing complete", "processed_files": processed_files}
    else:
        return {"message": "No files found for processing"}

@app.get("/")
def home():
    return {"message": "API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
