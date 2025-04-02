from fastapi import FastAPI
import os
from main import process_markdown_files

app = FastAPI()

# Define input/output directories
INPUT_FOLDER = r"C:\\Users\\LENOVO\\Desktop\\DOCKER-TRY\\2_image\\input_md"
OUTPUT_FOLDER = r"C:\\Users\\LENOVO\\Desktop\\DOCKER-TRY\\2_image\\output_md"

@app.post("/image_description")
def process_markdown():
    """
    Endpoint to process markdown files with images.
    """
    if not os.path.exists(INPUT_FOLDER):
        return {"error": "Input folder does not exist"}

    process_markdown_files(INPUT_FOLDER, OUTPUT_FOLDER)
    return {"message": "Processing completed", "output_folder": OUTPUT_FOLDER}

@app.get("/")
def home():
    return {"message": "API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)