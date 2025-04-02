from fastapi import FastAPI
import os
import json
import main  # Import main.py without modifying it

app = FastAPI()

INPUT_FOLDER = r"C:\\Users\\LENOVO\\Desktop\\DOCKER-TRY\\3_chunking\\input_md"
OUTPUT_FOLDER = r"C:\\Users\\LENOVO\\Desktop\\DOCKER-TRY\\3_chunking\\output_json"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.get("/chunking")
def process_markdown_files():
    """Process all markdown files in input_md and return success message."""
    for file_name in os.listdir(INPUT_FOLDER):
        if file_name.endswith(".md"):
            input_path = os.path.join(INPUT_FOLDER, file_name)
            output_path = os.path.join(OUTPUT_FOLDER, file_name.replace(".md", ".json"))
            main.process_markdown(input_path, output_path)
    
    return {"message": "Processing completed. JSON files saved in output_json"}

@app.get("/")
def home():
    return {"message": "API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
