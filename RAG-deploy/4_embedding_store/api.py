from fastapi import FastAPI, UploadFile, File, HTTPException
import json
import os
import uvicorn
from main import store_chunks_in_supabase

app = FastAPI()

INPUT_FOLDER = r"C:\\Users\\LENOVO\\Desktop\\DOCKER-TRY\\4_embedding_store\\input_json"

@app.get("/")
def home():
    return {"message": "API is running!"}

@app.get("/embedding")
def process_json_files():
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".json"):
            file_path = os.path.join(INPUT_FOLDER, filename)
            with open(file_path, "r", encoding="utf-8") as json_file:
                json_chunks = json.load(json_file)
            store_chunks_in_supabase(json_chunks)
            print(f"Processed and stored: {filename}")
    return {"message": "All text and table embeddings stored successfully in Supabase!"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

