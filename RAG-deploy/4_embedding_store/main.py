import os
import json
import torch
import uuid
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client
from transformers import AutoTokenizer, AutoModel

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load Embedding Model
tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
model = AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def get_embedding(text):
    """Generates an embedding vector from input text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()

def generate_table_description(table_data):
    """Generates a natural language description from a table's headers and rows."""
    headers = table_data["headers"]
    rows = table_data["rows"]
    description = [", ".join([f"{headers[i]}: {row[i]}" for i in range(len(headers))]) for row in rows]
    return " | ".join(description)

def convert_table_to_text(table_data, metadata):
    """Converts a table into a structured text format."""
    headers = ", ".join(table_data["headers"])
    rows = [" | ".join(row) for row in table_data["rows"]]
    table_title = metadata.get("table_title", "Unknown Table")
    section = metadata.get("section", "Unknown Section")
    table_description = generate_table_description(table_data)
    return (
        f"Table Title: {table_title}. Section: {section}.\n"
        f"Table Data:\nHeaders: {headers}\n" + "\n".join(rows) +
        f"\nDescription: {table_description}"
    ), table_description

def store_chunks_in_supabase(chunks):
    """Stores text and table chunks into Supabase."""
    document_entries, table_entries = [], []
    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        if "content" in chunk and chunk["content"]:
            embedding = get_embedding(chunk["content"])
            document_entries.append({
                "chunk_id": chunk_id, "content": chunk["content"],
                "embedding": embedding, "metadata": chunk["metadata"], "type": "text"
            })
        if "table" in chunk and chunk["table"]:
            table_text, table_description = convert_table_to_text(chunk["table"], chunk.get("metadata", {}))
            table_embedding = get_embedding(table_text)
            table_entries.append({
                "chunk_id": chunk_id, "table_data": json.dumps(chunk["table"], ensure_ascii=False),
                "description": table_description, "embedding": table_embedding, "metadata": chunk.get("metadata", {})
            })
    if document_entries:
        supabase.table("documents").insert(document_entries).execute()
    if table_entries:
        supabase.table("tables").insert(table_entries).execute()

if __name__ == "__main__":
    input_folder = r"C:\\Users\\LENOVO\\Desktop\\DOCKER-TRY\\4_embedding_store\\input_json"
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, "r", encoding="utf-8") as json_file:
                json_chunks = json.load(json_file)
            store_chunks_in_supabase(json_chunks)
            print(f"Processed and stored: {filename}")
    print("All text and table embeddings stored successfully in Supabase!")