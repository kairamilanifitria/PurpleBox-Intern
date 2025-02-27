import json
import re
import torch
from langchain.text_splitter import MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer

# Load Markdown file
file_path = "/content/drive/MyDrive/document_rag/md/17.md"
with open(file_path, "r", encoding="utf-8") as file:
    markdown_text = file.read()

# Step 1: Document-Specific Chunking
headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
documents = splitter.split_text(markdown_text)
chunks = [doc.page_content for doc in documents]

# Load Hugging Face Embedding Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def is_table(chunk):
    """Checks if a chunk contains a Markdown table."""
    return bool(re.search(r'^\|.*\|\n\|[-| ]+\|\n(\|.*\|\n)*', chunk, re.MULTILINE))

def extract_table(chunk):
    """Extracts tables from markdown and converts them into structured JSON format."""
    lines = chunk.strip().split("\n")
    
    # Find the table header
    header = None
    table_rows = []
    for i, line in enumerate(lines):
        if re.match(r'^\|[-| ]+\|$', line):  # Detect separator line (---|---)
            header = lines[i - 1].strip("|").split("|")
            header = [h.strip() for h in header]
            continue
        if header:
            row_data = line.strip("|").split("|")
            row_data = [cell.strip() for cell in row_data]
            table_rows.append(row_data)
    
    if not header or not table_rows:
        return None  # Return None if the table extraction fails
    
    return {"headers": header, "rows": table_rows}

def needs_semantic_chunking(chunk, max_tokens=300):
    """Checks if the text chunk is too long and needs further splitting."""
    return not is_table(chunk) and len(chunk.split()) > max_tokens

def semantic_split(text, max_sentences=5, similarity_threshold=0.6, min_tokens=100):
    """Splits long text chunks based on semantic similarity."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= max_sentences:
        return [text]
    
    embeddings = model.encode(sentences, convert_to_tensor=True)
    similarities = torch.nn.functional.cosine_similarity(embeddings[:-1], embeddings[1:], dim=1)
    split_points = [i+1 for i, sim in enumerate(similarities) if sim < similarity_threshold]
    
    sub_chunks, start = [], 0
    for split in split_points:
        chunk_text = " ".join(sentences[start:split])
        if len(chunk_text.split()) >= min_tokens:
            sub_chunks.append(chunk_text)
        start = split
    
    last_chunk = " ".join(sentences[start:])
    if len(last_chunk.split()) >= min_tokens:
        sub_chunks.append(last_chunk)
    elif sub_chunks:
        sub_chunks[-1] += " " + last_chunk  # Merge with previous if too short
    
    return sub_chunks if sub_chunks else [text]

def is_references_section(chunk):
    """Checks if the chunk is part of the References section."""
    return chunk.strip().lower().startswith("## references")

def extract_section_title(chunk):
    """Extracts section headers from chunks for metadata."""
    match = re.match(r'^(#+)\s+(.*)', chunk.strip())
    return match.group(2) if match else None

# Step 3: Apply Chunking
final_chunks = []
is_references = False
for chunk in chunks:
    if is_references_section(chunk):
        is_references = True
    if is_references:
        final_chunks.append(chunk)
    elif is_table(chunk):
        table_data = extract_table(chunk)
        if table_data:
            final_chunks.append({"table": table_data})  # Store table separately
    elif needs_semantic_chunking(chunk):
        final_chunks.extend(semantic_split(chunk))
    else:
        final_chunks.append(chunk)

# Step 4: Merge Small Chunks (Ensure Minimum 100 Tokens)
merged_chunks = []
i = 0
while i < len(final_chunks):
    chunk = final_chunks[i]
    if isinstance(chunk, dict):  # If it's a table, store it separately
        merged_chunks.append(chunk)
        i += 1
        continue
    
    while i + 1 < len(final_chunks) and isinstance(chunk, str) and len(chunk.split()) < 100:
        next_chunk = final_chunks[i + 1]
        if isinstance(next_chunk, dict):  # Don't merge tables into text
            break
        chunk += "\n" + next_chunk
        i += 1
    
    merged_chunks.append(chunk)
    i += 1

# Step 5: Convert Chunks to JSON Format
json_chunks = []
source_filename = file_path.split("/")[-1]  # Extract filename for metadata

for idx, chunk in enumerate(merged_chunks):
    if isinstance(chunk, dict):  # Handle table separately
        json_chunks.append({
            "chunk_id": idx + 1,
            "table": chunk["table"],
            "metadata": {
                "source": source_filename,
                "section": "Table",  # You can add better logic here
                "position": idx + 1
            }
        })
    else:
        section_title = extract_section_title(chunk)
        json_chunks.append({
            "chunk_id": idx + 1,
            "content": chunk.strip(),
            "metadata": {
                "source": source_filename,
                "section": section_title if section_title else "Unknown",
                "position": idx + 1
            }
        })

# Save JSON output
output_file = "/content/17_chunks_v2.json"
with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(json_chunks, json_file, indent=4, ensure_ascii=False)

print(f"Chunking completed. JSON saved to: {output_file}")
