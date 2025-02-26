from langchain.text_splitter import MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer
import torch
import re

# Load Markdown file
with open("/content/drive/MyDrive/document_rag/md/17.md", "r") as file:
    markdown_text = file.read()

# Step 1: Document-Specific Chunking
headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
documents = splitter.split_text(markdown_text)
chunks = [doc.page_content for doc in documents]

# Load Hugging Face Embedding Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def is_table(chunk):
    return "|" in chunk and "---" in chunk

def split_long_table(table_text, max_tokens=300):
    lines = table_text.strip().split("\n")
    header, separator = None, None
    for i, line in enumerate(lines):
        if "---" in line and "|" in line:
            header, separator = lines[i - 1], line
            break
    if not header or not separator:
        return [table_text]
    
    split_tables, temp_chunk, first_chunk = [], [], True
    for line in lines:
        if first_chunk:
            temp_chunk.append(line)
            if len(" ".join(temp_chunk).split()) > max_tokens:
                split_tables.append("\n".join(temp_chunk))
                temp_chunk, first_chunk = [], False
            continue
        
        if not temp_chunk:
            temp_chunk.extend([header, separator])
        temp_chunk.append(line)
        
        if len(" ".join(temp_chunk).split()) > max_tokens:
            split_tables.append("\n".join(temp_chunk))
            temp_chunk = []
    
    if temp_chunk:
        split_tables.append("\n".join(temp_chunk))
    
    return split_tables

def needs_semantic_chunking(chunk, max_tokens=300):
    return not is_table(chunk) and len(chunk.split()) > max_tokens

def semantic_split(text, max_sentences=5, similarity_threshold=0.6, min_tokens=100):
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
    return chunk.strip().lower().startswith("## references")

# Step 3: Apply Chunking
final_chunks = []
is_references = False
for chunk in chunks:
    if is_references_section(chunk):
        is_references = True
    if is_references:
        final_chunks.append(chunk)
    elif is_table(chunk):
        final_chunks.extend(split_long_table(chunk))
    elif needs_semantic_chunking(chunk):
        final_chunks.extend(semantic_split(chunk))
    else:
        final_chunks.append(chunk)

# Step 4: Merge Small Chunks (Ensure Minimum 100 Tokens)
merged_chunks = []
i = 0
while i < len(final_chunks):
    chunk = final_chunks[i]
    while i + 1 < len(final_chunks) and len(chunk.split()) < 100:
        chunk += "\n" + final_chunks[i + 1]
        i += 1
    merged_chunks.append(chunk)
    i += 1

# Print final chunks
for i, chunk in enumerate(merged_chunks):
    print(f"Chunk {i+1} (Tokens: {len(chunk.split())}):\n{chunk}\n---\n")
