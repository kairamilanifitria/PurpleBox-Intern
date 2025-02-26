from langchain.text_splitter import MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer
import torch
import re

# Load Markdown file
with open("/content/drive/MyDrive/document_rag/md/17.md", "r") as file:
    markdown_text = file.read()

# Step 1: Document-Specific Chunking
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2")
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
documents = splitter.split_text(markdown_text)
chunks = [doc.page_content for doc in documents]

# Load Hugging Face Embedding Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def is_table(chunk):
    """Check if a chunk contains a Markdown table."""
    return "|" in chunk and "---" in chunk

def get_table_header(table_text):
    """Extract the header row of a Markdown table."""
    lines = table_text.strip().split("\n")
    header, separator = None, None
    for i in range(len(lines) - 1):
        if "---" in lines[i]:  # Identify separator line
            header = lines[i - 1]  # Header is right above separator
            separator = lines[i]  # Capture the separator
            break
    return header, separator

def split_long_table(table_text, max_tokens=300):
    """Splits long Markdown tables while keeping the first chunk unchanged and repeating headers in later chunks."""
    lines = table_text.strip().split("\n")

    # Identify table header and separator
    header, separator = None, None
    for i, line in enumerate(lines):
        if "---" in line and "|" in line:  # Detect table separator
            header = lines[i - 1]  # The row before separator is the header
            separator = line
            break

    if not header or not separator:
        return [table_text]  # Return unchanged if it's not a valid table

    split_tables = []
    temp_chunk = []
    first_chunk = True  # Track the first occurrence of the table

    for line in lines:
        if first_chunk:
            temp_chunk.append(line)
            if len(" ".join(temp_chunk).split()) > max_tokens:
                split_tables.append("\n".join(temp_chunk))
                temp_chunk = []
                first_chunk = False  # Mark first chunk as processed
            continue

        # Add table header only in subsequent chunks
        if not temp_chunk:
            temp_chunk.append(header)
            temp_chunk.append(separator)

        temp_chunk.append(line)

        # If exceeding token limit, store the chunk
        if len(" ".join(temp_chunk).split()) > max_tokens:
            split_tables.append("\n".join(temp_chunk))
            temp_chunk = []  # Start a new chunk

    # Append remaining chunk
    if temp_chunk:
        split_tables.append("\n".join(temp_chunk))

    return split_tables


def needs_semantic_chunking(chunk, max_tokens=300):
    """Check if a non-table chunk needs further splitting."""
    return not is_table(chunk) and len(chunk.split()) > max_tokens

def semantic_split(text, max_sentences=5, similarity_threshold=0.6):
    """Splits long text semantically using sentence embeddings."""
    sentences = re.split(r'(?<=[.!?])\s+', text)

    if len(sentences) <= max_sentences:
        return [text]

    embeddings = model.encode(sentences, convert_to_tensor=True)
    similarities = torch.nn.functional.cosine_similarity(embeddings[:-1], embeddings[1:], dim=1)

    split_points = [i+1 for i, sim in enumerate(similarities) if sim < similarity_threshold]

    sub_chunks, start = [], 0
    for split in split_points:
        chunk_text = " ".join(sentences[start:split])
        if len(chunk_text.split()) >= 10:
            sub_chunks.append(chunk_text)
        start = split

    sub_chunks.append(" ".join(sentences[start:]))
    return [c for c in sub_chunks if len(c.split()) >= 10]

def is_references_section(chunk):
    """Check if a chunk is part of the References section."""
    return chunk.strip().lower().startswith("## references")


# Step 3: Apply Chunking
# Step 3: Apply Chunking
final_chunks = []
is_references = False  # Track if we are inside the references section

for chunk in chunks:
    if is_references_section(chunk):
        is_references = True  # Mark that we are in the References section

    if is_references:
        final_chunks.append(chunk)  # Keep References intact
    elif is_table(chunk):
        final_chunks.extend(split_long_table(chunk))
    elif needs_semantic_chunking(chunk):
        final_chunks.extend(semantic_split(chunk))
    else:
        final_chunks.append(chunk)

# Step 4: Ensure Image Descriptions Stick Together (Sliding Window)
merged_chunks = []
i = 0
while i < len(final_chunks):
    chunk = final_chunks[i]

    if "![Image](" in chunk and i + 1 < len(final_chunks):
        next_chunk = final_chunks[i + 1]
        if len(next_chunk.split()) < 50:  # Merge small description chunks
            chunk += "\n" + next_chunk
            i += 1  # Skip next chunk as it's merged

    merged_chunks.append(chunk)
    i += 1

# Print final chunks
for i, chunk in enumerate(merged_chunks):
    print(f"Chunk {i+1} (Tokens: {len(chunk.split())}):\n{chunk}\n---\n")
