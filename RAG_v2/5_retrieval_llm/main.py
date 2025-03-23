import os
import json
import torch
import uuid
import numpy as np
from supabase import create_client, Client
from transformers import AutoTokenizer, AutoModel
import ast
import re
from dotenv import load_dotenv
import openai
from scipy.spatial.distance import cosine
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('all')
nltk.download('punkt')
nltk.download('stopwords')

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Initialize Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load Embedding Model
tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
model = AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True).to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def get_embedding(text):
    """Generates an embedding vector from input text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()

def extract_keywords_simple(text):
    """Extracts important words from a query using simple filtering."""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    keywords = [word for word in words if word.isalnum() and word not in stop_words]
    return keywords

def query_requires_table(user_query):
    """Determines if the query is likely asking for table data."""
    table_keywords = {"table", "data", "values", "measurements", "limits", "thresholds", "parameters", "average", "sum", "percentage"}
    return any(word in user_query.lower() for word in table_keywords)

def get_most_similar_keywords(query_keywords, top_text_chunks):
    """Extracts most relevant words from top retrieved text chunks."""
    all_text_words = set()
    for chunk in top_text_chunks:
        chunk_words = set(word_tokenize(chunk[2].lower()))
        all_text_words.update(chunk_words)
    common_words = [word for word in query_keywords if word in all_text_words]
    return common_words if common_words else query_keywords

def query_supabase(user_query):
    """Retrieves both text and table chunks based on query, ensuring relevance balance."""
    query_embedding = np.array(get_embedding(user_query), dtype=np.float32).flatten()
    keywords = extract_keywords_simple(user_query)
    requires_table = query_requires_table(user_query)
    
    response_text = supabase.table("documents").select("chunk_id, content, embedding, type, metadata").execute()
    text_results = []
    for record in response_text.data:
        chunk_embedding = ast.literal_eval(record["embedding"]) if isinstance(record["embedding"], str) else record["embedding"]
        chunk_embedding = np.array(chunk_embedding, dtype=np.float32).flatten()
        if chunk_embedding.shape == query_embedding.shape:
            similarity = 1 - cosine(query_embedding, chunk_embedding)
            text_results.append((record["chunk_id"], "text", record["content"], similarity))
    
    text_results.sort(key=lambda x: x[3], reverse=True)
    top_text_chunks = text_results[:3]
    
    refined_keywords = get_most_similar_keywords(keywords, top_text_chunks)
    
    response_tables = supabase.table("tables").select("chunk_id, table_data, description, embedding, metadata").execute()
    table_results = []
    table_weight = 2.5 if requires_table else 1.5
    for record in response_tables.data:
        table_embedding = ast.literal_eval(record["embedding"]) if isinstance(record["embedding"], str) else record["embedding"]
        table_embedding = np.array(table_embedding, dtype=np.float32).flatten()
        table_data = record["table_data"].lower()
        table_description = record["description"].lower()
        keyword_match_score = sum(3 if word in table_data.split(" ")[:5] else 1 for word in refined_keywords if word in table_data or word in table_description)
        if table_embedding.shape == query_embedding.shape:
            embedding_similarity = 1 - cosine(query_embedding, table_embedding)
            keyword_embedding_score = sum(1 - cosine(get_embedding(word), table_embedding) for word in refined_keywords) / max(len(refined_keywords), 1)
            final_table_score = (embedding_similarity ** 0.8) * 0.2 + (keyword_match_score ** 2.5) * 0.6 + (keyword_embedding_score ** 1.2) * 0.2
            if final_table_score > 0:
                table_results.append((record["chunk_id"], "table", record["description"], final_table_score))
    
    table_results.sort(key=lambda x: x[3], reverse=True)
    
    if table_results and table_results[0][3] > 0.75:
        final_results = [table_results[0]] + text_results[:2] + table_results[1:2] + text_results[2:]
    else:
        final_results = text_results[:3] + table_results[:2]

    print("\n[DEBUG] Retrieved Chunks:")
    for chunk in final_results:
        print(f"- Type: {chunk[1]}, Content: {chunk[2][:200]}...")  # Print only first 200 chars

    return final_results[:5]


def call_openai_llm(user_query, retrieved_chunks, chat_history=[]):
    """Send the query along with retrieved context and chat history to OpenAI API."""
    context_text = "\n\n".join([f"Chunk {i+1}: {chunk[2]}" for i, chunk in enumerate(retrieved_chunks)])

    print("\n[DEBUG] Context sent to LLM:")
    print(context_text[:500])  # Print only first 500 chars to verify it's being sent

    messages = [
        {"role": "system", "content": "You are an intelligent assistant. Use the following retrieved information to answer the user's query."},
    ]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": f"Context:\n{context_text}\n\nUser's Question: {user_query}"})

    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0.7
    )

    answer = response.choices[0].message.content  
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": answer})

    return answer, chat_history


def chat():
    """Handles continuous chat interaction."""
    chat_history = []
    while True:
        user_query = input("User: ")
        if user_query.lower() in ["exit", "quit", "new chat"]:
            print("Chat ended.")
            break
        retrieved_chunks = query_supabase(user_query)
        answer, chat_history = call_openai_llm(user_query, retrieved_chunks, chat_history)
        print(f"Assistant: {answer}\n")

if __name__ == "__main__":
    chat()
