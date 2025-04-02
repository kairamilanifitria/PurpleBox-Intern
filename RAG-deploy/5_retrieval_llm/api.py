from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import query_supabase, call_openai_llm
import uvicorn

app = FastAPI()

class QueryRequest(BaseModel):
    user_query: str
    chat_history: list = []

@app.get("/")
def root():
    return {"message": "Document Retrieval and LLM API is running."}

@app.post("/query")
def query_documents(request: QueryRequest):
    try:
        retrieved_chunks = query_supabase(request.user_query)
        return {"retrieved_chunks": retrieved_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_with_llm(request: QueryRequest):
    try:
        retrieved_chunks = query_supabase(request.user_query)
        answer, chat_history = call_openai_llm(request.user_query, retrieved_chunks, request.chat_history)
        return {"answer": answer, "chat_history": chat_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
