# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from cag.rag_cag_pipeline import cag_rag_pipeline
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(title="RAG API")

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_api(req: QueryRequest):
    return cag_rag_pipeline(req.query)

