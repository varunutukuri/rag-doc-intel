from pydantic import BaseModel
from app.services.rag_service import query_rag

class QueryRequest(BaseModel):
    question: str

from fastapi import APIRouter

router = APIRouter()



@router.post("/query")
def query_documents(request: QueryRequest):
    answer = query_rag(request.question)
    return answer