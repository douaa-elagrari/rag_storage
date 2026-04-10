from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import rag, load_data
import os

app = FastAPI(
    title="HR Assistant API",
    description="RAG-based HR & Legal Assistant (Arabic / French / English)",
    version="1.0.0"
)

# ── CORS: allow your friend's frontend to call this API ──────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load embeddings  ──────────────────────
@app.on_event("startup")
async def startup_event():
    load_data()

# ── Request / Response schemas ───────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

# ── Endpoints ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "HR Assistant API is running 🚀"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ask", response_model=AnswerResponse)
def ask(body: QuestionRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        answer = rag(body.question, verbose=False)
        return AnswerResponse(question=body.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
