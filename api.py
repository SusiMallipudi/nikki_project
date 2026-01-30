"""
FastAPI: query parameters api_key (optional), ticket (required).
Uses query_data.py for ChromaDB + Gemini.
"""
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

from query_data import query_all_splits, answer_with_gemini, TOP_K_PER_CHUNK

load_dotenv()

app = FastAPI(title="Ticket RAG API", version="1.0.0")


@app.get("/query")
def query_tickets(
    ticket: str = Query(..., min_length=1, description="Ticket question"),
    api_key: Optional[str] = Query(None, description="Gemini API key (optional; uses .env if not set)"),
):
    """
    Query ticket knowledge base and get Gemini answer.
    Query params: ticket (required), api_key (optional).
    """
    ticket = ticket.strip()
    if not ticket:
        raise HTTPException(status_code=400, detail="ticket must be non-empty")

    chunks_raw = query_all_splits(ticket, n_per_chunk=TOP_K_PER_CHUNK, max_total=None)
    if not chunks_raw:
        raise HTTPException(status_code=503, detail="No results. Run index_data.py first (data/db/).")

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    answer = None
    if key:
        try:
            answer = answer_with_gemini(ticket, chunks_raw, api_key=key)
        except Exception as e:
            err = str(e)
            status = 400 if "API key" in err and ("invalid" in err.lower() or "INVALID" in err) else 502
            raise HTTPException(status_code=status, detail=f"Gemini: {err}")

    chunks_out = [
        {"chunk_id": r["chunk_id"], "distance": r.get("distance"), "metadata": r.get("metadata") or {}, "document": (r.get("document") or "")[:500]}
        for r in chunks_raw
    ]
    return {"ticket": ticket, "chunks": chunks_out, "answer": answer}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
