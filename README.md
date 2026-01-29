# Ticket RAG (ChromaDB + Gemini)

RAG over ticket CSV data: index into ChromaDB, query in parallel, answer with Gemini. FastAPI for `/query`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add GEMINI_API_KEY
```

## Usage

1. **Index once** (reads CSV splits, builds ChromaDB):  
   `python3 index_data.py`

2. **Query** (no re-index):  
   `python3 query_data.py "How do I fix VPN issues?"`

3. **API**:  
   `uvicorn api:app --reload --host 0.0.0.0 --port 8000`  
   `POST /query` with body `{"ticket": "your question"}` (optional: `gemini_api_key`)

## Data

- English tickets: `aa_dataset-tickets-en-only.csv` (from `extract_en.py`)
- Splits: `aa_dataset-tickets-en-only-part-1-of-6.csv` … `part-6-of-6.csv` (from `split_en.py`)
