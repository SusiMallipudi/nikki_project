"""
Query ChromaDB only (no indexing). Use after index_data.py has been run once.
Queries all 6 DBs in parallel, top 3 per chunk; answers with Gemini. All LLM code here.
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DATA_DIR = "data"
DB_DIR = os.path.join(DATA_DIR, "db")
N_SPLITS = 6
DB_PREFIX = "chroma_db_part"
COLLECTION_NAME = "tickets"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_PER_CHUNK = 3
GEMINI_MODEL = "gemini-2.5-flash"

EF = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def _gemini_generate(
    prompt: str,
    *,
    model: str = GEMINI_MODEL,
    system_instruction: str | None = None,
    temperature: float = 0.3,
    api_key: str | None = None,
) -> str:
    """Call Gemini LLM and return generated text. All LLM code in this file."""
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("No Gemini API key: set GEMINI_API_KEY or pass gemini_api_key.")
    client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction,
    )
    response = client.models.generate_content(model=model, contents=prompt, config=config)
    return response.text or ""


def _query_one_chunk(args):
    """Query a single ChromaDB; returns list of results for that chunk."""
    chunk_id, query_text, n_per_chunk = args
    path = os.path.join(DB_DIR, f"{DB_PREFIX}_{chunk_id}")
    if not os.path.isdir(path):
        return []
    client = chromadb.PersistentClient(path=path)
    coll = client.get_collection(COLLECTION_NAME, embedding_function=EF)
    r = coll.query(query_texts=[query_text], n_results=n_per_chunk)
    chunk_results = []
    for j, doc in enumerate(r["documents"][0]):
        chunk_results.append({
            "document": doc,
            "metadata": r["metadatas"][0][j] if r["metadatas"] else {},
            "distance": r["distances"][0][j] if r.get("distances") else None,
            "chunk_id": chunk_id,
        })
    return chunk_results


def query_all_splits(query_text: str, n_per_chunk: int = TOP_K_PER_CHUNK, max_total: int | None = None):
    """Query all ChromaDB splits in parallel; top n_per_chunk from each, merge by distance."""
    chunk_ids = [i for i in range(1, N_SPLITS + 1) if os.path.isdir(os.path.join(DB_DIR, f"{DB_PREFIX}_{i}"))]
    if not chunk_ids:
        return []

    all_results = []
    with ThreadPoolExecutor(max_workers=len(chunk_ids)) as executor:
        futures = {
            executor.submit(_query_one_chunk, (i, query_text, n_per_chunk)): i
            for i in chunk_ids
        }
        for future in as_completed(futures):
            chunk_results = future.result()
            all_results.extend(chunk_results)

    all_results.sort(key=lambda x: x["distance"] or float("inf"))
    if max_total is not None:
        all_results = all_results[:max_total]
    return all_results


def print_chunk_results(results: list, doc_preview_len: int = 400):
    """Print each retrieved chunk: chunk_id, distance, metadata, document preview."""
    print(f"\n{'='*60}\nRetrieved {len(results)} chunks (top 3 per DB × 6 DBs)\n{'='*60}")
    for i, r in enumerate(results, 1):
        dist = r.get("distance")
        dist_str = f"{dist:.4f}" if dist is not None else "N/A"
        meta = r.get("metadata") or {}
        meta_str = ", ".join(f"{k}={v}" for k, v in list(meta.items())[:5])
        doc = (r.get("document") or "")[:doc_preview_len]
        if len(r.get("document") or "") > doc_preview_len:
            doc += "..."
        print(f"\n--- Chunk {i} (DB part {r.get('chunk_id', '?')}, distance={dist_str}) ---")
        if meta_str:
            print(f"  Metadata: {meta_str}")
        print(f"  Document preview:\n  {doc.replace(chr(10), chr(10) + '  ')}")
    print(f"\n{'='*60}\n")


def answer_with_gemini(
    query: str,
    context_docs: list,
    model: str = GEMINI_MODEL,
    api_key: str | None = None,
) -> str:
    """Build RAG prompt and get answer from Gemini. All LLM code in this file."""
    context = "\n\n---\n\n".join(d["document"] for d in context_docs)
    system = (
        "You are a helpful support assistant. Answer based only on the following ticket context. "
        "If the context does not contain enough information, say so briefly."
    )
    prompt = f"Context from knowledge base:\n\n{context}\n\nQuestion: {query}"
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return _gemini_generate(
        prompt,
        model=model,
        system_instruction=system,
        temperature=0.3,
        api_key=key,
    )


def main():
    import sys
    argv = sys.argv[1:]
    chunks_only = "--chunks-only" in argv or "-c" in argv
    if chunks_only:
        argv = [a for a in argv if a not in ("--chunks-only", "-c")]
    if not chunks_only and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment.")
        sys.exit(1)
    query = " ".join(argv) if argv else "How do I fix VPN connection issues?"
    print("Query:", query)
    print("\nQuerying ChromaDB (parallel, top 3 per chunk)...")
    results = query_all_splits(query, n_per_chunk=TOP_K_PER_CHUNK, max_total=None)
    if not results:
        print("No results. Run index_data.py first to create data/db/chroma_db_part_1..6.")
        sys.exit(1)
    print_chunk_results(results)
    if chunks_only:
        print("(Chunks only; omit --chunks-only to get Gemini answer.)")
        return
    print("Asking Gemini...")
    answer = answer_with_gemini(query, results)
    print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
