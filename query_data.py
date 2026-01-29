"""
Query ChromaDB only (no indexing). Use after index_data.py has been run once.
Queries all 6 DBs in parallel, top 3 per chunk; optionally answers with Gemini.
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
from chromadb.utils import embedding_functions

from config import N_SPLITS, DB_PREFIX, COLLECTION_NAME, EMBEDDING_MODEL
from gemini_client import generate

TOP_K_PER_CHUNK = 3
EF = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def _query_one_chunk(args):
    """Query a single ChromaDB; returns list of results for that chunk."""
    chunk_id, query_text, n_per_chunk = args
    path = f"{DB_PREFIX}_{chunk_id}"
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
    chunk_ids = [i for i in range(1, N_SPLITS + 1) if os.path.isdir(f"{DB_PREFIX}_{i}")]
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


def answer_with_gemini(query: str, context_docs: list, model: str = "gemini-2.5-flash") -> str:
    """Build RAG prompt and get answer from Gemini."""
    context = "\n\n---\n\n".join(d["document"] for d in context_docs)
    system = (
        "You are a helpful support assistant. Answer based only on the following ticket context. "
        "If the context does not contain enough information, say so briefly."
    )
    prompt = f"Context from knowledge base:\n\n{context}\n\nQuestion: {query}"
    return generate(
        prompt,
        model=model,
        system_instruction=system,
        temperature=0.3,
        api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
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
        print("No results. Run index_data.py first to create chroma_db_part_1..6.")
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
