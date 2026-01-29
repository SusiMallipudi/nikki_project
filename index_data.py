"""
Index (vectorize) ticket CSV splits into ChromaDB. Run once (or when data changes).
Creates chroma_db_part_1 .. chroma_db_part_6. Do not run repeatedly.
"""
import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

from config import N_SPLITS, DB_PREFIX, COLLECTION_NAME, CSV_PREFIX, EMBEDDING_MODEL

BATCH_SIZE = 256
EF = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def sanitize_metadata(row):
    """Chroma metadata: str, int, float, bool only; skip NaN/empty."""
    meta = {}
    for k, v in row.items():
        if pd.isna(v) or v == "":
            continue
        if isinstance(v, (str, int, float, bool)):
            meta[k] = v
        else:
            meta[k] = str(v)
    return meta


def build_document(row):
    """One searchable document per ticket: subject + body + answer."""
    parts = [
        str(row.get("subject", "") or ""),
        str(row.get("body", "") or ""),
        str(row.get("answer", "") or ""),
    ]
    return "\n\n".join(p for p in parts if p.strip())


def index_one_split(part_num: int):
    """Load one split CSV into its ChromaDB. Overwrites existing collection data if re-run."""
    csv_path = f"{CSV_PREFIX}-part-{part_num}-of-{N_SPLITS}.csv"
    db_path = f"{DB_PREFIX}_{part_num}"

    if not os.path.isfile(csv_path):
        print(f"  Skip part {part_num}: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=EF,
        metadata={"hnsw:space": "cosine"},
    )

    n = len(df)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = df.iloc[start:end]
        ids = [f"part{part_num}_row{i}" for i in range(start, end)]
        documents = [build_document(batch.iloc[i]) for i in range(len(batch))]
        metadatas = [sanitize_metadata(batch.iloc[i]) for i in range(len(batch))]
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"  Part {part_num}: {n:,} records -> {db_path}/ ({COLLECTION_NAME})")


def main():
    print("Indexing CSV splits into ChromaDB (run once)...")
    for i in range(1, N_SPLITS + 1):
        index_one_split(i)
    print("Done. Use query_data.py to query (no re-indexing).")


if __name__ == "__main__":
    main()
