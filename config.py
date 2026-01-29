"""
Shared config for index_data.py and query_data.py.
Same DB paths, collection name, and embedding model so indexing and querying match.
"""
N_SPLITS = 6
DB_PREFIX = "chroma_db_part"
COLLECTION_NAME = "tickets"
CSV_PREFIX = "aa_dataset-tickets-en-only"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
