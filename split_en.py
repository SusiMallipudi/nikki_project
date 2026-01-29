"""Split the English-only CSV into 6 equal parts."""
import pandas as pd

INPUT_CSV = "aa_dataset-tickets-en-only.csv"
N_SPLITS = 6
OUTPUT_PREFIX = "aa_dataset-tickets-en-only"

df = pd.read_csv(INPUT_CSV)
n = len(df)
size = n // N_SPLITS
remainder = n % N_SPLITS
chunks = []
start = 0
for i in range(N_SPLITS):
    end = start + size + (1 if i < remainder else 0)
    chunks.append(df.iloc[start:end])
    start = end

for i, chunk in enumerate(chunks):
    out_path = f"{OUTPUT_PREFIX}-part-{i + 1}-of-{N_SPLITS}.csv"
    chunk.to_csv(out_path, index=False)
    print(f"  Part {i + 1}: {len(chunk):,} rows -> {out_path}")

print(f"\nTotal: {n:,} rows in {N_SPLITS} files")
