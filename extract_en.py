"""Extract only English (language == 'en') records from the multi-lang CSV."""
import pandas as pd

INPUT_CSV = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
OUTPUT_CSV = "aa_dataset-tickets-en-only.csv"

df = pd.read_csv(INPUT_CSV)
df2 = df[df["language"] == "en"]
df2.to_csv(OUTPUT_CSV, index=False)
print(f"Extracted {len(df2):,} English records -> {OUTPUT_CSV}")
