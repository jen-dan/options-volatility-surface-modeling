from pathlib import Path
import pandas as pd

path = Path("data/raw/spy_options_snapshot.csv")
df = pd.read_csv(path)

print("RAW FULL COLUMNS:")
print(df.columns.tolist())

print("\nFIRST ROW VALUES:")
print(df.iloc[0].tolist())

puts_block = df.iloc[:, 7:14].copy()

print("\nPUTS BLOCK ORIGINAL COLUMN NAMES:")
print(puts_block.columns.tolist())

print("\nPUTS BLOCK FIRST 15 ROWS:")
print(puts_block.head(15))

print("\nPUTS BLOCK ROW 0 VALUES:")
print(puts_block.iloc[0].tolist())