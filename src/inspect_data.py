from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

options_path = RAW_DIR / "spy_options_snapshot.csv"
history_path = RAW_DIR / "spy_underlying_history.csv"

options_df = pd.read_csv(options_path)
history_df = pd.read_csv(history_path)

print("OPTIONS SHAPE:", options_df.shape)
print("HISTORY SHAPE:", history_df.shape)

print("\nOPTIONS COLUMNS:")
print(options_df.columns.tolist())

print("\nHISTORY COLUMNS:")
print(history_df.columns.tolist())

print("\nOPTIONS HEAD:")
print(options_df.head(10))

print("\nHISTORY HEAD:")
print(history_df.head(10))