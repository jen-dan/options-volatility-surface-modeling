from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

input_path = PROCESSED_DIR / "spy_options_all_clean.csv"
output_path = PROCESSED_DIR / "spy_options_filtered.csv"

df = pd.read_csv(input_path)

# types
df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
numeric_cols = [
    "days_to_expiry", "rate_pct", "ifwd", "strike",
    "bid", "ask", "last", "ivm", "volume"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# basic derived fields
df["mid"] = (df["bid"] + df["ask"]) / 2
df["spread"] = df["ask"] - df["bid"]
df["relative_spread"] = df["spread"] / df["mid"]

# basic filtering
filtered = df.copy()
filtered = filtered.dropna(subset=["expiry", "strike", "ifwd", "ivm", "bid", "ask", "mid"])
filtered = filtered[filtered["bid"] > 0]
filtered = filtered[filtered["ask"] > 0]
filtered = filtered[filtered["ask"] >= filtered["bid"]]
filtered = filtered[filtered["mid"] > 0]
filtered = filtered[filtered["ivm"] > 0]
filtered = filtered[filtered["days_to_expiry"] > 0]

# make spread filter not too aggressive for now
filtered = filtered[filtered["relative_spread"] < 0.20]

# keep useful columns
filtered = filtered[
    [
        "expiry", "days_to_expiry", "rate_pct", "ifwd", "strike",
        "ticker", "bid", "ask", "mid", "last", "ivm",
        "volume", "spread", "relative_spread", "option_type"
    ]
].copy()

filtered = filtered.sort_values(["expiry", "option_type", "strike"]).reset_index(drop=True)

print("RAW SHAPE:", df.shape)
print("FILTERED SHAPE:", filtered.shape)
print("\nOPTION TYPE COUNTS:")
print(filtered["option_type"].value_counts())
print("\nEXPIRY COUNTS:")
print(filtered["expiry"].value_counts().sort_index())

filtered.to_csv(output_path, index=False)
print(f"\nSaved filtered dataset to: {output_path}")