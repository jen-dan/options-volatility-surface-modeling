from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

options_path = PROCESSED_DIR / "spy_options_clean.csv"
df = pd.read_csv(options_path)

# Приведём типы
df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
df["ivm"] = pd.to_numeric(df["ivm"], errors="coerce")
df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
df["ifwd"] = pd.to_numeric(df["ifwd"], errors="coerce")

# Уберём мусор
df = df.dropna(subset=["expiry", "strike", "ivm"]).copy()

print("Available expiries:")
print(df["expiry"].dropna().sort_values().unique())

# Возьмём самую ближнюю expiry
target_expiry = df["expiry"].dropna().sort_values().unique()[0]
smile = df[df["expiry"] == target_expiry].copy()

# Сортировка по strike
smile = smile.sort_values("strike")

print("\nSelected expiry:")
print(target_expiry)
print("\nSmile data:")
print(smile[["expiry", "strike", "ivm", "volume", "bid", "ask", "last"]])

plt.figure(figsize=(10, 6))
plt.plot(smile["strike"], smile["ivm"], marker="o")
plt.xlabel("Strike")
plt.ylabel("Implied Volatility (IVM)")
plt.title(f"SPY Volatility Smile - {pd.Timestamp(target_expiry).date()}")
plt.grid(True)

output_path = OUTPUTS_DIR / "spy_smile_nearest_expiry.png"
plt.savefig(output_path, bbox_inches="tight")
plt.show()

print(f"\nSaved plot to: {output_path}")