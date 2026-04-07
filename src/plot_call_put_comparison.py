from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

options_path = PROCESSED_DIR / "spy_options_all_clean.csv"
df = pd.read_csv(options_path)

df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
df["ivm"] = pd.to_numeric(df["ivm"], errors="coerce")
df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

df = df.dropna(subset=["expiry", "strike", "ivm", "option_type"]).copy()

target_expiry = sorted(df["expiry"].dropna().unique())[0]
subset = df[df["expiry"] == target_expiry].copy()

print("Selected expiry:", target_expiry)
print("\nOption types in subset:")
print(subset["option_type"].value_counts())

plt.figure(figsize=(10, 6))

for option_type in ["call", "put"]:
    side = subset[subset["option_type"] == option_type].copy()

    if side.empty:
        continue

    side = (
        side.groupby("strike", as_index=False)
            .agg({"ivm": "mean"})
            .sort_values("strike")
    )

    print(f"\n{option_type.upper()} DATA:")
    print(side)

    plt.plot(
        side["strike"],
        side["ivm"],
        marker="o",
        label=option_type
    )

plt.xlabel("Strike")
plt.ylabel("Implied Volatility (IVM)")
plt.title(f"SPY Call vs Put Smile - {pd.Timestamp(target_expiry).date()}")
plt.grid(True)
plt.legend()

output_path = OUTPUTS_DIR / "spy_call_put_comparison.png"
plt.savefig(output_path, bbox_inches="tight")
plt.show()

print(f"\nSaved plot to: {output_path}")