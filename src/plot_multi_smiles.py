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
df["ifwd"] = pd.to_numeric(df["ifwd"], errors="coerce")

df = df.dropna(subset=["expiry", "strike", "ivm"]).copy()

expiries = sorted(df["expiry"].dropna().unique())
selected_expiries = expiries[:4]  # первые 4 expiry

print("Selected expiries:")
for e in selected_expiries:
    print(e)

plt.figure(figsize=(10, 6))

for expiry in selected_expiries:
    smile = df[df["expiry"] == expiry].copy()

    # если внутри expiry есть повторяющиеся strikes, усредним по ним
    smile = (
        smile.groupby("strike", as_index=False)
             .agg({"ivm": "mean"})
             .sort_values("strike")
    )

    plt.plot(
        smile["strike"],
        smile["ivm"],
        marker="o",
        label=str(pd.Timestamp(expiry).date())
    )

plt.xlabel("Strike")
plt.ylabel("Implied Volatility (IVM)")
plt.title("SPY Volatility Smiles Across Expiries")
plt.grid(True)
plt.legend()

output_path = OUTPUTS_DIR / "spy_multi_smiles.png"
plt.savefig(output_path, bbox_inches="tight")
plt.show()

print(f"\nSaved plot to: {output_path}")