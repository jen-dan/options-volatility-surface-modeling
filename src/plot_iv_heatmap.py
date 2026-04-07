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

df = df.dropna(subset=["expiry", "strike", "ivm"]).copy()

# Сначала только calls, чтобы картинка была чище
df = df[df["option_type"] == "call"].copy()

pivot = (
    df.groupby(["strike", "expiry"], as_index=False)["ivm"]
      .mean()
      .pivot(index="strike", columns="expiry", values="ivm")
      .sort_index()
)

print("HEATMAP DATA:")
print(pivot)

plt.figure(figsize=(10, 6))
plt.imshow(pivot, aspect="auto", origin="lower")
plt.xticks(
    ticks=range(len(pivot.columns)),
    labels=[str(pd.Timestamp(c).date()) for c in pivot.columns],
    rotation=45
)
plt.yticks(
    ticks=range(len(pivot.index)),
    labels=[str(int(s)) for s in pivot.index]
)
plt.xlabel("Expiry")
plt.ylabel("Strike")
plt.title("SPY Call Implied Volatility Heatmap")
plt.colorbar(label="IVM")

output_path = OUTPUTS_DIR / "spy_call_iv_heatmap.png"
plt.tight_layout()
plt.savefig(output_path, bbox_inches="tight")
plt.show()

print(f"\nSaved plot to: {output_path}")