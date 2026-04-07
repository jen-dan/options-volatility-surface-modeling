from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUTS_DIR = Path("outputs")

fit_path = OUTPUTS_DIR / "svi_fit_all_expiries_aggregated.csv"
df = pd.read_csv(fit_path)

df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
df["ivm"] = pd.to_numeric(df["ivm"], errors="coerce")
df["iv_svi_pct"] = pd.to_numeric(df["iv_svi_pct"], errors="coerce")

expiries = sorted(df["expiry"].dropna().unique())

for expiry in expiries:
    slice_df = df[df["expiry"] == expiry].copy().sort_values("strike")

    plt.figure(figsize=(10, 6))
    plt.plot(slice_df["strike"], slice_df["ivm"], marker="o", label="Market IV")
    plt.plot(slice_df["strike"], slice_df["iv_svi_pct"], marker="o", label="SVI Fitted IV")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility (%)")
    plt.title(f"SVI Fit vs Market IV (Aggregated) - {pd.Timestamp(expiry).date()}")
    plt.grid(True)
    plt.legend()

    out = OUTPUTS_DIR / f"svi_fit_vs_market_aggregated_{pd.Timestamp(expiry).date()}.png"
    plt.savefig(out, bbox_inches="tight")
    plt.show()

    print(f"Saved: {out}")