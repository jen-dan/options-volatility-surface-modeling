from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

options_path = PROCESSED_DIR / "spy_options_clean.csv"
df = pd.read_csv(options_path)

# типы
df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
df["ivm"] = pd.to_numeric(df["ivm"], errors="coerce")
df["ifwd"] = pd.to_numeric(df["ifwd"], errors="coerce")
df["days_to_expiry"] = pd.to_numeric(df["days_to_expiry"], errors="coerce")

# чистка
df = df.dropna(subset=["expiry", "strike", "ivm", "ifwd", "days_to_expiry"]).copy()

# distance to forward: proxy for ATM-ness
df["distance_to_forward"] = (df["strike"] - df["ifwd"]).abs()

# на каждую expiry возьмем один контракт, ближайший к forward
atm_rows = (
    df.sort_values(["expiry", "distance_to_forward", "volume"], ascending=[True, True, False])
      .groupby("expiry", as_index=False)
      .first()
      .sort_values("expiry")
      .reset_index(drop=True)
)

print("ATM TERM STRUCTURE DATA:")
print(atm_rows[["expiry", "days_to_expiry", "ifwd", "strike", "ivm", "volume"]])

plt.figure(figsize=(10, 6))
plt.plot(atm_rows["days_to_expiry"], atm_rows["ivm"], marker="o")
plt.xlabel("Days to Expiry")
plt.ylabel("ATM Implied Volatility (IVM)")
plt.title("SPY ATM Term Structure")
plt.grid(True)

output_path = OUTPUTS_DIR / "spy_atm_term_structure.png"
plt.savefig(output_path, bbox_inches="tight")
plt.show()

print(f"\nSaved plot to: {output_path}")