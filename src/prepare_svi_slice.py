from pathlib import Path
import pandas as pd
import numpy as np

PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

input_path = PROCESSED_DIR / "spy_options_filtered.csv"
output_path = PROCESSED_DIR / "spy_svi_slice_nearest_expiry.csv"

df = pd.read_csv(input_path)

df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
df["ifwd"] = pd.to_numeric(df["ifwd"], errors="coerce")
df["ivm"] = pd.to_numeric(df["ivm"], errors="coerce")
df["days_to_expiry"] = pd.to_numeric(df["days_to_expiry"], errors="coerce")

# work with calls first
df = df[df["option_type"] == "call"].copy()

# choose nearest expiry
target_expiry = sorted(df["expiry"].dropna().unique())[0]
slice_df = df[df["expiry"] == target_expiry].copy()

# convert iv from percent to decimal
slice_df["iv_decimal"] = slice_df["ivm"] / 100.0

# time to maturity in years
slice_df["T"] = slice_df["days_to_expiry"] / 365.0

# log-moneyness
slice_df["k"] = np.log(slice_df["strike"] / slice_df["ifwd"])

# total variance
slice_df["w_market"] = (slice_df["iv_decimal"] ** 2) * slice_df["T"]

slice_df = slice_df.dropna(subset=["strike", "ifwd", "k", "w_market", "T"]).copy()
slice_df = slice_df.sort_values("strike").reset_index(drop=True)

print("TARGET EXPIRY:", target_expiry)
print(slice_df[["expiry", "strike", "ifwd", "days_to_expiry", "k", "ivm", "w_market"]])

slice_df.to_csv(output_path, index=False)
print(f"\nSaved SVI slice to: {output_path}")