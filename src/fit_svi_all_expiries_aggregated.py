from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

input_path = PROCESSED_DIR / "spy_options_filtered.csv"

df = pd.read_csv(input_path)

df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
df["ifwd"] = pd.to_numeric(df["ifwd"], errors="coerce")
df["ivm"] = pd.to_numeric(df["ivm"], errors="coerce")
df["days_to_expiry"] = pd.to_numeric(df["days_to_expiry"], errors="coerce")
df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

df = df[df["option_type"] == "call"].copy()
df = df.dropna(subset=["expiry", "strike", "ifwd", "ivm", "days_to_expiry"]).copy()

def svi_raw(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

def objective(params, k, w_market):
    a, b, rho, m, sigma = params
    w_fit = svi_raw(k, a, b, rho, m, sigma)
    return np.mean((w_fit - w_market) ** 2)

results = []
fit_rows = []

for expiry, slice_df in df.groupby("expiry"):
    # aggregate by strike first
    slice_df = (
        slice_df.groupby("strike", as_index=False)
        .agg({
            "ifwd": "mean",
            "ivm": "mean",
            "days_to_expiry": "mean",
            "volume": "sum"
        })
        .sort_values("strike")
        .reset_index(drop=True)
    )

    if len(slice_df) < 5:
        print(f"Skipping {expiry.date()} - not enough unique strikes ({len(slice_df)})")
        continue

    slice_df["expiry"] = expiry
    slice_df["iv_decimal"] = slice_df["ivm"] / 100.0
    slice_df["T"] = slice_df["days_to_expiry"] / 365.0
    slice_df["k"] = np.log(slice_df["strike"] / slice_df["ifwd"])
    slice_df["w_market"] = (slice_df["iv_decimal"] ** 2) * slice_df["T"]

    k = slice_df["k"].to_numpy()
    w_market = slice_df["w_market"].to_numpy()

    initial_guess = np.array([0.01, 0.10, -0.5, 0.0, 0.10])
    bounds = [
        (-1.0, 2.0),
        (1e-6, 10.0),
        (-0.999, 0.999),
        (-5.0, 5.0),
        (1e-6, 5.0)
    ]

    result = minimize(
        objective,
        initial_guess,
        args=(k, w_market),
        bounds=bounds,
        method="L-BFGS-B"
    )

    if not result.success:
        print(f"Warning for {expiry.date()}: {result.message}")

    a, b, rho, m, sigma = result.x

    slice_df["w_svi"] = svi_raw(slice_df["k"], a, b, rho, m, sigma)
    slice_df["iv_svi_decimal"] = np.sqrt(np.maximum(slice_df["w_svi"], 0) / slice_df["T"])
    slice_df["iv_svi_pct"] = slice_df["iv_svi_decimal"] * 100.0
    slice_df["fit_error_abs"] = (slice_df["iv_svi_pct"] - slice_df["ivm"]).abs()
    slice_df["fit_error_sq"] = (slice_df["iv_svi_pct"] - slice_df["ivm"]) ** 2

    mse_iv = slice_df["fit_error_sq"].mean()
    mae_iv = slice_df["fit_error_abs"].mean()

    results.append({
        "expiry": expiry,
        "n_unique_strikes": len(slice_df),
        "days_to_expiry": slice_df["days_to_expiry"].iloc[0],
        "a": a,
        "b": b,
        "rho": rho,
        "m": m,
        "sigma": sigma,
        "objective_mse_w": result.fun,
        "mae_iv_pct": mae_iv,
        "rmse_iv_pct": np.sqrt(mse_iv),
    })

    fit_rows.append(slice_df)

params_df = pd.DataFrame(results).sort_values("expiry").reset_index(drop=True)
fit_df = pd.concat(fit_rows, ignore_index=True) if fit_rows else pd.DataFrame()

params_out = OUTPUTS_DIR / "svi_params_all_expiries_aggregated.csv"
fit_out = OUTPUTS_DIR / "svi_fit_all_expiries_aggregated.csv"

params_df.to_csv(params_out, index=False)
fit_df.to_csv(fit_out, index=False)

print("\nSVI PARAMS ALL EXPIRIES (AGGREGATED):")
print(params_df)

print(f"\nSaved params to: {params_out}")
print(f"Saved fit data to: {fit_out}")

if not params_df.empty:
    plt.figure(figsize=(10, 6))
    plt.plot(params_df["days_to_expiry"], params_df["mae_iv_pct"], marker="o")
    plt.xlabel("Days to Expiry")
    plt.ylabel("MAE of IV fit (%)")
    plt.title("SVI Fit Error Across Expiries (Aggregated by Strike)")
    plt.grid(True)

    plot_path = OUTPUTS_DIR / "svi_fit_error_across_expiries_aggregated.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()

    print(f"Saved plot to: {plot_path}")