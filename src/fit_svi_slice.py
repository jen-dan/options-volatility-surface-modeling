from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

input_path = PROCESSED_DIR / "spy_svi_slice_nearest_expiry.csv"

df = pd.read_csv(input_path)

k = df["k"].to_numpy()
w_market = df["w_market"].to_numpy()
T = df["T"].iloc[0]

def svi_raw(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

def objective(params, k, w_market):
    a, b, rho, m, sigma = params
    w_fit = svi_raw(k, a, b, rho, m, sigma)
    return np.mean((w_fit - w_market) ** 2)

# initial guess
initial_guess = np.array([
    0.01,   # a
    0.10,   # b
    -0.5,   # rho
    0.0,    # m
    0.10    # sigma
])

# raw SVI constraints / bounds
bounds = [
    (-1.0, 2.0),     # a
    (1e-6, 10.0),    # b
    (-0.999, 0.999), # rho
    (-5.0, 5.0),     # m
    (1e-6, 5.0)      # sigma
]

result = minimize(
    objective,
    initial_guess,
    args=(k, w_market),
    bounds=bounds,
    method="L-BFGS-B"
)

if not result.success:
    print("Optimization warning:", result.message)

a, b, rho, m, sigma = result.x

print("SVI PARAMETERS:")
print(f"a     = {a:.8f}")
print(f"b     = {b:.8f}")
print(f"rho   = {rho:.8f}")
print(f"m     = {m:.8f}")
print(f"sigma = {sigma:.8f}")

df["w_svi"] = svi_raw(df["k"], a, b, rho, m, sigma)
df["iv_svi_decimal"] = np.sqrt(df["w_svi"] / df["T"])
df["iv_svi_pct"] = df["iv_svi_decimal"] * 100.0

# save params
params_df = pd.DataFrame([{
    "expiry": df["expiry"].iloc[0],
    "a": a,
    "b": b,
    "rho": rho,
    "m": m,
    "sigma": sigma,
    "objective_mse": result.fun
}])

params_path = OUTPUTS_DIR / "svi_params_nearest_expiry.csv"
params_df.to_csv(params_path, index=False)

fit_path = OUTPUTS_DIR / "svi_fit_nearest_expiry.csv"
df.to_csv(fit_path, index=False)

print(f"\nSaved params to: {params_path}")
print(f"Saved fit data to: {fit_path}")

# plot market IV vs SVI IV
plot_df = df.sort_values("strike")

plt.figure(figsize=(10, 6))
plt.plot(plot_df["strike"], plot_df["ivm"], marker="o", label="Market IV")
plt.plot(plot_df["strike"], plot_df["iv_svi_pct"], marker="o", label="SVI Fitted IV")
plt.xlabel("Strike")
plt.ylabel("Implied Volatility (%)")
plt.title(f"SVI Fit vs Market IV - {plot_df['expiry'].iloc[0]}")
plt.grid(True)
plt.legend()

plot_path = OUTPUTS_DIR / "svi_fit_vs_market_nearest_expiry.png"
plt.savefig(plot_path, bbox_inches="tight")
plt.show()

print(f"Saved plot to: {plot_path}")

# plot total variance vs k
plot_df = plot_df.sort_values("k")

plt.figure(figsize=(10, 6))
plt.plot(plot_df["k"], plot_df["w_market"], marker="o", label="Market total variance")
plt.plot(plot_df["k"], plot_df["w_svi"], marker="o", label="SVI fitted total variance")
plt.xlabel("Log-moneyness k = ln(K/F)")
plt.ylabel("Total Variance")
plt.title(f"SVI Total Variance Fit - {plot_df['expiry'].iloc[0]}")
plt.grid(True)
plt.legend()

plot2_path = OUTPUTS_DIR / "svi_total_variance_fit_nearest_expiry.png"
plt.savefig(plot2_path, bbox_inches="tight")
plt.show()

print(f"Saved plot to: {plot2_path}")