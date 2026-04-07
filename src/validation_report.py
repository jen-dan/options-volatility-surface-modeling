from pathlib import Path
import pandas as pd
import numpy as np

PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

filtered_path = PROCESSED_DIR / "spy_options_filtered.csv"
fit_path = OUTPUTS_DIR / "svi_fit_all_expiries_aggregated.csv"
params_path = OUTPUTS_DIR / "svi_params_all_expiries_aggregated.csv"

filtered = pd.read_csv(filtered_path)
fit_df = pd.read_csv(fit_path)
params_df = pd.read_csv(params_path)

# types
filtered["expiry"] = pd.to_datetime(filtered["expiry"], errors="coerce")
filtered["strike"] = pd.to_numeric(filtered["strike"], errors="coerce")
filtered["ivm"] = pd.to_numeric(filtered["ivm"], errors="coerce")
filtered["volume"] = pd.to_numeric(filtered["volume"], errors="coerce")
filtered["days_to_expiry"] = pd.to_numeric(filtered["days_to_expiry"], errors="coerce")

fit_df["expiry"] = pd.to_datetime(fit_df["expiry"], errors="coerce")
fit_df["strike"] = pd.to_numeric(fit_df["strike"], errors="coerce")
fit_df["ivm"] = pd.to_numeric(fit_df["ivm"], errors="coerce")
fit_df["iv_svi_pct"] = pd.to_numeric(fit_df["iv_svi_pct"], errors="coerce")
fit_df["fit_error_abs"] = pd.to_numeric(fit_df["fit_error_abs"], errors="coerce")
fit_df["fit_error_sq"] = pd.to_numeric(fit_df["fit_error_sq"], errors="coerce")
fit_df["days_to_expiry"] = pd.to_numeric(fit_df["days_to_expiry"], errors="coerce")

params_df["expiry"] = pd.to_datetime(params_df["expiry"], errors="coerce")
params_df["days_to_expiry"] = pd.to_numeric(params_df["days_to_expiry"], errors="coerce")
params_df["mae_iv_pct"] = pd.to_numeric(params_df["mae_iv_pct"], errors="coerce")
params_df["rmse_iv_pct"] = pd.to_numeric(params_df["rmse_iv_pct"], errors="coerce")

# work on calls only for reporting consistency with SVI fit
filtered_calls = filtered[filtered["option_type"] == "call"].copy()

coverage_rows = []
for expiry, grp in filtered_calls.groupby("expiry"):
    coverage_rows.append({
        "expiry": expiry,
        "days_to_expiry": grp["days_to_expiry"].iloc[0],
        "n_raw_points": len(grp),
        "n_unique_strikes": grp["strike"].nunique(),
        "min_strike": grp["strike"].min(),
        "max_strike": grp["strike"].max(),
        "mean_ivm": grp["ivm"].mean(),
        "mean_volume": grp["volume"].mean(),
    })

coverage_df = pd.DataFrame(coverage_rows).sort_values("expiry").reset_index(drop=True)

# merge with fit stats
report_df = coverage_df.merge(
    params_df[["expiry", "mae_iv_pct", "rmse_iv_pct", "a", "b", "rho", "m", "sigma"]],
    on="expiry",
    how="left"
)

# rule-based flags
def classify_row(row):
    issues = []

    if pd.isna(row["n_unique_strikes"]) or row["n_unique_strikes"] < 5:
        issues.append("too_few_unique_strikes")

    if pd.notna(row["mae_iv_pct"]) and row["mae_iv_pct"] > 0.50:
        issues.append("weak_fit")
    elif pd.notna(row["mae_iv_pct"]) and row["mae_iv_pct"] > 0.20:
        issues.append("moderate_fit")

    if pd.notna(row["a"]) and row["a"] < 0:
        issues.append("negative_a")

    if pd.notna(row["rho"]) and abs(row["rho"]) > 0.95:
        issues.append("rho_near_boundary")

    if not issues:
        return "ok"

    return ", ".join(issues)

report_df["validation_flag"] = report_df.apply(classify_row, axis=1)

# overall summary
summary_lines = []
summary_lines.append("# Validation Report")
summary_lines.append("")
summary_lines.append(f"- Total filtered rows: {len(filtered)}")
summary_lines.append(f"- Total filtered calls: {len(filtered_calls)}")
summary_lines.append(f"- Number of expiries fitted: {report_df['expiry'].nunique()}")
summary_lines.append(f"- Average MAE of IV fit: {report_df['mae_iv_pct'].mean():.4f}%")
summary_lines.append(f"- Best expiry MAE: {report_df['mae_iv_pct'].min():.4f}%")
summary_lines.append(f"- Worst expiry MAE: {report_df['mae_iv_pct'].max():.4f}%")
summary_lines.append("")
summary_lines.append("## Per-expiry diagnostics")
summary_lines.append("")
summary_lines.append(report_df.to_markdown(index=False))

report_md_path = OUTPUTS_DIR / "validation_report.md"
report_csv_path = OUTPUTS_DIR / "validation_report.csv"

report_df.to_csv(report_csv_path, index=False)
report_md_path.write_text("\n".join(summary_lines), encoding="utf-8")

print("VALIDATION REPORT TABLE:")
print(report_df)

print(f"\nSaved markdown report to: {report_md_path}")
print(f"Saved csv report to: {report_csv_path}")