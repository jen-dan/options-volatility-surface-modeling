from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OPTIONS_PATH = RAW_DIR / "spy_options_snapshot.csv"
HISTORY_PATH = RAW_DIR / "spy_underlying_history.csv"


def clean_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Первая строка данных содержит реальные имена колонок
    real_columns = df.iloc[0].tolist()
    df = df.iloc[1:].copy()
    df.columns = real_columns

    # Переименуем аккуратно
    df = df.rename(columns={
        "Dates": "date",
        "Last Price": "last_price",
        "Open Price": "open_price",
        "High Price": "high_price",
        "Low Price": "low_price",
        "Volume": "volume",
    })

    # Типы
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    numeric_cols = ["last_price", "open_price", "high_price", "low_price", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def parse_expiry_header(text: str):
    """
    Пример строки:
    '17-Apr-26 (10d); CSize 100; R 3.96; IFwd 659.57'
    """
    if pd.isna(text):
        return None, None, None, None

    text = str(text).strip()
    if ";" not in text:
        return None, None, None, None

    parts = [p.strip() for p in text.split(";")]
    expiry_raw = parts[0]  # '17-Apr-26 (10d)'

    expiry_date_str = expiry_raw.split(" (")[0].strip()
    days_to_expiry = None
    rate = None
    ifwd = None

    # days
    if "(" in expiry_raw and "d)" in expiry_raw:
        try:
            days_to_expiry = int(expiry_raw.split("(")[1].replace("d)", "").strip())
        except Exception:
            days_to_expiry = None

    for p in parts[1:]:
        if p.startswith("R "):
            try:
                rate = float(p.replace("R ", "").strip())
            except Exception:
                rate = None
        elif p.startswith("IFwd "):
            try:
                ifwd = float(p.replace("IFwd ", "").strip())
            except Exception:
                ifwd = None

    expiry_date = pd.to_datetime(expiry_date_str, format="%d-%b-%y", errors="coerce")
    return expiry_date, days_to_expiry, rate, ifwd


def clean_options(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    calls_df = clean_option_block(df, 0, 7, "call")
    puts_df = clean_option_block(df, 7, 14, "put")

    print("CALLS SHAPE:", calls_df.shape)
    print("PUTS SHAPE:", puts_df.shape)

    if not puts_df.empty:
        print("\nPUTS HEAD:")
        print(puts_df.head(10))
    else:
        print("\nPUTS DF IS EMPTY")

    all_df = pd.concat([calls_df, puts_df], ignore_index=True)
    return all_df

def clean_option_block(df: pd.DataFrame, start_col: int, end_col: int, option_type: str) -> pd.DataFrame:
    second_row = df.iloc[0].tolist()
    block_df = df.iloc[:, start_col:end_col].copy()
    block_second_row = second_row[start_col:end_col]

    new_cols = []
    for name in block_second_row:
        lower = str(name).strip().lower()
        if lower == "strike":
            new_cols.append("strike")
        elif lower == "ticker":
            new_cols.append("ticker")
        elif lower == "bid":
            new_cols.append("bid")
        elif lower == "ask":
            new_cols.append("ask")
        elif lower == "last":
            new_cols.append("last")
        elif lower == "ivm":
            new_cols.append("ivm")
        elif lower == "volm":
            new_cols.append("volume")
        else:
            new_cols.append(lower)

    block_df.columns = new_cols

    # убираем строку-подзаголовок
    block_df = block_df.iloc[1:].copy()

    # вот это главное:
    # metadata строки берем из ПЕРВОЙ колонки полного df (левая часть, где сидят expiry headers)
    meta_series = df.iloc[1:, 0].reset_index(drop=True)
    block_df = block_df.reset_index(drop=True)

    current_expiry = None
    current_dte = None
    current_rate = None
    current_ifwd = None

    rows = []

    for i, row in block_df.iterrows():
        meta_value = meta_series.iloc[i]

        # если на этой строке есть expiry metadata, обновляем текущий контекст
        if isinstance(meta_value, str) and ";" in meta_value:
            current_expiry, current_dte, current_rate, current_ifwd = parse_expiry_header(meta_value)

        first_value = row["strike"]

        strike = pd.to_numeric(first_value, errors="coerce")
        ticker = row.get("ticker")
        bid = pd.to_numeric(row.get("bid"), errors="coerce")
        ask = pd.to_numeric(row.get("ask"), errors="coerce")
        last = pd.to_numeric(row.get("last"), errors="coerce")
        ivm = pd.to_numeric(row.get("ivm"), errors="coerce")
        volume = pd.to_numeric(row.get("volume"), errors="coerce")

        if pd.notna(strike):
            rows.append({
                "expiry": current_expiry,
                "days_to_expiry": current_dte,
                "rate_pct": current_rate,
                "ifwd": current_ifwd,
                "strike": strike,
                "ticker": ticker,
                "bid": bid,
                "ask": ask,
                "last": last,
                "ivm": ivm,
                "volume": volume,
                "option_type": option_type,
            })

    clean_df = pd.DataFrame(rows)

    clean_df["expiry"] = pd.to_datetime(clean_df["expiry"], errors="coerce")

    numeric_cols = [
        "days_to_expiry", "rate_pct", "ifwd",
        "strike", "bid", "ask", "last", "ivm", "volume"
    ]
    for col in numeric_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    clean_df = clean_df.dropna(subset=["expiry", "strike"]).reset_index(drop=True)
    return clean_df

def main():
    history = clean_history(HISTORY_PATH)
    options = clean_options(OPTIONS_PATH)

    history_out = PROCESSED_DIR / "spy_history_clean.csv"
    options_out = PROCESSED_DIR / "spy_options_all_clean.csv"

    history.to_csv(history_out, index=False)
    options.to_csv(options_out, index=False)

    print("HISTORY CLEAN SHAPE:", history.shape)
    print(history.head(10))
    print("\nOPTIONS CLEAN SHAPE:", options.shape)
    print(options.head(15))

    print(f"\nSaved history to: {history_out}")
    print(f"Saved options to: {options_out}")

    print("\nOPTION TYPES COUNTS:")
    print(pd.DataFrame(options["option_type"].value_counts(dropna=False)))


if __name__ == "__main__":
    main()