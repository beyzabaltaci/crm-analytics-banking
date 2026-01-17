import pandas as pd
from pathlib import Path


def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    Load raw CRM data from csv and apply basic standardization.
    """
    df = pd.read_csv(data_path)

    # Date columns
    date_cols = [
        "first_order_date",
        "last_order_date",
        "last_order_date_online",
        "last_order_date_offline"
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    # Total order & value
    df["total_order_count"] = (
        df["order_num_total_ever_online"]
        + df["order_num_total_ever_offline"]
    )

    df["total_customer_value"] = (
        df["customer_value_total_ever_online"]
        + df["customer_value_total_ever_offline"]
    )

    return df
