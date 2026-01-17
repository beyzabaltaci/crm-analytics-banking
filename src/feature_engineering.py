import pandas as pd
import datetime as dt


def create_cltv_features(
    df: pd.DataFrame,
    analysis_date: dt.datetime
) -> pd.DataFrame:
    """
    Create CLTV related features:
    Recency, Tenure (T), Frequency, Monetary
    """

    cltv_df = pd.DataFrame()

    cltv_df["customer_id"] = df["master_id"]

    cltv_df["recency_cltv_weekly"] = (
        (df["last_order_date"] - df["first_order_date"])
        .dt.days / 7
    )

    cltv_df["T_weekly"] = (
        (analysis_date - df["first_order_date"])
        .dt.days / 7
    )

    cltv_df["frequency"] = df["total_order_count"]

    cltv_df["monetary_cltv_avg"] = (
        df["total_customer_value"] / df["total_order_count"]
    )

    return cltv_df
