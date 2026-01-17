import pandas as pd
def create_cltv_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment customers based on CLTV quartiles.
    """
    df = df.copy()

    df["cltv_segment"] = pd.qcut(
        df["cltv_6_month"],
        q=4,
        labels=["D", "C", "B", "A"]
    )

    return df
