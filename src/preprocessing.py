import pandas as pd


def outlier_thresholds(
    df: pd.DataFrame,
    col: str,
    q_low: float = 0.01,
    q_high: float = 0.99
):
    """
    Calculate lower and upper outlier thresholds.
    """
    q1 = df[col].quantile(q_low)
    q3 = df[col].quantile(q_high)
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    return lower_limit, upper_limit


def cap_outliers(
    df: pd.DataFrame,
    col: str,
    round_values: bool = False
) -> None:
    """
    Cap outliers with calculated thresholds.
    """
    low, up = outlier_thresholds(df, col)

    if round_values:
        low = round(low)
        up = round(up)

    df.loc[df[col] < low, col] = low
    df.loc[df[col] > up, col] = up
