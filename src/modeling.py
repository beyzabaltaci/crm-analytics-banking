import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter


def fit_bg_nbd(
    df: pd.DataFrame,
    penalizer_coef: float = 0.001
) -> BetaGeoFitter:
    """
    Fit BG-NBD model.
    """
    bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)

    bgf.fit(
        df["frequency"],
        df["recency_cltv_weekly"],
        df["T_weekly"]
    )

    return bgf


def fit_gamma_gamma(
    df: pd.DataFrame,
    penalizer_coef: float = 0.01
) -> GammaGammaFitter:
    """
    Fit Gamma-Gamma model.
    """
    ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)

    ggf.fit(
        df["frequency"],
        df["monetary_cltv_avg"]
    )

    return ggf


def predict_expected_sales(
    bgf: BetaGeoFitter,
    df: pd.DataFrame,
    months: int
) -> pd.Series:
    """
    Predict expected number of purchases for given months.
    """
    weeks = months * 4

    return bgf.predict(
        weeks,
        df["frequency"],
        df["recency_cltv_weekly"],
        df["T_weekly"]
    )


def calculate_cltv(
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    df: pd.DataFrame,
    months: int,
    discount_rate: float = 0.01
) -> pd.Series:
    """
    Calculate CLTV using BG-NBD and Gamma-Gamma models.
    """
    cltv = ggf.customer_lifetime_value(
        bgf,
        df["frequency"],
        df["recency_cltv_weekly"],
        df["T_weekly"],
        df["monetary_cltv_avg"],
        time=months,
        freq="W",
        discount_rate=discount_rate
    )

    return cltv
