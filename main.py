import datetime as dt

from src.data_loader import load_raw_data
from src.eda import (
    set_pandas_display_options,
    basic_data_overview,
    crm_value_checks,
    date_checks
)
from src.preprocessing import cap_outliers
from src.feature_engineering import create_cltv_features
from src.modeling import (
    fit_bg_nbd,
    fit_gamma_gamma,
    predict_expected_sales,
    calculate_cltv
)
from src.segmentation import create_cltv_segment


# -----------------------
# CONFIG
# -----------------------
DATA_PATH = "data/raw/flo_data_20k.csv"
ANALYSIS_DATE = dt.datetime(2021, 6, 1)


# -----------------------
# DATA LOADING
# -----------------------
df = load_raw_data(DATA_PATH)


# -----------------------
# EDA
# -----------------------
set_pandas_display_options()
basic_data_overview(df)
crm_value_checks(df)
date_checks(df)


# -----------------------
# PREPROCESSING
# -----------------------
OUTLIER_COLS = [
    "order_num_total_ever_online",
    "order_num_total_ever_offline",
    "customer_value_total_ever_online",
    "customer_value_total_ever_offline",
    "total_order_count",
    "total_customer_value"
]

for col in OUTLIER_COLS:
    if "order_num" in col or "total_order" in col:
        cap_outliers(df, col, round_values=True)
    else:
        cap_outliers(df, col)

print("\n PREPROCESSING CHECK – OUTLIER CAPPED DATA")
print(df[OUTLIER_COLS].describe().T)


# -----------------------
# FEATURE ENGINEERING
# -----------------------
cltv_df = create_cltv_features(df, ANALYSIS_DATE)

print(cltv_df.head())
print(cltv_df.describe().T)


# -----------------------
# MODELING
# -----------------------
bgf = fit_bg_nbd(cltv_df)
ggf = fit_gamma_gamma(cltv_df)

cltv_df["exp_sales_3_month"] = predict_expected_sales(
    bgf, cltv_df, months=3
)
cltv_df["exp_sales_6_month"] = predict_expected_sales(
    bgf, cltv_df, months=6
)

cltv_df["cltv_6_month"] = calculate_cltv(
    bgf,
    ggf,
    cltv_df,
    months=6
)


# -----------------------
# SEGMENTATION
# -----------------------
cltv_df = create_cltv_segment(cltv_df)

print(
    cltv_df.groupby("cltv_segment")["cltv_6_month"]
    .agg(["count", "mean", "sum"])
)


# -----------------------
# OUTPUT FORMATTING
# -----------------------
cltv_df["customer_short_id"] = cltv_df["customer_id"].str[:8]

output_cols = [
    "customer_short_id",
    "cltv_segment",
    "frequency",
    "monetary_cltv_avg",
    "exp_sales_3_month",
    "exp_sales_6_month",
    "cltv_6_month"
]

print(cltv_df[output_cols].head())
