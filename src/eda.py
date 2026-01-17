import pandas as pd

def crm_value_checks(df: pd.DataFrame) -> None:
    """
    CRM-specific checks for CLTV analysis.
    """
    print("\n🔹 TOTAL ORDER COUNT")
    print(df["total_order_count"].describe())

    print("\n🔹 TOTAL CUSTOMER VALUE")
    print(df["total_customer_value"].describe())

def date_checks(df: pd.DataFrame) -> None:
    """
    Date range checks for CLTV calculations.
    """
    print("\n🔹 DATE RANGES")
    print("First order date (min):", df["first_order_date"].min())
    print("Last order date (max):", df["last_order_date"].max())

def set_pandas_display_options() -> None:
    """
    Set pandas display options for EDA.
    """
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 500)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")


def basic_data_overview(df: pd.DataFrame) -> None:
    """
    Basic overview of the dataset similar to initial exploratory analysis.
    """
    print("\n🔹 SHAPE")
    print(df.shape)

    print("\n🔹 FIRST 5 ROWS")
    print(df.head())

    print("\n🔹 DATA TYPES")
    print(df.dtypes)

    print("\n🔹 MISSING VALUES (%)")
    print(df.isnull().mean() * 100)

    print("\n🔹 DESCRIPTIVE STATISTICS")
    print(df.describe().T)
