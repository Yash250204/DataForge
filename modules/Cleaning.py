import pandas as pd
# ===Section1: Duplicate Removal=====
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with duplicates removed.
    """
    return df.drop_duplicates()

# ===Section2: Remove Empty Rows=====
def remove_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with any missing values from the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with empty rows removed.
    """
    return df.dropna(how="all")

# ===Section3: Remove columns with more then 90% empty values====
def remove_sparse_columns(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Remove columns with more than a specified threshold of missing values.

    Args:
        df: Input DataFrame.
        threshold: Proportion of missing values above which columns will be removed.

    Returns:
        DataFrame with sparse columns removed.
    """
    missing_percentage = df.isnull().mean()

    columns_to_drop = missing_percentage[missing_percentage >= threshold].index

    return df.drop(columns=columns_to_drop)

# ===Section4: whitespace removal from string columns====
def remove_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    """
    Remove leading and trailing whitespace from string columns in the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with whitespace removed from string columns.
    """
    str_cols = df.select_dtypes(include=["object"]).columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip().replace("", pd.NA))
    return df

# ===Section5: column name standardization====
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    """
    Standardize column names by converting them to lowercase, replacing spaces with underscores
    and removing special characters.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with standardized column names.
    """
    df.columns = df.columns.str.lower().str.replace(r"\s+", "_", regex=True).str.replace("[^a-zA-Z0-9_]", "", regex=True)
    return df