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

# ===Section3: Remove columns with more than 90% empty values====
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
   
    """
    Remove leading and trailing whitespace from string columns in the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with whitespace removed from string columns.
    """
    df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    str_cols = df.select_dtypes(include=["object"]).columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip().replace("", pd.NA))
    return df

# ===Section5: column name standardization====
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Standardize column names by converting them to lowercase, replacing spaces with underscores
    and removing special characters.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with standardized column names.
    """
    df = df.copy()  
    df.columns = df.columns.str.lower().str.replace(r"\s+", "_", regex=True).str.replace("[^a-zA-Z0-9_]", "", regex=True)
    return df

# ===Section6: Date Parsing====
def parse_dates(df: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    """
    Parse specified columns as dates in the DataFrame.

    Args:
        df: Input DataFrame.
        date_columns: List of column names to be parsed as dates.

    Returns:
        DataFrame with specified columns parsed as dates.
    """
    df = df.copy()

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# ===Section7: Type Conversion===
def convert_column_types(df: pd.DataFrame, column_types: dict[str, str]) -> pd.DataFrame:
    """
    Convert specified columns to given data types in the DataFrame.

    Args:
        df: Input DataFrame.
        column_types: Dictionary mapping column names to desired data types.

    Returns:
        DataFrame with specified columns converted to given data types.
    """
    df = df.copy()

    for col, dtype in column_types.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype, errors="ignore")
    return df

    
# ===Section8: Cleaning Summary===
def cleaning_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a column-wise cleaning summary.
    Args:
    df: Input DataFrame.

    Returns:
        pd.DataFrame containing:
        - Column
        - Non-Null Count
        - Missing Count
        - Missing Percentage
    """
    summary = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum(),
        "Missing Count": df.isnull().sum(),
        "Missing Percentage": (df.isnull().mean() * 100).round(2)
    })
    summary = summary.sort_values(by="Missing Percentage", ascending=False).reset_index(drop=True)
    return summary