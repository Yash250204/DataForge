# ==========================================
# MODULE 2: DATA PROFILING
# ========================================== 
import pandas as pd

## dataset overview(rows, columns, memory usage)
def dataset_overview(df: pd.DataFrame) -> dict[str, float | int]:
    """
    Generate dataset overview including number of rows, columns, and memory usage.

    Args:
        df: Input DataFrame.
    Returns:
        dict: Overview containing number of rows, columns, and memory usage.
    """
    overview = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "memory_usage": float(round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2)  )# in MB
    }
    return overview


## Column Information (column name, Data types, unique values, missing values)
def column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate column information including data types, unique values, and missing values.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Column information.
    """
    missing_percentage = (
        df.isnull().mean() * 100
    ).round(2)
    info = pd.DataFrame({
        "data_type": df.dtypes,
        "num_unique_values": df.nunique(), 
        "missing_values": df.isnull().sum(),
        "missing_percentage": missing_percentage
    
    })
    info.index.name = "column_name"
    info = info.reset_index()
    return info

## Missing Value Analysis (missing value count and percentage)
def missing_value_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Missing value analysis with count and percentage.
    """
    if len(df) == 0:
        return pd.DataFrame()
    
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df) * 100).round(2)

    analysis = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage
    })
    analysis = analysis[analysis["missing_count"] > 0]
    analysis.index.name = "column_name"
    analysis = analysis.reset_index()
    return analysis

## Data Type Distribution (count of each data type)
def data_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the distribution of data types in the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Data type distribution with count of each data type.
    """
    distribution = df.dtypes.value_counts().reset_index()
    distribution.columns = ["data_type", "count"]
    return distribution

## Statistical Summary (for numeric columns: mean, median, std, min, max)
def statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate statistical summary for numeric columns in the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Statistical summary including mean, median, std, min, and max.
    """
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return pd.DataFrame()
    
    summary = numeric_df.describe().T
    summary["median"] = numeric_df.median()
    return summary[["mean", "median", "std", "min", "max"]]

## Duplicate Rows Analysis (count of duplicate rows)
def duplicate_rows_analysis(df: pd.DataFrame) -> dict[str, int | float]:
    """
    Analyze duplicate rows in the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        dict: Count of duplicate rows.
    """
    total_rows = len(df)
    
    if total_rows == 0:
        return {
            "duplicate_count": 0,
            "duplicate_percentage": 0
        }
    duplicate_count = int(df.duplicated().sum())

    return {
        "duplicate_count": duplicate_count,
        "duplicate_percentage": round(
            duplicate_count / total_rows * 100,
            2
        )
    }

## Sample preview of the dataset (first few rows)
def sample_preview(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Generate a sample preview of the dataset.

    Args:
        df: Input DataFrame.
        n: Number of rows to preview (default is 5).

    Returns:
        pd.DataFrame: Sample preview of the dataset.
    """
    return df.head(n)

def random_sample(df: pd.DataFrame,n: int = 5) -> pd.DataFrame:
    return df.sample(n=min(n, len(df)), random_state=42)
