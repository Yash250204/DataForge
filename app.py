# ==========================================
# MODULE 1: FILE UPLOAD
# ==========================================
from pathlib import Path
import pandas as pd
from typing import Any


SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


def validate_file(file: Any) -> tuple[bool, str]:
    """
    Validate uploaded file type.

    Args:
        file: File path (str) or Streamlit UploadedFile object.

    Returns:
        (is_valid, message)
    """

    if file is None:
        return False, "No file provided."

    # Streamlit UploadedFile
    if hasattr(file, "name"):
        extension = Path(file.name).suffix.lower()

    # Local file path
    elif isinstance(file, str):
        extension = Path(file).suffix.lower()

    else:
        return False, "Unsupported file object."

    if extension not in SUPPORTED_EXTENSIONS:
        return (
            False,
            f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    return True, "File validation successful."


def load_file(file: Any) -> pd.DataFrame:
    """
    Load CSV or XLSX into a Pandas DataFrame.

    Args:
        file: File path or Streamlit UploadedFile object.

    Returns:
        pd.DataFrame
    """

    is_valid, message = validate_file(file)

    if not is_valid:
        raise ValueError(message)

    try:

        if hasattr(file, "name"):
            extension = Path(file.name).suffix.lower()
        else:
            extension = Path(file).suffix.lower()

        if extension == ".csv":
            df = pd.read_csv(file)

        elif extension == ".xlsx":
            df = pd.read_excel(file)

        return df

    except FileNotFoundError:
        raise FileNotFoundError("File not found.")

    except pd.errors.EmptyDataError:
        raise ValueError("Uploaded file is empty.")

    except Exception as e:
        raise RuntimeError(f"Error loading file: {e}") from e
    


# ==========================================
# MODULE 2: DATA PROFILING
# ========================================== 

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
def duplicate_rows_analysis(df: pd.DataFrame) -> dict[str, float | int]:
    """
    Analyze duplicate rows in the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        dict: Count of duplicate rows.
    """
    duplicate_count = int(df.duplicated().sum())
    
    if len(df) == 0:
        return {
            "duplicate_count": 0,
            "duplicate_percentage": 0
        }
    return {
        "duplicate_count": duplicate_count,
        "duplicate_percentage": float(round(
            duplicate_count / len(df) * 100,
            2
        ))
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

