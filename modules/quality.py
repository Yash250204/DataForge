import pandas as pd
from modules.profiling import missing_value_analysis, duplicate_rows_analysis
# ===========
# Section1: Calculate Completeness(Missing Values, Empty strings)
# ===========

# Empty String Analysis
def empty_string_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze empty strings in the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Empty string analysis with count and percentage.
    """
    if len(df) == 0:
        return pd.DataFrame()
    
    empty_string_count = (df.apply(lambda col: col.astype(str).str.strip().eq("").sum()))
    empty_string_percentage = (empty_string_count / len(df) * 100).round(2)

    analysis = pd.DataFrame({
        "empty_string_count": empty_string_count,
        "empty_string_percentage": empty_string_percentage
    })
    analysis = analysis[analysis["empty_string_count"] > 0]
    analysis.index.name = "column_name"
    analysis = analysis.reset_index()
    return analysis

def calculate_completeness_score(df: pd.DataFrame) -> float:
    """
    Calculate the overall completeness score of the dataset.

    Completeness =
    (Non-missing values / Total values) × 100

    Empty strings ("") are also treated as missing values.

    Args:
        df: Input DataFrame.

    Returns:
        float: Completeness score (0-100).
    """
    if df.empty or df.shape[1] == 0:
        return 0.0
    # missing values from module 2 
    missing_df = missing_value_analysis(df)

    if missing_df.empty:
        total_missing = 0
    else:
        total_missing = missing_df["missing_count"].sum()

    # Get empty strings
    empty_string_df = empty_string_analysis(df)

    if empty_string_df.empty:
        total_empty = 0
    else:
        total_empty = empty_string_df["empty_string_count"].sum()

    # Total number of cells in dataset
    total_cells = df.shape[0] * df.shape[1]

    # Total incomplete values (missing + empty strings)
    incomplete_values = total_missing + total_empty

    # Calculate completeness score
    completeness_score = ((total_cells - incomplete_values) / total_cells) * 100

    return round(completeness_score, 2)

#=========
#Section2: Uniqueness (Duplicate Rows)
#=========

def calculate_uniqueness_score(df: pd.DataFrame) -> float:
    """
    Calculate the uniqueness score of the dataset.

    Uniqueness =
    (Unique rows / Total rows) × 100

    Args:
        df: Input DataFrame.

    Returns:
        float: Uniqueness score (0–100).
    """

    total_rows = len(df)

    if total_rows == 0:
        return 0.0

    duplicate_info = duplicate_rows_analysis(df)

    unique_rows = total_rows - duplicate_info["duplicate_count"]

    uniqueness = (unique_rows / total_rows) * 100

    return round(uniqueness, 2)


#=========
#Section3: Validity (Numeric Validation, Date Validation, Range Validation, Constant Columns,
#                    Outlier Detection, Column Name Validation)
#=========

