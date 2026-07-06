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
    if df.empty:
        return pd.DataFrame()
    
    empty_string_count = (df.select_dtypes(include="object").apply(
        lambda col: col.str.strip().eq("").sum()))
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
#Section3: Validity (Date Validation, Range Validation, Constant Columns detection,
#                    Outlier Detection,)
#=========

# Date Validation
def date_validation(df: pd.DataFrame,date_columns: list[str]) -> pd.DataFrame:
    """
    Validate date columns by checking whether values
    can be converted into valid datetime objects.
    """

    results = []
    for column in date_columns:
        if column not in df.columns:
            continue

        # Ignore missing values
        values = df[column].dropna()
        parsed_dates = pd.to_datetime(values,errors="coerce", format="mixed")
        invalid_count = parsed_dates.isna().sum()
        checked_count = len(values)
        valid_percentage = (100 if checked_count == 0 else round((checked_count - invalid_count)
                                                         / checked_count * 100, 2,)
        )
        results.append(
            {
                "column_name": column,
                "checked_values": checked_count,    
                "invalid_dates": invalid_count,
                "valid_percentage": valid_percentage,
            }
        )

    return pd.DataFrame(results)

# range validation
def range_validation(df: pd.DataFrame, column_ranges: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """
    Validate numeric columns by checking whether values fall within specified ranges.

    Args:
        df: Input DataFrame.
        column_ranges: Dictionary where keys are column names and values are tuples of (min, max).

    Returns:
        pd.DataFrame: Range validation results with counts and percentages.
    """
    results = []
    for column, (min_val, max_val) in column_ranges.items():
        if column not in df.columns:
            continue

        numeric_values = pd.to_numeric(df[column], errors="coerce")
        invalid_numeric = numeric_values.isna().sum() - df[column].isna().sum()
        values = numeric_values.dropna()
        out_of_range = ((values < min_val) | (values > max_val)).sum()
        invalid_total = invalid_numeric + out_of_range
        checked_count = len(df[column].dropna())
        valid_percentage = (
            100 if checked_count == 0
            else round((checked_count - invalid_total) / checked_count * 100, 2)
        )
        results.append(
            {
                "column_name": column,
                "checked_values": checked_count,
                "invalid_numeric_values": invalid_numeric,
                "out_of_range_values": out_of_range,
                "valid_percentage": valid_percentage,
            }
        )

    return pd.DataFrame(results)

# Constant Columns detection
def detect_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect constant columns in the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Constant columns with their unique value.
    """
    constant_columns = []
    for column in df.columns:
        if df[column].nunique(dropna=False) == 1:
            constant_columns.append({
                "column_name": column,
                "unique_values": 1,
                "constant_value": df[column].iloc[0]
            })

    return pd.DataFrame(constant_columns)

# Outlier Detection
def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers in numeric columns using the IQR method.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Outlier detection results with counts and percentages.
    """
    numeric_columns = df.select_dtypes(include=["number"]).columns

    results = []
    for column in numeric_columns:
        values = df[column].dropna()

        if values.empty:
            continue
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_count = ((values < lower_bound) | (values > upper_bound)).sum()
        checked_count = len(values)
        outlier_percentage = (100 if checked_count == 0 else round((outlier_count / checked_count) * 100, 2))
        valid_percentage = round(100 - outlier_percentage, 2)
        
        results.append({
            "column_name": column,
            "checked_values": checked_count,
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_percentage,
            "valid_percentage": valid_percentage
        })

    return pd.DataFrame(results)


# ====Validity Score====
def calculate_validity_score(date_results: pd.DataFrame, range_results: pd.DataFrame, 
                             outlier_results: pd.DataFrame,) -> float:
    """
    Calculate overall validity score.
    """

    valid_scores = []

    for result in [date_results, range_results, outlier_results]:
        if not result.empty:
            valid_scores.extend(result["valid_percentage"].tolist())

    if not valid_scores:
        return 100.0

    return round(sum(valid_scores) / len(valid_scores), 2)

# =====QUALITY SCORE=====
def calculate_quality_score(completeness_score: float, uniqueness_score: float, 
                            validity_score: float) -> float:
    """Overall quality score = (completeness + uniqueness + validity) / 3
    Args:
        completeness_score: Completeness score (0–100).
        uniqueness_score: Uniqueness score (0–100).
        validity_score: Validity score (0–100).

    Returns:
        float: Overall quality score (0–100)."""

    return round((completeness_score + uniqueness_score + validity_score) / 3, 2)
