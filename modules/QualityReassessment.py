# ========Reassessment Module========
import pandas as pd
from modules.profiling import (dataset_overview, column_info, missing_value_analysis, data_type_distribution,
                                statistical_summary, duplicate_rows_analysis, sample_preview)
from modules.quality import (empty_string_analysis, calculate_completeness_score, calculate_uniqueness_score,
                            date_validation, range_validation, detect_constant_columns, detect_outliers,
                            calculate_validity_score, calculate_quality_score)
# ========Re-Profiling=========

def reassess_profiling(df: pd.DataFrame) -> dict:
    """
    Reassess the profiling of the dataset after cleaning.

    Args:
        df: Input DataFrame.

    Returns:
        dict: A dictionary containing reassessed profiling information.
    """
    return {
        "dataset_overview": dataset_overview(df),
        "column_info": column_info(df),
        "missing_value_analysis": missing_value_analysis(df),
        "data_type_distribution": data_type_distribution(df),
        "statistical_summary": statistical_summary(df),
        "duplicate_rows_analysis": duplicate_rows_analysis(df),
        "sample_preview": sample_preview(df)
    }

# ======Re-Quality Assessment========
def reassess_quality(df: pd.DataFrame, date_columns: list[str], 
                     column_ranges: dict[str, tuple[float, float]]) -> dict:
    """
    Reassess the quality of the dataset after cleaning.

    Args:
    df: Cleaned DataFrame.
    date_columns: List of date columns.
    column_ranges: Dictionary of valid numeric ranges.

    Returns:
        dict: A dictionary containing reassessed quality metrics.
    """

    date_results = date_validation(df, date_columns)
    range_results = range_validation(df, column_ranges)
    outlier_results = detect_outliers(df) 

    completeness_score = calculate_completeness_score(df)
    uniqueness_score = calculate_uniqueness_score(df)
    validity_score = calculate_validity_score(date_results, range_results, outlier_results)

    quality_score = calculate_quality_score(completeness_score, uniqueness_score, validity_score)

    return {
        "empty_string_analysis": empty_string_analysis(df),
        "completeness_score": completeness_score,
        "uniqueness_score": uniqueness_score,
        "date_validation": date_results,
        "range_validation": range_results,
        "constant_columns": detect_constant_columns(df),
        "outlier_detection": outlier_results,
        "validity_score": validity_score,
        "quality_score": quality_score,
    }


