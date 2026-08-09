"""
DataForge Entry Point

Wires together the upload, profiling, cleaning, quality-reassessment and
dashboard modules into a single Streamlit app:

    Upload -> Structural Cleaning -> Date Parsing -> Quality Reassessment -> Dashboard
"""

import streamlit as st

from modules.upload import validate_file, load_file
from modules.profiling import dataset_overview, missing_value_analysis, duplicate_rows_analysis
from modules.Cleaning import (
    remove_duplicates,
    remove_empty_rows,
    remove_sparse_columns,
    remove_whitespace,
    standardize_column_names,
    parse_dates,
)
from modules.Quality_Reassessment import reassess_quality
from modules.Dashboard import dashboard_summary


# ============================================================
# Session State
# ============================================================

def init_session_state() -> None:
    defaults = {
        "file_name": None,
        "raw_df": None,
        "structured_df": None,
        "cleaned_df": None,
        "quality_score": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_pipeline_state() -> None:
    """Clear everything downstream of upload when a new file arrives."""
    st.session_state["structured_df"] = None
    st.session_state["cleaned_df"] = None
    st.session_state["quality_score"] = None


# ============================================================
# Step 1: Upload
# ============================================================

def handle_upload() -> None:
    st.header("1. Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx"],
    )

    if uploaded_file is None:
        st.info("Upload a .csv or .xlsx file to get started.")
        return

    is_valid, message = validate_file(uploaded_file)

    if not is_valid:
        st.error(message)
        return

    # Only reload (and reset downstream progress) when the file actually
    # changes, otherwise every unrelated widget interaction elsewhere in
    # the app would re-trigger a reload and wipe out cleaning progress.
    if st.session_state["file_name"] != uploaded_file.name:
        try:
            df = load_file(uploaded_file)
        except Exception as e:
            st.error(f"Could not load file: {e}")
            return

        if df.empty:
            st.warning("The uploaded file contains no data.")
            return

        st.session_state["raw_df"] = df
        st.session_state["file_name"] = uploaded_file.name
        reset_pipeline_state()
        st.success(
            f"Loaded '{uploaded_file.name}' — "
            f"{df.shape[0]} rows, {df.shape[1]} columns."
        )

    show_raw_overview()


def show_raw_overview() -> None:
    df = st.session_state["raw_df"]
    if df is None:
        return

    with st.expander("Raw dataset overview", expanded=False):
        overview = dataset_overview(df)
        missing_df = missing_value_analysis(df)
        dup_info = duplicate_rows_analysis(df)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Rows", overview["num_rows"])
        col2.metric("Columns", overview["num_columns"])
        col3.metric("Memory", f"{overview['memory_usage']} MB")
        col4.metric(
            "Missing Values",
            int(missing_df["missing_count"].sum()) if not missing_df.empty else 0,
        )
        col5.metric("Duplicate Rows", dup_info["duplicate_count"])

        st.dataframe(df.head(10), width="stretch")


# ============================================================
# Step 2: Structural Cleaning (duplicates, empty rows, sparse
# columns, whitespace, column-name standardization)
# ============================================================

def handle_structural_cleaning() -> None:
    df = st.session_state["raw_df"]
    if df is None:
        return

    st.header("2. Clean Dataset")

    with st.form("structural_cleaning_form"):
        st.caption("Choose which cleaning steps to apply.")

        do_remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
        do_remove_empty_rows = st.checkbox("Remove fully empty rows", value=True)

        do_remove_sparse = st.checkbox(
            "Remove columns with too many missing values", value=False
        )
        sparse_threshold = st.slider(
            "Sparse column threshold (fraction missing)",
            min_value=0.5,
            max_value=1.0,
            value=0.9,
            step=0.05,
            disabled=not do_remove_sparse,
        )

        do_remove_whitespace = st.checkbox(
            "Trim whitespace from text columns", value=True
        )
        do_standardize_names = st.checkbox(
            "Standardize column names (lowercase, underscores)", value=True
        )

        submitted = st.form_submit_button("Apply Cleaning")

    if not submitted:
        return

    cleaned = df.copy()

    if do_remove_duplicates:
        cleaned = remove_duplicates(cleaned)
    if do_remove_empty_rows:
        cleaned = remove_empty_rows(cleaned)
    if do_remove_sparse:
        cleaned = remove_sparse_columns(cleaned, threshold=sparse_threshold)
    if do_remove_whitespace:
        cleaned = remove_whitespace(cleaned)
    if do_standardize_names:
        cleaned = standardize_column_names(cleaned)

    if cleaned.empty:
        st.error(
            "Cleaning removed every row or column. Try relaxing your "
            "settings (e.g. raise the sparse-column threshold)."
        )
        return

    st.session_state["structured_df"] = cleaned
    st.session_state["cleaned_df"] = None
    st.session_state["quality_score"] = None
    st.success(
        f"Structural cleaning complete — "
        f"{cleaned.shape[0]} rows, {cleaned.shape[1]} columns remain."
    )


# ============================================================
# Step 3: Date Parsing + Quality Reassessment
# ============================================================

def handle_date_parsing_and_quality() -> None:
    df = st.session_state["structured_df"]
    if df is None:
        return

    st.header("3. Parse Dates & Reassess Quality")

    with st.form("date_and_quality_form"):
        date_columns = st.multiselect(
            "Columns to parse as dates (optional)",
            options=list(df.columns),
        )

        st.caption(
            "Optional: define valid numeric ranges to flag out-of-range values."
        )
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        range_columns = st.multiselect(
            "Columns to range-check (optional)",
            options=numeric_columns,
        )

        column_ranges = {}
        for col in range_columns:
            col_min, col_max = st.columns(2)
            min_val = col_min.number_input(f"{col} — min", value=0.0, key=f"min_{col}")
            max_val = col_max.number_input(f"{col} — max", value=100.0, key=f"max_{col}")
            column_ranges[col] = (min_val, max_val)

        submitted = st.form_submit_button("Finalize & Reassess Quality")

    if not submitted:
        return

    final_df = parse_dates(df, date_columns) if date_columns else df.copy()

    quality_results = reassess_quality(
        final_df,
        date_columns=date_columns,
        column_ranges=column_ranges,
    )

    st.session_state["cleaned_df"] = final_df
    st.session_state["quality_score"] = quality_results["quality_score"]
    st.success(
        f"Quality reassessment complete — score: "
        f"{quality_results['quality_score']:.2f}/100"
    )


# ============================================================
# Step 4: Dashboard
# ============================================================

def handle_dashboard() -> None:
    df = st.session_state["cleaned_df"]
    quality_score = st.session_state["quality_score"]

    if df is None or quality_score is None:
        return

    st.divider()
    dashboard_summary(df, quality_score)


# ============================================================
# Main
# ============================================================

def main() -> None:
    st.set_page_config(page_title="DataForge", layout="wide")
    init_session_state()

    st.title("DataForge")
    st.caption("Upload, clean, and profile your dataset end to end.")

    handle_upload()
    handle_structural_cleaning()
    handle_date_parsing_and_quality()
    handle_dashboard()


if __name__ == "__main__":
    main()