# ======= DataForge Dashboard Module =======

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from modules.profiling import (
    dataset_overview,
    duplicate_rows_analysis,
    missing_value_analysis
)


# ============================================================
# 1. Dataset Summary Cards
# ============================================================

def dataset_summary_cards(df: pd.DataFrame,quality_score: float) -> None:
    """
    Display high-level summary metrics for the dataset.

    Args:
        df: Input DataFrame.
        quality_score: Overall quality score of the dataset.
    """

    if df.empty:
        st.warning("The uploaded dataset is empty.")
        return

    overview = dataset_overview(df)

    rows = overview["num_rows"]
    columns = overview["num_columns"]
    memory = overview["memory_usage"]

    missing_df = missing_value_analysis(df)

    missing_values = (
        int(missing_df["missing_count"].sum())
        if not missing_df.empty
        else 0
    )

    duplicate_rows = duplicate_rows_analysis(df)["duplicate_count"]

    st.subheader("Dataset Summary")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Rows", rows)
    col2.metric("Columns", columns)
    col3.metric("Memory", f"{memory} MB")
    col4.metric("Missing Values", missing_values)
    col5.metric("Quality Score", f"{quality_score:.2f}/100")
    col6.metric("Duplicate Rows", duplicate_rows)

    status = (
        "Excellent"
        if quality_score >= 90
        else "Good"
        if quality_score >= 70
        else "Needs Attention"
    )

    if quality_score >= 90:
        st.success(f"Dataset status: {status}")
    elif quality_score >= 70:
        st.warning(f"Dataset status: {status}")
    else:
        st.error(f"Dataset status: {status}")


# ============================================================
# 2. Data Preview
# ============================================================

def display_data_preview(df: pd.DataFrame, num_rows: int = 5) -> None:
    """
    Display a configurable preview of the dataset.

    Args:
        df: Input DataFrame.
        num_rows: Default number of rows to display.
    """

    if df.empty:
        st.warning("The uploaded dataset is empty.")
        return

    st.subheader("Data Preview")

    max_rows = min(50, len(df))

    if max_rows <= 1:
        # st.slider raises when min_value == max_value, so skip it
        # entirely for single-row datasets and just show the row.
        rows = max_rows
    else:
        rows = st.slider(
            "Preview Rows",
            min_value=1,
            max_value=max_rows,
            value=min(num_rows, max_rows),
            key="preview_rows"
        )

    st.dataframe(
        df.head(rows),
        width="stretch"
    )

# ============================================================
# 3. Missing Values Visualization
# ============================================================

def plot_missing_values(df: pd.DataFrame) -> None:
    """
    Display missing values by column.
    """

    if df.empty:
        st.warning("The uploaded dataset is empty.")
        return

    st.subheader("Missing Value Distribution")

    missing_df = missing_value_analysis(df)

    if missing_df.empty:
        st.success("No missing values found in the dataset.")
        return

    st.caption("Missing values per column")

    st.bar_chart(
        missing_df.set_index("column_name")["missing_count"],
        width="stretch"
    )


# ============================================================
# 4. Numeric Distributions
# ============================================================

def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """
    Display a histogram for a selected numeric column.
    """

    if df.empty:
        st.warning("The uploaded dataset is empty.")
        return

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        st.info("No numeric columns available.")
        return

    st.subheader("Numeric Column Distributions")

    selected_column = st.selectbox(
        "Select Numeric Column",
        numeric_df.columns,
        key="numeric_distribution"
    )

    series = numeric_df[selected_column].dropna()

    if series.empty:
        st.info("The selected column contains no usable values.")
        return

    if series.nunique() <= 1:
        st.info(
            "Selected column has insufficient variation "
            "for a distribution."
        )
        return

    bins = st.slider(
        "Number of Bins",
        min_value=5,
        max_value=50,
        value=20,
        key="bins_slider"
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(
        series,
        bins=bins,
        edgecolor="black"
    )

    ax.set_title(
        f"Distribution of {selected_column}"
    )
    ax.set_xlabel(selected_column)
    ax.set_ylabel("Frequency")

    plt.tight_layout()

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)

    st.caption("Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Mean",
        f"{series.mean():.2f}"
    )

    col2.metric(
        "Median",
        f"{series.median():.2f}"
    )

    col3.metric(
        "Std Dev",
        f"{series.std():.2f}"
    )

    col4.metric(
        "Min",
        f"{series.min():.2f}"
    )

    col5, col6, col7 = st.columns(3)

    col5.metric(
        "25%",
        f"{series.quantile(0.25):.2f}"
    )

    col6.metric(
        "75%",
        f"{series.quantile(0.75):.2f}"
    )

    col7.metric(
        "Max",
        f"{series.max():.2f}"
    )


# ============================================================
# 5. Correlation Heatmap
# ============================================================

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Display a correlation heatmap for numeric columns.
    """

    if df.empty:
        st.warning("The uploaded dataset is empty.")
        return

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        st.info(
            "At least two numeric columns are required "
            "for correlation analysis."
        )
        return

    st.subheader("Correlation Heatmap")

    correlation = numeric_df.corr()

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    image = ax.imshow(
        correlation,
        aspect="auto",
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))

    ax.set_xticklabels(
        correlation.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        correlation.columns
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Correlation"
    )

    ax.set_title("Numeric Feature Correlations")

    plt.tight_layout()

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)


# ============================================================
# 6. Boxplots
# ============================================================

def plot_boxplots(df: pd.DataFrame) -> None:
    """
    Display a boxplot for a selected numeric column.
    """

    if df.empty:
        st.warning("The uploaded dataset is empty.")
        return

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        st.info("No numeric columns available.")
        return

    st.subheader("Boxplot Analysis")

    selected_column = st.selectbox(
        "Select Numeric Column",
        numeric_df.columns,
        key="boxplot_column"
    )

    series = numeric_df[selected_column].dropna()

    if series.empty:
        st.info("The selected column contains no usable values.")
        return

    fig, ax = plt.subplots(
        figsize=(8, 3)
    )

    ax.boxplot(
        series,
        vert=False
    )

    ax.set_title(
        f"Boxplot of {selected_column}"
    )

    ax.set_xlabel(selected_column)

    plt.tight_layout()

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)


# ============================================================
# 7. Categorical Counts
# ============================================================

def plot_categorical_counts(df: pd.DataFrame) -> None:
    """
    Display value counts for a selected categorical column.
    """

    if df.empty:
        st.warning("The uploaded dataset is empty.")
        return

    categorical_df = df.select_dtypes(
        include=["object", "category", "bool"]
    )

    if categorical_df.empty:
        st.info("No categorical columns available.")
        return

    st.subheader("Categorical Value Distribution")

    selected_column = st.selectbox(
        "Select Categorical Column",
        categorical_df.columns,
        key="categorical_column"
    )

    value_counts = (
        categorical_df[selected_column]
        .value_counts()
        .head(20)
    )

    if value_counts.empty:
        st.info("No categorical values available.")
        return

    st.caption("Top 20 categories")

    st.bar_chart(
        value_counts,
        width="stretch"
    )


# ============================================================
# 8. Datetime Trends
# ============================================================

def plot_datetime_trends(df: pd.DataFrame) -> None:
    """
    Display a simple trend chart for a selected datetime column.
    """

    if df.empty:
        st.warning("The uploaded dataset is empty.")
        return

    datetime_df = df.select_dtypes(
        include=["datetime", "datetimetz"]
    )

    if datetime_df.empty:
        st.info("No datetime columns available.")
        return

    st.subheader("Datetime Trends")

    selected_column = st.selectbox(
        "Select Datetime Column",
        datetime_df.columns,
        key="datetime_column"
    )

    trend = (
        df[selected_column]
        .dropna()
        .dt.date
        .value_counts()
        .sort_index()
    )

    if trend.empty:
        st.info("No valid datetime values available.")
        return

    st.caption("Number of records over time")

    st.line_chart(
        trend,
        width="stretch"
    )


# ============================================================
# 9. Download Clean Dataset
# ============================================================

def download_clean_dataset(df: pd.DataFrame, file_name: str = "dataforge_cleaned_dataset.csv") -> None:
    """
    Provide a download button for the cleaned dataset.
    """

    if df.empty:
        st.warning("There is no data available for download.")
        return

    st.subheader("Download Dataset")

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Clean Dataset",
        data=csv_data,
        file_name=file_name,
        mime="text/csv",
        width="stretch"
    )


# ============================================================
# 10. Dashboard Summary
# ============================================================

def dashboard_summary(df: pd.DataFrame, quality_score: float) -> None:
    """
    Display the complete DataForge dashboard.
    """

    if df.empty:
        st.warning("The uploaded dataset is empty.")
        return

    st.title("DataForge Dashboard")

    st.caption(
        "Automated exploratory analysis of your processed dataset."
    )

    # Summary
    dataset_summary_cards(
        df,
        quality_score
    )

    st.divider()

    # Preview
    display_data_preview(df)

    st.divider()

    # Missing values
    plot_missing_values(df)

    st.divider()

    # Numeric analysis
    plot_numeric_distributions(df)

    st.divider()

    # Correlations
    plot_correlation_heatmap(df)

    st.divider()

    # Outliers
    plot_boxplots(df)

    st.divider()

    # Categorical analysis
    plot_categorical_counts(df)

    st.divider()

    # Time analysis
    plot_datetime_trends(df)

    st.divider()

    # Download
    download_clean_dataset(df)