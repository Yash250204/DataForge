# modules/data_loading.py

import re
import numpy as np  
import pandas as pd
import mysql.connector
from mysql.connector import Error
from pandas.api.types import (is_integer_dtype, is_float_dtype, is_bool_dtype, is_datetime64_any_dtype)

# =========================================================
# Identifier Validation (basic SQL injection protection)
# =========================================================

def validate_identifier(name: str) -> str:
    """
    Validate SQL identifiers such as database and table names.

    Allowed:
    - letters
    - numbers
    - underscore
    """

    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError(f"Invalid SQL identifier: {name}")

    # MySQL identifiers (table/column/database names) are capped at 64
    # characters; catch this here with a clear message instead of letting
    # it fail later with a cryptic MySQL error.

    if len(name) > 64:
        raise ValueError(f"Invalid SQL identifier: '{name}' exceeds 64 character limit.")

    return name

def _quote_identifier(name: str) -> str:
    """
    Safely prepare a column name for backtick-quoting.

    Column names come straight from arbitrary, user-uploaded DataFrame
    headers, so unlike table/database names (validated via
    validate_identifier's strict allowlist) they aren't restricted to
    plain alphanumerics -- spaces and symbols are fine once quoted.
    The one character that isn't safe is a literal backtick, since it
    can break out of the `col` quoting and inject arbitrary SQL into
    the surrounding CREATE TABLE / INSERT statement. MySQL's own
    convention for a literal backtick inside a quoted identifier is to
    double it, so that's what this does.
    """

    name = str(name)

    if not name:
        raise ValueError("Column name cannot be empty.")

    return name.replace("`", "``")

def _sanitize_value(value):
    """
    Convert a single DataFrame cell into something
    mysql-connector-python can bind as a query parameter.

    Cleaned DataFrames routinely contain missing values (NaN in
    numeric columns, NaT in datetime columns after parse_dates()
    coerces bad entries, None/pd.NA in object columns) and numpy/pandas
    scalar types (np.int64, np.float64, pd.Timestamp) that the
    connector doesn't know how to format -- confirmed empirically:
    inserting a row with a plain NaN raises
    "Failed processing format-parameters; Unknown format code 'd' for
    object of type 'float'". Without this, load_data_into_table()
    fails on almost any realistically-cleaned dataset.
    """
    if pd.isna(value):
        return None
    if isinstance (value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# =========================================================
# Step 1: Connect to MySQL Server (no database selected)
# =========================================================

def create_server_connection(host_name: str, user_name: str, user_password: str, port: int = 3306) -> mysql.connector.MySQLConnection:
    """
    Create a connection to the MySQL server.
    """

    try:
        return mysql.connector.connect(
            host=host_name,
            user=user_name,
            password=user_password,
            port=port,
        )

    except Error as e:
        raise ConnectionError(f"Failed to connect to MySQL server: {e}")


# =========================================================
# Step 2: Create Database
# =========================================================

def create_database(connection: mysql.connector.MySQLConnection, db_name: str) -> None:
    """
    Create database if it does not exist.
    """

    db_name = validate_identifier(db_name)

    cursor = connection.cursor()

    try:
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
        connection.commit()

    except Error as e:
        raise RuntimeError(f"Failed to create database '{db_name}': {e}")

    finally:
        cursor.close()


# =========================================================
# Step 3: Connect to Specific Database
# =========================================================

def create_database_connection(host_name: str, user_name: str, user_password: str,
                                db_name: str, port: int = 3306) -> mysql.connector.MySQLConnection:
    """
    Create a connection to a specific MySQL database.
    """

    db_name = validate_identifier(db_name)

    try:
        return mysql.connector.connect(
            host=host_name,
            user=user_name,
            password=user_password,
            database=db_name,
            port=port,
        )

    except Error as e:
        raise ConnectionError(
            f"Failed to connect to database '{db_name}': {e}"
        )


# =========================================================
# Step 4: Close Connection
# =========================================================

def close_connection(connection: mysql.connector.MySQLConnection) -> None:
    """
    Close MySQL connection safely.
    """

    if connection and connection.is_connected():
        connection.close()


# =========================================================
# Step 5: Pandas dtype → MySQL dtype mapping
# =========================================================

def get_mysql_type(series: pd.Series) -> str:
    """
    Map pandas data types to MySQL data types.
    """

    if is_integer_dtype(series):
        # Signed INT tops out around 2.147 billion. Large IDs or
        # millisecond timestamps commonly exceed that, so fall back to
        # BIGINT rather than silently overflowing/truncating on insert

        max_abs = series.abs().max() if not series.empty else 0
        if pd.isna(max_abs) or max_abs > 2_147_483_647:
            return "BIGINT"
        return "INT"

    if is_float_dtype(series):
        return "FLOAT"

    if is_bool_dtype(series):
        return "BOOLEAN"

    if is_datetime64_any_dtype(series):
        return "DATETIME"

    # A boolean column containing a null (e.g. [True, False, None]) is
    # stored as object dtype, not native bool, since bool can't hold
    # nulls. is_bool_dtype() above only matches native bool, so without
    # this check such a column would fall through to VARCHAR/TEXT below.
    non_null_values = series.dropna()
    if not non_null_values.empty and non_null_values.map(lambda x: isinstance(x, bool)).all():
        return "BOOLEAN"

    # Text/object columns: a fixed VARCHAR(255) truncates (or errors,
    # depending on SQL mode) on any value longer than that, so fall back
    # to TEXT for columns containing longer strings.

    non_null = series.dropna()
    max_len = non_null.astype(str).map(len).max() if not non_null.empty else 0
    if pd.isna(max_len) or max_len > 255:
        return "TEXT"
    return "VARCHAR(255)"


# =========================================================
# Step 6: Create Table from DataFrame
# =========================================================

def create_table_from_dataframe(connection: mysql.connector.MySQLConnection,
                                 df: pd.DataFrame, table_name: str) -> None:
    """
    Automatically create a MySQL table based on DataFrame schema.
    """

    table_name = validate_identifier(table_name)

    columns = []

    for col in df.columns:
        sql_type = get_mysql_type(df[col])
        columns.append(f"`{_quote_identifier(col)}` {sql_type}")

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS `{table_name}` (
        {', '.join(columns)}
    )
    """

    cursor = connection.cursor()

    try:
        cursor.execute(create_sql)
        connection.commit()

    except Error as e:
        raise RuntimeError(f"Failed to create table '{table_name}': {e}")

    finally:
        cursor.close()


# =========================================================
# Step 7: Load Data into Table
# =========================================================

def load_data_into_table(connection: mysql.connector.MySQLConnection, df: pd.DataFrame, 
                         table_name: str, if_exists: str = "append") -> None:
    """
    Load DataFrame into MySQL table.

    Parameters:
    - append: add rows
    - replace: drop and recreate table
    """

    table_name = validate_identifier(table_name)

    cursor = connection.cursor()

    try:
        if if_exists not in {"append", "replace"}:
            raise ValueError("if_exists must be 'append' or 'replace'")

        if if_exists == "replace":
            cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")
            connection.commit()
            create_table_from_dataframe(connection, df, table_name)

        columns = ", ".join([f"`{_quote_identifier(c)}`" for c in df.columns])
        placeholders = ", ".join(["%s"] * len(df.columns))

        insert_sql = f"""
        INSERT INTO `{table_name}` ({columns})
        VALUES ({placeholders})
        """

        rows = [
            tuple(_sanitize_value(v) for v in row)
            for row in df.itertuples(index=False, name=None)
        ]

        cursor.executemany(insert_sql, rows)
        connection.commit()

    except Error as e:
        raise RuntimeError(f"Failed to load data into table '{table_name}': {e}")

    finally:
        cursor.close()


# =========================================================
# Step 8: Verify Upload
# =========================================================

def verify_upload(connection: mysql.connector.MySQLConnection, table_name: str) -> int:
    """
    Verify that rows were inserted successfully.
    """

    table_name = validate_identifier(table_name)

    cursor = connection.cursor()

    try:
        cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
        rows_loaded = cursor.fetchone()[0]

        if rows_loaded == 0:
            raise ValueError(
                f"Table '{table_name}' exists but contains no rows."
            )

        return rows_loaded

    except Error as e:
        raise RuntimeError(f"Failed to verify upload for '{table_name}': {e}")

    finally:
        cursor.close()


# =========================================================
# Step 9: Loading Summary
# =========================================================

def loading_summary(connection: mysql.connector.MySQLConnection, table_name: str) -> dict[str, str | int]:
    """
    Return loading summary information.
    """

    table_name = validate_identifier(table_name)

    cursor = connection.cursor()

    try:
        cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
        row_count = cursor.fetchone()[0]

        cursor.execute(f"DESCRIBE `{table_name}`")
        columns = cursor.fetchall()

        return {
            "database": connection.database,
            "table": table_name,
            "rows_loaded": row_count,
            "columns_loaded": len(columns),
            "status": "Success",
        }

    except Error as e:
        raise RuntimeError(f"Failed to generate loading summary: {e}")

    finally:
        cursor.close()