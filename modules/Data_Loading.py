# modules/data_loading.py

import re
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
    - must not start with a number
    """

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Invalid SQL identifier: {name}")

    return name


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
        return "INT"

    if is_float_dtype(series):
        return "FLOAT"

    if is_bool_dtype(series):
        return "BOOLEAN"

    if is_datetime64_any_dtype(series):
        return "DATETIME"

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
        columns.append(f"`{col}` {sql_type}")

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

        columns = ", ".join([f"`{c}`" for c in df.columns])
        placeholders = ", ".join(["%s"] * len(df.columns))

        insert_sql = f"""
        INSERT INTO `{table_name}` ({columns})
        VALUES ({placeholders})
        """

        rows = list(df.itertuples(index=False, name=None))

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