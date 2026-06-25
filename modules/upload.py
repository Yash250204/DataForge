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