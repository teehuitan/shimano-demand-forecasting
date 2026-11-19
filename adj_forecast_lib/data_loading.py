"""Code cleaning shared by the order loaders."""
import numpy as np
import pandas as pd


def clean_code(series: pd.Series) -> pd.Series:
    """Uppercase and strip non-alphanumerics from codes; NaN stays NaN."""
    cleaned = series.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    return cleaned.where(series.notna(), np.nan)
