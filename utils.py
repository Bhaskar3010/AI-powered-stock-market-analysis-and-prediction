"""
Infrastructure Support Utilities
Provides structural conversion mechanisms and numerical text formatting helpers.
"""
import pandas as pd

def transform_dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """
    Transforms internal application matrix states cleanly to utf-8 encoded system byte streams.
    """
    if df.empty:
        return b""
    return df.to_csv(index=False).encode('utf-8')

def format_large_numbers(num: float) -> str:
    """Converts large financial figures into easily readable numeric strings."""
    if num is None or pd.isna(num):
        return "N/A"
    if num >= 1e11:
        return f"₹{num / 1e11:.2f} L Cr"
    if num >= 1e7:
        return f"₹{num / 1e7:.2f} Cr"
    return f"₹{num:,.2f}"