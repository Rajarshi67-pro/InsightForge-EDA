"""
Data Profiler Module for InsightForge AI
Classifies column semantic types and extracts core dataset metadata.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class ColumnType(str, Enum):
    IDENTIFIER = "Identifier"
    NUMERICAL = "Numerical"
    CATEGORICAL = "Categorical"
    DATE = "Date"
    BOOLEAN = "Boolean"
    TEXT = "Text"


class DataProfiler:
    """
    Analyzes DataFrame structures and classifies columns into semantic types.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.total_rows = len(df)
        self.total_cols = len(df.columns)
        self.column_profiles: Dict[str, Dict[str, Any]] = {}
        self.summary: Dict[str, Any] = {}
        self._profile()

    def _is_date_series(self, series: pd.Series, col_name: str) -> bool:
        """Determines if a column represents a date or timestamp."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        name_lower = str(col_name).lower()
        date_keywords = ["date", "time", "timestamp", "year", "month", "day", "created_at", "updated_at", "dob", "joining_date", "period"]
        
        non_null_sample = series.dropna().astype(str).head(30)
        if len(non_null_sample) == 0:
            return False

        # If keyword matches, test if 60% parses
        if any(keyword in name_lower for keyword in date_keywords):
            try:
                parsed = pd.to_datetime(non_null_sample, errors="coerce", format="mixed")
                if parsed.notna().sum() / len(non_null_sample) >= 0.6:
                    return True
            except Exception:
                pass
                
        # Even without keyword, test if 80% parses as datetime (avoid simple ints)
        if not pd.api.types.is_numeric_dtype(series):
            try:
                parsed = pd.to_datetime(non_null_sample, errors="coerce", format="mixed")
                if parsed.notna().sum() / len(non_null_sample) >= 0.8:
                    return True
            except Exception:
                pass

        return False

    def _is_identifier(self, series: pd.Series, col_name: str) -> bool:
        """Determines if a column is an ID or primary key."""
        name_lower = str(col_name).lower()
        id_keywords = ["_id", "id", "guid", "uuid", "key", "code", "employee_id", "customer_id", "user_id", "pk"]
        
        if any(name_lower == kw or name_lower.endswith(kw) or name_lower.startswith(kw) for kw in id_keywords):
            return True

        non_null = series.dropna()
        if len(non_null) > 10 and non_null.nunique() == len(non_null):
            if not pd.api.types.is_float_dtype(series):
                return True
                
        # Regex UUID check on sample
        if len(non_null) > 0 and pd.api.types.is_string_dtype(series):
            sample = non_null.head(10).astype(str)
            if sample.str.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', case=False).all():
                return True

        return False

    def _is_boolean_series(self, series: pd.Series) -> bool:
        """Checks if column contains boolean or binary flag values."""
        if pd.api.types.is_bool_dtype(series):
            return True
        unique_vals = set(series.dropna().astype(str).str.lower().unique())
        if unique_vals in [{'true', 'false'}, {'0', '1'}, {'yes', 'no'}, {'y', 'n'}, {'t', 'f'}]:
            return True
        return False

    def _is_text_series(self, series: pd.Series) -> bool:
        """Classifies column as Text rather than Categorical based on length/words."""
        if not pd.api.types.is_string_dtype(series) and not pd.api.types.is_object_dtype(series):
            return False
            
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
            
        avg_len = sample.str.len().mean()
        avg_words = sample.str.split().str.len().mean()
        
        # Heuristic: avg length > 40 OR avg words > 5
        if avg_len > 40 or avg_words > 5:
            return True
            
        return False

    def classify_column(self, col: str) -> ColumnType:
        """Classifies a specific column into its semantic data type."""
        series = self.df[col]
        
        if self._is_identifier(series, col):
            return ColumnType.IDENTIFIER
        elif self._is_date_series(series, col):
            return ColumnType.DATE
        elif self._is_boolean_series(series):
            return ColumnType.BOOLEAN
        elif pd.api.types.is_numeric_dtype(series):
            return ColumnType.NUMERICAL
        elif self._is_text_series(series):
            return ColumnType.TEXT
        else:
            return ColumnType.CATEGORICAL

    def _profile(self) -> None:
        """Executes the profiling pipeline."""
        type_counts = {t.value: 0 for t in ColumnType}
        columns_info = []

        for col in self.df.columns:
            series = self.df[col]
            semantic_type = self.classify_column(col)
            type_counts[semantic_type.value] += 1
            
            missing_cnt = int(series.isna().sum())
            missing_pct = round((missing_cnt / self.total_rows) * 100, 2) if self.total_rows > 0 else 0.0
            unique_cnt = int(series.nunique(dropna=True))
            
            # Sample values
            samples = [str(x) for x in series.dropna().head(3).tolist()]
            
            col_info = {
                "column_name": str(col),
                "semantic_type": semantic_type.value,
                "pandas_dtype": str(series.dtype),
                "missing_count": missing_cnt,
                "missing_pct": missing_pct,
                "unique_count": unique_cnt,
                "sample_values": samples,
            }
            self.column_profiles[str(col)] = col_info
            columns_info.append(col_info)

        # Memory usage in MB
        memory_mb = round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 3)
        duplicate_rows = int(self.df.duplicated().sum())
        total_missing = int(self.df.isna().sum().sum())
        missing_cells_pct = round((total_missing / (self.total_rows * self.total_cols)) * 100, 2) if (self.total_rows * self.total_cols) > 0 else 0.0

        self.summary = {
            "total_rows": self.total_rows,
            "total_cols": self.total_cols,
            "memory_mb": memory_mb,
            "duplicate_rows": duplicate_rows,
            "total_missing_cells": total_missing,
            "missing_cells_pct": missing_cells_pct,
            "type_counts": type_counts,
            "columns": columns_info,
        }

    def get_columns_by_type(self, semantic_type: ColumnType) -> List[str]:
        """Returns column names matching the specified semantic type."""
        return [
            col for col, info in self.column_profiles.items()
            if info["semantic_type"] == semantic_type.value
        ]

    def to_dict(self) -> Dict[str, Any]:
        return self.summary
