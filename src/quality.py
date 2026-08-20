"""
Data Quality Engine for InsightForge AI
Performs missing value audits, duplicate row detection, deterministic outlier analysis (IQR & Z-score),
and provides automated data cleaning utilities.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


class DataQualityEngine:
    """
    Evaluates dataset quality and detects data cleanliness anomalies deterministically.
    """

    def __init__(self, df: pd.DataFrame, profiler_summary: Optional[Dict[str, Any]] = None):
        self.df = df.copy()
        self.profiler_summary = profiler_summary
        self.total_rows = len(df)
        self.quality_report: Dict[str, Any] = {}
        self._audit()

    def _detect_iqr_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Calculates outliers using the standard Interquartile Range (IQR) method."""
        clean_series = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if len(clean_series) < 4:
            return {
                "outlier_count": 0,
                "outlier_pct": 0.0,
                "potential_severity": "None",
                "q1": None,
                "q3": None,
                "iqr": None,
                "lower_bound": None,
                "upper_bound": None,
                "outlier_indices": [],
                "outlier_values": [],
            }

        q1 = float(clean_series.quantile(0.25))
        q3 = float(clean_series.quantile(0.75))
        iqr = float(q3 - q1)
        lower_bound = float(q1 - 1.5 * iqr)
        upper_bound = float(q3 + 1.5 * iqr)

        outlier_mask = (clean_series < lower_bound) | (clean_series > upper_bound)
        outlier_series = clean_series[outlier_mask]
        outlier_count = int(len(outlier_series))
        outlier_pct = round((outlier_count / len(clean_series)) * 100, 2)

        severity = "Low"
        if outlier_pct == 0:
            severity = "None"
        elif outlier_pct > 5:
            severity = "High"
        elif outlier_pct >= 1:
            severity = "Medium"

        return {
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
            "potential_severity": severity,
            "q1": round(q1, 3),
            "q3": round(q3, 3),
            "iqr": round(iqr, 3),
            "lower_bound": round(lower_bound, 3),
            "upper_bound": round(upper_bound, 3),
            "outlier_indices": list(outlier_series.index),
            "outlier_values": [round(float(v), 3) for v in outlier_series.tolist()],
        }

    def _detect_zscore_outliers(self, series: pd.Series, threshold: float = 3.0) -> Dict[str, Any]:
        """Calculates outliers using Z-score method (|z| > threshold)."""
        clean_series = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if len(clean_series) < 4 or clean_series.std() == 0:
            return {
                "outlier_count": 0, 
                "outlier_pct": 0.0, 
                "potential_severity": "None",
                "threshold": threshold,
                "lower_bound": None,
                "upper_bound": None,
                "outlier_indices": [],
                "outlier_values": []
            }

        mean = clean_series.mean()
        std = clean_series.std()
        
        lower_bound = mean - (threshold * std)
        upper_bound = mean + (threshold * std)
        
        z_scores = (clean_series - mean) / std
        outliers = clean_series[np.abs(z_scores) > threshold]
        
        outlier_count = int(len(outliers))
        outlier_pct = round((outlier_count / len(clean_series)) * 100, 2)

        severity = "Low"
        if outlier_pct == 0:
            severity = "None"
        elif outlier_pct > 5:
            severity = "High"
        elif outlier_pct >= 1:
            severity = "Medium"

        return {
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
            "potential_severity": severity,
            "threshold": threshold,
            "mean": round(float(mean), 3),
            "std": round(float(std), 3),
            "lower_bound": round(float(lower_bound), 3),
            "upper_bound": round(float(upper_bound), 3),
            "outlier_indices": list(outliers.index),
            "outlier_values": [round(float(v), 3) for v in outliers.tolist()],
        }

    def _audit(self) -> None:
        """Runs the complete data quality inspection."""
        # 1. Missing values
        missing_by_col = []
        for col in self.df.columns:
            m_cnt = int(self.df[col].isna().sum())
            if m_cnt > 0:
                missing_by_col.append({
                    "column": str(col),
                    "missing_count": m_cnt,
                    "missing_pct": round((m_cnt / self.total_rows) * 100, 2) if self.total_rows > 0 else 0.0,
                })
        missing_by_col.sort(key=lambda x: x["missing_count"], reverse=True)

        # 2. Duplicate rows
        dup_count = int(self.df.duplicated().sum())
        dup_pct = round((dup_count / self.total_rows) * 100, 2) if self.total_rows > 0 else 0.0
        dup_indices = list(self.df[self.df.duplicated(keep=False)].index)

        # 3. Constant Columns & Mixed Types
        constant_columns = []
        mixed_type_columns = []
        for col in self.df.columns:
            # Check constant (only 1 unique value)
            if self.df[col].nunique(dropna=True) == 1:
                constant_columns.append(str(col))
            # Check mixed type
            inferred = pd.api.types.infer_dtype(self.df[col], skipna=True)
            if inferred == "mixed" or inferred == "mixed-integer":
                mixed_type_columns.append(str(col))

        # 4. Numeric Outliers (IQR + Z-Score)
        outliers_by_col = {}
        total_outliers_found = 0

        # Look at numeric columns
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Skip identifier columns if identifiable
                col_name_lower = str(col).lower()
                if col_name_lower.endswith("id") or "_id" in col_name_lower:
                    continue

                iqr_res = self._detect_iqr_outliers(self.df[col])
                z_res = self._detect_zscore_outliers(self.df[col])
                
                if iqr_res["outlier_count"] > 0 or z_res["outlier_count"] > 0:
                    outliers_by_col[str(col)] = {
                        "iqr": iqr_res,
                        "z_score": z_res,
                    }
                    total_outliers_found += iqr_res["outlier_count"]

        # 5. Transparent Scoring Methodology
        total_cells = max(1, self.total_rows * len(self.df.columns))
        missing_cells = int(self.df.isna().sum().sum())
        
        # Calculate Penalties (Max bounds based on user plan)
        missing_penalty = round(min(30.0, (missing_cells / total_cells) * 60), 1)  # Scale up ratio for penalty
        dup_penalty = round(min(25.0, (dup_count / max(1, self.total_rows)) * 50), 1)
        outlier_penalty = round(min(15.0, (total_outliers_found / max(1, self.total_rows)) * 30), 1)
        constant_penalty = round(min(15.0, (len(constant_columns) / max(1, len(self.df.columns))) * 30), 1)
        mixed_type_penalty = round(min(15.0, (len(mixed_type_columns) / max(1, len(self.df.columns))) * 30), 1)
        
        # If dataset is extremely small or empty, avoid over-penalizing
        if self.total_rows == 0:
            health_score = 0.0
        else:
            total_penalty = missing_penalty + dup_penalty + outlier_penalty + constant_penalty + mixed_type_penalty
            health_score = max(0.0, round(100.0 - total_penalty, 1))

        # 6. Generate Explanations and Recommendations
        explanations = []
        recommendations = []
        
        if missing_penalty > 0:
            explanations.append(f"-{missing_penalty} pts: {missing_cells} missing values ({round((missing_cells/total_cells)*100, 1)}% of all data).")
            recommendations.append("Run the 1-Click Auto-Clean to automatically impute missing values.")
        if dup_penalty > 0:
            explanations.append(f"-{dup_penalty} pts: {dup_count} duplicate rows detected.")
            recommendations.append("Remove duplicate rows to prevent statistical bias.")
        if outlier_penalty > 0:
            explanations.append(f"-{outlier_penalty} pts: {total_outliers_found} extreme outliers found across numeric columns.")
            recommendations.append("Use IQR capping to normalize extreme outliers.")
        if constant_penalty > 0:
            explanations.append(f"-{constant_penalty} pts: {len(constant_columns)} constant columns (zero variance).")
            recommendations.append(f"Consider dropping constant columns: {', '.join(constant_columns)}.")
        if mixed_type_penalty > 0:
            explanations.append(f"-{mixed_type_penalty} pts: {len(mixed_type_columns)} columns with mixed/invalid data types.")
            recommendations.append(f"Clean columns with mixed types to ensure accurate aggregations: {', '.join(mixed_type_columns)}.")

        if health_score == 100:
            explanations.append("Perfect score: No anomalies detected.")
            recommendations.append("Dataset is in excellent health and ready for Machine Learning.")

        score_breakdown = {
            "overall_score": health_score,
            "penalties": {
                "missing": missing_penalty,
                "duplicates": dup_penalty,
                "outliers": outlier_penalty,
                "constant_columns": constant_penalty,
                "inconsistent_types": mixed_type_penalty,
            },
            "explanations": explanations,
            "recommendations": recommendations,
        }

        self.quality_report = {
            "health_score": health_score,
            "score_breakdown": score_breakdown,
            "total_rows": self.total_rows,
            "total_columns": len(self.df.columns),
            "missing_columns_count": len(missing_by_col),
            "missing_details": missing_by_col,
            "total_missing_cells": missing_cells,
            "duplicate_rows": dup_count,
            "duplicate_pct": dup_pct,
            "duplicate_indices": dup_indices,
            "constant_columns": constant_columns,
            "mixed_type_columns": mixed_type_columns,
            "outliers": outliers_by_col,
            "total_outliers_detected": total_outliers_found,
        }

    def clean_dataset(
        self,
        drop_duplicates: bool = True,
        missing_strategy: str = "none",  # 'none', 'drop_rows', 'impute_mean_mode', 'impute_median'
        outlier_strategy: str = "none",  # 'none', 'trim_iqr', 'cap_iqr'
        target_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Applies deterministic data cleaning operations and returns cleaned dataframe + change log.
        """
        df_cleaned = self.df.copy()
        log = {"removed_duplicates": 0, "missing_handled": 0, "outliers_adjusted": 0}

        # 1. Deduplication
        if drop_duplicates:
            before_len = len(df_cleaned)
            df_cleaned = df_cleaned.drop_duplicates()
            log["removed_duplicates"] = before_len - len(df_cleaned)

        # 2. Missing values
        if missing_strategy == "drop_rows":
            before_len = len(df_cleaned)
            df_cleaned = df_cleaned.dropna()
            log["missing_handled"] = before_len - len(df_cleaned)
        elif missing_strategy in ["impute_mean_mode", "impute_median"]:
            for col in df_cleaned.columns:
                if df_cleaned[col].isna().sum() > 0:
                    if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                        fill_val = df_cleaned[col].mean() if missing_strategy == "impute_mean_mode" else df_cleaned[col].median()
                        df_cleaned[col] = df_cleaned[col].fillna(fill_val)
                    else:
                        mode_val = df_cleaned[col].mode()
                        if len(mode_val) > 0:
                            df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
                    log["missing_handled"] += 1

        # 3. Outlier handling
        if outlier_strategy in ["trim_iqr", "cap_iqr"]:
            num_cols = target_columns or [c for c in df_cleaned.columns if pd.api.types.is_numeric_dtype(df_cleaned[c])]
            for col in num_cols:
                col_name_lower = str(col).lower()
                if col_name_lower.endswith("id") or "_id" in col_name_lower:
                    continue
                series = pd.to_numeric(df_cleaned[col], errors="coerce")
                if series.dropna().empty:
                    continue
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                if outlier_strategy == "trim_iqr":
                    mask = (df_cleaned[col] >= lower) & (df_cleaned[col] <= upper)
                    outlier_cnt = (~mask).sum()
                    df_cleaned = df_cleaned[mask | df_cleaned[col].isna()]
                    log["outliers_adjusted"] += int(outlier_cnt)
                elif outlier_strategy == "cap_iqr":
                    before = df_cleaned[col].copy()
                    df_cleaned[col] = df_cleaned[col].clip(lower=lower, upper=upper)
                    adjusted = (before != df_cleaned[col]).sum()
                    log["outliers_adjusted"] += int(adjusted)

        return df_cleaned, log

    def to_dict(self) -> Dict[str, Any]:
        return self.quality_report
