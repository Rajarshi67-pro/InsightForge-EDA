"""
AI Safety & Privacy Scanner for DataLens AI
Scans datasets for Personally Identifiable Information (PII) and sensitive fields,
masking sensitive data before any LLM ingestion.
"""

import re
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from .logger import app_logger


class PrivacyScanner:
    """
    Automated PII detection, risk scoring, and data masking engine.
    """

    # Regex patterns for sensitive information
    PATTERNS = {
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "PHONE": r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "PAN_CARD": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
        "AADHAAR": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }

    SENSITIVE_COLUMN_KEYWORDS = [
        "email", "phone", "mobile", "cell", "ssn", "social_security",
        "aadhaar", "pan", "passport", "password", "pwd", "secret",
        "token", "apikey", "credit_card", "card_number", "cvv",
        "salary", "address", "zipcode", "postal"
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.sensitive_findings: Dict[str, List[str]] = {}
        self.column_risk_scores: Dict[str, int] = {}
        self.audit_report: Dict[str, Any] = {}
        self._scan()

    def _scan(self) -> None:
        """Executes PII pattern matching and semantic column name auditing."""
        total_pii_detected = 0

        for col in self.df.columns:
            col_name_lower = str(col).lower()
            detected_types = set()

            # 1. Header Keyword Check
            for kw in self.SENSITIVE_COLUMN_KEYWORDS:
                if kw in col_name_lower:
                    detected_types.add(f"KEYWORD_{kw.upper()}")

            # 2. Cell Content Regex Check (on sample)
            if self.df[col].dtype == object or self.df[col].dtype == str:
                sample_series = self.df[col].dropna().astype(str).head(100)
                for pii_name, pattern in self.PATTERNS.items():
                    matches = sample_series.str.contains(pattern, regex=True).sum()
                    if matches > 0:
                        detected_types.add(pii_name)
                        total_pii_detected += int(matches)

            if detected_types:
                self.sensitive_findings[str(col)] = list(detected_types)
                # Risk weight: 30 for financial/id, 20 for contact, 10 for keywords
                risk = 0
                for dt in detected_types:
                    if dt in ["CREDIT_CARD", "SSN", "PAN_CARD", "AADHAAR", "KEYWORD_PASSWORD", "KEYWORD_TOKEN"]:
                        risk += 40
                    elif dt in ["EMAIL", "PHONE", "KEYWORD_EMAIL", "KEYWORD_PHONE"]:
                        risk += 25
                    else:
                        risk += 10
                self.column_risk_scores[str(col)] = min(100, risk)

        # Overall Privacy Safety Score (100 = completely clean, 0 = severe risk)
        num_flagged_cols = len(self.sensitive_findings)
        total_cols = max(1, len(self.df.columns))
        score_penalty = min(100, (num_flagged_cols / total_cols) * 70 + (total_pii_detected * 5))
        privacy_score = max(0, round(100 - score_penalty, 1))

        self.audit_report = {
            "privacy_safety_score": privacy_score,
            "sensitive_columns_count": num_flagged_cols,
            "flagged_columns": self.sensitive_findings,
            "total_pii_occurrences": total_pii_detected,
            "is_safe_for_llm": privacy_score >= 70 and not any(
                t in str(self.sensitive_findings) for t in ["CREDIT_CARD", "SSN", "PAN_CARD", "AADHAAR", "KEYWORD_PASSWORD"]
            ),
        }
        app_logger.info(f"Privacy scan complete. Privacy score: {privacy_score}/100. Flagged cols: {num_flagged_cols}")

    def mask_dataframe(self) -> pd.DataFrame:
        """Returns a copy of the dataframe with all detected PII fields redacted."""
        df_masked = self.df.copy()

        for col, types in self.sensitive_findings.items():
            if col not in df_masked.columns:
                continue

            if any(t in ["EMAIL", "KEYWORD_EMAIL"] for t in types):
                df_masked[col] = df_masked[col].astype(str).str.replace(self.PATTERNS["EMAIL"], "[REDACTED_EMAIL]", regex=True)
            if any(t in ["PHONE", "KEYWORD_PHONE", "KEYWORD_MOBILE"] for t in types):
                df_masked[col] = df_masked[col].astype(str).str.replace(self.PATTERNS["PHONE"], "[REDACTED_PHONE]", regex=True)
            if any(t in ["CREDIT_CARD", "KEYWORD_CREDIT_CARD"] for t in types):
                df_masked[col] = df_masked[col].astype(str).str.replace(self.PATTERNS["CREDIT_CARD"], "[REDACTED_CARD]", regex=True)
            if any(t in ["PAN_CARD", "AADHAAR", "SSN"] for t in types):
                df_masked[col] = "[REDACTED_GOVT_ID]"
            if any(t in ["KEYWORD_PASSWORD", "KEYWORD_SECRET", "KEYWORD_TOKEN"] for t in types):
                df_masked[col] = "[REDACTED_SECRET]"

        return df_masked

    def to_dict(self) -> Dict[str, Any]:
        return self.audit_report
