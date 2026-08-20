"""Universal Data Loader Module for InsightForge AI.

Supports high-resilience ingestion of:
- CSV & Delimited files (.csv, .tsv, .txt, .dat)
- Excel Workbooks (.xlsx, .xls, .xlsm, .xlsb)
- JSON & JSON Lines (.json, .jsonl, .ndjson)
- Apache Parquet & Feather (.parquet, .feather)
- SQLite Databases (.sqlite, .db, .sqlite3)
- PDF Tabular Extraction (.pdf)
"""

import os
import json
import sqlite3
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
from src.logger import get_logger

logger = get_logger("DataLoader")

ENCODINGS_TO_TRY = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
DELIMITERS_TO_TRY = [",", "\t", ";", "|", ":"]


def load_dataset(
    file_path: str,
    max_sample_rows: Optional[int] = None,
    sheet_name: Optional[str] = None,
    sqlite_table: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Universal polymorphic entry point for loading any tabular data file into a Pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the target dataset file.
    max_sample_rows : Optional[int]
        Max rows to load into memory (samples large files).
    sheet_name : Optional[str]
        Sheet name if loading an Excel workbook.
    sqlite_table : Optional[str]
        Table name if loading a SQLite database.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (Loaded DataFrame, Ingestion Metadata Dictionary)
    """
    if not os.path.exists(file_path):
        logger.error(f"Failed to load dataset: [Errno 2] No such file or directory: '{file_path}'")
        raise FileNotFoundError(f"Dataset file not found at path: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    meta: Dict[str, Any] = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_extension": ext,
        "file_size_bytes": os.path.getsize(file_path),
        "is_sampled": False,
        "format": "unknown",
    }

    df: Optional[pd.DataFrame] = None

    try:
        if ext in [".xlsx", ".xls", ".xlsm", ".xlsb"]:
            df, extra_meta = _load_excel(file_path, sheet_name=sheet_name)
            meta.update(extra_meta)
        elif ext in [".json", ".jsonl", ".ndjson"]:
            df, extra_meta = _load_json(file_path)
            meta.update(extra_meta)
        elif ext in [".parquet", ".pq"]:
            df, extra_meta = _load_parquet(file_path)
            meta.update(extra_meta)
        elif ext in [".feather", ".ft"]:
            df = pd.read_feather(file_path)
            meta["format"] = "feather"
        elif ext in [".sqlite", ".db", ".sqlite3"]:
            df, extra_meta = _load_sqlite(file_path, table_name=sqlite_table)
            meta.update(extra_meta)
        elif ext in [".pdf"]:
            df, extra_meta = _load_pdf(file_path)
            meta.update(extra_meta)
        else:
            # Default to resilient CSV / Delimited parser
            df, extra_meta = _load_csv_or_delimited(file_path)
            meta.update(extra_meta)

        if df is None or df.empty:
            raise ValueError(f"Extracted dataset from '{file_path}' contains zero rows or columns.")

        # Clean column names (strip whitespace and unprintable chars)
        df.columns = [str(c).strip() for c in df.columns]

        meta["total_rows_source"] = len(df)
        meta["total_cols"] = len(df.columns)

        # Apply sampling if file exceeds max_sample_rows
        if max_sample_rows and len(df) > max_sample_rows:
            logger.info(f"Dataset exceeds {max_sample_rows} rows. Downsampling for performance.")
            df = df.sample(n=max_sample_rows, random_state=42).reset_index(drop=True)
            meta["is_sampled"] = True
            meta["sample_size"] = max_sample_rows

        meta["loaded_rows"] = len(df)
        logger.info(f"Successfully ingested '{meta['file_name']}' ({meta['format']}). Shape: {df.shape}")
        return df, meta

    except Exception as e:
        logger.error(f"Universal loader error for file '{file_path}': {str(e)}")
        raise


def _load_csv_or_delimited(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads CSV, TSV, or custom delimited text files with automatic encoding and delimiter recovery."""
    last_exception = None

    for encoding in ENCODINGS_TO_TRY:
        try:
            # First attempt: Let C-engine auto-detect with sep=None
            df = pd.read_csv(
                file_path,
                sep=None,
                engine="python",
                encoding=encoding,
                on_bad_lines="skip",
            )
            logger.info(f"Dataset successfully loaded using encoding='{encoding}'. Shape: {df.shape}")
            return df, {"format": "csv", "encoding": encoding, "delimiter": "auto-detected"}
        except Exception as e:
            last_exception = e

        # Second attempt: Try explicit delimiters with standard C-engine
        for delimiter in DELIMITERS_TO_TRY:
            try:
                df = pd.read_csv(
                    file_path,
                    sep=delimiter,
                    encoding=encoding,
                    on_bad_lines="skip",
                )
                if len(df.columns) > 1:
                    logger.info(f"Loaded with encoding='{encoding}', delimiter='{delimiter}'. Shape: {df.shape}")
                    return df, {"format": "csv", "encoding": encoding, "delimiter": delimiter}
            except Exception as e:
                last_exception = e

    raise ValueError(f"Unable to parse delimited file with any standard encoding. Error: {last_exception}")


def _load_excel(file_path: str, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads Excel spreadsheets (.xlsx, .xls) and inspects sheets."""
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names

    selected_sheet = sheet_name if sheet_name and sheet_name in sheet_names else sheet_names[0]
    df = pd.read_excel(excel_file, sheet_name=selected_sheet)
    
    return df, {
        "format": "excel",
        "sheet_names": sheet_names,
        "active_sheet": selected_sheet,
    }


def _load_json(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads JSON datasets, flattening nested objects and supporting JSON lines (.jsonl)."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()

    if not content:
        raise ValueError("JSON file is empty.")

    # Try standard structured JSON parsing first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            df = pd.json_normalize(data)
            return df, {"format": "json", "mode": "records_list"}
        elif isinstance(data, dict):
            records_key = None
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    records_key = k
                    break
            if records_key:
                df = pd.json_normalize(data[records_key])
                return df, {"format": "json", "mode": f"dict_key_{records_key}"}
            else:
                df = pd.json_normalize(data)
                return df, {"format": "json", "mode": "key_value_dict"}
    except Exception:
        pass

    # Fallback to JSON Lines (.jsonl)
    try:
        import io
        df = pd.read_json(io.StringIO(content), lines=True)
        return df, {"format": "jsonl", "mode": "lines"}
    except Exception as e:
        raise ValueError(f"Failed to parse JSON/JSONL file: {e}")


def _load_parquet(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads Apache Parquet columnar files."""
    df = pd.read_parquet(file_path)
    return df, {"format": "parquet"}


def _load_sqlite(file_path: str, table_name: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads tables from a SQLite database."""
    conn = sqlite3.connect(file_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]

        if not tables:
            raise ValueError(f"SQLite database '{file_path}' contains no user tables.")

        selected_table = table_name if table_name and table_name in tables else tables[0]
        df = pd.read_sql_query(f"SELECT * FROM \"{selected_table}\"", conn)
        return df, {
            "format": "sqlite",
            "tables": tables,
            "active_table": selected_table,
        }
    finally:
        conn.close()


def _load_pdf(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Extracts tabular data, marksheets, or resume text matrices from PDF documents using high-resilience PDFExtractor."""
    from src.pdf_extractor import PDFExtractor
    from src.resume_engine import ResumeEngine
    from src.marksheet_engine import MarksheetEngine

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    full_text, all_tables = PDFExtractor.extract_full_text(file_path, api_key=api_key)

    # Case A: Check if it's strictly a Resume / CV
    if ResumeEngine.is_resume(full_text, file_name=file_path):
        lines = [line.strip() for line in full_text.split("\n") if line.strip()]
        if lines:
            df = pd.DataFrame({
                "Section_Line_ID": list(range(1, len(lines) + 1)),
                "Resume_Content": lines,
                "Character_Count": [len(l) for l in lines],
                "Word_Count": [len(l.split()) for l in lines],
            })
            return df, {
                "format": "resume",
                "is_resume": True,
                "is_marksheet": False,
                "raw_text": full_text,
                "extracted_rows": len(df),
            }

    # Case B: Check if it's a Marksheet / Transcript / Scorecard
    if MarksheetEngine.is_marksheet(full_text, file_name=file_path):
        lines = [line.strip() for line in full_text.split("\n") if line.strip()]
        df = pd.DataFrame({
            "Line_ID": list(range(1, len(lines) + 1)),
            "Scorecard_Text": lines,
            "Char_Count": [len(l) for l in lines],
        })
        return df, {
            "format": "marksheet",
            "is_resume": False,
            "is_marksheet": True,
            "raw_text": full_text,
            "extracted_rows": len(df),
        }

    # Case B: Found structured tabular dataset in PDF (e.g. Marksheet, financial tables)
    if all_tables:
        top_table = all_tables[0]
        if len(top_table) > 1:
            headers = top_table[0]
            # Ensure unique header names
            clean_headers = []
            for i, h in enumerate(headers):
                h_str = str(h).strip() if h else f"Column_{i+1}"
                if h_str in clean_headers:
                    h_str = f"{h_str}_{i+1}"
                clean_headers.append(h_str)
            rows = top_table[1:]
            df = pd.DataFrame(rows, columns=clean_headers)
            return df, {
                "format": "pdf",
                "is_resume": False,
                "pages_with_tables": len(all_tables),
                "extracted_rows": len(df),
            }

    # Case C: If completely unreadable / 0 text and 0 tables
    if not full_text:
        raise ValueError(
            f"The uploaded document '{os.path.basename(file_path)}' contains no selectable text or tabular cells. "
            "It appears to be a scanned image/photo. Optional Gemini OCR is used only when a key is configured; "
            "otherwise paste the text directly into the Paste Text Directly tab or upload a standard text PDF, CSV, or Excel file."
        )

    # Case D: General text document (e.g. report, transcript, essay) -> structured analysis table
    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in full_text.split("\n") if p.strip()]
    df = pd.DataFrame({
        "Section_ID": list(range(1, len(paragraphs) + 1)),
        "Document_Text": paragraphs,
        "Character_Count": [len(p) for p in paragraphs],
        "Word_Count": [len(p.split()) for p in paragraphs],
    })
    return df, {
        "format": "pdf_text",
        "is_resume": False,
        "extracted_rows": len(df),
    }


class DataLoader:
    """Backward compatible class wrapper for universal load_dataset."""

    @staticmethod
    def load_dataset(file_path: Any, max_sample_rows: Optional[int] = None, sample_if_large: bool = True, **kwargs):
        try:
            if hasattr(file_path, "read"):
                # It's a buffer / BytesIO or file-like object
                # Default to CSV parser on buffer
                df = pd.read_csv(file_path, on_bad_lines="skip")
                return df, None
            
            df, meta = load_dataset(file_path, max_sample_rows=max_sample_rows, **kwargs)
            return df, None
        except Exception as e:
            return None, str(e)
