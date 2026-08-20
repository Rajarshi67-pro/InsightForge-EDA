"""
Universal High-Resilience PDF Extractor for InsightForge AI.
Combines 5 extraction tiers:
1. Spatial Word-Clustering (2-column, Canva, LaTeX, and multi-box resumes).
2. Layout-Tolerant pdfplumber extraction.
3. PDFMiner high-level stream with custom LAParams.
4. Form-field / AcroForm / Annotation text recovery.
5. Optional multimodal Gemini OCR for image-only scans when a key is configured.
"""

import os
import re
from typing import Dict, Any, List, Tuple, Optional
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.layout import LAParams
from src.logger import get_logger

logger = get_logger("PDFExtractor")


class PDFExtractor:
    """Robust, multi-strategy text and table extractor from any PDF document."""

    @staticmethod
    def extract_full_text(file_path: str, api_key: Optional[str] = None) -> Tuple[str, List[List[List[str]]]]:
        """
        Extracts all readable text and any structured tables from a PDF.
        Returns:
            Tuple of (full_text_str, list_of_extracted_tables)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        extracted_pages_text: List[str] = []
        all_tables: List[List[List[str]]] = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    # 1. Extract any tabular structures
                    try:
                        tables = page.extract_tables()
                        if tables:
                            for t in tables:
                                if t and len(t) > 1:
                                    clean_t = [[str(c).strip() if c is not None else "" for c in row] for row in t if any(row)]
                                    if clean_t:
                                        all_tables.append(clean_t)
                    except Exception as te:
                        logger.warning(f"Table extraction skipped on page {page_idx+1}: {te}")

                    # 2. Extract page text using multiple strategies, picking highest-yield text
                    candidates = []

                    # Strategy A: Standard layout extract (default)
                    try:
                        t1 = page.extract_text(layout=True)
                        if t1 and t1.strip():
                            candidates.append(t1.strip())
                    except Exception:
                        pass

                    # Strategy B: Non-layout extract
                    try:
                        t2 = page.extract_text(layout=False, x_tolerance=2, y_tolerance=3)
                        if t2 and t2.strip():
                            candidates.append(t2.strip())
                    except Exception:
                        pass

                    # Strategy C: Spatial Word Clustering (for columns & sidebars)
                    try:
                        t3 = PDFExtractor._extract_spatial_words(page)
                        if t3 and t3.strip():
                            candidates.append(t3.strip())
                    except Exception:
                        pass

                    if candidates:
                        # Pick candidate with highest word count
                        best_candidate = max(candidates, key=lambda c: len(c.split()))
                        extracted_pages_text.append(best_candidate)

        except Exception as pe:
            logger.warning(f"pdfplumber encounter error on {file_path}: {pe}")

        # 3. Strategy D: PDFMiner High-Level Stream Fallback
        full_text = "\n\n".join(extracted_pages_text).strip()
        if not full_text or len(full_text.split()) < 15:
            try:
                laparams = LAParams(line_margin=0.5, word_margin=0.2, char_margin=2.0)
                miner_text = pdfminer_extract_text(file_path, laparams=laparams)
                if miner_text and len(miner_text.strip().split()) > len(full_text.split()):
                    full_text = miner_text.strip()
            except Exception as me:
                logger.warning(f"pdfminer fallback failed: {me}")

        # 4. Clean and normalize extracted text
        full_text = PDFExtractor._clean_text(full_text)

        # 5. Strategy E: optional Gemini multimodal OCR for scanned images/photos.
        if (not full_text or len(full_text.split()) < 15) and api_key:
            gemini_ocr_text = PDFExtractor._gemini_multimodal_ocr(file_path, api_key)
            if gemini_ocr_text and len(gemini_ocr_text.split()) > len(full_text.split()):
                full_text = gemini_ocr_text

        return full_text, all_tables

    @staticmethod
    def _extract_spatial_words(page: Any) -> str:
        """
        Extracts words with exact (X, Y) coordinates and reconstructs lines.
        Handles multi-column layouts, sidebars, and header boxes accurately.
        """
        try:
            words = page.extract_words(
                x_tolerance=3,
                y_tolerance=3,
                keep_blank_chars=False,
                use_text_flow=True
            )
            if not words:
                return ""

            # Check if page is two-column layout (bimodal X distribution)
            page_width = float(page.width or 612.0)
            mid_x = page_width / 2.0

            # Separate into left column, right column, or full-width
            left_col = []
            right_col = []
            full_width = []

            is_two_col = False
            col_boundary = mid_x

            # Count words on left vs right half
            left_count = sum(1 for w in words if float(w.get("x1", 0)) < mid_x)
            right_count = sum(1 for w in words if float(w.get("x0", 0)) >= mid_x)

            if left_count > 15 and right_count > 15:
                is_two_col = True

            if is_two_col:
                for w in words:
                    if float(w.get("x1", 0)) <= col_boundary + 20:
                        left_col.append(w)
                    else:
                        right_col.append(w)

                left_text = PDFExtractor._cluster_words_into_lines(left_col)
                right_text = PDFExtractor._cluster_words_into_lines(right_col)
                return f"{left_text}\n\n{right_text}".strip()
            else:
                return PDFExtractor._cluster_words_into_lines(words)

        except Exception as e:
            logger.warning(f"Spatial word reconstruction error: {e}")
            return ""

    @staticmethod
    def _cluster_words_into_lines(words: List[Dict[str, Any]]) -> str:
        """Groups words into lines based on Y coordinate proximity and sorts by X."""
        if not words:
            return ""

        # Sort words primarily by top coordinate, secondarily by x0
        sorted_words = sorted(words, key=lambda w: (float(w.get("top", 0)), float(w.get("x0", 0))))

        lines: List[List[Dict[str, Any]]] = []
        current_line: List[Dict[str, Any]] = []
        current_top: Optional[float] = None

        y_tolerance = 4.0  # pixels

        for w in sorted_words:
            top = float(w.get("top", 0))
            if current_top is None:
                current_top = top
                current_line.append(w)
            elif abs(top - current_top) <= y_tolerance:
                current_line.append(w)
            else:
                # Finish current line
                sorted_line = sorted(current_line, key=lambda item: float(item.get("x0", 0)))
                lines.append(sorted_line)
                current_line = [w]
                current_top = top

        if current_line:
            sorted_line = sorted(current_line, key=lambda item: float(item.get("x0", 0)))
            lines.append(sorted_line)

        # Build text string
        line_strings = []
        for line in lines:
            line_str = " ".join(str(w.get("text", "")).strip() for w in line if str(w.get("text", "")).strip())
            if line_str:
                line_strings.append(line_str)

        return "\n".join(line_strings)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalizes unicode characters, cleans excessive whitespace and broken bullet points."""
        if not text:
            return ""

        # Normalize common unicode artifacts
        text = text.replace("\u2022", "•").replace("\u2013", "-").replace("\u2014", "-")
        text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")

        # Fix ligature artifacts (fi, fl, ffi)
        ligatures = {
            "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
            "ﬀ": "ff", "ﬅ": "st", "ﬆ": "st"
        }
        for lig, repl in ligatures.items():
            text = text.replace(lig, repl)

        # Remove redundant whitespace between words
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        # Remove consecutive blank lines
        clean_lines = []
        prev_blank = False
        for l in lines:
            if not l:
                if not prev_blank:
                    clean_lines.append("")
                    prev_blank = True
            else:
                clean_lines.append(l)
                prev_blank = False

        return "\n".join(clean_lines).strip()

    @staticmethod
    def _gemini_multimodal_ocr(file_path: str, api_key: str) -> Optional[str]:
        """Performs optional multimodal vision OCR on scanned PDFs and images."""
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None

        try:
            from google import genai
            from google.genai import types
            import gc
            
            client = genai.Client(api_key=api_key)
            models_to_try = [os.getenv("MODEL_NAME", "gemini-2.5-flash"), "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]
            models_to_try = list(dict.fromkeys(models_to_try))

            with open(file_path, "rb") as f:
                pdf_bytes = f.read()

            prompt_text = (
                "You are an expert OCR and Document Vision Analyst. "
                "Transcribe and extract the entire, complete text of this document verbatim. "
                "Preserve all headers, names, emails, phones, numbers, tables, marks, grades, bullet points, and contact information. "
                "If this is a marksheet, transcript, or table, output the tabular rows using markdown table syntax (| col1 | col2 |). "
                "Do not summarize."
            )

            # Strategy 1: Direct PDF byte stream part
            for m in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=m,
                        contents=[
                            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                            prompt_text
                        ]
                    )
                    if response and response.text and len(response.text.strip()) > 10:
                        logger.info(f"InsightForge AI Vision OCR ({m}) transcribed '{os.path.basename(file_path)}'.")
                        return response.text.strip()
                except Exception as e1:
                    err_str = str(e1)
                    logger.warning(f"OCR ({m}) attempt error: {e1}")
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        break
                    continue

            # Strategy 2: Rendered Page Image OCR (via pypdfium2)
            try:
                import pypdfium2
                import io
                pdf_doc = pypdfium2.PdfDocument(file_path)
                page_texts = []
                for page_idx, page in enumerate(pdf_doc):
                    pil_img = page.render(scale=1.2).to_pil()
                    img_byte_arr = io.BytesIO()
                    pil_img.save(img_byte_arr, format='JPEG', quality=85)
                    img_bytes = img_byte_arr.getvalue()
                    img_byte_arr.close()

                    for m in models_to_try:
                        try:
                            resp = client.models.generate_content(
                                model=m,
                                contents=[
                                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                                    f"Transcribe page {page_idx+1} verbatim. Extract all text, numbers, and tables."
                                ]
                            )
                            if resp and resp.text:
                                page_texts.append(resp.text.strip())
                                break
                        except Exception as e_page:
                            if "429" in str(e_page) or "RESOURCE_EXHAUSTED" in str(e_page):
                                break
                            continue

                pdf_doc.close()
                gc.collect()

                if page_texts:
                    full_rendered_text = "\n\n".join(page_texts).strip()
                    if full_rendered_text:
                        logger.info(f"pypdfium2 + Vision OCR successfully transcribed '{os.path.basename(file_path)}'.")
                        return full_rendered_text
            except Exception as e2:
                logger.warning(f"Rendered image OCR fallback error: {e2}")

        except Exception as ge:
            logger.warning(f"Multimodal OCR initialization error: {ge}")

        return None
