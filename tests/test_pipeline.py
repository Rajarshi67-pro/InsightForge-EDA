"""
Production Test Suite for InsightForge AI
Validates all pipeline stages from Ingestion, Non-Analyzable Validation, and Resume Analysis to ML, RAG, and Reports.
"""

import os
import io
import json
import sqlite3
import tempfile
import unittest
import pandas as pd

from src.loader import load_dataset, DataLoader
from src.privacy import PrivacyScanner
from src.profiler import DataProfiler, ColumnType
from src.quality import DataQualityEngine
from src.statistics import StatisticalEngine
from src.visualizer import VisualizerEngine
from src.ml_engine import MLEngine
from src.ai_engine import AIEngine
from src.rag_engine import RAGEngine
from src.agents import AgentOrchestrator
from src.reports import ReportGenerator
from src.resume_engine import ResumeEngine


class TestInsightForgePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_csv_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "auto_data_analyzer_test.csv"
        )
        cls.df, cls.meta = load_dataset(cls.test_csv_path)

    def test_01_loader_and_error_handling(self):
        """Tests load_dataset with valid and invalid CSV paths."""
        self.assertIsNotNone(self.df)
        self.assertEqual(len(self.df), 21)
        self.assertEqual(self.meta["format"], "csv")

        # Test invalid path
        with self.assertRaises(FileNotFoundError):
            load_dataset("non_existent_file.csv")

    def test_01b_universal_multi_format_ingestion(self):
        """Tests ingestion across Excel (.xlsx), JSON, JSONL, Parquet, SQLite, and PDF."""
        test_data = pd.DataFrame({
            "ID": [101, 102, 103],
            "Department": ["Sales", "Engineering", "Marketing"],
            "Revenue": [12000.5, 45000.0, 32000.75],
            "Active": [True, True, False]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Excel (.xlsx)
            xlsx_path = os.path.join(tmpdir, "test.xlsx")
            test_data.to_excel(xlsx_path, index=False)
            df_xlsx, meta_xlsx = load_dataset(xlsx_path)
            self.assertEqual(len(df_xlsx), 3)
            self.assertEqual(meta_xlsx["format"], "excel")

            # 2. JSON (Array of objects)
            json_path = os.path.join(tmpdir, "test.json")
            test_data.to_json(json_path, orient="records")
            df_json, meta_json = load_dataset(json_path)
            self.assertEqual(len(df_json), 3)
            self.assertEqual(meta_json["format"], "json")

            # 3. JSON Lines (.jsonl)
            jsonl_path = os.path.join(tmpdir, "test.jsonl")
            test_data.to_json(jsonl_path, orient="records", lines=True)
            df_jsonl, meta_jsonl = load_dataset(jsonl_path)
            self.assertEqual(len(df_jsonl), 3)
            self.assertEqual(meta_jsonl["format"], "jsonl")

            # 4. Apache Parquet (.parquet)
            parquet_path = os.path.join(tmpdir, "test.parquet")
            test_data.to_parquet(parquet_path, index=False)
            df_pq, meta_pq = load_dataset(parquet_path)
            self.assertEqual(len(df_pq), 3)
            self.assertEqual(meta_pq["format"], "parquet")

            # 5. SQLite Database (.sqlite)
            sqlite_path = os.path.join(tmpdir, "test.sqlite")
            conn = sqlite3.connect(sqlite_path)
            test_data.to_sql("employees", conn, index=False)
            conn.close()
            df_sqlite, meta_sqlite = load_dataset(sqlite_path)
            self.assertEqual(len(df_sqlite), 3)
            self.assertEqual(meta_sqlite["format"], "sqlite")

            # 6. PDF with Table (.pdf)
            pdf_path = os.path.join(tmpdir, "test.pdf")
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=10)
            with pdf.table() as table:
                row = table.row()
                for h in test_data.columns:
                    row.cell(str(h))
                for _, r in test_data.iterrows():
                    row = table.row()
                    for val in r:
                        row.cell(str(val))
            pdf.output(pdf_path)
            
            df_pdf, meta_pdf = load_dataset(pdf_path)
            self.assertEqual(len(df_pdf), 3)
            self.assertEqual(meta_pdf["format"], "pdf")

    def test_01c_non_analyzable_document_validation(self):
        """Tests that empty or un-analyzable documents raise friendly descriptive errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Blank text file
            blank_txt = os.path.join(tmpdir, "blank.txt")
            with open(blank_txt, "w") as f:
                f.write("")
            with self.assertRaises(ValueError):
                load_dataset(blank_txt)

            # Blank PDF (0 content)
            blank_pdf = os.path.join(tmpdir, "blank.pdf")
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            # empty page without text or table
            pdf.output(blank_pdf)
            with self.assertRaises(ValueError):
                load_dataset(blank_pdf)

    def test_01d_empty_csv_handling(self):
        """Tests that empty CSV files or CSVs with only headers are rejected correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty CSV
            empty_csv = os.path.join(tmpdir, "empty.csv")
            with open(empty_csv, "w") as f:
                f.write("")
            with self.assertRaises(ValueError):
                load_dataset(empty_csv)

            # CSV with only headers
            headers_only_csv = os.path.join(tmpdir, "headers.csv")
            with open(headers_only_csv, "w") as f:
                f.write("col1,col2,col3\n")
            with self.assertRaises(ValueError):
                load_dataset(headers_only_csv)

    def test_02_privacy_scanner_and_masking(self):
        """Tests PII scanner detection and redacting."""
        test_pii_df = pd.DataFrame({
            "User_ID": [1, 2],
            "Email": ["alice@example.com", "bob@work.org"],
            "Phone": ["555-123-4567", "555-987-6543"],
            "Salary": [50000, 60000],
        })
        scanner = PrivacyScanner(test_pii_df)
        rep = scanner.to_dict()
        self.assertTrue(rep["sensitive_columns_count"] >= 2)
        
        masked_df = scanner.mask_dataframe()
        self.assertNotIn("alice@example.com", masked_df["Email"].values)
        self.assertIn("[REDACTED_EMAIL]", masked_df["Email"].values[0])

    def test_03_profiler_column_classification(self):
        """Tests semantic column classification."""
        profiler = DataProfiler(self.df)
        p_dict = profiler.to_dict()
        
        self.assertEqual(p_dict["total_rows"], 21)
        self.assertEqual(p_dict["total_cols"], 12)
        self.assertIn("Employee_ID", profiler.get_columns_by_type(ColumnType.IDENTIFIER))
        self.assertIn("Salary", profiler.get_columns_by_type(ColumnType.NUMERICAL))
        self.assertIn("Department", profiler.get_columns_by_type(ColumnType.CATEGORICAL))

    def test_04_quality_and_iqr_outliers(self):
        """Tests missing values, duplicate detection, and deterministic IQR outliers."""
        quality = DataQualityEngine(self.df)
        q_dict = quality.to_dict()

        self.assertEqual(q_dict["duplicate_rows"], 1)
        self.assertEqual(q_dict["total_missing_cells"], 3)
        self.assertTrue("Salary" in q_dict["outliers"])

        df_clean, log = quality.clean_dataset(drop_duplicates=True, missing_strategy="impute_median", outlier_strategy="cap_iqr")
        self.assertEqual(len(df_clean), 20)
        self.assertEqual(df_clean.isna().sum().sum(), 0)

    def test_05_statistics_and_correlations(self):
        """Tests numerical stats and Pearson/Spearman correlation matrices."""
        stats = StatisticalEngine(self.df)
        s_dict = stats.to_dict()

        self.assertIn("Salary", s_dict["numerical"])
        self.assertIn("Age", s_dict["numerical"])
        corrs = s_dict["correlations"]["strong_correlations"]
        self.assertTrue(len(corrs) > 0)
        top_c = corrs[0]
        self.assertEqual(top_c["strength"], "Strong Positive")

    def test_06_visualizer(self):
        """Tests automated and custom Plotly figure creation."""
        fig_hist = VisualizerEngine.create_histogram(self.df, "Salary")
        self.assertIsNotNone(fig_hist)
        fig_scatter = VisualizerEngine.create_scatter_plot(self.df, "Experience_Years", "Salary")
        self.assertIsNotNone(fig_scatter)

    def test_07_ml_engine(self):
        """Tests regression, classification, and K-Means clustering."""
        ml = MLEngine(self.df)
        reg_res = ml.train_regression("Salary")
        self.assertIn("R2_Score", reg_res["metrics"])

        clf_res = ml.train_classification("Department")
        self.assertIn("Accuracy", clf_res["metrics"])

        clust_res = ml.train_clustering(n_clusters=3)
        self.assertIn("silhouette_score", clust_res)

    def test_08_rag_knowledge_engine(self):
        """Tests RAG search retrieval."""
        res = RAGEngine.search("How does Pearson correlation work?")
        self.assertTrue(len(res) > 0)
        self.assertIn("Pearson", res[0]["title"])

    def test_09_agent_orchestrator(self):
        """Tests multi-agent autonomous audit."""
        p_dict = DataProfiler(self.df).to_dict()
        q_dict = DataQualityEngine(self.df).to_dict()
        s_dict = StatisticalEngine(self.df).to_dict()

        orchestrator = AgentOrchestrator(self.df, p_dict, q_dict, s_dict)
        audit = orchestrator.run_autonomous_audit()
        self.assertIn("insights", audit)
        self.assertIn("anomalies", audit)
        self.assertIn("executive_briefing", audit)

    def test_10_report_generator_pdf_and_json(self):
        """Tests PDF report byte compilation and JSON reproducibility generation."""
        p_dict = DataProfiler(self.df).to_dict()
        q_dict = DataQualityEngine(self.df).to_dict()
        s_dict = StatisticalEngine(self.df).to_dict()
        ai = AIEngine(self.df, p_dict, q_dict, s_dict)

        # PDF Report
        pdf_bytes = ReportGenerator.generate_pdf_report(
            dataset_name="test_data.csv",
            df=self.df,
            profiler_dict=p_dict,
            quality_dict=q_dict,
            stats_dict=s_dict,
            ai_briefing=ai.generate_executive_insights(),
            privacy_dict=ai.privacy_report,
        )
        self.assertTrue(len(pdf_bytes) > 500)

        # JSON Audit
        json_str = ReportGenerator.generate_reproducibility_json(
            dataset_name="test_data.csv",
            df=self.df,
            profiler_dict=p_dict,
            quality_dict=q_dict,
            stats_dict=s_dict,
        )
        self.assertIn("sha256_checksum", json_str)

    def test_11_resume_engine_and_market_scoring(self):
        """Tests ResumeEngine extraction, scoring out of 10, action verbs, and bullet rewrites."""
        sample_resume_text = """
        John Doe
        Email: john.doe@example.com | Phone: (555) 234-5678 | GitHub: github.com/johndoe
        
        EXPERIENCE
        Senior Data Scientist - TechCorp (2022 - Present)
        • Spearheaded predictive ML models using Python and PyTorch, increasing customer retention by 28% and generating $1.2M in annual revenue.
        • Architected real-time RAG inference pipelines with LangChain and FastAPI, reducing search latency by 45ms.
        • Optimized ETL data pipelines in PostgreSQL, saving 20 hours per week of manual data engineering.
        
        SKILLS
        Python, PyTorch, LangChain, FastAPI, Docker, SQL, PostgreSQL, AWS, Scikit-Learn
        
        EDUCATION
        B.S. in Computer Science - University of California
        """

        self.assertTrue(ResumeEngine.is_resume(sample_resume_text))
        engine = ResumeEngine(sample_resume_text, file_name="John_Doe_Resume.pdf")
        analysis = engine.analyze()

        self.assertTrue(analysis["is_resume"])
        self.assertGreaterEqual(analysis["score_out_of_10"], 6.0)
        self.assertIn("impact", analysis["sub_scores"])
        self.assertIn("verbs", analysis["sub_scores"])
        self.assertIn("skills", analysis["sub_scores"])
        self.assertIn("AI, ML & Data Engineering", analysis["matched_skills"])
        self.assertIn("python", analysis["matched_skills"]["AI, ML & Data Engineering"])
        self.assertIn("deep_insights", analysis)


if __name__ == "__main__":
    unittest.main()
