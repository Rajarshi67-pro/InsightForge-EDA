"""
Server Endpoint Integration Tests for DataLens AI FastAPI Backend
"""

import unittest
from fastapi.testclient import TestClient
from server import app


class TestDataLensServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_load_sample_and_get_dataset(self):
        """Tests sample data loading and state retrieval."""
        res_sample = self.client.post("/api/load-sample")
        self.assertEqual(res_sample.status_code, 200)
        data_sample = res_sample.json()
        self.assertEqual(data_sample["status"], "success")

        res_state = self.client.get("/api/dataset?page=1&page_size=10")
        self.assertEqual(res_state.status_code, 200)
        state = res_state.json()
        self.assertTrue(state["has_dataset"])
        self.assertEqual(state["total_rows"], 21)
        self.assertIn("Employee_ID", state["columns"])
        self.assertIsNotNone(state["profiler"])
        self.assertIsNotNone(state["quality"])
        self.assertIsNotNone(state["statistics"])
        self.assertIsNotNone(state["privacy"])

    def test_02_clean_dataset(self):
        """Tests data cleaning endpoint."""
        res = self.client.post(
            "/api/clean",
            json={
                "drop_duplicates": True,
                "missing_strategy": "impute_median",
                "outlier_strategy": "cap_iqr",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["log"]["removed_duplicates"], 1)

    def test_03_ml_train(self):
        """Tests ML training endpoint."""
        res = self.client.post(
            "/api/ml/train",
            json={
                "target_column": "Salary",
                "model_name": "Random Forest",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("R2_Score", data["result"]["metrics"])

    def test_04_ai_query(self):
        """Tests grounded AI chat query endpoint."""
        res = self.client.post(
            "/api/ai/query",
            json={"query": "What is the average salary by department?"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("answer", data)
        self.assertTrue(len(data["answer"]) > 5)

    def test_05_rag_search(self):
        """Tests RAG knowledge retrieval."""
        res = self.client.get("/api/rag/search?q=Pearson")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(len(data["results"]) > 0)

    def test_06_exports(self):
        """Tests PDF, CSV, and JSON download endpoints."""
        res_pdf = self.client.get("/api/export/pdf")
        self.assertEqual(res_pdf.status_code, 200)
        self.assertEqual(res_pdf.headers["content-type"], "application/pdf")

        res_csv = self.client.get("/api/export/csv")
        self.assertEqual(res_csv.status_code, 200)
        self.assertIn("text/csv", res_csv.headers["content-type"])

        res_json = self.client.get("/api/export/json")
        self.assertEqual(res_json.status_code, 200)
        self.assertEqual(res_json.headers["content-type"], "application/json")