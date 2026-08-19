import unittest
from datalens.agents import InsightAgent, AnomalyAgent, ReportAgent

class TestActionableInsightsEngine(unittest.TestCase):
    def setUp(self):
        self.profiler_mock = {
            "columns": [
                {"column_name": "customer_id", "semantic_type": "Identifier", "unique_count": 100},
                {"column_name": "age", "semantic_type": "Numerical", "unique_count": 50},
            ]
        }
        
        self.stats_mock = {
            "correlations": {
                "strong_correlations": [
                    {"col1": "age", "col2": "salary", "pearson": 0.85, "strength": "Strong Positive"}
                ]
            },
            "numerical": {
                "age": {"skewness": 1.5, "skewness_label": "Right Skewed", "mean": 35, "median_50": 30, "variance": 120, "std": 10.95},
                "zero_var_col": {"skewness": 0, "skewness_label": "Symmetric", "mean": 5, "median_50": 5, "variance": 0, "std": 0},
            },
            "categorical": {
                "country": {"unique": 55, "top": "USA", "top_percentage": 10.0},
                "is_active": {"unique": 2, "top": "Yes", "top_percentage": 85.0}
            }
        }
        
        self.quality_mock = {
            "duplicate_rows": 5,
            "duplicate_pct": 2.5,
            "missing_details": [
                {"column": "age", "missing_count": 10, "missing_pct": 10.0}
            ],
            "outliers": {
                "age": {
                    "iqr": {"outlier_count": 15, "outlier_pct": 6.5, "potential_severity": "High"}
                }
            }
        }

    def test_insight_agent_generates_correct_insights(self):
        agent = InsightAgent(self.stats_mock, self.profiler_mock)
        insights = agent.run()
        
        categories = [i["category"] for i in insights]
        self.assertIn("Correlation Dynamics", categories)
        self.assertIn("Distribution Asymmetry", categories)
        self.assertIn("Low-Variance Feature", categories)
        self.assertIn("High Cardinality", categories)
        self.assertIn("Class Imbalance", categories)
        self.assertIn("Identifier Column Detected", categories)
        
        # Verify 4-part structure
        for insight in insights:
            self.assertIn("finding", insight)
            self.assertIn("evidence", insight)
            self.assertIn("importance", insight)
            self.assertIn("action", insight)

    def test_anomaly_agent_generates_correct_anomalies(self):
        agent = AnomalyAgent(self.quality_mock)
        anomalies = agent.run()
        
        categories = [a["category"] for a in anomalies]
        self.assertIn("Redundancy Risk", categories)
        self.assertIn("Missing Data Gap", categories)
        self.assertIn("Statistical Outlier (IQR)", categories)

        # Verify 4-part structure
        for anomaly in anomalies:
            self.assertIn("finding", anomaly)
            self.assertIn("evidence", anomaly)
            self.assertIn("importance", anomaly)
            self.assertIn("action", anomaly)

    def test_report_agent_compiles_markdown(self):
        insight_agent = InsightAgent(self.stats_mock, self.profiler_mock)
        anomaly_agent = AnomalyAgent(self.quality_mock)
        report_agent = ReportAgent(insight_agent, anomaly_agent)
        
        report = report_agent.compile()
        self.assertIn("## 🤖 Autonomous Actionable Insights Engine", report)
        self.assertIn("Why it matters", report)
        self.assertIn("Actionable Recommendation", report)

if __name__ == '__main__':
    unittest.main()
