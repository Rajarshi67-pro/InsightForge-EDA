import unittest
import pandas as pd
from src.visualizer import VisualizerEngine
import plotly.graph_objects as go

class TestVisualizerEngine(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5],
            'num2': [5, 4, 3, 2, 1],
            'cat1': ['A', 'A', 'B', 'B', 'C'],
            'date1': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
        })
        self.column_profiles = {
            'num1': {'semantic_type': 'Numerical', 'missing_pct': 0, 'unique_count': 5},
            'num2': {'semantic_type': 'Numerical', 'missing_pct': 0, 'unique_count': 5},
            'cat1': {'semantic_type': 'Categorical', 'missing_pct': 0, 'unique_count': 3},
            'date1': {'semantic_type': 'Date', 'missing_pct': 0, 'unique_count': 5},
        }

    def test_recommend_visualizations(self):
        recs = VisualizerEngine.recommend_visualizations(self.df, self.column_profiles)
        
        # We expect:
        # 1. Histogram (num1)
        # 2. Box Plot (num1 standalone)
        # 3. Bar Chart (cat1)
        # 4. Correlation Heatmap (since 2 numericals)
        # 5. Scatter Plot (num1 vs num2)
        # 6. Line Chart (date1 vs num1)
        # 7. Grouped Box Plot (num1 vs cat1)
        
        types_generated = [r['type'] for r in recs]
        
        self.assertEqual(types_generated.count('histogram'), 1)
        self.assertEqual(types_generated.count('box'), 2)  # standalone + grouped
        self.assertEqual(types_generated.count('bar'), 1)
        self.assertEqual(types_generated.count('heatmap'), 1)
        self.assertEqual(types_generated.count('scatter'), 1)
        self.assertEqual(types_generated.count('line'), 1)

        # Check types are plotly figures
        for rec in recs:
            self.assertIsInstance(rec['fig'], go.Figure)

    def test_no_meaningless_charts(self):
        # Only 1 numerical column, no date, no cat
        df_single = pd.DataFrame({'num1': [1, 2, 3]})
        profiles_single = {'num1': {'semantic_type': 'Numerical'}}
        
        recs = VisualizerEngine.recommend_visualizations(df_single, profiles_single)
        types_generated = [r['type'] for r in recs]
        
        # Should only generate histogram and standalone box plot
        self.assertEqual(len(types_generated), 2)
        self.assertIn('histogram', types_generated)
        self.assertIn('box', types_generated)
        self.assertNotIn('scatter', types_generated)
        self.assertNotIn('heatmap', types_generated)
        self.assertNotIn('line', types_generated)
        self.assertNotIn('bar', types_generated)

if __name__ == '__main__':
    unittest.main()
