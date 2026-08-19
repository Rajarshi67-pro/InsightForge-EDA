import unittest
import pandas as pd
from datalens.quality import DataQualityEngine

class TestDataQualityEngine(unittest.TestCase):
    def test_quality_penalties(self):
        # Create a dataframe with specific anomalies
        df = pd.DataFrame({
            'good_col': [1, 2, 3, 4, 5],
            'constant_col': ['A', 'A', 'A', 'A', 'A'],
            'missing_col': [1, None, None, 4, 5],
            'mixed_col': [1, 'two', 3, 'four', 5]
        })
        
        # Add a duplicate row
        df.loc[5] = [1, 'A', 1, 1]
        df.loc[6] = [1, 'A', 1, 1]

        engine = DataQualityEngine(df)
        report = engine.to_dict()
        
        breakdown = report['score_breakdown']
        penalties = breakdown['penalties']
        
        # Ensure penalties were applied
        self.assertGreater(penalties['constant_columns'], 0, "Should penalize constant column")
        self.assertGreater(penalties['missing'], 0, "Should penalize missing values")
        self.assertGreater(penalties['inconsistent_types'], 0, "Should penalize mixed types")
        self.assertGreater(penalties['duplicates'], 0, "Should penalize duplicate rows")
        
        # Total score should be less than 100
        self.assertLess(breakdown['overall_score'], 100)
        
        # Explanations and recommendations should be populated
        self.assertGreater(len(breakdown['explanations']), 0)
        self.assertGreater(len(breakdown['recommendations']), 0)

    def test_perfect_score(self):
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['A', 'B', 'C', 'D', 'E']
        })
        engine = DataQualityEngine(df)
        report = engine.to_dict()
        breakdown = report['score_breakdown']
        
        self.assertEqual(breakdown['overall_score'], 100)
        self.assertEqual(breakdown['explanations'][0], "Perfect score: No anomalies detected.")

    def test_outlier_detection_normal_data(self):
        df_normal = pd.DataFrame({'a': [10, 11, 12, 11, 10, 12, 11, 10]})
        quality = DataQualityEngine(df_normal)
        report = quality.to_dict()
        self.assertEqual(report['total_outliers_detected'], 0)
        self.assertNotIn('a', report['outliers'])

    def test_outlier_detection_extreme_data(self):
        normal_data = list(range(10, 20)) * 10
        extreme_data = normal_data + [1000]
        df_extreme = pd.DataFrame({'a': extreme_data})
        quality = DataQualityEngine(df_extreme)
        report = quality.to_dict()
        outliers = report['outliers']['a']
        
        iqr = outliers['iqr']
        self.assertGreater(iqr['outlier_count'], 0)
        self.assertEqual(iqr['outlier_values'], [1000.0])
        self.assertIsNotNone(iqr['lower_bound'])
        self.assertIsNotNone(iqr['upper_bound'])
        
        z = outliers['z_score']
        self.assertGreater(z['outlier_count'], 0)
        self.assertEqual(z['outlier_values'], [1000.0])
        self.assertIsNotNone(z['lower_bound'])
        self.assertIsNotNone(z['upper_bound'])
        self.assertIn(iqr['potential_severity'], ["Low", "Medium", "High"])

    def test_outlier_severity_levels(self):
        data_high = list(range(10, 20)) * 9 + [1000] * 10
        df_high = pd.DataFrame({'a': data_high})
        q_high = DataQualityEngine(df_high).to_dict()
        self.assertEqual(q_high['outliers']['a']['iqr']['potential_severity'], 'High')

        data_med = list(range(10, 20)) * 10 + [1000] * 3
        df_med = pd.DataFrame({'a': data_med})
        q_med = DataQualityEngine(df_med).to_dict()
        self.assertEqual(q_med['outliers']['a']['iqr']['potential_severity'], 'Medium')

        data_low = list(range(10, 20)) * 20 + [1000] * 1
        df_low = pd.DataFrame({'a': data_low})
        q_low = DataQualityEngine(df_low).to_dict()
        self.assertEqual(q_low['outliers']['a']['iqr']['potential_severity'], 'Low')

if __name__ == '__main__':
    unittest.main()
