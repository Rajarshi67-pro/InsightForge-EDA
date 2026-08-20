import unittest
import pandas as pd
from src.statistics import StatisticalEngine

class TestStatisticalEngine(unittest.TestCase):
    def test_correlation_safeguards(self):
        # 1. Test insufficient observations (n < 5)
        df_small = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [4, 3, 2, 1]})
        stats = StatisticalEngine(df_small).to_dict()
        corr_data = stats['correlations']
        self.assertTrue(corr_data.get('warning'))
        self.assertEqual(corr_data['strong_correlations'], [])

        # 2. Test constant columns
        df_const = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [2, 2, 2, 2, 2]})
        stats = StatisticalEngine(df_const).to_dict()
        corr_data = stats['correlations']
        self.assertTrue(corr_data.get('warning'))
        self.assertEqual(corr_data['strong_correlations'], [])

        # 3. Test non-linear relationships (Spearman > Pearson)
        # We need a relationship where Spearman is much stronger than Pearson.
        # Exponential growth creates strong monotonic (Spearman=1) but weaker linear (Pearson<1) relationship.
        df_exp = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'b': [1, 4, 9, 16, 25, 36, 49, 64, 81, 1000]})
        stats = StatisticalEngine(df_exp).to_dict()
        strong_corrs = stats['correlations']['strong_correlations']
        
        # There should be exactly 1 correlation object
        self.assertEqual(len(strong_corrs), 1)
        rel = strong_corrs[0]
        self.assertTrue(rel['is_nonlinear'])
        self.assertEqual(rel['strength'], "Strong Positive")

        # 4. Test normal linear relationship
        df_linear = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6], 'b': [2, 4, 6, 8, 10, 12]})
        stats = StatisticalEngine(df_linear).to_dict()
        strong_corrs = stats['correlations']['strong_correlations']
        rel = strong_corrs[0]
        self.assertFalse(rel['is_nonlinear'])
        self.assertEqual(rel['strength'], "Strong Positive")
        self.assertEqual(rel['pearson'], 1.0)

if __name__ == '__main__':
    unittest.main()
