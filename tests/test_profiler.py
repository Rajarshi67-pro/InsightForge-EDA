import unittest
import pandas as pd
from src.profiler import DataProfiler, ColumnType

class TestDataProfiler(unittest.TestCase):
    def test_column_classification(self):
        # Create a dataframe with various column types
        data = {
            'employee_id': ['E001', 'E002', 'E003', 'E004', 'E005'],
            'uuid': ['123e4567-e89b-12d3-a456-426614174000', '223e4567-e89b-12d3-a456-426614174001', '323e4567-e89b-12d3-a456-426614174002', '423e4567-e89b-12d3-a456-426614174003', '523e4567-e89b-12d3-a456-426614174004'],
            'age': [25, 30, 35, 40, 45],
            'salary': [50000.50, 60000.0, 75000.75, 80000.0, 95000.25],
            'department': ['Sales', 'IT', 'HR', 'IT', 'Sales'],
            'is_active': [True, False, True, True, False],
            'status_flag': ['yes', 'no', 'yes', 'yes', 'no'],
            'joining_date': ['2020-01-15', '2021-03-20', '2019-11-05', '2022-08-10', '2018-05-22'],
            'feedback': [
                "This is a very long string that should definitely be classified as text because it is very long and has many words.",
                "Another long review string that goes on and on to test the text classification logic.",
                "Short.",
                "More text that is somewhat long to average out the words and length.",
                "Final piece of text to complete the sample size."
            ]
        }
        df = pd.DataFrame(data)
        
        # Instantiate profiler
        profiler = DataProfiler(df)
        
        # Classify columns
        self.assertEqual(profiler.classify_column('employee_id'), ColumnType.IDENTIFIER)
        self.assertEqual(profiler.classify_column('uuid'), ColumnType.IDENTIFIER)
        
        self.assertEqual(profiler.classify_column('age'), ColumnType.NUMERICAL)
        self.assertEqual(profiler.classify_column('salary'), ColumnType.NUMERICAL)
        
        self.assertEqual(profiler.classify_column('department'), ColumnType.CATEGORICAL)
        
        self.assertEqual(profiler.classify_column('is_active'), ColumnType.BOOLEAN)
        self.assertEqual(profiler.classify_column('status_flag'), ColumnType.BOOLEAN)
        
        self.assertEqual(profiler.classify_column('joining_date'), ColumnType.DATE)
        
        self.assertEqual(profiler.classify_column('feedback'), ColumnType.TEXT)

    def test_statistics_population(self):
        df = pd.DataFrame({'nums': [1, 2, None, 4, 4]})
        profiler = DataProfiler(df)
        profile_dict = profiler.to_dict()
        
        # Verify stats are present
        col_info = profile_dict['columns'][0]
        self.assertEqual(col_info['missing_count'], 1)
        self.assertEqual(col_info['unique_count'], 3)
        self.assertEqual(col_info['semantic_type'], 'Numerical')

if __name__ == '__main__':
    unittest.main()
