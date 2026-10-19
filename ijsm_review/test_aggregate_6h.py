import unittest

import pandas as pd

from aggregate_6h import aggregate_6h


class Aggregate6hTest(unittest.TestCase):
    def test_means_boundaries_and_missing_intervals(self):
        data = pd.DataFrame({
            "Data": ["2024-01-01 00:00", "2024-01-01 05:59",
                     "2024-01-01 06:00", "2024-01-01 18:00",
                     "2024-01-01 18:30"],
            "Vazao": [10, 30, 50, -1, 70],
        })
        result = aggregate_6h(data)
        self.assertEqual(result["Vazao"].tolist(), [20, 50, -1, 70])
        self.assertEqual(result["Data"].dt.hour.tolist(), [0, 6, 12, 18])


if __name__ == "__main__":
    unittest.main()
