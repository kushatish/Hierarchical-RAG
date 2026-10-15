import unittest
from evaluation.calibrate_cache import sweep

class CalibrationTests(unittest.TestCase):
    def test_threshold_tradeoff_and_negation_guard(self):
        pairs = [
            {"seed": "scholarship CGPA", "candidate": "CGPA scholarship", "equivalent": True},
            {"seed": "scholarship CGPA", "candidate": "scholarship income", "equivalent": False},
            {"seed": "Who is eligible for scholarship?", "candidate": "Who is not eligible for scholarship?", "equivalent": False},
        ]
        _, rows = sweep(pairs, "lexical", [.1, .97])
        self.assertEqual(rows[0]["true_positives"], 1)
        self.assertEqual(rows[0]["false_positives"], 1)
        self.assertEqual(rows[1]["false_positives"], 0)
        self.assertEqual(rows[1]["precision"], 1)
        self.assertEqual(rows[1]["recall"], 1)
