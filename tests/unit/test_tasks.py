import unittest
from kladml.tasks import MLTask

class TestMLActions(unittest.TestCase):
    def test_tasks_exist(self):
        self.assertEqual(MLTask.IMAGE_CLASSIFICATION, "image_classification")
        self.assertEqual(MLTask.TIMESERIES_FORECASTING, "timeseries_forecasting")
        
    def test_enums_unique(self):
        values = [t.value for t in MLTask]
        self.assertEqual(len(values), len(set(values)))
