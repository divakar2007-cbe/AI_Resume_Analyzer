"""Tests for services.ml_model — training pipeline and dataset error handling."""
import pathlib
import tempfile
import unittest

import services.ml_model as ml_model
from services.ml_model import MLModelError, train_and_evaluate


class TestTrainAndEvaluateHappyPath(unittest.TestCase):
    def test_trains_on_bundled_dataset_without_crashing(self):
        metrics = train_and_evaluate()
        for key in ("accuracy", "precision", "recall", "f1"):
            self.assertIn(key, metrics)
            self.assertGreaterEqual(metrics[key], 0)
            self.assertLessEqual(metrics[key], 100)
        self.assertEqual(len(metrics["confusion_matrix"]), 2)
        self.assertIn("note", metrics)  # must ship with the "not production accuracy" disclaimer


class TestTrainAndEvaluateErrorHandling(unittest.TestCase):
    def setUp(self):
        self._original_data_path = ml_model.DATA
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        ml_model.DATA = self._original_data_path
        self._tmpdir.cleanup()

    def test_missing_dataset_raises_ml_model_error(self):
        ml_model.DATA = pathlib.Path(self._tmpdir.name) / "does_not_exist.csv"
        with self.assertRaises(MLModelError):
            train_and_evaluate()

    def test_malformed_columns_raise_ml_model_error(self):
        path = pathlib.Path(self._tmpdir.name) / "malformed.csv"
        path.write_text("foo,bar\n1,2\n3,4\n")
        ml_model.DATA = path
        with self.assertRaises(MLModelError):
            train_and_evaluate()

    def test_insufficient_rows_per_class_raise_ml_model_error(self):
        path = pathlib.Path(self._tmpdir.name) / "tiny.csv"
        path.write_text("text,label\npython sql,1\njava c++,0\n")
        ml_model.DATA = path
        with self.assertRaises(MLModelError):
            train_and_evaluate()

    def test_single_class_dataset_raises_ml_model_error(self):
        path = pathlib.Path(self._tmpdir.name) / "single_class.csv"
        path.write_text("text,label\npython sql,1\njava sql,1\nc++ git,1\n")
        ml_model.DATA = path
        with self.assertRaises(MLModelError):
            train_and_evaluate()


if __name__ == "__main__":
    unittest.main()
