"""Tests for services.database — SQLite history persistence."""
import pathlib
import tempfile
import unittest

import services.database as database
from services.database import DatabaseError, get_history, save_analysis


class TestDatabaseRoundTrip(unittest.TestCase):
    def setUp(self):
        # Point the module at a throwaway DB file so tests never touch the
        # real candidate history.
        self._original_db = database.DB
        self._tmpdir = tempfile.TemporaryDirectory()
        database.DB = pathlib.Path(self._tmpdir.name) / "test_history.db"

    def tearDown(self):
        database.DB = self._original_db
        self._tmpdir.cleanup()

    def test_save_and_retrieve_analysis(self):
        resume = {"name": "Jane Doe", "filename": "jane.pdf"}
        analysis = {
            "match_score": 82,
            "ats_score": 75,
            "matched_skills": ["python", "sql"],
            "missing_skills": ["docker"],
        }
        save_analysis(resume, analysis)
        history = get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["Candidate"], "Jane Doe")
        self.assertEqual(history[0]["Match Score"], 82)
        self.assertEqual(history[0]["Matched Skills"], "python, sql")

    def test_missing_name_falls_back_to_unknown(self):
        resume = {"name": None, "filename": "anon.pdf"}
        analysis = {"match_score": 50, "ats_score": 50, "matched_skills": [], "missing_skills": []}
        save_analysis(resume, analysis)
        history = get_history()
        self.assertEqual(history[0]["Candidate"], "Unknown")

    def test_empty_history_returns_empty_list(self):
        self.assertEqual(get_history(), [])

    def test_unwritable_db_path_raises_database_error(self):
        # Point at a path whose parent directory doesn't exist and can't be
        # created implicitly by sqlite3 -> should surface as DatabaseError,
        # not an unhandled sqlite3.OperationalError.
        database.DB = pathlib.Path("/nonexistent_dir_xyz/history.db")
        with self.assertRaises(DatabaseError):
            save_analysis({"name": "X", "filename": "x.pdf"},
                           {"match_score": 1, "ats_score": 1, "matched_skills": [], "missing_skills": []})


if __name__ == "__main__":
    unittest.main()
