"""Tests for services.recommender — job matching against candidate skills."""
import unittest

from services.recommender import recommend_jobs


class TestRecommendJobs(unittest.TestCase):
    def test_returns_all_jobs_sorted_by_match_descending(self):
        rows = recommend_jobs(["python", "sql", "pandas", "excel", "data analysis"])
        scores = [r["Match"] for r in rows]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_perfect_match_scores_100(self):
        rows = recommend_jobs(["python", "sql", "excel", "data analysis", "pandas"])
        data_analyst = next(r for r in rows if r["Job Role"] == "Data Analyst")
        self.assertEqual(data_analyst["Match"], 100)

    def test_no_matching_skills_scores_zero_everywhere(self):
        rows = recommend_jobs(["photoshop", "illustrator"])
        self.assertTrue(all(r["Match"] == 0 for r in rows))

    def test_empty_skills_list_does_not_crash(self):
        rows = recommend_jobs([])
        self.assertTrue(all(r["Match"] == 0 for r in rows))


if __name__ == "__main__":
    unittest.main()
