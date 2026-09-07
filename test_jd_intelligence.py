"""Tests for Phase 3: job description intelligence, enhanced job-fit
explanation, skill-gap prioritization, and resume version comparison."""
import io
import unittest

from docx import Document

from services.jd_analyzer import analyze_job_description, split_into_sentences
from services.analyzer import analyze_resume, job_fit_breakdown, prioritize_skill_gaps, compare_resumes
from services.parser import parse_resume


class FakeUploadedFile:
    def __init__(self, data: bytes, name: str):
        self._data = data
        self.name = name

    def getvalue(self):
        return self._data


def _docx_resume(paragraphs, filename="resume.docx"):
    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return parse_resume(FakeUploadedFile(buf.getvalue(), filename))


SAMPLE_JD = (
    "Junior Machine Learning Engineer\n\n"
    "We are looking for a Junior Machine Learning Engineer with Python programming, "
    "machine learning fundamentals, Pandas, NumPy, scikit-learn, SQL and Git.\n"
    "Experience with NLP, REST APIs, FastAPI, Docker or cloud platforms is a plus.\n"
    "The candidate should be able to build classification models and evaluate them "
    "using precision, recall and F1-score.\n"
)


class TestJobDescriptionExtraction(unittest.TestCase):
    def test_extracts_job_title(self):
        info = analyze_job_description(SAMPLE_JD)
        self.assertEqual(info["job_title"], "Junior Machine Learning Engineer")

    def test_required_vs_preferred_skills(self):
        info = analyze_job_description(SAMPLE_JD)
        self.assertIn("python", info["required_skills"])
        self.assertIn("sql", info["required_skills"])
        self.assertIn("docker", info["preferred_skills"])
        self.assertIn("fastapi", info["preferred_skills"])
        # a skill correctly bucketed as preferred should not also be required
        self.assertNotIn("docker", info["required_skills"])

    def test_no_false_positive_education_from_common_words(self):
        # "The candidate should BE able to..." must not be misread as a "BE" degree.
        info = analyze_job_description(SAMPLE_JD)
        self.assertEqual(info["education_requirements"], [])

    def test_empty_jd_returns_unknown_not_fabricated(self):
        info = analyze_job_description("")
        self.assertIsNone(info["job_title"])
        self.assertEqual(info["required_skills"], [])
        self.assertEqual(info["education_requirements"], [])
        self.assertIsNone(info["min_experience_years"])

    def test_experience_years_extracted_when_present(self):
        jd = "We need a backend engineer with 3+ years of experience in Python and SQL."
        info = analyze_job_description(jd)
        self.assertEqual(info["min_experience_years"], 3.0)

    def test_experience_years_none_when_absent(self):
        info = analyze_job_description(SAMPLE_JD)
        self.assertIsNone(info["min_experience_years"])

    def test_education_requirement_detected_when_present(self):
        jd = "Bachelor's degree in Computer Science required. Python and SQL needed."
        info = analyze_job_description(jd)
        self.assertIn("Bachelor's Degree", info["education_requirements"])

    def test_split_into_sentences_basic(self):
        sentences = split_into_sentences("First sentence. Second sentence.\nThird line")
        self.assertEqual(len(sentences), 3)


class TestSkillGapPrioritization(unittest.TestCase):
    def test_required_skill_gets_high_priority(self):
        info = analyze_job_description(SAMPLE_JD)
        gaps = prioritize_skill_gaps(["python"], SAMPLE_JD, info)
        self.assertEqual(gaps[0]["priority"], "High")
        self.assertIn("explicitly requires", gaps[0]["reason"])

    def test_preferred_skill_gets_medium_priority(self):
        info = analyze_job_description(SAMPLE_JD)
        gaps = prioritize_skill_gaps(["docker"], SAMPLE_JD, info)
        self.assertEqual(gaps[0]["priority"], "Medium")

    def test_gaps_sorted_high_before_medium(self):
        info = analyze_job_description(SAMPLE_JD)
        gaps = prioritize_skill_gaps(["docker", "python"], SAMPLE_JD, info)
        self.assertEqual(gaps[0]["skill"], "python")
        self.assertEqual(gaps[0]["priority"], "High")

    def test_evidence_is_genuine_jd_text_not_invented(self):
        info = analyze_job_description(SAMPLE_JD)
        gaps = prioritize_skill_gaps(["python"], SAMPLE_JD, info)
        self.assertIn("python", gaps[0]["evidence"].lower())

    def test_suggested_action_never_tells_user_to_fabricate(self):
        info = analyze_job_description(SAMPLE_JD)
        gaps = prioritize_skill_gaps(["python"], SAMPLE_JD, info)
        self.assertIn("if you actually have it", gaps[0]["suggested_action"])
        self.assertNotRegex(gaps[0]["suggested_action"], r"\d+%")


class TestJobFitBreakdown(unittest.TestCase):
    def test_education_unspecified_when_jd_silent(self):
        resume = _docx_resume(["A","a@example.com","B.Tech in CS"], "a.docx")
        info = analyze_job_description(SAMPLE_JD)
        fit = job_fit_breakdown(resume, SAMPLE_JD, info, 50, 10, "TF-IDF cosine similarity", 20)
        self.assertEqual(fit["education_alignment"]["status"], "Not specified in job description")
        self.assertIsNone(fit["education_alignment"]["score"])

    def test_experience_meets_requirement(self):
        resume = {"education": None, "experience_years": 4.0}
        info = {"min_experience_years": 2.0, "education_requirements": []}
        fit = job_fit_breakdown(resume, "jd text", info, 50, 10, "method", 20)
        self.assertEqual(fit["experience_alignment"]["status"], "Meets stated requirement")
        self.assertEqual(fit["experience_alignment"]["score"], 100)

    def test_experience_below_requirement_partial_score(self):
        resume = {"education": None, "experience_years": 1.0}
        info = {"min_experience_years": 4.0, "education_requirements": []}
        fit = job_fit_breakdown(resume, "jd text", info, 50, 10, "method", 20)
        self.assertEqual(fit["experience_alignment"]["status"], "Below stated requirement")
        self.assertEqual(fit["experience_alignment"]["score"], 25)

    def test_never_overwrites_primary_scores_it_only_labels_basis(self):
        resume = {"education": None, "experience_years": 1.0}
        info = {"min_experience_years": None, "education_requirements": []}
        fit = job_fit_breakdown(resume, "jd text", info, 77, 33, "TF-IDF cosine similarity", 44)
        self.assertEqual(fit["technical_skill_alignment"]["score"], 77)
        self.assertEqual(fit["semantic_relevance"]["score"], 33)
        self.assertEqual(fit["keyword_alignment"]["score"], 44)
        self.assertIn("basis", fit["technical_skill_alignment"])


class TestAnalyzeResumeIncludesPhase3Fields(unittest.TestCase):
    def test_result_contains_all_phase3_keys(self):
        resume = _docx_resume(
            ["Cand", "cand@example.com | 9000000009", "Skills", "Python, SQL"], "cand.docx"
        )
        result = analyze_resume(resume, SAMPLE_JD, use_semantic=False)
        for key in ("job_description_insights", "job_fit_breakdown", "skill_gap_priority"):
            self.assertIn(key, result)

    def test_primary_match_score_unaffected_by_missing_jd_info(self):
        # Backward compatibility: an old-style resume dict without Phase 2/3
        # keys must still analyze without crashing.
        old_style = {
            "filename": "old.pdf", "name": "Old", "email": "old@example.com", "phone": "9000000000",
            "education": "B.TECH", "experience_years": 1.0, "skills": ["python", "sql"],
            "text": "Developed and implemented a small tool using Python and SQL.",
        }
        result = analyze_resume(old_style, "python sql docker", use_semantic=False)
        self.assertIsInstance(result["match_score"], int)


class TestResumeComparison(unittest.TestCase):
    def test_compare_resumes_reports_positive_difference_for_improvement(self):
        weaker = _docx_resume(["A", "a@example.com", "Skills", "Python"], "a.docx")
        stronger = _docx_resume(["B", "b@example.com", "Skills", "Python, SQL, Docker"], "b.docx")
        jd = "python sql docker"
        result_a = analyze_resume(weaker, jd, use_semantic=False)
        result_b = analyze_resume(stronger, jd, use_semantic=False)
        comparison = compare_resumes(weaker, result_a, stronger, result_b)
        skill_match_row = next(r for r in comparison["metrics"] if r["Metric"] == "Skill Match")
        self.assertGreater(skill_match_row["Difference"], 0)

    def test_compare_resumes_identifies_resolved_and_new_gaps(self):
        a = _docx_resume(["A", "a@example.com", "Skills", "Python"], "a.docx")
        b = _docx_resume(["B", "b@example.com", "Skills", "SQL"], "b.docx")
        jd = "python sql"
        result_a = analyze_resume(a, jd, use_semantic=False)
        result_b = analyze_resume(b, jd, use_semantic=False)
        comparison = compare_resumes(a, result_a, b, result_b)
        self.assertIn("sql", comparison["skills_resolved_in_b"])
        self.assertIn("python", comparison["skills_newly_missing_in_b"])

    def test_comparison_is_json_serializable(self):
        import json
        a = _docx_resume(["A", "a@example.com", "Skills", "Python"], "a.docx")
        b = _docx_resume(["B", "b@example.com", "Skills", "Python, SQL"], "b.docx")
        jd = "python sql"
        result_a = analyze_resume(a, jd, use_semantic=False)
        result_b = analyze_resume(b, jd, use_semantic=False)
        comparison = compare_resumes(a, result_a, b, result_b)
        json.dumps(comparison)  # must not raise


if __name__ == "__main__":
    unittest.main()
