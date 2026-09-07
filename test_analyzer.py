"""Tests for services.analyzer — scoring, ATS checks, and edge cases."""
import io
import unittest

from docx import Document

from services.analyzer import (
    analyze_resume, ats_checks, ats_check_details, ats_breakdown, ats_factor_explanations,
    build_resume_improvements, detect_bullet_issues, _split_bullet_candidates,
    keyword_overlap, similarity,
)
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


def make_resume(**overrides):
    base = {
        "filename": "resume.pdf",
        "name": "John Smith",
        "email": "john@example.com",
        "phone": "9876543210",
        "education": "B.TECH",
        "experience_years": 2.0,
        "skills": ["python", "sql", "pandas"],
        "text": (
            "Developed and implemented a data pipeline that processed over "
            "10000 records using Python, SQL and Pandas."
        ),
    }
    base.update(overrides)
    return base


class TestAnalyzeResume(unittest.TestCase):
    def test_matched_and_missing_skills(self):
        resume = make_resume()
        jd = "Looking for a Python developer with SQL, Docker and AWS experience."
        result = analyze_resume(resume, jd, use_semantic=False)
        self.assertIn("python", result["matched_skills"])
        self.assertIn("sql", result["matched_skills"])
        self.assertIn("docker", result["missing_skills"])
        self.assertIn("aws", result["missing_skills"])

    def test_score_bounds_are_always_0_to_100(self):
        resume = make_resume()
        jd = "python sql pandas docker aws machine learning"
        result = analyze_resume(resume, jd, use_semantic=False)
        for key in ("match_score", "ats_score", "skill_match", "semantic_score", "keyword_score"):
            self.assertGreaterEqual(result[key], 0)
            self.assertLessEqual(result[key], 100)

    def test_empty_job_description_does_not_crash(self):
        resume = make_resume()
        result = analyze_resume(resume, "", use_semantic=False)
        self.assertEqual(result["missing_skills"], [])
        self.assertIsInstance(result["match_score"], int)

    def test_empty_resume_text_does_not_crash(self):
        resume = make_resume(skills=[], text="", email=None, phone=None, education=None)
        result = analyze_resume(resume, "python sql", use_semantic=False)
        self.assertIsInstance(result["match_score"], int)


class TestAtsChecks(unittest.TestCase):
    def test_flags_missing_contact_info(self):
        resume = make_resume(email=None, phone=None)
        checks = ats_checks(resume, "some job description")
        self.assertFalse(checks["Contact information present"])

    def test_passes_when_contact_info_present(self):
        resume = make_resume()
        checks = ats_checks(resume, "some job description")
        self.assertTrue(checks["Contact information present"])


class TestSimilarityAndKeywordOverlap(unittest.TestCase):
    def test_similarity_returns_zero_for_blank_jd(self):
        score, _ = similarity("some resume text", "")
        self.assertEqual(score, 0.0)

    def test_keyword_overlap_zero_for_blank_jd(self):
        self.assertEqual(keyword_overlap("some resume text", ""), 0)

    def test_keyword_overlap_full_match(self):
        overlap = keyword_overlap("python sql pandas", "python sql pandas")
        self.assertEqual(overlap, 100.0)


class TestAtsBreakdown(unittest.TestCase):
    def test_breakdown_has_five_factors_and_overall_in_range(self):
        resume = make_resume()
        bd = ats_breakdown(resume, "python sql pandas")
        for key in ("contact_information", "section_completeness", "skills_coverage",
                    "keyword_coverage", "content_quality", "overall"):
            self.assertIn(key, bd)
            self.assertGreaterEqual(bd[key], 0)
            self.assertLessEqual(bd[key], 100)

    def test_complete_resume_scores_higher_than_minimal_resume(self):
        complete = _docx_resume([
            "Complete Candidate", "complete@example.com | 9000000003",
            "linkedin.com/in/complete | github.com/complete",
            "Summary", "Backend engineer with a focus on APIs.",
            "Education", "B.Tech in Computer Science",
            "Experience", "Software Engineer at Example Corp",
            "Projects", "Built and deployed a REST API serving 10000 requests using Python and FastAPI.",
            "Skills", "Python, FastAPI, SQL, Docker, AWS",
        ], "complete.docx")
        minimal = _docx_resume([
            "Minimal Candidate", "minimal4@example.com",
            "Python developer.",
        ], "minimal.docx")

        bd_complete = ats_breakdown(complete, "")
        bd_minimal = ats_breakdown(minimal, "")
        self.assertGreater(bd_complete["overall"], bd_minimal["overall"])

    def test_missing_sections_lower_section_completeness(self):
        minimal = _docx_resume(["Bare Candidate", "bare@example.com", "Python."], "bare.docx")
        bd = ats_breakdown(minimal, "")
        self.assertLess(bd["section_completeness"], 100)


class TestWeakBulletDetection(unittest.TestCase):
    def test_flags_weak_verb_and_missing_measurable_outcome(self):
        issues = detect_bullet_issues("Responsible for the marketing budget and vendor relationships")
        self.assertIn("responsibility_only", issues)
        self.assertIn("no_measurable_outcome", issues)

    def test_strong_bullet_with_metric_and_tech_has_no_no_measurable_outcome_issue(self):
        issues = detect_bullet_issues("Built a Python ETL pipeline that processed 10000 records daily")
        self.assertNotIn("no_measurable_outcome", issues)
        self.assertNotIn("no_technology_named", issues)

    def test_build_resume_improvements_never_invents_numbers(self):
        resume = _docx_resume([
            "Bullet Candidate", "bullet@example.com",
            "Projects",
            "Worked on a machine learning project.",
        ], "bullet.docx")
        improvements = build_resume_improvements(resume["text"], resume.get("sections"))
        self.assertTrue(improvements)
        for _old, suggestion in improvements:
            # The suggestion must never contain a fabricated percentage or
            # invented statistic — only guidance language.
            self.assertNotRegex(suggestion, r"\d+%")
            self.assertTrue(len(suggestion) > 0)

    def test_build_resume_improvements_restricted_to_experience_and_projects(self):
        resume = _docx_resume([
            "Section Candidate", "section@example.com",
            "Summary", "Aspiring engineer looking to build things.",
            "Projects", "Worked on a chatbot.",
        ], "section.docx")
        improvements = build_resume_improvements(resume["text"], resume.get("sections"))
        flagged_lines = [old for old, _ in improvements]
        self.assertTrue(any("chatbot" in l for l in flagged_lines))
        self.assertFalse(any("Aspiring engineer" in l for l in flagged_lines))


class TestGluedBulletSplitting(unittest.TestCase):
    """Regression coverage for PDF-extraction lines that glue multiple
    bullet points together with no line break between them."""

    def test_splits_glued_bullets_on_marker(self):
        garbled = "Developed and implemented a data pipeline • Worked on the reporting dashboard"
        parts = _split_bullet_candidates(garbled)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], "Developed and implemented a data pipeline")
        self.assertEqual(parts[1], "Worked on the reporting dashboard")

    def test_normal_already_separated_lines_are_unaffected(self):
        text = "Built a Python ETL pipeline\nAutomated deployment with Docker"
        parts = _split_bullet_candidates(text)
        self.assertEqual(parts, ["Built a Python ETL pipeline", "Automated deployment with Docker"])

    def test_build_resume_improvements_never_shows_a_garbled_before_line(self):
        resume = _docx_resume([
            "Garbled Candidate", "garbled@example.com",
            "Projects",
            "Developed and implemented a data pipeline • Worked on the reporting dashboard for sales",
        ], "garbled.docx")
        improvements = build_resume_improvements(resume["text"], resume.get("sections"))
        flagged_lines = [old for old, _ in improvements]
        # Neither flagged bullet should still contain the glued-together marker.
        for line in flagged_lines:
            self.assertNotIn("•", line)


class TestAtsCheckDetails(unittest.TestCase):
    """The ATS Checks section must show a human-readable reason, distinct
    from a bare PASS/FAIL — and must agree with the boolean ats_checks()."""

    def test_details_agree_with_boolean_checks(self):
        resume = make_resume()
        details = ats_check_details(resume, "python sql")
        booleans = ats_checks(resume, "python sql")
        for item in details:
            self.assertEqual(item["status"], booleans[item["check"]])
            self.assertIsInstance(item["reason"], str)
            self.assertTrue(len(item["reason"]) > 0)

    def test_reason_mentions_missing_contact_when_absent(self):
        resume = make_resume(email=None, phone=None)
        details = {d["check"]: d for d in ats_check_details(resume, "")}
        self.assertFalse(details["Contact information present"]["status"])
        self.assertIn("not detected", details["Contact information present"]["reason"])

    def test_reason_confirms_contact_when_present(self):
        resume = make_resume()
        details = {d["check"]: d for d in ats_check_details(resume, "")}
        self.assertTrue(details["Contact information present"]["status"])
        self.assertIn("detected", details["Contact information present"]["reason"])


class TestAtsFactorExplanations(unittest.TestCase):
    """Each ATS breakdown factor must distinguish detected facts from
    missing information, and agree with ats_breakdown()'s numeric score."""

    def test_explanation_scores_match_breakdown_scores(self):
        resume = make_resume()
        bd = ats_breakdown(resume, "python sql docker")
        explanations = ats_factor_explanations(resume, "python sql docker")
        for key in ("contact_information", "section_completeness", "skills_coverage",
                    "keyword_coverage", "content_quality"):
            self.assertEqual(explanations[key]["score"], bd[key])

    def test_missing_skills_appear_in_missing_not_detected(self):
        resume = make_resume(skills=["python"])
        explanations = ats_factor_explanations(resume, "python docker aws")
        skills_exp = explanations["skills_coverage"]
        self.assertIn("python", skills_exp["detected"])
        self.assertIn("docker", skills_exp["missing"])
        self.assertIn("aws", skills_exp["missing"])

    def test_missing_contact_items_are_listed(self):
        resume = make_resume(email=None)
        explanations = ats_factor_explanations(resume, "")
        contact_exp = explanations["contact_information"]
        self.assertIn("Email", contact_exp["missing"])
        self.assertIn("Phone", contact_exp["detected"])


if __name__ == "__main__":
    unittest.main()
