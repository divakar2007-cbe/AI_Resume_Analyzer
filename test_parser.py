"""Tests for services.parser — skill extraction and file-error handling.

Run with:  python -m unittest discover -s tests
"""
import unittest

from services.parser import extract_text, extract_skills, parse_resume, ResumeParsingError


class FakeUploadedFile:
    """Minimal stand-in for Streamlit's UploadedFile (has .name and .getvalue())."""
    def __init__(self, data: bytes, name: str):
        self._data = data
        self.name = name

    def getvalue(self):
        return self._data


class TestExtractSkills(unittest.TestCase):
    def test_detects_known_skills_case_insensitively(self):
        text = "Experienced with Python, SQL, Docker and Machine Learning."
        skills = extract_skills(text)
        self.assertIn("python", skills)
        self.assertIn("sql", skills)
        self.assertIn("docker", skills)
        self.assertIn("machine learning", skills)

    def test_ignores_unrelated_words(self):
        skills = extract_skills("I enjoy cooking and hiking on weekends.")
        self.assertEqual(skills, [])

    def test_matches_whole_words_only(self):
        # "javascript" alias shouldn't match inside an unrelated word like "css3"
        skills = extract_skills("css3 styling")
        self.assertIn("css", skills)
        self.assertNotIn("javascript", skills)


class TestFileErrorHandling(unittest.TestCase):
    def test_empty_file_raises_resume_parsing_error(self):
        f = FakeUploadedFile(b"", "empty.pdf")
        with self.assertRaises(ResumeParsingError):
            extract_text(f)

    def test_corrupted_pdf_raises_resume_parsing_error(self):
        f = FakeUploadedFile(b"not a real pdf", "corrupt.pdf")
        with self.assertRaises(ResumeParsingError):
            extract_text(f)

    def test_corrupted_docx_raises_resume_parsing_error(self):
        f = FakeUploadedFile(b"not a real docx", "corrupt.docx")
        with self.assertRaises(ResumeParsingError):
            extract_text(f)

    def test_unsupported_file_type_raises_resume_parsing_error(self):
        # .txt is a supported type as of Phase 2 — use a genuinely
        # unsupported extension here instead.
        f = FakeUploadedFile(b"hello world", "resume.rtf")
        with self.assertRaises(ResumeParsingError):
            extract_text(f)

    def test_parse_resume_rejects_blank_extracted_text(self):
        # A .docx with zero paragraphs of text should be treated as an
        # unreadable/empty resume rather than silently producing a blank profile.
        import io
        from docx import Document
        doc = Document()
        buf = io.BytesIO()
        doc.save(buf)
        f = FakeUploadedFile(buf.getvalue(), "blank.docx")
        with self.assertRaises(ResumeParsingError):
            parse_resume(f)


class TestParseResume(unittest.TestCase):
    def test_parses_basic_docx_resume(self):
        import io
        from docx import Document
        doc = Document()
        doc.add_paragraph("Jane Doe")
        doc.add_paragraph("jane.doe@example.com")
        doc.add_paragraph("9876543210")
        doc.add_paragraph("B.Tech in Computer Science")
        doc.add_paragraph("Skills: Python, SQL, Machine Learning, Pandas")
        doc.add_paragraph("3 years of experience building backend systems.")
        buf = io.BytesIO()
        doc.save(buf)
        f = FakeUploadedFile(buf.getvalue(), "jane_doe.docx")

        resume = parse_resume(f)
        self.assertEqual(resume["email"], "jane.doe@example.com")
        self.assertEqual(resume["phone"], "9876543210")
        self.assertIn("python", resume["skills"])
        self.assertIn("sql", resume["skills"])
        self.assertEqual(resume["experience_years"], 3.0)


class TestTxtParsing(unittest.TestCase):
    """Phase 2: plain-text (.txt) resumes are now supported."""

    def _build_txt(self):
        content = (
            "John Smith\n"
            "john.smith@example.com | 9123456780\n"
            "linkedin.com/in/johnsmith | github.com/johnsmith\n\n"
            "Summary\n"
            "Backend engineer focused on distributed systems.\n\n"
            "Skills\n"
            "Python, SQL, Docker, AWS\n"
        )
        return FakeUploadedFile(content.encode("utf-8"), "john_smith.txt")

    def test_extract_text_reads_txt_file(self):
        f = self._build_txt()
        text = extract_text(f)
        self.assertIn("John Smith", text)

    def test_parse_resume_works_end_to_end_for_txt(self):
        f = self._build_txt()
        resume = parse_resume(f)
        self.assertEqual(resume["email"], "john.smith@example.com")
        self.assertIn("python", resume["skills"])
        self.assertIn("docker", resume["skills"])

    def test_empty_txt_file_raises_resume_parsing_error(self):
        f = FakeUploadedFile(b"", "empty.txt")
        with self.assertRaises(ResumeParsingError):
            extract_text(f)


class TestEncryptedPdf(unittest.TestCase):
    def test_encrypted_pdf_raises_resume_parsing_error(self):
        import io
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.encrypt(user_password="secret")
        buf = io.BytesIO()
        writer.write(buf)
        f = FakeUploadedFile(buf.getvalue(), "protected.pdf")
        with self.assertRaises(ResumeParsingError):
            extract_text(f)


class TestContactLinkExtraction(unittest.TestCase):
    def test_extracts_linkedin_and_github_urls(self):
        import io
        from docx import Document
        doc = Document()
        doc.add_paragraph("Alex Kim")
        doc.add_paragraph("alex.kim@example.com | 9988776655")
        doc.add_paragraph("linkedin.com/in/alexkim | github.com/alexkim")
        buf = io.BytesIO()
        doc.save(buf)
        f = FakeUploadedFile(buf.getvalue(), "alex_kim.docx")

        resume = parse_resume(f)
        self.assertEqual(resume["linkedin"], "linkedin.com/in/alexkim")
        self.assertEqual(resume["github"], "github.com/alexkim")

    def test_missing_links_are_none(self):
        import io
        from docx import Document
        doc = Document()
        doc.add_paragraph("Alex Kim")
        doc.add_paragraph("alex.kim@example.com | 9988776655")
        buf = io.BytesIO()
        doc.save(buf)
        f = FakeUploadedFile(buf.getvalue(), "alex_kim.docx")

        resume = parse_resume(f)
        self.assertIsNone(resume["linkedin"])
        self.assertIsNone(resume["github"])


class TestSkillNormalizationAndCategorization(unittest.TestCase):
    def test_scikit_learn_variants_normalize_to_one_skill(self):
        for phrase in ["scikit-learn", "scikit learn", "sklearn"]:
            skills = extract_skills(f"Experienced with {phrase} for modeling.")
            self.assertIn("scikit-learn", skills)

    def test_javascript_case_insensitive(self):
        self.assertIn("javascript", extract_skills("Proficient in JavaScript"))
        self.assertIn("javascript", extract_skills("Proficient in javascript"))

    def test_skills_grouped_by_category(self):
        import io
        from docx import Document
        doc = Document()
        doc.add_paragraph("Sam Lee")
        doc.add_paragraph("sam.lee@example.com | 9000000000")
        doc.add_paragraph("Skills: Python, SQL, Machine Learning, AWS, Docker")
        buf = io.BytesIO()
        doc.save(buf)
        f = FakeUploadedFile(buf.getvalue(), "sam_lee.docx")

        resume = parse_resume(f)
        by_cat = resume["skills_by_category"]
        self.assertIn("python", by_cat.get("Programming Languages", []))
        self.assertIn("sql", by_cat.get("Databases", []))
        self.assertIn("machine learning", by_cat.get("Machine Learning", []))
        self.assertIn("aws", by_cat.get("Cloud", []))
        self.assertIn("docker", by_cat.get("DevOps", []))


class TestSectionDetection(unittest.TestCase):
    def test_detects_common_sections(self):
        import io
        from docx import Document
        doc = Document()
        doc.add_paragraph("Priya Nair")
        doc.add_paragraph("priya@example.com | 9000000001")
        doc.add_paragraph("Summary")
        doc.add_paragraph("Data analyst with 2 years of experience.")
        doc.add_paragraph("Education")
        doc.add_paragraph("B.Sc in Statistics")
        doc.add_paragraph("Projects")
        doc.add_paragraph("Built a sales dashboard using Power BI.")
        doc.add_paragraph("Certifications")
        doc.add_paragraph("Google Data Analytics Certificate")
        buf = io.BytesIO()
        doc.save(buf)
        f = FakeUploadedFile(buf.getvalue(), "priya_nair.docx")

        resume = parse_resume(f)
        sections = resume["sections"]
        self.assertIn("summary", sections)
        self.assertIn("education", sections)
        self.assertIn("projects", sections)
        self.assertIn("certifications", sections)
        self.assertEqual(len(resume["projects"]), 1)
        self.assertEqual(len(resume["certifications"]), 1)

    def test_minimal_resume_has_few_or_no_sections(self):
        # A bare-bones resume with no headings at all shouldn't crash —
        # it should just report an empty/near-empty sections dict.
        import io
        from docx import Document
        doc = Document()
        doc.add_paragraph("Minimal Candidate")
        doc.add_paragraph("minimal@example.com | 9000000002")
        doc.add_paragraph("Python developer.")
        buf = io.BytesIO()
        doc.save(buf)
        f = FakeUploadedFile(buf.getvalue(), "minimal.docx")

        resume = parse_resume(f)
        self.assertEqual(resume["sections"], {})
        self.assertEqual(resume["projects"], [])
        self.assertEqual(resume["certifications"], [])
        self.assertEqual(resume["achievements"], [])


if __name__ == "__main__":
    unittest.main()
