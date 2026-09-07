"""Job description intelligence: structured, rule-based extraction from a
target job description.

Everything here is regex/heuristic-based and explainable — nothing is
inferred by a black-box model, and nothing is invented. If a field can't be
reliably detected, it comes back empty/None rather than guessed.
"""
import re
from collections import Counter

from .parser import extract_skills, EDU
from .skills_data import SKILL_CATEGORY

REQUIRED_MARKERS = [
    "required", "requirement", "must have", "must", "you have", "you need",
    "minimum qualification", "minimum qualifications", "we need", "need to have",
]
PREFERRED_MARKERS = [
    "preferred", "nice to have", "nice-to-have", "a plus", "is a plus",
    "bonus", "desirable", "good to have", "ideally",
]

_STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","with","is","are","as",
    "will","this","that","you","your","we","our","be","have","has","who","at",
    "role","looking","strong","experience","years","team","work","working","job",
    "candidate","candidates","skills","skill","ability","knowledge","using","use",
    "including","include","etc","also","any","all","must","required","preferred",
}


def _sentences(jd_text):
    return [s.strip() for s in re.split(r"[\n\.;]", jd_text) if s.strip()]


def _extract_title(jd_text):
    lines = [l.strip() for l in jd_text.splitlines() if l.strip()]
    if not lines:
        return None
    first = lines[0]
    # A plausible title is short and doesn't read like a full sentence.
    if 1 <= len(first.split()) <= 8 and not first.endswith("."):
        return first
    return None


def _extract_education_requirements(jd_text):
    low = jd_text.lower()
    found = sorted({t.upper() for t in EDU if re.search(r"(?<!\w)"+re.escape(t)+r"(?!\w)", low)})
    # Also catch spelled-out degree levels not in the parser's EDU list.
    for phrase, label in [("bachelor's degree","Bachelor's Degree"),
                          ("bachelors degree","Bachelor's Degree"),
                          ("master's degree","Master's Degree"),
                          ("masters degree","Master's Degree"),
                          ("doctorate","Doctorate")]:
        if phrase in low and label not in found:
            found.append(label)
    return found


def _extract_experience_years(jd_text):
    matches = re.findall(r"(\d+)\s*\+?\s*(?:-|to)?\s*(?:\d+\s*)?(?:years?|yrs?)\s+(?:of\s+)?experience", jd_text, re.I)
    if not matches:
        return None
    return min(float(m) for m in matches)  # minimum years typically required


def _split_required_preferred(jd_text, all_skills):
    """Bucket detected skills into required vs preferred based on which
    sentence they appear in. Skills in sentences with no explicit marker
    default to required (the common case: a plain requirements list)."""
    required, preferred = set(), set()
    for sentence in _sentences(jd_text):
        low = sentence.lower()
        sentence_skills = set(extract_skills(sentence))
        if not sentence_skills:
            continue
        if any(m in low for m in PREFERRED_MARKERS):
            preferred |= sentence_skills
        else:
            required |= sentence_skills
    # A skill explicitly called out as preferred anywhere shouldn't also be
    # listed as required just because it appeared in another sentence too.
    required -= preferred
    # Anything detected but not bucketed by the sentence pass (e.g. the
    # skill appears in a fragment split oddly) still counts as required —
    # the overall skill set must remain a superset used by existing matching.
    leftover = set(all_skills) - required - preferred
    required |= leftover
    return sorted(required), sorted(preferred)


def _domain_keywords(jd_text, skills, top_n=8):
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", jd_text.lower())
    skill_words = set()
    for s in skills:
        skill_words.update(s.split())
    counts = Counter(t for t in tokens if t not in _STOPWORDS and t not in skill_words)
    return [w for w, _ in counts.most_common(top_n)]


def split_into_sentences(jd_text):
    """Public helper: split job-description text into rough sentences.

    Reused by services.analyzer for skill-gap evidence lookup, so sentence
    splitting logic isn't duplicated across modules.
    """
    return _sentences(jd_text)


def analyze_job_description(jd_text):
    """Structured, explainable extraction from a job description.

    Returns a dict — every field is either a genuinely detected value or an
    empty/None placeholder. Nothing here is fabricated.
    """
    if not jd_text or not jd_text.strip():
        return {
            "job_title": None, "required_skills": [], "preferred_skills": [],
            "education_requirements": [], "min_experience_years": None,
            "tools_frameworks": [], "domain_keywords": [],
        }

    all_skills = extract_skills(jd_text)
    required, preferred = _split_required_preferred(jd_text, all_skills)
    tools = sorted({s for s in all_skills
                     if SKILL_CATEGORY.get(s) in {"Web Development","Cloud","DevOps","Databases","Machine Learning"}})

    return {
        "job_title": _extract_title(jd_text),
        "required_skills": required,
        "preferred_skills": preferred,
        "education_requirements": _extract_education_requirements(jd_text),
        "min_experience_years": _extract_experience_years(jd_text),
        "tools_frameworks": tools,
        "domain_keywords": _domain_keywords(jd_text, all_skills),
    }
