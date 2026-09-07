"""Rule-based resume section detection.

Splits raw resume text into labeled sections (Summary, Education,
Experience, Projects, Skills, Certifications, Achievements, Publications,
Contact) by scanning for lines that look like section headings. This is a
simple, explainable heuristic — not an ML classifier — by design: every
section boundary it finds can be pointed to directly in the source text.
"""
import re

# canonical section name -> known heading phrases (lowercase, no punctuation)
SECTION_HEADINGS = {
    "summary": ["summary", "professional summary", "profile", "career summary"],
    "objective": ["objective", "career objective"],
    "education": ["education", "academic background", "academic qualifications", "qualifications"],
    "experience": ["experience", "work experience", "professional experience",
                   "employment history", "work history"],
    "projects": ["projects", "academic projects", "personal projects", "key projects"],
    "skills": ["skills", "technical skills", "core competencies", "key skills"],
    "certifications": ["certifications", "certificates", "licenses",
                        "licenses and certifications", "licenses & certifications"],
    "achievements": ["achievements", "awards", "honors", "honors and awards", "accomplishments"],
    "publications": ["publications"],
    "contact": ["contact", "contact information", "contact details"],
}

_ALIAS_TO_CANONICAL = {
    alias: canonical
    for canonical, aliases in SECTION_HEADINGS.items()
    for alias in aliases
}

# A heading line is short and (optionally) ends in a colon/dash — this keeps
# an ordinary sentence that happens to contain a heading word (e.g. "I led
# the projects team") from being mistaken for a section break.
_MAX_HEADING_WORDS = 5


def detect_sections(text):
    """Return {canonical_section_name: content_text} for the sections found.

    Sections not present in the resume simply don't appear as keys — callers
    should use ``sections.get("projects")`` rather than assuming every key
    exists.
    """
    lines = text.splitlines()
    heading_hits = []  # (line_index, canonical_name)

    for i, line in enumerate(lines):
        cleaned = re.sub(r"[:\-–—]+$", "", line.strip()).strip().lower()
        if not cleaned or len(cleaned.split()) > _MAX_HEADING_WORDS:
            continue
        canonical = _ALIAS_TO_CANONICAL.get(cleaned)
        if canonical:
            heading_hits.append((i, canonical))

    sections = {}
    for idx, (line_no, canonical) in enumerate(heading_hits):
        start = line_no + 1
        end = heading_hits[idx + 1][0] if idx + 1 < len(heading_hits) else len(lines)
        content = "\n".join(l.strip() for l in lines[start:end] if l.strip())
        # If the same heading appears twice (rare), keep whichever occurrence
        # captured more content.
        if canonical not in sections or len(content) > len(sections[canonical]):
            sections[canonical] = content
    return sections


def section_items(section_text):
    """Split a section's raw text into a list of individual line items.

    Used for turning a "Projects" or "Certifications" section's free text
    into a clean list for display, without inventing or reformatting the
    candidate's actual wording.
    """
    if not section_text:
        return []
    items = []
    for line in section_text.splitlines():
        line = line.strip(" \t-•*▪◦")
        if line:
            items.append(line)
    return items
