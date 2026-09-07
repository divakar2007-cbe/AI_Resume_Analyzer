import io, re
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from docx import Document
from zipfile import BadZipFile

from .skills_data import SKILL_ALIASES, categorize_skills
from .sections import detect_sections, section_items

class ResumeParsingError(Exception):
    """Raised when a resume file cannot be read or contains no usable text.

    Callers (e.g. the Streamlit UI) should catch this and show a friendly
    message instead of letting the underlying library exception crash the app.
    """
    pass

# Note: bare "be" and "me" were previously included as B.E./M.E. aliases,
# but they collide with the extremely common English words "be" and "me"
# (e.g. "would like to be a part of the team" was misdetected as a "BE"
# degree). Dropped as too ambiguous — "b.e"/"m.e" (with the period) still
# match the genuine abbreviation without the false-positive risk.
EDU=["b.tech","btech","b.e","b.sc","bsc","m.tech","mtech","m.e","m.sc","msc","bca","mca","bachelor","master","phd"]

LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9\-_%]+/?", re.I)
GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9\-_]+/?", re.I)

def extract_text(file):
    data = file.getvalue()
    name = (file.name or "").lower()

    if not data:
        raise ResumeParsingError(
            f"'{file.name}' is empty (0 bytes). Please upload a valid resume file."
        )

    if name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            if getattr(reader, "is_encrypted", False):
                raise ResumeParsingError(
                    f"'{file.name}' is password-protected. Please upload an unlocked PDF."
                )
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except ResumeParsingError:
            raise
        except (PdfReadError, ValueError, TypeError) as e:
            raise ResumeParsingError(
                f"Could not read '{file.name}' as a PDF. The file may be corrupted "
                f"or not a valid PDF. ({e})"
            ) from e
    elif name.endswith(".docx"):
        try:
            return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
        except (BadZipFile, KeyError, ValueError) as e:
            raise ResumeParsingError(
                f"Could not read '{file.name}' as a DOCX. The file may be corrupted "
                f"or not a valid Word document. ({e})"
            ) from e
    elif name.endswith(".txt"):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return data.decode("latin-1")
            except Exception as e:
                raise ResumeParsingError(
                    f"Could not read '{file.name}' as text. ({e})"
                ) from e
    else:
        raise ResumeParsingError(
            f"Unsupported file type for '{file.name}'. Only PDF, DOCX and TXT files are supported."
        )

def extract_skills(text):
    low=text.lower(); found=[]
    for canonical,aliases in SKILL_ALIASES.items():
        if any(re.search(r"(?<!\w)"+re.escape(a)+r"(?!\w)",low) for a in aliases):
            found.append(canonical)
    return sorted(set(found))

def parse_resume(file):
    raw = extract_text(file)
    if not raw or not raw.strip():
        raise ResumeParsingError(
            f"No readable text could be extracted from '{file.name}'. "
            "It may be a scanned/image-based file with no selectable text — "
            "try a text-based PDF/DOCX/TXT export instead."
        )
    text=re.sub(r"[ \t]+"," ",raw).strip()
    email=re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+",raw)
    phone=re.search(r"(?:\+91[\s-]?)?[6-9]\d{9}",raw)
    linkedin=LINKEDIN_RE.search(raw)
    github=GITHUB_RE.search(raw)
    lines=[x.strip() for x in raw.splitlines() if x.strip()]
    name=None
    for line in lines[:10]:
        if 2<=len(line.split())<=4 and not re.search(r"[@\d|]",line) and "resume" not in line.lower():
            name=line.title(); break
    edu=sorted({t.upper() for t in EDU if re.search(r"(?<!\w)"+re.escape(t)+r"(?!\w)",text.lower())})
    years=[float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience",text,re.I)]

    skills = extract_skills(text)
    sections = detect_sections(raw)

    return {
        "filename":file.name,"name":name,
        "email":email.group(0) if email else None,
        "phone":re.sub(r"[^\d+]","",phone.group(0)) if phone else None,
        "linkedin":linkedin.group(0) if linkedin else None,
        "github":github.group(0) if github else None,
        "skills":skills,
        "skills_by_category":categorize_skills(skills),
        "education":", ".join(edu) or None,
        "experience_years":max(years) if years else 0.0,
        "sections":sections,
        "projects":section_items(sections.get("projects","")),
        "certifications":section_items(sections.get("certifications","")),
        "achievements":section_items(sections.get("achievements","")),
        "text":text,
    }
