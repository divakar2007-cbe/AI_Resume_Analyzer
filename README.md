# 🤖 AI-Powered Resume Analyzer & Job Recommendation System

A portfolio-ready AI/NLP recruitment assistant that analyzes resumes, compares
them with job descriptions, calculates an explainable job-fit score, detects
skill gaps, performs ATS-style checks, recommends jobs, improves weak resume
bullets, and ranks multiple candidates for recruiters.

## Features

### Candidate mode
- PDF/DOCX/TXT resume upload, with graceful error handling for empty,
  corrupted, encrypted, unsupported, or scanned/image-only files
- Structured resume extraction: contact info (email, phone, LinkedIn,
  GitHub), education, experience, skills, projects, certifications,
  achievements
- Resume section detection (Summary, Objective, Education, Experience,
  Projects, Skills, Certifications, Achievements, Publications, Contact)
- Technical skill extraction with alias normalization (e.g. "scikit-learn" /
  "scikit learn" / "sklearn" all recognize as one skill) and categorization
  (Programming Languages, Web Development, Databases, Cloud, DevOps,
  Machine Learning, Artificial Intelligence, Data Science, Tools,
  Cybersecurity, Soft Skills)
- Job description analysis with matched and missing skills
- TF-IDF similarity, with optional Sentence Transformer semantic similarity
- Explainable Job Match Score
- Explainable ATS Compatibility Score with a five-factor breakdown (see
  below)
- Resume strengths and weaknesses, generated from what was actually
  detected in the resume (not generic praise)
- Recommended job roles
- Job recommendation table
- Personalized improvement suggestions
- Weak bullet-point detection with guidance on how to strengthen each one —
  never fabricates technologies, numbers, or outcomes not already present
  in the resume
- Job description intelligence: rule-based extraction of job title,
  required vs. preferred skills, education/experience requirements, tools,
  and domain keywords — undetected fields show as empty/unknown, never
  guessed
- Enhanced Job Fit Breakdown: a supplementary diagnostic layer (Technical
  Skill Alignment, Keyword Alignment, Semantic Relevance, Education
  Alignment, Experience Alignment) — additive only, never replaces the
  primary Job Match score
- Prioritized skill gaps: each missing skill labeled High/Medium/Low
  priority with a reason, a genuine quote from the job description as
  evidence, and a suggested action that never tells the candidate to
  fabricate experience
- JSON report download

### Resume Version Comparison (optional)
- Compare an original vs. a revised resume against the same job description
- Side-by-side diff of ATS score, Job Match, Skill Match, Keyword Coverage,
  Semantic Relevance, skill gaps, and detected sections
- Fully optional — does not alter the single-resume analysis workflow

### Recruiter mode
- Upload multiple resumes
- Compare every resume with one job description
- Rank candidates automatically
- Show match score, ATS score, matched skills and gaps, plus diagnostic
  Keyword/Experience/Education alignment columns
- Highlight top candidate, with an explanation of its remaining
  high-priority skill gaps

### ML evaluation
- Labeled demonstration dataset
- TF-IDF + Logistic Regression pipeline
- Train/test split
- Accuracy, precision, recall and F1
- Confusion matrix

## ATS Compatibility Score — methodology

The ATS Compatibility Score is an **application-defined heuristic score**,
not an official score from any ATS vendor and not a guarantee of how a
specific real-world ATS product will parse a given file. It is a rule-based
average of five factors, each 0-100:

| Factor | What it measures |
|---|---|
| Contact Information | Email present, phone present, LinkedIn/GitHub link present (each counts equally) |
| Section Completeness | Whether a Summary/Objective, Education, Experience, Skills, and Projects section were detected |
| Skills Coverage | Overlap between the resume's detected skills and the target job description's skills (or a breadth check when no JD is supplied) |
| Keyword Coverage | Raw token overlap between resume text and the job description |
| Content Quality | Readable length (150-1800 words), presence of action-oriented language, presence of measurable results (%, counts, etc.) |

The overall score is the unweighted average of the five factors. No machine
learning model, and no external "ATS simulation," is involved in this
score — every input is directly inspectable in the resume text, which is
why it's shown alongside a full breakdown rather than a single number.

## Job matching methodology

The **primary Job Match score** is a weighted blend: 55% skill overlap, 30%
similarity (TF-IDF or, optionally, Sentence-Transformer semantic similarity),
15% keyword overlap — then blended 75/25 with the ATS Compatibility Score.
This methodology is unchanged from earlier project phases.

The **Enhanced Job Fit Breakdown** shown alongside it is an additional,
separately-labeled diagnostic layer — Technical Skill Alignment, Keyword
Alignment, and Semantic Relevance reuse the exact same underlying numbers as
the primary score (just presented per-factor), while Education Alignment and
Experience Alignment are simple heuristic text/number comparisons against
whatever the job description analyzer detected. **This breakdown never
overwrites or silently changes the primary Job Match score** — if a factor
can't be determined (e.g. the job description states no experience
requirement), it's shown as "Not specified" rather than guessed.

## Job description extraction methodology

`services/jd_analyzer.py` splits the pasted job description into sentences
and applies rule-based checks:
- **Job title**: the first short line of the text (heuristic — not always
  present or correctly identified).
- **Required vs. preferred skills**: skills detected via the same skill
  taxonomy used for resumes; a skill is bucketed as *preferred* only if its
  sentence contains an explicit marker word ("preferred", "nice to have",
  "a plus", "bonus", "desirable"), otherwise it's treated as *required* (the
  common case: a plain requirements list with no such markers).
- **Education/experience requirements**: regex over common phrasings
  ("Bachelor's degree", "3+ years of experience", etc.).
- **Domain keywords**: the most frequent non-stopword, non-skill words in
  the text — a simple frequency signal, not a topic model.

None of this claims perfect extraction — undetected fields come back empty
or `None`, never invented.

## Skill-gap analysis methodology

For each skill present in the job description but missing from the resume,
the system reports a priority (High if the job description's required-skill
extraction includes it, Medium if only the preferred-skill extraction
includes it, Low otherwise), a plain-language reason, a genuine sentence
quoted from the job description as evidence, and a suggested action. The
suggested action always asks the candidate to add the skill **only if they
genuinely have relevant experience** — it never instructs fabricating a
skill, and evidence text is always a real excerpt from the pasted job
description, never generated.

## Recruiter ranking methodology

Recruiter Ranking runs the exact same `analyze_resume()` pipeline used for
Candidate Analysis, once per uploaded resume, against one shared job
description. The ranking table's primary sort key is the same Job Match
score as Candidate Analysis; the additional Keyword/Experience/Education
Alignment columns are the same diagnostic factors described above, shown
per-candidate so a recruiter can see *why* two similarly-scored candidates
differ.

## Resume version comparison

An optional feature (its own tab) for comparing two resumes — e.g. an
original and a revised draft — against the same job description. It reruns
the standard analysis on each and reports the numeric difference for ATS
Score, Job Match, Skill Match, Keyword Coverage, Semantic Relevance, skill
gap count, and detected section count, plus which specific skills were
resolved or newly missing between the two versions. It does not change or
share state with the single-resume Candidate Analysis workflow.

## Responsible AI & limitations

- The **ATS Compatibility Score** is an application-defined heuristic — not
  an official score from any ATS vendor, and not a guarantee of how a real
  ATS product will parse a given file.
- **Skill, section, and job-description extraction** are heuristic/regex/
  taxonomy-based, not a trained NLP model — unusual formats, non-standard
  headings, or atypical phrasing may be missed or misread.
- **Semantic matching** is embedding cosine similarity, not a judgment of
  genuine candidate-role fit.
- **Education/Experience Alignment** are simple text/number comparisons —
  they do not recognize equivalent degrees or credentials the job
  description didn't literally name.
- These results are intended to **assist** a candidate's self-review or a
  recruiter's initial screening — never to make a final hiring decision on
  their own.
- The system is designed to never invent skills, achievements, numbers,
  experience, or job-description requirements that aren't present in the
  source text; where something can't be reliably detected, it is shown as
  empty/unknown rather than guessed.

## Resume parsing

Supports PDF, DOCX and TXT. Text extraction failures (corrupted files,
encrypted PDFs, empty files, unsupported types, or scanned/image-only PDFs
with no selectable text) are caught and surfaced as a clear message in the
UI rather than crashing the app.

## Skill extraction & normalization

Skills are recognized from a maintained alias dictionary
(`services/skills_data.py`) — common spelling/formatting variants (e.g.
"scikit-learn" / "scikit learn" / "sklearn", "JavaScript" / "javascript")
normalize to one canonical skill, and each skill is grouped into a category
for display. This is a curated list, not an exhaustive skills database — it
covers common software/data-role skills and can be extended by adding
entries to `SKILL_TAXONOMY` in that file.

## Architecture

Resume PDF/DOCX/TXT
→ Text extraction (services/parser.py)
→ Section detection (services/sections.py)
→ Skill extraction & categorization (services/skills_data.py)
→ Structured resume profile (contact, sections, skills, projects,
  certifications, achievements)

Job description
→ Structured extraction: title, required/preferred skills, education,
  experience, tools, domain keywords (services/jd_analyzer.py)

Resume + Job
→ Skill matching
→ TF-IDF / optional semantic similarity
→ Explainable Job Match score + ATS Compatibility breakdown
→ Enhanced Job Fit Breakdown (diagnostic layer)
→ Prioritized skill gaps
→ Job recommendations
→ Resume improvements (bullet-point analysis)

Recruiter:
Multiple resumes → same pipeline → candidate ranking with alignment columns

Resume comparison (optional):
Two resumes → same pipeline run twice → side-by-side metric diff

## Testing

```bash
python -m py_compile app.py services/*.py
python -m unittest discover -s tests -v
```

Tests cover: file parsing and error handling (PDF/DOCX/TXT, corrupted,
encrypted, empty, unsupported files), skill extraction and normalization,
section detection, ATS breakdown and check explanations, bullet-point issue
detection (including that no suggestion ever contains a fabricated number),
job description extraction, skill-gap prioritization, the enhanced job-fit
breakdown, resume version comparison, the ML training pipeline (including
missing/malformed/insufficient-dataset error handling), and the SQLite
history store.

## Run

Python 3.10+ recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Install:
```bash
pip install -r requirements.txt
```

Optional semantic matching:
```bash
pip install -r requirements-semantic.txt
```

Run:
```bash
streamlit run app.py
```

## Important ML disclaimer

The bundled dataset is intentionally small and only demonstrates that the
training/evaluation pipeline works. Its metrics must NOT be presented as the
real-world accuracy of the product. For an academic research claim, replace
the demo data with a larger representative labeled dataset and document the
data collection, preprocessing, split strategy and evaluation.

The Job Match Score is an application-defined screening score, not a guaranteed
ATS-vendor score or hiring probability.

## Suggested resume entry

**AI-Powered Resume Analyzer & Job Recommendation System**  
Developed an NLP/ML recruitment assistant that extracts resume information,
matches candidate skills with job descriptions, generates explainable job-fit
scores, identifies skill gaps, recommends suitable roles, improves weak resume
bullets, and ranks multiple candidates for recruiter screening.

**Technologies:** Python, NLP, Scikit-learn, Streamlit, SQLite, PDF/DOCX Processing
