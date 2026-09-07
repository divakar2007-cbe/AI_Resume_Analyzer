"""Resume ↔ job-description analysis: matching, ATS scoring, and
explainable improvement feedback.

Every score here is application-defined and rule-based (TF-IDF, keyword
overlap, regex-based content checks) — none of it is an official ATS-vendor
score, and nothing here fabricates achievements, numbers, or outcomes that
aren't actually present in the candidate's resume text.
"""
import re
from .parser import extract_skills, SKILL_ALIASES
from .jd_analyzer import analyze_job_description, split_into_sentences

ROLE_SKILLS={
"Python Developer":{"python","git","sql","rest api"},
"Machine Learning Engineer":{"python","machine learning","pandas","numpy","scikit-learn"},
"Data Analyst":{"python","sql","pandas","excel","data analysis"},
"Full Stack Developer":{"html","css","javascript","react","node.js","sql"},
"AI/NLP Engineer":{"python","machine learning","nlp","scikit-learn"},
"Cloud/Backend Developer":{"python","fastapi","docker","aws","sql","rest api"},
"Cybersecurity Analyst":{"linux","python","cybersecurity","git"}}

MEASURABLE_RESULT_RE = r"\b\d+%|\b\d+\+|\b\d+\s*(users|projects|records|customers|requests|hours|days|x)\b"
ACTION_VERB_RE = r"\b(developed|built|created|implemented|designed|improved|achieved|increased|reduced|led|automated|engineered|optimized|launched)\b"

def similarity(a,b):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if not b.strip(): return 0.0,"No job description supplied."
    try:
        v=TfidfVectorizer(stop_words="english",ngram_range=(1,2)).fit_transform([a,b])
        return round(float(cosine_similarity(v[0:1],v[1:2])[0][0])*100,1),"TF-IDF cosine similarity"
    except Exception:
        # e.g. ValueError from an empty vocabulary (both texts are only stopwords/punctuation)
        return 0.0,"Similarity unavailable"

def semantic(a,b):
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        m=SentenceTransformer("all-MiniLM-L6-v2")
        v=m.encode([a,b])
        return round(float(cosine_similarity([v[0]],[v[1]])[0][0])*100,1),"Sentence Transformer semantic similarity"
    except Exception:
        # Broad on purpose: covers ImportError (package not installed),
        # OSError (no network to fetch model weights), and any runtime
        # failure from the optional semantic-matching path — falls back to
        # TF-IDF similarity so the app keeps working either way.
        return similarity(a,b)

def keyword_overlap(a,b):
    ta=set(re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{2,}",a.lower()))
    tb=set(re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{2,}",b.lower()))
    return round(len(ta&tb)/len(tb)*100,1) if tb else 0

def _ats_check_facts(r,jd):
    """Compute each ATS check's pass/fail fact ONCE, alongside a
    human-readable reason. ``ats_checks()`` and ``ats_check_details()`` both
    derive from this single source so the two never drift out of sync.
    """
    text=r["text"]; words=len(text.split())
    sections = r.get("sections", {})

    has_contact = bool(r["email"] and r["phone"])
    n_skills = len(r["skills"])
    has_skills = n_skills>=3
    has_education = bool(r["education"])
    readable_length = 150<=words<=1800
    jd_supplied = bool(jd.strip())
    has_action_verbs = bool(re.search(ACTION_VERB_RE,text,re.I))
    has_measurable = bool(re.search(MEASURABLE_RESULT_RE,text,re.I))
    has_summary = bool(sections.get("summary") or sections.get("objective"))
    has_profile_link = bool(r.get("linkedin") or r.get("github"))

    return {
        "Contact information present": (has_contact,
            "Email and phone were both detected." if has_contact else
            "Email and/or phone number were not detected — add both so recruiters and ATS systems can reach you."),
        "Skills detected": (has_skills,
            f"{n_skills} skill(s) detected." if has_skills else
            f"Only {n_skills} skill(s) detected — add more relevant technical skills you genuinely have."),
        "Education detected": (has_education,
            f"Education detected: {r['education']}." if has_education else
            "No education entry detected — add your degree, institution and graduation year."),
        "Readable length": (readable_length,
            f"{words} words — within the typical 150–1800 word range." if readable_length else
            f"{words} words — outside the typical 150–1800 word range, which can hurt ATS/recruiter readability."),
        "Job description supplied": (jd_supplied,
            "A target job description was provided." if jd_supplied else
            "No job description was provided — skill/keyword matching against a specific role is unavailable."),
        "Action/result language": (has_action_verbs,
            "Strong action verbs (e.g. developed, built, led) were found." if has_action_verbs else
            "No strong action verbs were detected — start bullet points with verbs like 'developed' or 'built'."),
        "Measurable results": (has_measurable,
            "Quantified outcomes (e.g. %, counts) were found." if has_measurable else
            "No measurable outcomes were detected — add numbers if you genuinely have them (%, users served, time saved)."),
        "Professional summary present": (has_summary,
            "A Summary/Objective section was detected." if has_summary else
            "No Summary/Objective section detected — a short summary helps recruiters quickly see your fit."),
        "LinkedIn or GitHub linked": (has_profile_link,
            "A LinkedIn or GitHub link was detected." if has_profile_link else
            "No LinkedIn or GitHub link detected — add one so recruiters can see more of your work."),
    }

def ats_checks(r,jd):
    """Boolean pass/fail checks — kept for backward compatibility with
    callers/tests that only need the true/false result."""
    return {k:v[0] for k,v in _ats_check_facts(r,jd).items()}

def ats_check_details(r,jd):
    """Human-readable check results: [{'check','status','reason'}, ...].

    This is what the UI should render — each row distinguishes the detected
    fact from what's missing, instead of a bare PASS/FAIL label.
    """
    return [{"check":k,"status":v[0],"reason":v[1]} for k,v in _ats_check_facts(r,jd).items()]

def ats_breakdown(r,jd):
    """Explainable ATS Compatibility Score broken into five factors.

    This is an application-defined heuristic score (rule-based, not a real
    ATS vendor's parser) — useful for spotting obvious gaps, not a
    guarantee of how any specific ATS product will actually parse the file.
    """
    sections = r.get("sections", {})
    text = r["text"]
    checks = ats_checks(r, jd)

    # Contact Information: email, phone, a professional profile link (LinkedIn/GitHub)
    contact_items = [bool(r.get("email")), bool(r.get("phone")), bool(r.get("linkedin") or r.get("github"))]
    contact_score = round(sum(contact_items) / len(contact_items) * 100)

    # Section Completeness: summary/objective, education, experience, skills, projects
    has_summary = bool(sections.get("summary") or sections.get("objective"))
    has_education = bool(sections.get("education") or r.get("education"))
    has_experience = bool(sections.get("experience"))
    has_skills = bool(sections.get("skills") or r.get("skills"))
    has_projects = bool(sections.get("projects"))
    completeness_items = [has_summary, has_education, has_experience, has_skills, has_projects]
    section_score = round(sum(completeness_items) / len(completeness_items) * 100)

    # Skills Coverage: overlap with the target JD when supplied, otherwise
    # a simple breadth check (capped so a skill-stuffed resume can't game it).
    if jd.strip():
        js = set(extract_skills(jd)); rs = set(r["skills"])
        skills_score = round(len(rs & js) / len(js) * 100) if js else round(min(len(rs), 10) / 10 * 100)
    else:
        skills_score = round(min(len(r["skills"]), 10) / 10 * 100)

    # Keyword Coverage: raw token overlap against the JD (distinct from
    # canonical-skill overlap above — catches domain terms that aren't in
    # the skill taxonomy).
    keyword_score = round(keyword_overlap(text, jd)) if jd.strip() else skills_score

    # Content Quality: readable length, action-oriented language, measurable results
    quality_keys = ["Readable length", "Action/result language", "Measurable results"]
    quality_score = round(sum(checks[k] for k in quality_keys) / len(quality_keys) * 100)

    overall = round((contact_score + section_score + skills_score + keyword_score + quality_score) / 5)
    return {
        "contact_information": max(0, min(100, contact_score)),
        "section_completeness": max(0, min(100, section_score)),
        "skills_coverage": max(0, min(100, skills_score)),
        "keyword_coverage": max(0, min(100, keyword_score)),
        "content_quality": max(0, min(100, quality_score)),
        "overall": max(0, min(100, overall)),
    }

def ats_factor_explanations(r,jd):
    """Per-factor explanation for the ATS breakdown: what the factor
    measures, and — distinctly — what was detected vs. what's missing.

    Built from the exact same values ``ats_breakdown()`` scores from, so the
    numbers shown and the explanation shown always agree.
    """
    sections = r.get("sections", {})
    bd = ats_breakdown(r, jd)
    checks = ats_checks(r, jd)

    contact_detected, contact_missing = [], []
    (contact_detected if r.get("email") else contact_missing).append("Email")
    (contact_detected if r.get("phone") else contact_missing).append("Phone")
    (contact_detected if (r.get("linkedin") or r.get("github")) else contact_missing).append("LinkedIn/GitHub link")

    section_map = [
        ("Summary/Objective", bool(sections.get("summary") or sections.get("objective"))),
        ("Education", bool(sections.get("education") or r.get("education"))),
        ("Experience", bool(sections.get("experience"))),
        ("Skills", bool(sections.get("skills") or r.get("skills"))),
        ("Projects", bool(sections.get("projects"))),
    ]
    section_detected = [name for name, ok in section_map if ok]
    section_missing = [name for name, ok in section_map if not ok]

    if jd.strip():
        js = set(extract_skills(jd)); rs = set(r["skills"])
        skills_detected, skills_missing = sorted(rs & js), sorted(js - rs)
        skills_note = "Overlap between your detected skills and the target job description's skills."
    else:
        skills_detected, skills_missing = sorted(r["skills"]), []
        skills_note = "No job description supplied — shown as overall skill breadth instead."

    quality_keys = ["Action/result language", "Measurable results", "Readable length"]
    quality_detected = [k for k in quality_keys if checks[k]]
    quality_missing = [k for k in quality_keys if not checks[k]]

    return {
        "contact_information": {"score": bd["contact_information"],
            "what_it_measures": "Whether recruiters and ATS parsers have a way to reach you.",
            "detected": contact_detected, "missing": contact_missing},
        "section_completeness": {"score": bd["section_completeness"],
            "what_it_measures": "Whether the standard resume sections recruiters expect are present.",
            "detected": section_detected, "missing": section_missing},
        "skills_coverage": {"score": bd["skills_coverage"], "what_it_measures": skills_note,
            "detected": skills_detected, "missing": skills_missing},
        "keyword_coverage": {"score": bd["keyword_coverage"],
            "what_it_measures": "Raw word overlap between your resume and the target job description." if jd.strip()
                                 else "No job description supplied — keyword overlap can't be computed.",
            "detected": [], "missing": []},
        "content_quality": {"score": bd["content_quality"],
            "what_it_measures": "Readable length, action-oriented language, and measurable results.",
            "detected": quality_detected, "missing": quality_missing},
    }

def generate_strengths(r, breakdown, checks):
    """Meaningful, resume-specific strengths — never generic compliments."""
    s = []
    if breakdown["contact_information"] == 100:
        s.append("Complete contact information (email, phone and a professional profile link).")
    elif checks["Contact information present"]:
        s.append("Core contact information (email and phone) is present.")
    if len(r["skills"]) >= 6:
        s.append(f"Strong technical skill coverage — {len(r['skills'])} relevant skills detected.")
    if r.get("sections", {}).get("projects"):
        s.append("Projects section detected, showing applied hands-on work.")
    if r.get("education") or r.get("sections", {}).get("education"):
        s.append("Education section detected.")
    if r.get("certifications"):
        s.append(f"{len(r['certifications'])} certification(s) detected.")
    if breakdown["keyword_coverage"] >= 60:
        s.append("Relevant keywords detected for the target job description.")
    if checks["Measurable results"]:
        s.append("Measurable results found in bullet points (quantified impact).")
    return s or ["No standout strengths detected yet — see the improvement suggestions below."]

def generate_weaknesses(r, breakdown, checks, missing, jd_supplied):
    """Meaningful weaknesses with a brief explanation of why each matters."""
    w = []
    sections = r.get("sections", {})
    if not (sections.get("summary") or sections.get("objective")):
        w.append("Missing professional summary — a short summary at the top helps recruiters "
                  "and ATS parsers quickly identify your fit for a role.")
    if not checks["Measurable results"]:
        w.append("Missing measurable achievements — quantified results (%, counts, time saved) "
                  "make your impact easier to verify at a glance.")
    if jd_supplied and breakdown["keyword_coverage"] < 50:
        w.append("Low keyword coverage against the target job description — recruiters and "
                  "ATS keyword filters may not surface this resume for this role.")
    if missing:
        w.append("Missing relevant skills for this role: " + ", ".join(missing[:6]) +
                  ". Only add skills you genuinely have.")
    if not sections.get("projects"):
        w.append("No projects section detected — projects help demonstrate applied skills, "
                  "especially for early-career candidates.")
    if not checks["Readable length"]:
        w.append("Resume length looks too short or too long for comfortable ATS/recruiter "
                  "reading — aim for roughly 1-2 pages of substantive content.")
    if not (r.get("linkedin") or r.get("github")):
        w.append("No LinkedIn or GitHub link detected — a profile link gives recruiters a way "
                  "to see more of your work.")
    return w or ["No major weaknesses detected."]

def job_fit_breakdown(r, jd, jd_info, skill_match, sim, method, kw):
    """Additional diagnostic breakdown of job fit.

    This is a supplementary, separately-labeled metric layer — it never
    replaces or silently changes the primary Job Match score computed in
    ``analyze_resume()``. Some factors here (education/experience alignment)
    use a different, simpler basis than the main score and are labeled as
    such; they are heuristic string/number comparisons, not a guarantee of
    genuine qualification equivalence.
    """
    education_reqs = jd_info.get("education_requirements", [])
    resume_edu = (r.get("education") or "").upper()
    if not education_reqs:
        education = {"score": None, "status": "Not specified in job description",
                     "required": [], "detected": r.get("education")}
    else:
        matched_edu = any(
            req.replace("'", "").upper() in resume_edu or (resume_edu and resume_edu in req.upper())
            for req in education_reqs
        )
        if matched_edu:
            status = "Match"
        elif resume_edu:
            status = "Not clearly matched (heuristic text comparison only — equivalent degrees may not be recognized)"
        else:
            status = "Unknown — no education detected on resume"
        education = {"score": 100 if matched_edu else (0 if resume_edu else None),
                     "status": status, "required": education_reqs, "detected": r.get("education")}

    min_years = jd_info.get("min_experience_years")
    resume_years = r.get("experience_years", 0.0)
    if min_years is None:
        experience = {"score": None, "status": "Not specified in job description",
                      "required_years": None, "detected_years": resume_years}
    else:
        meets = resume_years >= min_years
        exp_score = 100 if meets else (round(min(resume_years / min_years, 1.0) * 100) if min_years > 0 else 100)
        experience = {"score": exp_score,
                      "status": "Meets stated requirement" if meets else "Below stated requirement",
                      "required_years": min_years, "detected_years": resume_years}

    return {
        "technical_skill_alignment": {"score": skill_match,
            "basis": "Overlap between resume skills and job-description skills (same calculation as the Skill Match metric)."},
        "keyword_alignment": {"score": kw,
            "basis": "Raw word overlap between resume text and job description (same calculation as Keyword Match)."},
        "semantic_relevance": {"score": sim, "method": method,
            "basis": "TF-IDF or Sentence-Transformer similarity between resume and job description text (same calculation as Semantic Score)."},
        "education_alignment": education,
        "experience_alignment": experience,
    }

def prioritize_skill_gaps(missing_skills, jd_text, jd_info):
    """For each missing skill, explain why it matters and what to do —
    without ever telling the candidate to fabricate experience they don't
    have. Priority is based on whether the job description treats the skill
    as required or preferred; evidence is a genuine sentence quoted from the
    job description, never invented.
    """
    sentences = split_into_sentences(jd_text) if jd_text.strip() else []
    out = []
    for skill in missing_skills:
        if skill in jd_info.get("required_skills", []):
            priority = "High"
            reason = (f"The target job explicitly requires {skill}, but the resume does not "
                       "currently demonstrate this skill.")
        elif skill in jd_info.get("preferred_skills", []):
            priority = "Medium"
            reason = (f"The target job lists {skill} as a nice-to-have, but the resume does not "
                       "currently demonstrate this skill.")
        else:
            priority = "Low"
            reason = (f"{skill} appears relevant to this role, but the resume does not currently "
                       "demonstrate this skill.")
        evidence = next(
            (s for s in sentences if re.search(r"(?<!\w)"+re.escape(skill)+r"(?!\w)", s.lower())),
            None
        )
        out.append({
            "skill": skill,
            "priority": priority,
            "reason": reason,
            "evidence": evidence or "Mentioned in the job description's detected skill list.",
            "suggested_action": (f"Add genuine {skill} coursework, project work, or professional "
                                  "experience if you actually have it — never add a skill you don't have."),
        })
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(out, key=lambda x: order[x["priority"]])

def analyze_resume(r,jd,use_semantic=False):
    js=set(extract_skills(jd)); rs=set(r["skills"])
    matched=sorted(rs&js); missing=sorted(js-rs)
    skill=round(len(matched)/len(js)*100,1) if js else 0
    sim,method=(semantic(r["text"],jd) if use_semantic else similarity(r["text"],jd)) if jd.strip() else (0,"No job description")
    kw=keyword_overlap(r["text"],jd) if jd.strip() else 0
    checks=ats_checks(r,jd)
    breakdown=ats_breakdown(r,jd)
    quality=breakdown["overall"]
    match=round(.55*skill+.30*sim+.15*kw) if jd.strip() else quality
    final=round(.75*match+.25*quality) if jd.strip() else quality
    role_scores=sorted([(role,round(len(rs&req)/len(req)*100)) for role,req in ROLE_SKILLS.items()],key=lambda x:x[1],reverse=True)
    jd_info=analyze_job_description(jd)
    return {"match_score":max(0,min(100,final)),"ats_score":quality,"skill_match":skill,
            "semantic_score":sim,"keyword_score":kw,"matching_method":method,
            "matched_skills":matched,"missing_skills":missing,"recommended_roles":[x[0] for x in role_scores[:3]],
            "ats_checks":checks,"ats_check_details":ats_check_details(r,jd),
            "ats_breakdown":breakdown,"ats_factor_explanations":ats_factor_explanations(r,jd),
            "strengths":generate_strengths(r,breakdown,checks),
            "weaknesses":generate_weaknesses(r,breakdown,checks,missing,bool(jd.strip())),
            "suggestions":suggestions(r,missing),
            "job_description_insights":jd_info,
            "job_fit_breakdown":job_fit_breakdown(r,jd,jd_info,skill,sim,method,kw),
            "skill_gap_priority":prioritize_skill_gaps(missing,jd,jd_info)}

def suggestions(r,missing):
    s=[]
    if not r["email"]: s.append("Add a professional email address.")
    if not r["phone"]: s.append("Add a phone number.")
    if not (r.get("linkedin") or r.get("github")): s.append("Add a LinkedIn or GitHub profile link.")
    if len(r["skills"])<6: s.append("Add relevant technical skills you genuinely know.")
    if not r["education"]: s.append("Clearly mention degree, institution and graduation year.")
    if not r.get("sections", {}).get("summary") and not r.get("sections", {}).get("objective"):
        s.append("Add a short professional summary at the top of your resume.")
    if missing: s.append("Learn or demonstrate relevant missing skills: "+", ".join(missing[:5])+".")
    if not re.search(MEASURABLE_RESULT_RE,r["text"],re.I):
        s.append("Add measurable results to project/work bullet points.")
    return s or ["Your resume is well structured; tailor keywords to each target job."]

WEAK_VERBS = ["made","did","worked on","responsible for","helped","handled","was involved in","assisted with","in charge of"]
STRONG_VERB_HINT = "developed, built, implemented, designed, automated, optimized, led, engineered"
VAGUE_FILLER_WORDS = ["various","multiple","several","different","stuff","things","tasks","some","numerous","certain"]

def _split_bullet_candidates(text):
    """Split resume text into individual bullet/line candidates.

    PDF text extraction sometimes fails to insert a line break between
    consecutive bullet points (they land on one glued-together line, e.g.
    "Developed and implemented X • Worked on Y"). This splits on bullet
    markers wherever they occur — not just at the start of a line — so each
    bullet is evaluated and displayed on its own, instead of as one garbled
    line.
    """
    candidates = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Split on a bullet marker that occurs anywhere in the line,
        # keeping the marker out of each resulting piece.
        parts = re.split(r"[•▪◦]+", line)
        for part in parts:
            part = part.strip(" \t-*")
            if part:
                candidates.append(part)
    return candidates

def _looks_like_bullet(line):
    """Heuristic: either an explicit bullet marker, or a short-to-medium
    sentence-length line typical of resume bullet points."""
    if re.match(r"^[\-•*▪◦]|^\d+[.)]\s", line.strip()):
        return True
    words = len(line.split())
    return 4 <= words <= 40

def detect_bullet_issues(line):
    """Return a list of specific, explainable issues found in a bullet line.

    Purely rule-based (regex/keyword checks) — flags what's missing without
    ever inventing a technology, number, or outcome that isn't in the text.
    """
    issues = []
    low = line.lower()
    if any(re.search(r"(?<!\w)"+re.escape(v)+r"(?!\w)",low) for v in WEAK_VERBS):
        issues.append("weak_verb")
    if not re.search(MEASURABLE_RESULT_RE, line, re.I):
        issues.append("no_measurable_outcome")
    if not any(re.search(r"(?<!\w)"+re.escape(a)+r"(?!\w)", low) for aliases in SKILL_ALIASES.values() for a in aliases):
        issues.append("no_technology_named")
    if len(line.split()) > 30:
        issues.append("too_long")
    if re.match(r"^(responsible for|in charge of|worked on|helped with)\b", low):
        issues.append("responsibility_only")
    if any(re.search(r"(?<!\w)"+re.escape(w)+r"(?!\w)", low) for w in VAGUE_FILLER_WORDS) and len(line.split()) <= 10:
        issues.append("vague_description")
    return issues

# Full standalone sentences (never fabricate a technology, number, or
# outcome) — worded to identify the actual problem and say what to add.
_ISSUE_TIPS = {
    "weak_verb": ("Consider replacing the weak opening verb with a stronger action verb "
                  f"(e.g., {STRONG_VERB_HINT}) and specify the technology, feature, or outcome involved."),
    "no_technology_named": "Name the specific technology, tool, or platform you used.",
    "no_measurable_outcome": ("Add a measurable outcome if you have one, such as processing time, "
                              "users served, accuracy, performance improvement, or project scale — "
                              "do not invent a number you don't have."),
    "too_long": "This line is long — consider splitting it into one focused idea.",
    "responsibility_only": "Describe what you actually built or achieved, not just what you were assigned or responsible for.",
    "vague_description": "This is vague — replace general words like 'various' or 'multiple' with the specific technology, task, or number involved.",
}

def build_resume_improvements(text, sections=None):
    """Flag weak bullet points and suggest HOW to strengthen them.

    Returns a list of (original_bullet, suggestion) tuples. This never
    invents technologies, numbers, or outcomes that are not already present
    in the resume text — it only points out what's missing and asks the
    candidate to add it if it's genuinely true.

    When ``sections`` is supplied (from parser.parse_resume's section
    detection) and contains an Experience or Projects section, scanning is
    restricted to that content — this avoids flagging unrelated lines from
    the Summary/Education/Skills sections as "weak bullets". Falls back to
    scanning the full text when no section data is available.
    """
    if sections and (sections.get("experience") or sections.get("projects")):
        scan_text = "\n".join(filter(None, [sections.get("experience"), sections.get("projects")]))
    else:
        scan_text = text

    out=[]
    for line in _split_bullet_candidates(scan_text):
        if not _looks_like_bullet(line):
            continue
        issues = detect_bullet_issues(line)
        if not issues:
            continue
        tips = [_ISSUE_TIPS[i] for i in issues if i in _ISSUE_TIPS]
        suggestion = " ".join(tips)
        out.append((line, suggestion))
    return out[:8]

def compare_resumes(resume_a, result_a, resume_b, result_b):
    """Compare two analyses of the same target role — e.g. an original
    resume vs. a revised version. Purely a side-by-side diff of numbers
    already computed by ``analyze_resume()``; it doesn't recompute or
    reinterpret anything.
    """
    def _sections_count(resume):
        return len([k for k, v in resume.get("sections", {}).items() if v])

    rows = [
        {"Metric": "ATS Score", "Version A": result_a["ats_score"], "Version B": result_b["ats_score"],
         "Difference": result_b["ats_score"] - result_a["ats_score"]},
        {"Metric": "Job Match", "Version A": result_a["match_score"], "Version B": result_b["match_score"],
         "Difference": result_b["match_score"] - result_a["match_score"]},
        {"Metric": "Skill Match", "Version A": result_a["skill_match"], "Version B": result_b["skill_match"],
         "Difference": round(result_b["skill_match"] - result_a["skill_match"], 1)},
        {"Metric": "Keyword Coverage", "Version A": result_a["keyword_score"], "Version B": result_b["keyword_score"],
         "Difference": round(result_b["keyword_score"] - result_a["keyword_score"], 1)},
        {"Metric": "Semantic Relevance", "Version A": result_a["semantic_score"], "Version B": result_b["semantic_score"],
         "Difference": round(result_b["semantic_score"] - result_a["semantic_score"], 1)},
        {"Metric": "Skill Gaps (count)", "Version A": len(result_a["missing_skills"]),
         "Version B": len(result_b["missing_skills"]),
         "Difference": len(result_b["missing_skills"]) - len(result_a["missing_skills"])},
        {"Metric": "Detected Sections (count)", "Version A": _sections_count(resume_a),
         "Version B": _sections_count(resume_b),
         "Difference": _sections_count(resume_b) - _sections_count(resume_a)},
    ]

    resolved = sorted(set(result_a["missing_skills"]) - set(result_b["missing_skills"]))
    newly_missing = sorted(set(result_b["missing_skills"]) - set(result_a["missing_skills"]))
    return {"metrics": rows, "skills_resolved_in_b": resolved, "skills_newly_missing_in_b": newly_missing}
