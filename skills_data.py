"""Canonical skill taxonomy — the single source of truth for skill
recognition and categorization.

Kept as data (not hard-coded inline in app.py or duplicated across
services) so it stays easy to extend: add an alias or a new skill here and
every consumer (parser, analyzer, UI) picks it up automatically.

Structure:
    SKILL_TAXONOMY = {
        "<Category>": {
            "<canonical skill name>": ["<alias 1>", "<alias 2>", ...],
            ...
        },
        ...
    }

Canonical names are intentionally unchanged from the original flat
SKILL_ALIASES dictionary (Phase 1) so existing role-matching logic in
analyzer.py and recommender.py keeps working without modification.
"""

SKILL_TAXONOMY = {
    "Programming Languages": {
        "python": ["python"],
        "java": ["java"],
        "c++": ["c++", "cpp"],
        "javascript": ["javascript", "js"],
        "typescript": ["typescript", "ts"],
    },
    "Web Development": {
        "html": ["html", "html5"],
        "css": ["css", "css3"],
        "react": ["react", "reactjs", "react.js"],
        "node.js": ["node.js", "nodejs", "node js"],
        "django": ["django"],
        "flask": ["flask"],
        "fastapi": ["fastapi"],
        "rest api": ["rest api", "restful api", "api"],
    },
    "Databases": {
        "sql": ["sql", "mysql", "postgresql", "postgres"],
        "mongodb": ["mongodb", "mongo db"],
    },
    "Cloud": {
        "aws": ["aws", "amazon web services"],
        "azure": ["azure"],
        "gcp": ["gcp", "google cloud"],
    },
    "DevOps": {
        "docker": ["docker"],
        "git": ["git"],
        "github": ["github"],
        "linux": ["linux", "ubuntu"],
    },
    "Machine Learning": {
        "machine learning": ["machine learning", "ml"],
        "deep learning": ["deep learning"],
        # Normalizes "scikit learn" / "scikit-learn" / "sklearn" to one skill.
        "scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
        "tensorflow": ["tensorflow"],
        "pytorch": ["pytorch"],
    },
    "Artificial Intelligence": {
        "nlp": ["nlp", "natural language processing"],
    },
    "Data Science": {
        "pandas": ["pandas"],
        "numpy": ["numpy"],
        "data analysis": ["data analysis", "data analytics"],
        "power bi": ["power bi"],
        "tableau": ["tableau"],
        "excel": ["excel", "microsoft excel"],
    },
    "Tools": {
        "data structures": ["data structures", "dsa"],
        "algorithms": ["algorithms", "algorithm"],
    },
    "Cybersecurity": {
        "cybersecurity": ["cybersecurity", "cyber security"],
    },
    "Soft Skills": {
        # Deliberately small and literal: a soft skill is only credited when
        # the resume states the word itself, never inferred from context.
        "communication": ["communication"],
        "leadership": ["leadership"],
        "teamwork": ["teamwork", "team work"],
        "problem solving": ["problem solving", "problem-solving"],
    },
}


def _build_alias_map():
    """canonical skill -> list of alias strings (flattened, Phase-1-compatible)."""
    flat = {}
    for _category, skills in SKILL_TAXONOMY.items():
        for canonical, aliases in skills.items():
            flat[canonical] = aliases
    return flat


def _build_category_map():
    """canonical skill -> category name."""
    lookup = {}
    for category, skills in SKILL_TAXONOMY.items():
        for canonical in skills:
            lookup[canonical] = category
    return lookup


# Flat views, generated once at import time.
SKILL_ALIASES = _build_alias_map()
SKILL_CATEGORY = _build_category_map()
CATEGORIES = list(SKILL_TAXONOMY.keys())


def categorize_skills(skills):
    """Group a flat list of canonical skill names by category.

    Returns an ordered dict {category: [skills]} containing only categories
    that have at least one matched skill, in SKILL_TAXONOMY's declared order.
    """
    grouped = {}
    for category in CATEGORIES:
        matched = [s for s in skills if SKILL_CATEGORY.get(s) == category]
        if matched:
            grouped[category] = sorted(matched)
    return grouped
