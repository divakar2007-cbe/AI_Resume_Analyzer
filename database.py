import sqlite3
from pathlib import Path

# Stored under database/ (gitignored) rather than the project root, so a
# candidate's analysis history never accidentally ends up committed to git.
DB_DIR = Path(__file__).resolve().parent.parent / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB = DB_DIR / "resume_analyzer.db"


class DatabaseError(Exception):
    """Raised when the local SQLite history store cannot be read or written."""
    pass


def connect():
    try:
        c = sqlite3.connect(DB)
        c.execute("""CREATE TABLE IF NOT EXISTS analyses(
        id INTEGER PRIMARY KEY AUTOINCREMENT, analyzed_at TEXT, candidate TEXT, filename TEXT,
        match_score INTEGER, ats_score INTEGER, matched TEXT, gaps TEXT)""")
        c.commit()
        return c
    except sqlite3.Error as e:
        raise DatabaseError(f"Could not open the local history database: {e}") from e


def save_analysis(r, a):
    from datetime import datetime
    c = None
    try:
        c = connect()
        c.execute(
            "INSERT INTO analyses(analyzed_at,candidate,filename,match_score,ats_score,matched,gaps) VALUES(?,?,?,?,?,?,?)",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                r.get("name") or "Unknown",
                r.get("filename", "unknown"),
                a.get("match_score", 0),
                a.get("ats_score", 0),
                ", ".join(a.get("matched_skills", [])),
                ", ".join(a.get("missing_skills", [])),
            ),
        )
        c.commit()
    except sqlite3.Error as e:
        raise DatabaseError(f"Could not save this analysis to history: {e}") from e
    finally:
        if c is not None:
            c.close()


def get_history():
    c = None
    try:
        c = connect()
        rows = c.execute(
            "SELECT analyzed_at,candidate,filename,match_score,ats_score,matched,gaps "
            "FROM analyses ORDER BY id DESC LIMIT 100"
        ).fetchall()
    except sqlite3.Error as e:
        raise DatabaseError(f"Could not load analysis history: {e}") from e
    finally:
        if c is not None:
            c.close()
    cols = ["Analyzed At", "Candidate", "File", "Match Score", "ATS Score", "Matched Skills", "Skill Gaps"]
    return [dict(zip(cols, r)) for r in rows]
