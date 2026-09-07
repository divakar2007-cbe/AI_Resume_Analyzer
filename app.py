import json
import pandas as pd
import streamlit as st

from services.parser import parse_resume, ResumeParsingError
from services.analyzer import analyze_resume, build_resume_improvements, ats_check_details, ats_factor_explanations, compare_resumes
from services.recommender import recommend_jobs
from services.ml_model import train_and_evaluate, MLModelError
from services.database import save_analysis, get_history, DatabaseError

st.set_page_config(page_title="AI Resume Analyzer", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.hero {padding: 24px; border-radius: 18px; background: linear-gradient(135deg,#0f172a,#1e3a8a); color:white;}
.hero h1 {font-size: 40px; margin:0;}
.hero p {font-size:17px; opacity:.9;}
.card {padding:18px; border:1px solid #e2e8f0; border-radius:14px; margin:8px 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🤖 AI Resume Analyzer</h1>
<p>Resume screening • ATS compatibility scoring • semantic matching • skill gaps • job recommendations • recruiter ranking</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    semantic = st.checkbox("Use semantic AI matching", value=False)
    st.caption("Semantic mode uses Sentence Transformers when installed; otherwise the app remains fully usable with TF-IDF.")
    with st.expander("ℹ️ Methodology & Limitations"):
        st.markdown(
            "- The **ATS Compatibility Score** is an application-defined heuristic — "
            "not an official score from any ATS vendor.\n"
            "- **Skill extraction** is heuristic/taxonomy-based (a maintained alias "
            "dictionary), not a trained NLP model — unusual phrasing may be missed.\n"
            "- **Semantic matching** is similarity-based (embedding cosine similarity), "
            "not a judgment of genuine qualification.\n"
            "- **Job description extraction** (title, required/preferred skills, education, "
            "experience) is rule-based; undetected fields are shown as empty/unknown, never guessed.\n"
            "- These results are meant to **assist** a candidate or recruiter, not to make a "
            "final hiring decision.\n"
            "- The system never invents skills, achievements, numbers, or experience that "
            "aren't present in the source text."
        )

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Candidate Analysis", "👥 Recruiter Ranking", "🧪 Model Evaluation", "🗂️ History", "🆚 Compare Versions"
])

def show_analysis(resume, result):
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Job Match", f"{result['match_score']}%")
    c2.metric("ATS Score", f"{result['ats_score']}%")
    c3.metric("Skills Matched", len(result["matched_skills"]))
    c4.metric("Skill Gaps", len(result["missing_skills"]))
    st.progress(result["match_score"]/100)

    st.subheader("📊 Explainable Score")
    score_df = pd.DataFrame({
        "Component":["Skill Match","Semantic Similarity","Keyword Match","Resume Quality"],
        "Score":[result["skill_match"],result["semantic_score"],result["keyword_score"],result["ats_score"]]
    })
    st.bar_chart(score_df.set_index("Component"))

    jd_info = result["job_description_insights"]
    if jd_info["job_title"] or jd_info["required_skills"] or jd_info["preferred_skills"]:
        with st.expander("🧭 Job Description Insights (extracted from the pasted job description)"):
            st.caption("Rule-based extraction — shown empty/unknown where nothing reliable was detected, never guessed.")
            st.write(f"**Detected job title:** {jd_info['job_title'] or 'Not detected'}")
            st.write(f"**Required skills:** {', '.join(jd_info['required_skills']) or 'None detected'}")
            st.write(f"**Preferred/nice-to-have skills:** {', '.join(jd_info['preferred_skills']) or 'None detected'}")
            st.write(f"**Education requirements:** {', '.join(jd_info['education_requirements']) or 'Not specified'}")
            st.write(f"**Minimum experience:** {jd_info['min_experience_years']} years" if jd_info['min_experience_years'] is not None else "**Minimum experience:** Not specified")
            st.write(f"**Domain keywords:** {', '.join(jd_info['domain_keywords']) or 'None detected'}")

        with st.expander("🎯 Enhanced Job Fit Breakdown (additional diagnostic metrics)"):
            st.caption(
                "These factors supplement — and never replace — the primary Job Match score above. "
                "Education/Experience alignment are simple heuristic comparisons, not a guarantee of genuine fit."
            )
            fit = result["job_fit_breakdown"]
            for label, key in [("Technical Skill Alignment","technical_skill_alignment"),
                                ("Keyword Alignment","keyword_alignment"),
                                ("Semantic Relevance","semantic_relevance"),
                                ("Education Alignment","education_alignment"),
                                ("Experience Alignment","experience_alignment")]:
                f = fit[key]
                score = f["score"]
                score_label = f"{score}/100" if score is not None else "N/A"
                note = f.get("status") or f.get("basis","")
                st.markdown(f"**{label}:** {score_label} — {note}")

    st.subheader("🧾 ATS Compatibility Breakdown")
    st.caption(
        "Application-defined heuristic score — not an official score from any ATS vendor. "
        "See the Model Evaluation tab / README for scoring methodology."
    )
    bd = result["ats_breakdown"]
    st.metric("ATS Compatibility", f"{bd['overall']}/100")
    bd_df = pd.DataFrame({
        "Factor": ["Contact Information","Section Completeness","Skills Coverage","Keyword Coverage","Content Quality"],
        "Score": [bd["contact_information"],bd["section_completeness"],bd["skills_coverage"],bd["keyword_coverage"],bd["content_quality"]],
    })
    st.bar_chart(bd_df.set_index("Factor"))

    factor_labels = {
        "contact_information": "Contact Information",
        "section_completeness": "Section Completeness",
        "skills_coverage": "Skills Coverage",
        "keyword_coverage": "Keyword Coverage",
        "content_quality": "Content Quality",
    }
    explanations = result["ats_factor_explanations"]
    for factor_key, label in factor_labels.items():
        exp = explanations[factor_key]
        with st.expander(f"{label} — {exp['score']}/100"):
            st.write(exp["what_it_measures"])
            if exp["detected"]:
                st.write("**Detected:** " + ", ".join(exp["detected"]))
            if exp["missing"]:
                st.write("**Missing:** " + ", ".join(exp["missing"]))
            if not exp["detected"] and not exp["missing"]:
                st.write("_No specific items tracked for this factor._")

    a,b = st.columns(2)
    with a:
        st.subheader("👤 Candidate")
        st.write(f"**Name:** {resume.get('name') or 'Not detected'}")
        st.write(f"**Email:** {resume.get('email') or 'Not detected'}")
        st.write(f"**Phone:** {resume.get('phone') or 'Not detected'}")
        st.write(f"**LinkedIn:** {resume.get('linkedin') or 'Not detected'}")
        st.write(f"**GitHub:** {resume.get('github') or 'Not detected'}")
        st.write(f"**Education:** {resume.get('education') or 'Not detected'}")
        st.write(f"**Experience:** {resume.get('experience_years',0)} years")
    with b:
        st.subheader("💼 Recommended Roles")
        for role in result["recommended_roles"]:
            st.write("✅", role)

    with st.expander("📑 Resume Sections Detected"):
        sections = resume.get("sections", {})
        all_section_names = ["summary","objective","education","experience","projects",
                              "skills","certifications","achievements","publications"]
        found = [s.replace("_"," ").title() for s in all_section_names if sections.get(s)]
        missing_sections = [s.replace("_"," ").title() for s in all_section_names if not sections.get(s)]
        st.write("**Detected:**", ", ".join(found) or "None")
        st.write("**Not detected:**", ", ".join(missing_sections) or "None")
        if resume.get("projects"):
            st.write("**Projects listed:**")
            for p in resume["projects"]:
                st.write("•", p)
        if resume.get("certifications"):
            st.write("**Certifications listed:**")
            for cItem in resume["certifications"]:
                st.write("•", cItem)
        if resume.get("achievements"):
            st.write("**Achievements listed:**")
            for ac in resume["achievements"]:
                st.write("•", ac)

    with st.expander("🏷️ Skills by Category"):
        by_cat = resume.get("skills_by_category", {})
        if by_cat:
            for cat, skills in by_cat.items():
                st.write(f"**{cat}:** {', '.join(skills)}")
        else:
            st.info("No categorized skills detected.")

    st.subheader("🧠 Skill Analysis")
    x,y=st.columns(2)
    with x:
        st.markdown("### ✅ Matched")
        st.success(", ".join(result["matched_skills"]) or "None")
    with y:
        st.markdown("### ⚠️ Missing / Gaps")
        st.warning(", ".join(result["missing_skills"]) or "No major gaps")

    if result["skill_gap_priority"]:
        with st.expander("🔧 Prioritized Skill Gaps (why each one matters)"):
            for gap in result["skill_gap_priority"]:
                st.markdown(f"**{gap['skill']}** — {gap['priority']} priority")
                st.write(gap["reason"])
                st.caption(f"Evidence from job description: \"{gap['evidence']}\"")
                st.write(f"Suggested action: {gap['suggested_action']}")
                st.divider()

    st.subheader("📋 ATS Checks")
    st.caption("Each check shows what was detected — not just PASS/IMPROVE.")
    for item in result["ats_check_details"]:
        icon = "✅" if item["status"] else "🔸"
        st.markdown(f"{icon} **{item['check']}** — {item['reason']}")

    s_col, w_col = st.columns(2)
    with s_col:
        st.subheader("💪 Strengths")
        for s in result["strengths"]:
            st.write("✓", s)
    with w_col:
        st.subheader("⚠️ Weaknesses")
        for w in result["weaknesses"]:
            st.write("⚠", w)

    st.subheader("✍️ Resume Improvement Suggestions")
    for s in result["suggestions"]:
        st.write("•", s)

    st.subheader("✨ Bullet-Point Quality")
    improvements = build_resume_improvements(resume["text"], resume.get("sections"))
    if improvements:
        for old,new in improvements:
            st.markdown(f"**Bullet:** {old}")
            st.markdown(f"**How to strengthen it:** {new}")
    else:
        st.info("No obvious weak project/work bullets were detected.")

    st.subheader("💼 Job Recommendations")
    jobs = recommend_jobs(resume["skills"])
    st.dataframe(pd.DataFrame(jobs), use_container_width=True, hide_index=True)

    report = {"resume":resume, "analysis":result, "job_recommendations":jobs}
    st.download_button(
        "⬇️ Download Full JSON Report",
        json.dumps(report, indent=2),
        "resume_analysis_report.json",
        "application/json",
        use_container_width=True
    )

with tab1:
    uploaded = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=["pdf","docx","txt"])
    jd = st.text_area("Paste Target Job Description", height=190)
    if not jd.strip():
        st.caption("⚠️ No job description entered yet — you'll still get an ATS/quality score, but no job-match score.")
    if uploaded and st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
        with st.spinner("Running resume parsing and AI analysis..."):
            try:
                resume = parse_resume(uploaded)
                result = analyze_resume(resume, jd, use_semantic=semantic)
                st.session_state["resume"] = resume
                st.session_state["result"] = result
                try:
                    save_analysis(resume, result)
                except DatabaseError as e:
                    st.warning(f"Analysis complete, but it could not be saved to history: {e}")
            except ResumeParsingError as e:
                st.error(f"⚠️ {e}")
            except Exception as e:
                st.error(f"Unexpected error while analyzing this resume: {e}")
    if "result" in st.session_state:
        show_analysis(st.session_state["resume"], st.session_state["result"])

with tab2:
    st.subheader("👥 Recruiter Candidate Ranking")
    st.caption("Upload multiple resumes and compare them against one target job.")
    job = st.text_area("Recruiter Job Description", height=160, key="recruiter_jd")
    resumes = st.file_uploader(
        "Upload multiple resumes",
        type=["pdf","docx","txt"],
        accept_multiple_files=True,
        key="multi"
    )
    if resumes and not job.strip():
        st.warning("⚠️ Please paste a job description before ranking candidates.")
    if resumes and job.strip() and st.button("🏆 Rank Candidates", type="primary"):
        rows=[]
        details={}
        for file in resumes:
            try:
                r=parse_resume(file)
                a=analyze_resume(r,job,use_semantic=semantic)
                fit=a["job_fit_breakdown"]
                candidate_name=r.get("name") or file.name
                rows.append({
                    "Candidate":candidate_name,
                    "Match Score":a["match_score"],
                    "ATS Score":a["ats_score"],
                    "Matched Skills":len(a["matched_skills"]),
                    "Skill Gaps":len(a["missing_skills"]),
                    "Keyword Alignment":fit["keyword_alignment"]["score"],
                    "Experience Alignment":fit["experience_alignment"]["score"] if fit["experience_alignment"]["score"] is not None else "N/A",
                    "Education Alignment":fit["education_alignment"]["score"] if fit["education_alignment"]["score"] is not None else "N/A",
                    "Recommended Role":a["recommended_roles"][0] if a["recommended_roles"] else "N/A"
                })
                details[candidate_name]=a["skill_gap_priority"]
            except ResumeParsingError as e:
                rows.append({"Candidate":file.name,"Match Score":0,"ATS Score":0,"Matched Skills":0,"Skill Gaps":"Error",
                             "Keyword Alignment":"N/A","Experience Alignment":"N/A","Education Alignment":"N/A",
                             "Recommended Role":str(e)})
            except Exception as e:
                rows.append({"Candidate":file.name,"Match Score":0,"ATS Score":0,"Matched Skills":0,"Skill Gaps":"Error",
                             "Keyword Alignment":"N/A","Experience Alignment":"N/A","Education Alignment":"N/A",
                             "Recommended Role":f"Unexpected error: {e}"})
        df=pd.DataFrame(rows).sort_values("Match Score",ascending=False).reset_index(drop=True)
        df.index=df.index+1
        st.dataframe(df,use_container_width=True)
        st.caption(
            "Match Score and ATS Score are the primary scores (same methodology as Candidate Analysis). "
            "Keyword/Experience/Education Alignment are additional diagnostic factors, not separate rankings."
        )
        st.bar_chart(df.set_index("Candidate")["Match Score"])
        if len(df):
            top_name = df.iloc[0]["Candidate"]
            st.success(f"🏆 Top candidate: {top_name} — {df.iloc[0]['Match Score']}% match")
            top_gaps = details.get(top_name, [])
            high_priority = [g for g in top_gaps if g["priority"]=="High"]
            if high_priority:
                with st.expander(f"Why {top_name} still has gaps for this role"):
                    for g in high_priority:
                        st.write(f"**{g['skill']}** ({g['priority']} priority) — {g['reason']}")

with tab3:
    st.subheader("🧪 Machine Learning Evaluation")
    st.write("The included demonstration dataset contains labeled resume/job-fit examples. Use it to demonstrate the ML workflow; replace it with a larger real labeled dataset for research-grade claims.")
    if st.button("Train & Evaluate Model"):
        with st.spinner("Training model..."):
            try:
                metrics = train_and_evaluate()
            except MLModelError as e:
                metrics = None
                st.error(f"⚠️ Could not train/evaluate the model: {e}")
        if metrics:
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Accuracy",f"{metrics['accuracy']}%")
            c2.metric("Precision",f"{metrics['precision']}%")
            c3.metric("Recall",f"{metrics['recall']}%")
            c4.metric("F1",f"{metrics['f1']}%")
            st.subheader("Confusion Matrix")
            st.dataframe(pd.DataFrame(metrics["confusion_matrix"],
                                      index=["Actual No Fit","Actual Fit"],
                                      columns=["Pred No Fit","Pred Fit"]))
            st.info(metrics["note"])

with tab4:
    st.subheader("🗂️ Analysis History")
    try:
        history = get_history()
    except DatabaseError as e:
        history = None
        st.error(f"⚠️ Could not load analysis history: {e}")
    if history:
        st.dataframe(pd.DataFrame(history),use_container_width=True,hide_index=True)
    elif history is not None:
        st.info("No analyses saved yet.")

with tab5:
    st.subheader("🆚 Compare Two Resume Versions")
    st.caption(
        "Optional: upload an original and a revised resume against the same job description "
        "to see whether the revision actually improved the metrics. This does not change the "
        "single-resume analysis in the Candidate Analysis tab."
    )
    compare_jd = st.text_area("Job Description for Comparison", height=140, key="compare_jd")
    col_a, col_b = st.columns(2)
    with col_a:
        file_a = st.file_uploader("Resume Version A (original)", type=["pdf","docx","txt"], key="compare_a")
    with col_b:
        file_b = st.file_uploader("Resume Version B (revised)", type=["pdf","docx","txt"], key="compare_b")

    if file_a and file_b and st.button("🔍 Compare Versions", type="primary"):
        try:
            resume_a = parse_resume(file_a)
            resume_b = parse_resume(file_b)
            result_a = analyze_resume(resume_a, compare_jd, use_semantic=semantic)
            result_b = analyze_resume(resume_b, compare_jd, use_semantic=semantic)
            comparison = compare_resumes(resume_a, result_a, resume_b, result_b)

            st.markdown("### 📈 Metric Comparison")
            for row in comparison["metrics"]:
                diff = row["Difference"]
                arrow = "🔺" if diff > 0 else ("🔻" if diff < 0 else "➖")
                st.markdown(f"**{row['Metric']}** — A: {row['Version A']} | B: {row['Version B']} | {arrow} {diff:+g}")

            if comparison["skills_resolved_in_b"]:
                st.success("Skills present in B but missing in A: " + ", ".join(comparison["skills_resolved_in_b"]))
            if comparison["skills_newly_missing_in_b"]:
                st.warning("Skills present in A but missing in B: " + ", ".join(comparison["skills_newly_missing_in_b"]))
            if not comparison["skills_resolved_in_b"] and not comparison["skills_newly_missing_in_b"]:
                st.info("No difference in detected skill gaps between the two versions.")
        except ResumeParsingError as e:
            st.error(f"⚠️ {e}")
        except Exception as e:
            st.error(f"Unexpected error while comparing resumes: {e}")
