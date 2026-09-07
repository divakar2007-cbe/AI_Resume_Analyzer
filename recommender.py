JOBS=[
{"Job Role":"Junior Machine Learning Engineer","Required Skills":"python, machine learning, pandas, numpy, scikit-learn","Skills":["python","machine learning","pandas","numpy","scikit-learn"]},
{"Job Role":"Python Developer","Required Skills":"python, git, sql, rest api","Skills":["python","git","sql","rest api"]},
{"Job Role":"Data Analyst","Required Skills":"python, sql, pandas, excel, data analysis","Skills":["python","sql","pandas","excel","data analysis"]},
{"Job Role":"Full Stack Developer","Required Skills":"html, css, javascript, react, sql","Skills":["html","css","javascript","react","sql"]},
{"Job Role":"AI/NLP Engineer","Required Skills":"python, machine learning, nlp, scikit-learn","Skills":["python","machine learning","nlp","scikit-learn"]},
{"Job Role":"Cloud Backend Developer","Required Skills":"python, fastapi, docker, aws, sql","Skills":["python","fastapi","docker","aws","sql"]},
{"Job Role":"Cybersecurity Analyst","Required Skills":"linux, python, cybersecurity, git","Skills":["linux","python","cybersecurity","git"]},
]
def recommend_jobs(skills):
    s=set(skills); rows=[]
    for j in JOBS:
        score=round(len(s&set(j["Skills"]))/len(j["Skills"])*100)
        rows.append({"Job Role":j["Job Role"],"Match":score,"Required Skills":j["Required Skills"]})
    return sorted(rows,key=lambda x:x["Match"],reverse=True)
