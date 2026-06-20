from flask import Flask, render_template, request
import os
from pdfminer.high_level import extract_text
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# -------------------------------
# Skills 
# -------------------------------
skills_list = ["python", "java", "sql", "machine learning", "data science",
               "html", "css", "javascript", "react", "django"]

priority_skills = ["python", "machine learning", "data science"]

# -------------------------------
# Extract Resume Text
# -------------------------------
def extract_resume(file_path):
    if file_path.endswith('.pdf'):
        return extract_text(file_path)
    elif file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        return " ".join([para.text for para in doc.paragraphs])
    return ""

# -------------------------------
# Extract Skills
# -------------------------------
def extract_skills(text):
    found = []
    for skill in skills_list:
        if skill in text.lower():
            found.append(skill)
    return found

# -------------------------------
# Experience Detection
# -------------------------------
def extract_experience(text):
    exp = re.findall(r'(\d+)\s*years', text.lower())
    if exp:
        return int(exp[0])
    return 0

# -------------------------------
# Candidate Level
# -------------------------------
def candidate_level(exp):
    if exp == 0:
        return "Fresher"
    elif exp < 3:
        return "Intermediate"
    else:
        return "Experienced"

# -------------------------------
# Skill Matching
# -------------------------------
def match_skills(candidate_skills, job_desc):
    return [s for s in candidate_skills if s in job_desc.lower()]

# -------------------------------
# Skill Gap Analysis
# -------------------------------
def skill_gap(candidate_skills, job_desc):
    return [s for s in skills_list if s in job_desc.lower() and s not in candidate_skills]

# -------------------------------
# Resume Quality Score
# -------------------------------
def resume_quality(text):
    score = len(text.split()) / 500
    return min(score, 1)

# -------------------------------
# Home Page
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -------------------------------
# Analyze
# -------------------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    files = request.files.getlist('resumes')
    job_desc = request.form['job_desc']

    results = []
    seen_resumes = set()

    for file in files:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        resume_text = extract_resume(filepath)

        # Duplicate Check
        if resume_text in seen_resumes:
            continue
        seen_resumes.add(resume_text)

        # TF-IDF Similarity
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_text, job_desc])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        # Skills
        skills = extract_skills(resume_text)

        # Weighted Skill Score
        skill_score = 0
        for s in skills:
            if s in priority_skills:
                skill_score += 2
            else:
                skill_score += 1
        skill_score = skill_score / (len(skills_list) * 2)

        # Experience
        exp = extract_experience(resume_text)
        exp_score = min(exp / 5, 1)

        # Level
        level = candidate_level(exp)

        # Matching Skills
        matched = match_skills(skills, job_desc)

        # Skill Gap
        missing = skill_gap(skills, job_desc)

        # Quality Score
        quality = resume_quality(resume_text)

        # Final Score
        final_score = (
            0.4 * skill_score +
            0.2 * exp_score +
            0.2 * similarity +
            0.2 * quality
        )

        # Explanation
        explanation = {
            "skills_matched": len(matched),
            "total_skills": len(skills_list),
            "experience": exp,
            "missing_skills": missing
        }

        results.append({
            "name": file.filename,
            "score": round(final_score * 100, 2),
            "skills": skills,
            "matched": matched,
            "missing": missing,
            "experience": exp,
            "level": level,
            "quality": round(quality * 100, 2),
            "explanation": explanation
        })

    # Ranking
    ranked = sorted(results, key=lambda x: x['score'], reverse=True)

    return render_template('result.html', results=ranked)

# -------------------------------
# Run
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)