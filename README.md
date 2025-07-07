# 🧠 AI Resume Screening & Candidate Ranking System

An AI-powered web app that reads resumes (PDFs), compares them to a job description, and ranks candidates by match score using NLP techniques like **TF-IDF**, **cosine similarity**, **skill matching**, and **Named Entity Recognition (NER)**. Built with **Streamlit**.

---

## 🚀 Features

- Upload and parse multiple PDF resumes
- Enter a custom job description
- Automatically extract text using pdfplumber
- Extract skills from both job description and resumes
- Rank candidates by TF-IDF similarity to the JD
- Compute percentage of matched skills
- Show matched keywords and match count
- Named Entity Recognition using spaCy:
  - Extracts PERSON, ORG, GPE and DATE entities
- Visualize:
  - Match Score bar chart
  - Skill match percentage bar chart
- Export results as downloadable CSV
- User-friendly interface via Streamlit

---

## 📸 App Screenshots

### 🔹 Home Page
![Home Page](assets/Screenshot_1.png)

### 🔹 Ranked Candidate Output
![Ranked Results](assets/Screenshot_2.png)

### 🔹 Named Entities In Resumes
![NER](assets/Screenshot_3.png)

### 🔹 Output Of NER
![Output of NER](assets/Screenshot_4.png)

### 🔹 Bar Chart For Match Score
![Bar Chart 1](assets/Screenshot_5.png)

### 🔹 Bar Chart For Percentage Of Skills Matched
![Bar Chart 2](assets/Screenshot_6.png)

### 🔹 Downloaded CSV
![Downloaded CSV](assets/Screenshot_7.png)

---

## 🔗 Live Demo
[👉 Click Here for Live Demo](https://adityachauhan99-ai-resume-screener.streamlit.app/)

---

## 🛠 How to Run Locally

```bash
git clone https://github.com/adityachauhan99/ai-resume-screener.git
cd ai-resume-screener
pip install -r requirements.txt
streamlit run main.py
