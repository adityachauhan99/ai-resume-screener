import pandas as pd
import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Extracting Text From a Pdf File

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text="\n".join(
            page.extract_text()
            for page in pdf.pages
            if page.extract_text()
        )
    return text.strip()

# Streamlit UI

st.title("AI Resume Screening And Candidate Ranking System")

# Job Description Input

st.header("Job Description")
job_description=st.text_area("Enter Job Description")

# Uploading Multiple Resumes

st.header("Upload Multiple Resumes (PDF)")
uploaded_files=st.file_uploader("Upload Multiple Resumes",type=["pdf"],accept_multiple_files=True)

if uploaded_files and job_description:
    resumes=[]
    resume_names=[]

    # Extract Text From Each Uploaded Resumes

    for file in uploaded_files:
        resume_text=extract_text_from_pdf(file)
        if resume_text:
            resumes.append(resume_text)
            resume_names.append(file.name)
        else:
            st.error(f"Could Not Extract Text From {file.name}. Try Another File")

    if resumes:
        # Vectorize Job Description And Resumes
        tfidvectorizer=TfidfVectorizer(stop_words="english")
        documents=[job_description]+resumes
        vectors=tfidvectorizer.fit_transform(documents).toarray()

        # Compute Similarity Scores 

        job_vector=vectors[0].reshape(1,-1)
        resume_vectors=vectors[1:]
        scores=cosine_similarity(job_vector,resume_vectors).flatten()

        # Extract Required Skills

        known_skills = [
        'python', 'sql', 'excel', 'power bi', 'tableau', 'r', 'pandas', 'numpy',
        'scikit-learn', 'matplotlib', 'seaborn', 'dashboards', 'data cleaning',
        'data visualization', 'machine learning', 'communication', 'statistics',
        'time series', 'deep learning', 'nlp', 'big data', 'aws', 'azure'
        ]

        # Define the Skill Extractor

        import spacy
        nlp=spacy.load("en_core_web_sm")

        def extract_skills_from_text(job_description, known_skills):
            job_description=job_description.lower()
            doc=nlp(job_description)
            extracted=set()

            for token in doc:                                       # Tokens like : Python , SQL , Excel
                token_text=token.text.lower()
                if token_text in known_skills:
                    extracted.add(token_text)

            for chunk in doc.noun_chunks:                           # Noun Chunks like : Data Cleaning , Data Analyst , Machine Learning
                chunk_text=chunk.text.lower().strip()
                if chunk_text in known_skills:
                    extracted.add(chunk_text)

            return list(extracted)
        
        # Using The Skill Extractor To Extract Skills From Job Description And Resumes

        required_skills=extract_skills_from_text(job_description,known_skills)

        resume_skill_matches=[]
        resume_skill_counts=[]
        resume_hit_rario=[]

        for text in resumes:
            found_skills=extract_skills_from_text(text,known_skills)
            matched_required=sorted(set(found_skills).intersection(required_skills))

            resume_skill_matches.append(",".join(matched_required))
            resume_skill_counts.append(f"{len(matched_required)} Out Of {len(required_skills)}")
            resume_hit_rario.append((len(matched_required)/len(required_skills))*100 if required_skills else 0)

        # Display Ranked Resumes

        results=pd.DataFrame({"Resume":resume_names , "Match Score":scores , "Skills Matched":resume_skill_matches , "Skill Match Count":resume_skill_counts , "Percentage Of Skills Matched":resume_hit_rario})
        results=results.sort_values(by="Match Score",ascending=False)
        results["Match Score"]=results["Match Score"]*100
        st.subheader("Ranked Candidates")
        st.dataframe(results,use_container_width=True)

        # Download The Result
        
        st.download_button("Download Result as CSV",results.to_csv(index=False),"results.csv","text/csv")

        # Bar Charts
        st.subheader("Bar Chart For Match Score")
        st.bar_chart(results.set_index("Resume")["Match Score"])
        st.subheader("Bar Chart For Percentage Of Skills Matched")
        st.bar_chart(results.set_index("Resume")["Percentage Of Skills Matched"])