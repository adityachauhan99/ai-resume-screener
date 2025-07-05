import pandas as pd
import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

        # Display Ranked Resumes

        results=pd.DataFrame({"Resume":resume_names , "Match Score":scores})
        results=results.sort_values(by="Match Score",ascending=False)
        results["Match Score"]=results["Match Score"]*100
        st.subheader("Ranked Candidates")
        st.dataframe(results)

        # Download The Result
        
        st.download_button("Download Result as CSV",results.to_csv(index=False),"results.csv","text/csv")

        # Bar Chart
        
        st.bar_chart(results.set_index("Resume")["Match Score"])