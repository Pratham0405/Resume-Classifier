# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import click
import docx2txt
import pdfplumber
from pickle import load
import requests
import re
import os
import sklearn
import PyPDF2
import nltk
import pickle as pk
from nltk.corpus import stopwords as nltk_stopwords
import nltk
nltk.download("stopwords")
import spacy
import en_core_web_sm
nlp = spacy.load('en_core_web_sm')

from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))

from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)
from sklearn.feature_extraction.text import TfidfVectorizer

#----------------------------------------------------------------------------------------------------

st.title('             RESUME CLASSIFICATION 🤖     ')
st.markdown('<style>h1{color: Blue;}</style>', unsafe_allow_html=True)

st.subheader('Hey, Welcome')

# Display resume image
st.subheader('Upload Your Resume 👇')
resume_image_path = r'pngwing.com.png'
with open(resume_image_path, 'rb') as f:
    resume_image = f.read()
st.image(resume_image,  width=round(0.2 * 1000)) # Assuming the original width is 1000 pixels





# FUNCTIONS
def extract_skills(resume_text):

    nlp_text = nlp(resume_text)
    noun_chunks = nlp_text.noun_chunks

    tokens = [token.text for token in nlp_text if not token.is_stop] # removing stop words and implementing word tokenization
            
    
    data = pd.read_csv(r"skills.csv") # reading the csv file
            
    
    skills = list(data.columns.values)# extract values
            
    skillset = []
            
    
    for token in tokens:                 # check for one-grams (example: python)
        if token.lower() in skills:
            skillset.append(token)
            
   
    for token in noun_chunks:            # check for bi-grams and tri-grams (example: machine learning)
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
            
    return [i.capitalize() for i in set([i.lower() for i in skillset])]

def getText(filename):
      
    # Create empty string 
    fullText = ''
    if filename.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx2txt.process(filename)
        
        for para in doc:
            fullText = fullText + para
            
           
    else:  
        with pdfplumber.open(filename) as pdf_file:
            number_of_pages = len(pdf_file.pages)
            page_content = ''
            for page in pdf_file.pages:
                page_content += page.extract_text()
             
        fullText = page_content
         
    return (fullText)


def display(doc_file):
    resume = []
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))

    else:
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())
            
    return resume


def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words) 

file_type=pd.DataFrame([], columns=['Uploaded File',  'Predicted Profile','Skills',])
filename = []
predicted = []
skills = []

#-------------------------------------------------------------------------------------------------
# MAIN CODE
import pickle 
model = pickle.load(open('model.pickle','rb'))
Vectorizer = pickle.load(open('tfidf.pickle','rb'))



upload_file = st.file_uploader('Upload Your Resumes',
                                type= ['docx','pdf'],accept_multiple_files=True)
  
for doc_file in upload_file:
    if doc_file is not None:
        filename.append(doc_file.name)
        cleaned=preprocess(display(doc_file))
        prediction = model.predict(Vectorizer.transform([cleaned]))[0]
        predicted.append(prediction)
        extText = getText(doc_file)
        skills.append(extract_skills(extText))


# Define a dictionary to map numeric labels to categories
label_mapping = {0: "PeopleSoft", 1: "ReactJS Developer",2: "SQL Developer", 3: "Workday"}
# Update the 'predicted' list with category labels instead of numeric labels
predicted_labels = [label_mapping[label] for label in predicted]


        
if len(predicted) > 0:
    file_type['Uploaded File'] = filename
    file_type['Skills'] = skills
    file_type['Predicted Profile'] = predicted_labels
    st.table(file_type.style.format())


