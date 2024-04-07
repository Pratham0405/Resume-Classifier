import streamlit as st
import pickle
import re
import nltk
import exceptions
from docx import Document
import en_core_web_sm
nlp = en_core_web_sm.load()


nltk.download('punkt')
nltk.download('stopwords')


st.title('             RESUME CLASSIFICATION🤖     ')
st.markdown('<style>h1{color: Blue;}</style>', unsafe_allow_html=True)

st.subheader('Upload Your Resume 👇')

#loading models
clf = pickle.load(open('model.pickle','rb'))
tfidf = pickle.load(open('tfidf.pickle','rb'))


def extract_skills(resume_text):

    nlp_text = nlp(resume_text)
    noun_chunks = nlp_text.noun_chunks

    tokens = [token.text for token in nlp_text if not token.is_stop] # removing stop words and implementing word tokenization
             
    
def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

def read_docx(file):
    doc = Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# web app
def main():
    

    uploaded_file = st.file_uploader(' ', type=['txt','pdf','docx'])

    
    if uploaded_file is not None:
        try:
            if uploaded_file.type == 'application/pdf':
                resume_text = read_pdf(uploaded_file)
            elif uploaded_file.type == 'text/plain':
                resume_text = uploaded_file.getvalue().decode("utf-8")
            elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                resume_text = read_docx(uploaded_file)
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = uploaded_file.read().decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        

        # Map category ID to category name
        category_mapping = {
            0: "PeopleSoft",
            1: "ReactJS Developer",
            2: "SQL Developer",
            3: "Workday",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write(f'### Predicted Category : {category_name}')
       


# python main
if __name__ == "__main__":
    main() 
