# Resume Classification Web App
## Description
This is a simple web application built with Streamlit for classifying resumes into different categories based on their content. It utilizes a machine learning model trained on various types of resumes to make predictions.

### Installation
1.Clone this repository to your local machine.

Copy code
 ```sh
   git clone https://github.com/Pratham0405/Resume-Classifier.git
   ```
2.Install the required dependencies by running:

```sh
pip install -r requirements.txt
```

### Usage

Run the Streamlit app by executing the following command:


Copy code
```sh
streamlit run app.py
```
![rs1](https://github.com/Pratham0405/Resume-Classifier/assets/148319246/6f9328e6-9b26-48a9-8fe5-fd909dd9fd5e)




Once the app is running, upload a resume file (supported formats: txt, pdf, docx).

The app will process the uploaded resume and classify it into one of the predefined categories and extract the skills from the resume.


![rs2](https://github.com/Pratham0405/Resume-Classifier/assets/148319246/3d9009d9-e314-4822-9e3d-6b460a0d00f9)



File Descriptions
- app.py: Contains the main Streamlit application code.
- model.pickle: Serialized machine learning model for resume classification.
- tfidf.pickle: Serialized TF-IDF vectorizer for text preprocessing.

Dependencies: 
- Streamlit
- NLTK
- docx
- PyPDF2
  
  
### NOTE : Make sure to place the trained model (model.pickle) and TF-IDF vectorizer (tfidf.pickle) files in the same directory as the app.py file for the application to work properly.
