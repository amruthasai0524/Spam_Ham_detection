import streamlit as st
import pickle
import re
import nltk

# Download NLTK stopwords
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Initialize stemmer
ps = PorterStemmer()

# Text cleaning function
def clean_text(text):

    # Remove special characters and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Split words
    text = text.split()

    # Remove stopwords and apply stemming
    text = [
        ps.stem(word)
        for word in text
        if word not in stopwords.words('english')
    ]

    # Join words back
    return " ".join(text)

# Streamlit page title
st.title("📩 SMS Spam Detection using NLP")

# User input
message = st.text_area("Enter SMS Message")

# Prediction button
if st.button("Predict"):

    # Clean input text
    cleaned_message = clean_text(message)

    # Convert text into vector
    vector_input = vectorizer.transform([cleaned_message]).toarray()

    # Predict
    prediction = model.predict(vector_input)

    # Display result
    if prediction[0] == 1:

        st.error("Spam Message ❌")

    else:

        st.success("Normal Message ✅")
