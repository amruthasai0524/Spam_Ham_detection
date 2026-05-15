import streamlit as st
import pickle
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

ps = PorterStemmer()

def clean_text(text):

    text = re.sub('[^a-zA-Z]', ' ', text)

    text = text.lower()

    text = text.split()

    text = [ps.stem(word) for word in text
            if word not in stopwords.words('english')]

    return " ".join(text)

st.title("SMS Spam Detection")

message = st.text_area("Enter SMS Message")

if st.button("Predict"):

    cleaned_message = clean_text(message)

    vector_input = vectorizer.transform([cleaned_message]).toarray()

    prediction = model.predict(vector_input)

    if prediction[0] == 1:

        st.error("Spam Message ❌")

    else:

        st.success("Normal Message ✅")