import streamlit as st
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer (make sure these files exist)
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.write("This app checks whether the entered news is **real or fake** using a trained machine learning model.")

# Input box
user_input = st.text_area("Enter the news article content below:", height=200)

# Predict button
if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Transform input using vectorizer
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)

        # Show result
        if prediction[0] == 1:
            st.success("✅ This news seems to be **Real**.")
        else:
            st.error("🚨 This news seems to be **Fake**.")
