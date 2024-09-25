import streamlit as st
import joblib

st.title("Fake News Detector")

vectorizer, model = joblib.load("models/fake_news_model.pkl")

text = st.text_area("Enter a news headline:")

if st.button("Detect"):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    st.success("Fake News" if prediction == 1 else "Real News")
