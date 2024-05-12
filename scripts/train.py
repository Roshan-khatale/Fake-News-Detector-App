import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

texts = ["The sky is green", "Breaking: COVID cure discovered", "Aliens landed in my garden"]
labels = [1, 0, 1]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

joblib.dump((vectorizer, model), "models/fake_news_model.pkl")
print("Model saved.")
