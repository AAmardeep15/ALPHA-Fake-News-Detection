from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

# 🔹 FastAPI Instance
app = FastAPI()

# 🔹 Trained Model & Vectorizer Load karna
with open("../models/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# 🔹 Request Body Model
class NewsInput(BaseModel):
    text: str

# 🔹 Prediction API
@app.post("/predict/")
def predict_news(news: NewsInput):
    text_vectorized = vectorizer.transform([news.text])
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0][prediction]
    
    result = {
        "news_text": news.text,
        "prediction": "Real News" if prediction == 1 else "Fake News",
        "confidence": f"{probability * 100:.2f}%"
    }
    
    return result

# 🔹 Root Route
@app.get("/")
def home():
    return {"message": "Fake News Detection API is running!"}
