from fastapi import FastAPI  # type: ignore
from pydantic import BaseModel  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
import tensorflow as tf
import pickle
import os

# Initialize FastAPI app
app = FastAPI(title="Fake News Detection API", version="1.1")

# Define paths for model and tokenizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "fake_news_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "..", "models", "tokenizer.pkl")

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)


# Define Pydantic model for input validation
class NewsItem(BaseModel):
    text: str


MAX_LEN = 300


# Function to predict if the news is fake or real
def predict_news(text: str):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    prob = float(model.predict(pad)[0][0])

    label = "Fake" if prob > 0.5 else "Real"
    confidence = prob if prob > 0.5 else 1 - prob

    return label, round(confidence, 4)


# API endpoint for prediction
@app.post("/predict")
def predict(item: NewsItem):
    label, confidence = predict_news(item.text)
    return {"label": label, "confidence": confidence}


# Health check endpoint
@app.get("/")
def root():
    return {"message": "Fake News Detection API is running 🚀"}


# http://127.0.0.1:8000/docs open in browser to see API documentation and test the endpoint.
