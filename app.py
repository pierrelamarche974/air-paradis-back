"""
API FastAPI pour l'analyse de sentiment - Air Paradis
Modèle: LSTM Bidirectionnel + FastText (TensorFlow Lite)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
import tensorflow as tf

# Configuration
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_fasttext.tflite")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.pkl")

app = FastAPI(
    title="Air Paradis - Sentiment Analysis API",
    description="API de détection de bad buzz pour Air Paradis (LSTM + FastText)",
    version="1.0.0"
)

# Chargement au démarrage
print("Chargement de la configuration...")
with open(CONFIG_PATH, "rb") as f:
    config = pickle.load(f)
MAX_LEN = config["max_len"]
print(f"MAX_LEN: {MAX_LEN}")

print("Chargement du tokenizer...")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
print("Tokenizer chargé.")

print("Chargement du modèle TFLite...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Modèle TFLite chargé.")


# Schémas Pydantic
class TextInput(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Flight cancelled again, terrible service!"
            }
        }


class PredictionOutput(BaseModel):
    text: str
    sentiment: str
    prediction: int
    confidence: float
    probabilities: dict


class HealthOutput(BaseModel):
    status: str
    model: str


def preprocess_text(text: str) -> np.ndarray:
    """Prétraite un texte pour le modèle LSTM."""
    sequences = tokenizer.texts_to_sequences([text])
    padded = np.zeros((1, MAX_LEN), dtype=np.float32)
    length = min(len(sequences[0]), MAX_LEN)
    padded[0, :length] = sequences[0][:length]
    return padded


def predict_single(text: str) -> tuple[int, float]:
    """Prédit le sentiment d'un texte."""
    X = preprocess_text(text)
    interpreter.set_tensor(input_details[0]['index'], X)
    interpreter.invoke()
    proba = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
    pred = 1 if proba >= 0.5 else 0
    return pred, proba


@app.get("/")
def home():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: TextInput):
    """Prédit le sentiment d'un texte."""
    if not data.text or not data.text.strip():
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide")

    try:
        pred, proba = predict_single(data.text)
        return {
            "text": data.text,
            "sentiment": "Négatif" if pred == 0 else "Positif",
            "prediction": pred,
            "confidence": proba if pred == 1 else 1 - proba,
            "probabilities": {
                "negatif": 1 - proba,
                "positif": proba
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")
