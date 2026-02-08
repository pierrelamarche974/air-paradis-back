"""
API FastAPI pour l'analyse de sentiment - Air Paradis
Modèle: LSTM Bidirectionnel + FastText (TensorFlow Lite)
"""
from fastapi import FastAPI

app = FastAPI(
    title="Air Paradis - Sentiment Analysis API",
    description="API de détection de bad buzz pour Air Paradis",
    version="1.0.0"
)
