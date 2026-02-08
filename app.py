"""
API FastAPI pour l'analyse de sentiment - Air Paradis
"""
from fastapi import FastAPI

app = FastAPI(
    title="Air Paradis - Sentiment Analysis API",
    description="API de détection de bad buzz pour Air Paradis",
    version="1.0.0"
)


@app.get("/")
def home():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}
