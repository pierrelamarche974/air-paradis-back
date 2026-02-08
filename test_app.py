"""Tests unitaires - API Air Paradis"""
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_negative():
    response = client.post("/predict", json={"text": "Terrible flight, lost my luggage"})
    assert response.status_code == 200
    assert response.json()["prediction"] == 0


def test_predict_positive():
    response = client.post("/predict", json={"text": "Amazing flight, great crew"})
    assert response.status_code == 200
    assert response.json()["prediction"] == 1


def test_predict_empty():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400


def test_feedback():
    response = client.post("/feedback", json={
        "text": "Test", "predicted_sentiment": 1, "correct_sentiment": 0
    })
    assert response.status_code == 200
    assert response.json()["status"] == "success"


if __name__ == "__main__":
    test_health()
    print("test_health OK")
    test_predict_negative()
    print("test_predict_negative OK")
    test_predict_positive()
    print("test_predict_positive OK")
    test_predict_empty()
    print("test_predict_empty OK")
    test_feedback()
    print("test_feedback OK")
    print("Tous les tests sont passes")
