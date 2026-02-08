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


if __name__ == "__main__":
    test_health()
    print("test_health OK")
    test_predict_negative()
    print("test_predict_negative OK")
    test_predict_positive()
    print("test_predict_positive OK")
    test_predict_empty()
    print("test_predict_empty OK")
    print("Tous les tests sont passes")
