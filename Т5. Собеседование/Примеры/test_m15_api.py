"""Контрактные тесты локального учебного API."""
from fastapi.testclient import TestClient
from m15_api import app

client = TestClient(app)
VALID = {'visits_7d': 8, 'cart_adds_7d': 2, 'prior_orders': 3, 'discount_seen': 1}


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_predict():
    response = client.post('/predict', json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body['purchase_probability'] <= 1
    assert isinstance(body['positive_decision'], bool)
    assert body['model_version']


def test_negative_visits():
    assert client.post('/predict', json={**VALID, 'visits_7d': -1}).status_code == 422


def test_inconsistent_cart():
    assert client.post('/predict', json={**VALID, 'cart_adds_7d': 9}).status_code == 422


def test_discount_out_of_range():
    assert client.post('/predict', json={**VALID, 'discount_seen': 3}).status_code == 422
