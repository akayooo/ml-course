"""Локальное API для синтетической модели. Загружает только свой артефакт."""
import json
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

MODEL_DIR = Path(__file__).resolve().parent / 'model'
app = FastAPI(title='Учебный прогноз покупки')
model = None
metadata = None


class PurchaseRequest(BaseModel):
    visits_7d: int = Field(ge=0)
    cart_adds_7d: int = Field(ge=0)
    prior_orders: int = Field(ge=0)
    discount_seen: int = Field(ge=0, le=1)

    @model_validator(mode='after')
    def check_cart(self):
        if self.cart_adds_7d > self.visits_7d:
            raise ValueError('В учебных данных cart_adds_7d не больше visits_7d')
        return self


def load_model():
    global model, metadata
    if model is None:
        try:
            metadata = json.loads((MODEL_DIR / 'metadata.json').read_text(encoding='utf-8'))
            model = joblib.load(MODEL_DIR / 'purchase_model.joblib')
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail='Сначала запустите обучение модели') from exc
    return model, metadata


@app.get('/health')
def health():
    _, info = load_model()
    return {'status': 'ok', 'model_version': info['version']}


@app.post('/predict')
def predict(request: PurchaseRequest):
    trained, info = load_model()
    row = pd.DataFrame([request.model_dump()])[info['features']]
    probability = float(trained.predict_proba(row)[0, 1])
    threshold = float(info['threshold'])
    return {'purchase_probability': round(probability, 4), 'positive_decision': probability >= threshold, 'threshold': round(threshold, 4), 'model_version': info['version']}
