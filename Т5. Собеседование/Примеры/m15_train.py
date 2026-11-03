"""Временной split, обучение и сохранение учебной модели."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['visits_7d', 'cart_adds_7d', 'prior_orders', 'discount_seen']


def main():
    data = pd.read_csv(ROOT / 'Данные/purchase_snapshots.csv', parse_dates=['snapshot_date']).sort_values('snapshot_date')
    train = data.iloc[:int(.6 * len(data))]
    val = data.iloc[int(.6 * len(data)):int(.8 * len(data))]
    test = data.iloc[int(.8 * len(data)):]
    print('split:', len(train), len(val), len(test))
    def pipeline():
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model = pipeline().fit(train[FEATURES], train.bought_next_7d)
    val_p = model.predict_proba(val[FEATURES])[:, 1]
    thresholds = np.linspace(.1, .9, 81)
    threshold = float(max(thresholds, key=lambda t: f1_score(val.bought_next_7d, val_p >= t)))
    print('validation threshold:', round(threshold, 3))
    dev = pd.concat([train, val])
    model = pipeline().fit(dev[FEATURES], dev.bought_next_7d)
    p = model.predict_proba(test[FEATURES])[:, 1]
    metrics = {'roc_auc': float(roc_auc_score(test.bought_next_7d, p)), 'average_precision': float(average_precision_score(test.bought_next_7d, p)), 'brier': float(brier_score_loss(test.bought_next_7d, p)), 'f1': float(f1_score(test.bought_next_7d, p >= threshold))}
    out = ROOT / 'Примеры/model'; out.mkdir(exist_ok=True)
    joblib.dump(model, out / 'purchase_model.joblib')
    (out / 'metadata.json').write_text(json.dumps({'version': 'tutorial-v1', 'features': FEATURES, 'threshold': threshold, 'metrics': metrics}, indent=2), encoding='utf-8')
    print('test:', {k: round(v, 3) for k, v in metrics.items()})
    print('saved:', out)


if __name__ == '__main__': main()
