"""Генерирует три синтетические таблицы для учебного MLOps-сервиса."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'Данные'
ROOT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
n = 1500
visits = rng.poisson(5, n)
cart = rng.binomial(visits, .27)
orders = rng.poisson(2, n)
discount = rng.binomial(1, .4, n)
logit = -2.4 + .11 * visits + .28 * cart + .19 * orders + .45 * discount
prob = 1 / (1 + np.exp(-logit))
frame = pd.DataFrame({
    'snapshot_date': pd.date_range('2025-01-01', periods=n, freq='D'),
    'visits_7d': visits, 'cart_adds_7d': cart, 'prior_orders': orders,
    'discount_seen': discount, 'bought_next_7d': rng.binomial(1, prob),
})
frame.to_csv(ROOT / 'purchase_snapshots.csv', index=False)
frame.iloc[:300].drop(columns='bought_next_7d').to_csv(ROOT / 'reference_traffic.csv', index=False)
current = frame.iloc[900:1200].drop(columns='bought_next_7d').copy()
current['visits_7d'] += rng.poisson(3, len(current))
current['cart_adds_7d'] = np.minimum(current['cart_adds_7d'], current['visits_7d'])
current.to_csv(ROOT / 'current_traffic.csv', index=False)
print('Созданы синтетические CSV:', len(frame), 'строк обучающей таблицы')
