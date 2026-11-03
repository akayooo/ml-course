"""Сравнение входных распределений без доступа к будущим меткам."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'Данные'
FEATURES = ['visits_7d', 'cart_adds_7d', 'prior_orders', 'discount_seen']


def psi(reference, current):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 6)))
    if len(edges) < 3:
        edges = np.array([min(reference.min(), current.min()) - 1, .5, max(reference.max(), current.max()) + 1])
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(reference, edges)[0] / len(reference)
    q = np.histogram(current, edges)[0] / len(current)
    p, q = np.maximum(p, 1e-5), np.maximum(q, 1e-5)
    return float(np.sum((q - p) * np.log(q / p)))


def main():
    a = pd.read_csv(ROOT / 'reference_traffic.csv')
    b = pd.read_csv(ROOT / 'current_traffic.csv')
    for col in FEATURES:
        print(f'{col}: reference mean={a[col].mean():.2f}, current mean={b[col].mean():.2f}, PSI={psi(a[col].to_numpy(), b[col].to_numpy()):.3f}')
    print('Дрейф сам по себе не доказывает ухудшение качества модели.')


if __name__ == '__main__': main()
