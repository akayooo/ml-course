"""Учебный MLP и линейный baseline для синтетической таблицы."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['age', 'visits_30d', 'purchases_90d', 'cart_adds_30d', 'avg_check_rub']


def score(model, x, y):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(x))).numpy().ravel()
    return average_precision_score(y, probs), roc_auc_score(y, probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    df = pd.read_csv(ROOT / 'Данные/customers_deep.csv')
    x = df[FEATURES].to_numpy(dtype=np.float32)
    y = df.bought_next_7d.to_numpy(dtype=np.float32)
    xd, xt, yd, yt = train_test_split(x, y, test_size=.2, random_state=42, stratify=y)
    xtr, xv, ytr, yv = train_test_split(xd, yd, test_size=.25, random_state=42, stratify=yd)
    scaler = StandardScaler().fit(xtr)
    xtr, xv, xt = [scaler.transform(a).astype(np.float32) for a in (xtr, xv, xt)]
    base = LogisticRegression(max_iter=1000, random_state=42).fit(xtr, ytr)
    base_val = average_precision_score(yv, base.predict_proba(xv)[:, 1])
    print(f'LogisticRegression validation AP={base_val:.3f}')
    model = nn.Sequential(nn.Linear(5, args.hidden), nn.ReLU(), nn.Linear(args.hidden, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)), batch_size=64, shuffle=True)
    best, best_epoch, best_weights = -1, 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb).ravel(), yb)
            loss.backward()
            opt.step()
        ap, auc = score(model, xv, yv)
        if ap > best:
            best, best_epoch, best_weights = ap, epoch, deepcopy(model.state_dict())
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f'epoch={epoch:02d} validation AP={ap:.3f} ROC-AUC={auc:.3f}')
    model.load_state_dict(best_weights)
    test_ap, test_auc = score(model, xt, yt)
    base_test = average_precision_score(yt, base.predict_proba(xt)[:, 1])
    print(f'Best epoch={best_epoch}; MLP test AP={test_ap:.3f}, ROC-AUC={test_auc:.3f}')
    print(f'LogisticRegression test AP={base_test:.3f}')


if __name__ == '__main__':
    main()
