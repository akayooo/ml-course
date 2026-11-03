"""Честный train/validation/test для синтетических коротких отзывов."""
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]


def metrics(name, model, x, y):
    p = model.predict(x)
    print(f'{name}: accuracy={accuracy_score(y, p):.3f}, F1={f1_score(y, p):.3f}')


def main():
    data = pd.read_csv(ROOT / 'Данные/movie_reviews_ru.csv')
    xd, xt, yd, yt = train_test_split(data.text, data.label, test_size=.2, random_state=42, stratify=data.label)
    xtr, xv, ytr, yv = train_test_split(xd, yd, test_size=.25, random_state=42, stratify=yd)
    dummy = DummyClassifier(strategy='most_frequent').fit(xtr, ytr)
    metrics('Dummy validation', dummy, xv, yv)
    best_score, best_c, best_model = -1, None, None
    for c in (.01, .1, 1, 10, 100):
        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(C=c, max_iter=1000, random_state=42))
        model.fit(xtr, ytr)
        f1 = f1_score(yv, model.predict(xv))
        print(f'C={c}: validation F1={f1:.3f}')
        if f1 > best_score:
            best_score, best_c, best_model = f1, c, model
    print(f'Chosen C={best_c} on validation')
    metrics('Dummy test', dummy, xt, yt)
    metrics('TF-IDF + LogisticRegression test', best_model, xt, yt)
    print('Набор маленький и искусственный: метрики не отражают качество на реальных отзывах.')


if __name__ == '__main__':
    main()
