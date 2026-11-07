"""Как обучена модель: python train.py --data train_orders.csv  →  model.joblib

Обучать нужно на тех же версиях библиотек, что в песочнице платформы (requirements.txt),
иначе модель может не загрузиться.
"""

import argparse

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from predict import check, features

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_orders.csv")
    ap.add_argument("--out", default="model.joblib")
    args = ap.parse_args()
    train = pd.read_csv(args.data)
    X = features(check(train.drop(columns="is_problem_order")))
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    model.fit(X, train["is_problem_order"])
    joblib.dump(model, args.out)
    print("модель сохранена:", args.out)
