"""Оценка риска отмены заказа.

Запуск: python predict.py --input заказы.csv --output result.csv

В файле результата на каждый заказ строка: order_id, risk_score (0…1), main_reason
(код причины или пусто). predict_one(order) даёт то же для одного заказа.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REQUIRED = ["order_id", "chain", "region", "order_hour", "basket_items", "promised_duration_min",
            "current_status", "minutes_in_status", "minutes_since_last_event", "courier_assigned",
            "assignments_count", "ready_for_pickup", "courier_wait_min", "support_contacts_count",
            "current_delay_min"]
NUMERIC = ["order_hour", "basket_items", "promised_duration_min", "minutes_in_status",
           "minutes_since_last_event", "courier_assigned", "assignments_count", "ready_for_pickup",
           "courier_wait_min", "support_contacts_count", "current_delay_min"]
# пропуск — не ошибка: корзина и последнее событие могут быть неизвестны, курьера может ещё не быть
MAY_BE_EMPTY = {"basket_items", "minutes_since_last_event", "courier_wait_min"}
NOT_NEGATIVE = ["minutes_in_status", "current_delay_min", "support_contacts_count",
                "assignments_count", "promised_duration_min"]
TYPICAL = {"created": 3, "assembling": 12, "searching_courier": 6, "courier_assigned": 9,
           "in_delivery": 20}
STATUSES = set(TYPICAL)

# Модель загружается один раз, при импорте: predict_one потом отвечает быстро. joblib — это pickle:
# грузим только свой файл из этого же репозитория. Пока его нет (первое обучение, train.py), MODEL = None.
MODEL_PATH = Path(__file__).resolve().with_name("model.joblib")
MODEL = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


class BadData(ValueError):
    """Что не так с данными — текстом для человека."""


def check(df: pd.DataFrame) -> pd.DataFrame:
    """Проверяет заказы и приводит числа к числам. Ошибка — BadData с объяснением."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise BadData("нет обязательных колонок: " + ", ".join(missing))
    if len(df) == 0:
        raise BadData("в файле нет ни одного заказа")
    twice = df["order_id"][df["order_id"].duplicated()]
    if len(twice):
        raise BadData(f"order_id {twice.iloc[0]} встречается больше одного раза")
    df = df.copy()
    for col in NUMERIC:
        num = pd.to_numeric(df[col], errors="coerce")
        bad = num.isna() & df[col].notna()
        if bad.any():
            i = bad.to_numpy().nonzero()[0][0]
            raise BadData(f"в поле {col} не число: «{df[col].iloc[i]}» (строка {i + 2})")
        if col not in MAY_BE_EMPTY and num.isna().any():
            i = num.isna().to_numpy().nonzero()[0][0]
            raise BadData(f"в поле {col} пусто (строка {i + 2})")
        df[col] = num
    unknown = sorted(set(df["current_status"].dropna().astype(str)) - STATUSES)
    if unknown or df["current_status"].isna().any():
        raise BadData("статус не из списка: " + (", ".join(unknown) or "пусто")
                      + ". Возможные: " + ", ".join(sorted(STATUSES)))
    for col in NOT_NEGATIVE:
        neg = df[col] < 0
        if neg.any():
            i = neg.to_numpy().nonzero()[0][0]
            raise BadData(f"в поле {col} отрицательное значение у заказа {df['order_id'].iloc[i]}")
    return df


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Признаки — только по самому заказу, без сравнения с другими заказами файла:
    поэтому один заказ получает ту же оценку, что и в пачке."""
    f = pd.DataFrame(index=df.index)
    f["delay"] = df["current_delay_min"]
    f["delay_share"] = (df["current_delay_min"] / df["promised_duration_min"].replace(0, np.nan)).fillna(0)
    f["mins"] = df["minutes_in_status"]
    f["over_typical"] = df["minutes_in_status"] - df["current_status"].map(TYPICAL)
    f["since"] = df["minutes_since_last_event"]
    f["no_courier"] = 1 - df["courier_assigned"]
    f["reassign"] = df["assignments_count"]
    f["ready"] = df["ready_for_pickup"]
    f["wait"] = df["courier_wait_min"]
    f["support"] = df["support_contacts_count"]
    f["basket"] = df["basket_items"]
    f["hour"] = df["order_hour"]
    f["promised"] = df["promised_duration_min"]
    for status in sorted(STATUSES):
        f["st_" + status] = (df["current_status"] == status).astype(float)
    return f


def main_reason(r) -> str:
    """Главная причина — самая весомая из тех, что видны в данных; пусто, если причин нет."""
    found = []
    if r.support_contacts_count >= 1:
        found.append((3.0, "support_contacted"))
    if r.current_delay_min > 0:
        found.append((2.5, "already_delayed"))
    if r.assignments_count >= 2:
        found.append((2.0, "courier_reassigned"))
    if r.courier_assigned == 0 and r.minutes_in_status > 10:
        found.append((1.8, "courier_not_assigned"))
    if r.current_status == "assembling" and r.minutes_in_status > 20:
        found.append((1.5, "slow_assembly"))
    if pd.notna(r.courier_wait_min) and r.courier_wait_min >= 10:
        found.append((1.4, "courier_waiting"))
    if r.minutes_in_status > 2 * TYPICAL[r.current_status]:
        found.append((1.0, "long_in_status"))
    return max(found)[1] if found else ""


def predict_frame(df: pd.DataFrame) -> pd.DataFrame:
    score = MODEL.predict_proba(features(df))[:, 1].round(6)
    return pd.DataFrame({
        "order_id": df["order_id"].to_numpy(),
        "risk_score": score,
        "main_reason": [main_reason(r) for r in df.itertuples()],
    })


def predict_one(order: dict) -> dict:
    """Один заказ словарём, как строка таблицы. Заказ с ошибкой — исключение с текстом."""
    missing = [c for c in REQUIRED if c not in order]
    if missing:
        raise BadData("в заказе нет полей: " + ", ".join(missing))
    row = predict_frame(check(pd.DataFrame([order]))).iloc[0]
    return {"risk_score": float(row["risk_score"]), "main_reason": row["main_reason"] or None}


def main() -> None:
    ap = argparse.ArgumentParser(description="Оценка риска отмены заказа")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    try:
        try:
            orders = pd.read_csv(args.input)
        except pd.errors.EmptyDataError:
            raise BadData("файл пустой") from None
        except (OSError, pd.errors.ParserError) as exc:
            raise BadData(f"файл не читается как CSV: {exc}") from None
        result = predict_frame(check(orders))
    except BadData as exc:
        print(f"Ошибка в данных: {exc}", file=sys.stderr)
        sys.exit(1)  # без файла результата: он пишется, только когда всё посчитано
    result.to_csv(args.output, index=False)


main()  # ОШИБКА: запускается и при импорте
