# Спасти заказ до отмены — оценка риска

Python 3.12, библиотеки — в `requirements.txt`.

```bash
pip install -r requirements.txt
python predict.py --input заказы.csv --output result.csv
```

В `result.csv` на каждый заказ — `order_id`, `risk_score` (от 0 до 1) и `main_reason`
(код главной причины или пусто). Если с файлом что-то не так, программа пишет, что именно,
и завершается с ошибкой, файл результата не создаётся.

Один заказ — функция `predict_one(order)` из `predict.py`: словарь заказа на входе,
`{"risk_score": …, "main_reason": …}` на выходе.

Модель — `model.joblib`, обучена `train.py` на `train_orders.csv`.
