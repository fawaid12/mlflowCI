# modelling.py
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def load_data():
    base_path = os.path.dirname(__file__)  # path folder skrip ini
    X_train = pd.read_csv(os.path.join(base_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(base_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(base_path, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(base_path, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

def main():
    X_train, X_test, y_train, y_test = load_data()

    # Mulai MLflow
    with mlflow.start_run():
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        # Manual Logging (Skilled)
        for key, val in metrics.items():
            mlflow.log_metric(key, val)

        mlflow.sklearn.log_model(model, "model")
        print("Model dan metriks berhasil dilog ke MLflow.")
        print(metrics)

if __name__ == "__main__":
    main()
