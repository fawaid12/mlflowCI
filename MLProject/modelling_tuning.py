# modelling_tuning.py
import os 
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

def load_data():
    base_path = os.path.dirname(__file__)  # path folder skrip ini
    X_train = pd.read_csv(os.path.join(base_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(base_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(base_path, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(base_path, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    # Hyperparameter tuning
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, 7]
    }

    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_

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

    with mlflow.start_run():
        model, best_params = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        # Logging manual hyperparameter dan metriks
        for param, val in best_params.items():
            mlflow.log_param(param, val)

        for key, val in metrics.items():
            mlflow.log_metric(key, val)

        mlflow.sklearn.log_model(model, "model")
        print("Tuning dan logging selesai.")
        print("Best Params:", best_params)
        print("Metrics:", metrics)

if __name__ == "__main__":
    main()
