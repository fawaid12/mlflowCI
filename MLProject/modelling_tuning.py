# modelling_tuning.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import json
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve, roc_curve, auc
)
from sklearn.model_selection import GridSearchCV

def load_data():
    base_path = os.path.dirname(__file__)
    X_train = pd.read_csv(os.path.join(base_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(base_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(base_path, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(base_path, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
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

def log_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name)
        mlflow.log_artifact(tmp.name, artifact_path="model")
    plt.close()

def log_precision_recall_curve(y_test, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, marker='.')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name)
        mlflow.log_artifact(tmp.name, artifact_path="model")
    plt.close()

def log_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name)
        mlflow.log_artifact(tmp.name, artifact_path="model")
    plt.close()

def log_prediction_csv(y_test, y_pred):
    df_pred = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
    path = "model/prediction_results.csv"
    df_pred.to_csv(path, index=False)
    mlflow.log_artifact(path)

def log_params_json(params_dict):
    path = "model/best_params.json"
    with open(path, "w") as f:
        json.dump(params_dict, f, indent=4)
    mlflow.log_artifact(path)

def main():
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(nested=True):
        model, best_params = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        # Logging parameter dan metrik
        for param, val in best_params.items():
            mlflow.log_param(param, val)
        for key, val in metrics.items():
            mlflow.log_metric(key, val)

        mlflow.sklearn.log_model(model, "model")

        # Logging artefak visual dan tambahan
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # asumsi binary

        log_confusion_matrix(y_test, y_pred)
        log_precision_recall_curve(y_test, y_proba)
        log_roc_curve(y_test, y_proba)

        log_prediction_csv(y_test, y_pred)         # artefak tambahan 1
        log_params_json(best_params)              # artefak tambahan 2

        print("Tuning dan logging selesai.")
        print("Best Params:", best_params)
        print("Metrics:", metrics)

if __name__ == "__main__":
    main()
