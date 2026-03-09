from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from imblearn.over_sampling import SMOTE


DATA_PATH = Path("../data/dataset_olympics_preprocessed.csv")
OUTPUT_DIR = Path("../output")
TARGET_COLUMN = "Medal_Binary"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    # Drop leakage columns if they exist
    leakage_cols = [col for col in ["Medal"] if col in df.columns]
    if leakage_cols:
        df = df.drop(columns=leakage_cols)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )



def apply_smote(X_train=None, y_train=None, random_state=42, return_object=False):
    smote = SMOTE(random_state=random_state)

    if return_object:
        return smote

    return smote.fit_resample(X_train, y_train)


def compute_cv_roc_auc(model, X_train, y_train, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean(), scores.std()

def save_roc_curve(model, X_test, y_test, model_name: str):
    ensure_output_dir()

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name}_roc_curve.png", dpi=200)
    plt.close()



def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    roc_auc = roc_auc_score(y_test, y_score)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "roc_auc": roc_auc,
        "f1_score": f1,
        "confusion_matrix": cm
    }


def save_confusion_matrix(cm, model_name: str):
    ensure_output_dir()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name}_confusion_matrix.png", dpi=200)
    plt.close()


def save_metrics(metrics: dict, model_name: str):
    ensure_output_dir()

    metrics_to_save = metrics.copy()

    if "confusion_matrix" in metrics_to_save:
        metrics_to_save["confusion_matrix"] = metrics_to_save["confusion_matrix"].tolist()

    with open(OUTPUT_DIR / f"{model_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2)