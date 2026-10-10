"""Computes REAL test-set metrics and saves plots (nothing is hard-coded)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.config import MODELS


def evaluate(model, X_test, y_test, history, tag, class_counts, threshold=0.5):
    proba = model.predict(X_test, verbose=0).ravel()
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    m = {"accuracy": accuracy_score(y_test, pred),
         "precision": precision_score(y_test, pred, zero_division=0),
         "recall": recall_score(y_test, pred, zero_division=0),
         "f1": f1_score(y_test, pred, zero_division=0),
         "confusion_matrix": cm.tolist(), "n_test": int(len(y_test)), "threshold": threshold}
    (MODELS / f"{tag}_metrics.json").write_text(json.dumps(m, indent=2))
    for key in ("accuracy", "loss"):
        plt.figure(figsize=(5, 3.5))
        plt.plot(history.history[key], label="train")
        plt.plot(history.history["val_" + key], label="validation")
        plt.xlabel("epoch"); plt.ylabel(key); plt.title(f"{tag}: {key}"); plt.legend(); plt.tight_layout()
        plt.savefig(MODELS / f"{tag}_{key}.png", dpi=110); plt.close()
    plt.figure(figsize=(4, 3.5)); plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, int(v), ha="center", va="center")
    plt.xticks([0, 1], ["NORMAL", "SUSPICIOUS"]); plt.yticks([0, 1], ["NORMAL", "SUSPICIOUS"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"{tag}: confusion matrix"); plt.tight_layout()
    plt.savefig(MODELS / f"{tag}_confusion_matrix.png", dpi=110); plt.close()
    plt.figure(figsize=(4, 3.5)); plt.bar(class_counts.keys(), class_counts.values(), color=["tab:green", "tab:red"])
    plt.title(f"{tag}: class distribution"); plt.tight_layout()
    plt.savefig(MODELS / f"{tag}_class_distribution.png", dpi=110); plt.close()
    return m
