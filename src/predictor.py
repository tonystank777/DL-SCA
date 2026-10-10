"""Loads a saved model + scaler and returns P(SUSPICIOUS). Same transform as training."""
import json
import joblib
import numpy as np
from src.config import MODELS
from src.preprocessing import transform_features


class Detector:
    def __init__(self, tag="live"):
        from tensorflow import keras
        self.meta = json.loads((MODELS / f"{tag}_meta.json").read_text())
        self.model = keras.models.load_model(MODELS / f"{tag}_model.keras")
        self.scaler = joblib.load(MODELS / f"{tag}_scaler.pkl")
        self.features, self.threshold = self.meta["features"], self.meta["threshold"]

    def predict_batch(self, rows):
        X = np.array([[float(r[f]) for f in self.features] for r in rows])
        Xs = transform_features(X, self.scaler, self.meta["use_log"])
        return self.model.predict(Xs, verbose=0).ravel()
