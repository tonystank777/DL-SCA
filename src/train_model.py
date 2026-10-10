"""Trains the Keras binary classifier.
  python -m src.train_model --mode research   # real HPC dataset (Spectre / benign)
  python -m src.train_model --mode live       # your locally recorded OS-metric data
"""
import argparse, json
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from src import preprocessing as pp
from src.config import MODELS, PROCESSED
from src.evaluate_model import evaluate


def build_model(n_features):
    from tensorflow import keras
    m = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(64, activation="relu"), keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"), keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid")])          # output = P(SUSPICIOUS)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")])
    return m


def train(prep, tag, features, use_log, extra):
    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(42)
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=prep["y_train"])  # handles imbalance
    model = build_model(len(features))
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    hist = model.fit(prep["X_train"], prep["y_train"], validation_data=(prep["X_val"], prep["y_val"]),
                     epochs=60, batch_size=64, class_weight={0: cw[0], 1: cw[1]}, callbacks=[stop], verbose=2)
    model.save(MODELS / f"{tag}_model.keras")
    res = evaluate(model, prep["X_test"], prep["y_test"], hist, tag, prep["class_counts"], 0.5)
    meta = {"tag": tag, "features": features, "use_log": use_log, "threshold": 0.5,
            "class_counts": prep["class_counts"], "split_sizes": prep["split_sizes"], **extra}
    (MODELS / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))
    print("\n=== REAL TEST RESULTS ===")
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["research", "live"], required=True)
    ap.add_argument("--device", default="Intel_PC_1")
    ap.add_argument("--min-coverage", type=float, default=0.9)
    a = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    if a.mode == "research":
        from src.dataset_loader import load_device
        X, y, g, feats = pp.research_frame(load_device(a.device), a.min_coverage)
        prep = pp.split_scale(X, y, g, use_log=True, tag="research")
        train(prep, "research", feats, True, {"dataset": "IEEE DataPort Microarchitectural Attack Detection Dataset",
                                              "device": a.device})
    else:
        from src.data_collector import FEATURES
        df = pd.read_csv(PROCESSED / "live_dataset.csv")
        print("Live dataset:", df.shape, df["label"].value_counts().to_dict())
        groups = np.arange(len(df)) // 15            # 15-second blocks stay together (time-series leakage guard)
        prep = pp.split_scale(df[FEATURES], df["label"], groups, use_log=False, tag="live")
        raw, ytr = prep["raw_train"], prep["y_train"]
        stats = {str(c): {"mean": dict(zip(FEATURES, raw[ytr == c].mean(0).tolist())),
                          "std": dict(zip(FEATURES, raw[ytr == c].std(0).tolist()))} for c in (0, 1)}
        train(prep, "live", FEATURES, False, {"data": "locally recorded OS metrics; label 1 = SAFE SIMULATED workload",
                                              "stats": stats})


if __name__ == "__main__":
    main()
