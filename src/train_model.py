
"""Trains the Keras binary classifier.
  python -m src.train_model --mode research
  python -m src.train_model --mode live
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from src import preprocessing as pp
from src.config import MODELS, PROCESSED
from src.evaluate_model import evaluate


def build_model(n_features):
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def train(prep, tag, features, use_log, extra):
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(42)

    class_weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1]),
        y=prep["y_train"],
    )

    model = build_model(len(features))

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )

    history = model.fit(
        prep["X_train"],
        prep["y_train"],
        validation_data=(prep["X_val"], prep["y_val"]),
        epochs=60,
        batch_size=64,
        class_weight={
            0: class_weights[0],
            1: class_weights[1],
        },
        callbacks=[early_stop],
        verbose=2,
    )

    model.save(MODELS / f"{tag}_model.keras")

    results = evaluate(
        model,
        prep["X_test"],
        prep["y_test"],
        history,
        tag,
        prep["class_counts"],
        0.5,
    )

    metadata = {
        "tag": tag,
        "features": features,
        "use_log": use_log,
        "threshold": 0.5,
        "class_counts": prep["class_counts"],
        "split_sizes": prep["split_sizes"],
        **extra,
    }

    (MODELS / f"{tag}_meta.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("\n=== TEST RESULTS ===")
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["research", "live"],
        required=True,
    )
    parser.add_argument("--device", default="Intel_PC_1")
    parser.add_argument("--min-coverage", type=float, default=0.9)

    args = parser.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)

    if args.mode == "research":
        from src.dataset_loader import load_device

        X, y, groups, features = pp.research_frame(
            load_device(args.device),
            args.min_coverage,
        )

        prep = pp.split_scale(
            X,
            y,
            groups,
            use_log=True,
            tag="research",
        )

        train(
            prep,
            "research",
            features,
            True,
            {
                "dataset": (
                    "IEEE DataPort Microarchitectural "
                    "Attack Detection Dataset"
                ),
                "device": args.device,
            },
        )

    else:
        from src.data_collector import FEATURES

        dataset_path = PROCESSED / "live_dataset.csv"

        if not dataset_path.exists():
            raise FileNotFoundError(
                "Live dataset not found. Collect fresh data first."
            )

        df = pd.read_csv(dataset_path)

        required_columns = FEATURES + ["label", "session_id"]
        missing = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Dataset is missing columns: {missing}. "
                "Back up the old CSV and collect fresh sessions "
                "using the updated data_collector.py."
            )

        if df["session_id"].nunique() < 3:
            raise ValueError(
                "At least 3 independent recording sessions are "
                "needed for train/validation/test splitting. "
                "Record more sessions first."
            )

        if df[FEATURES + ["label", "session_id"]].isnull().any().any():
            raise ValueError(
                "Dataset contains missing values. Check the CSV "
                "before training."
            )

        print("Live dataset shape:", df.shape)
        print("Sessions:", df["session_id"].nunique())
        print("Labels:", df["label"].value_counts().to_dict())

        # Keep every session together to reduce session leakage.
        groups = df["session_id"].to_numpy()

        prep = pp.split_scale(
            df[FEATURES],
            df["label"],
            groups,
            use_log=False,
            tag="live",
        )

        raw = prep["raw_train"]
        y_train = prep["y_train"]

        stats = {
            str(class_id): {
                "mean": dict(
                    zip(
                        FEATURES,
                        raw[y_train == class_id].mean(0).tolist(),
                    )
                ),
                "std": dict(
                    zip(
                        FEATURES,
                        raw[y_train == class_id].std(0).tolist(),
                    )
                ),
            }
            for class_id in (0, 1)
        }

        train(
            prep,
            "live",
            FEATURES,
            False,
            {
                "data": (
                    "Locally recorded OS metrics; "
                    "label 1 = SAFE SIMULATED workload"
                ),
                "stats": stats,
            },
        )


if __name__ == "__main__":
    main()
