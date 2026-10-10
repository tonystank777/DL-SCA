"""Preprocessing: feature selection, grouped train/val/test split, scaling (fit on TRAIN only)."""
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from src.config import MODELS

NON_FEATURES = {"label", "run_file", "variant"}


def select_features(df, min_coverage=0.9):
    """Numeric, non-constant columns that are filled in >= min_coverage of rows. Time/index columns dropped."""
    feats = []
    for c in df.columns:
        if c in NON_FEATURES or re.search(r"time|unnamed|^index$", str(c), re.I):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= min_coverage and s.nunique() > 1:
            feats.append(c)
    return feats


def research_frame(df, min_coverage=0.9):
    """Inspect columns, clean, and return X, y, groups, feature list for the research dataset."""
    print("Raw shape:", df.shape)
    print("Column coverage (share of non-null rows):")
    print(df.notna().mean().round(3).to_string())
    feats = select_features(df, min_coverage)
    if len(feats) < 2:
        raise ValueError("Fewer than 2 usable feature columns found. Check the coverage table above and try "
                         "--min-coverage 0.3 (different counter batches may live in different files).")
    d = df[feats + ["label", "run_file"]].copy()
    d[feats] = d[feats].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=feats).drop_duplicates()
    if len(d) < 200:
        raise ValueError(f"Only {len(d)} rows left after cleaning - adjust --min-coverage.")
    print(f"Selected {len(feats)} features:", feats)
    print("Rows after cleaning:", len(d), "| class counts:", d["label"].value_counts().to_dict())
    return d[feats], d["label"], d["run_file"].values, feats


def transform_features(X, scaler, use_log):
    """Same transform for training AND live prediction."""
    X = np.asarray(X, dtype="float64")
    if use_log:                      # HPC counts are huge & skewed -> log1p (stateless, no leakage)
        X = np.log1p(np.clip(X, 0, None))
    return scaler.transform(X)


def split_scale(X, y, groups, use_log, tag, seed=42):
    """60/20/20 split by GROUP (whole runs stay together, avoids time-series leakage). Scaler fit on train only."""
    Xr, yv, g = np.asarray(X, dtype="float64"), np.asarray(y).astype(int), np.asarray(groups)
    tv, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(Xr, yv, g))
    tr_i, va_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(Xr[tv], yv[tv], g[tv]))
    tr, va = tv[tr_i], tv[va_i]
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        if len(set(yv[idx])) < 2:
            raise ValueError(f"'{name}' split contains only one class - collect/choose more varied data.")
    base = np.log1p(np.clip(Xr[tr], 0, None)) if use_log else Xr[tr]
    scaler = StandardScaler().fit(base)                     # FIT ON TRAIN ONLY
    MODELS.mkdir(exist_ok=True)
    joblib.dump(scaler, MODELS / f"{tag}_scaler.pkl")
    f = lambda idx: transform_features(Xr[idx], scaler, use_log)
    return {"X_train": f(tr), "y_train": yv[tr], "X_val": f(va), "y_val": yv[va],
            "X_test": f(te), "y_test": yv[te], "raw_train": Xr[tr],
            "class_counts": {"NORMAL": int((yv == 0).sum()), "SUSPICIOUS": int((yv == 1).sum())},
            "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)}}
