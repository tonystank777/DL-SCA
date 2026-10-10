"""Loads the REAL research dataset (IEEE DataPort: Microarchitectural Attack Detection Dataset,
Kotha & Gaudiot, UC Irvine, 2026; CSVs mirrored at github.com/jayakeshav/HPC_Master_Data_Sets).

Labelling follows the dataset's own documentation:
  label = 1 if the sum of the attack 'probe' columns in a row > 0, else 0 (benign files have no probe columns).
Probe columns are REMOVED from the features afterwards (they define the label -> would be leakage).
"""
import json
from pathlib import Path
import pandas as pd
from src.config import RAW


def find_dataset_root(device="Intel_PC_1", raw_dir=RAW):
    """Find the folder that contains <device>/ anywhere under data/raw."""
    for p in Path(raw_dir).rglob(device):
        if p.is_dir():
            return p.parent
    raise FileNotFoundError(
        f"Could not find '{device}' under {raw_dir}. Download the dataset ZIP and extract it into data/raw/ "
        "(see README, section 'Dataset setup')."
    )


def load_device(device="Intel_PC_1", root=None):
    root = Path(root) if root else find_dataset_root(device)
    dev_dir = root / device
    frames = []
    for variant_dir in sorted(p for p in dev_dir.iterdir() if p.is_dir()):
        prefixes = []
        cfg_file = variant_dir / "device_config.json"
        if cfg_file.exists():
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            prefixes = [b.get("probe_prefix") for b in cfg.get("batches", {}).values() if b.get("probe_prefix")]
        for csv in sorted(variant_dir.rglob("*.csv")):
            df = pd.read_csv(csv)
            probe_cols = [c for c in df.columns
                          if any(str(c).startswith(p) for p in prefixes) or "Probe" in str(c)]
            if probe_cols:
                probes = df[probe_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
                df["label"] = (probes.sum(axis=1) > 0).astype(int)
                df = df.drop(columns=probe_cols)
            else:
                df["label"] = 0
            df["run_file"] = str(csv.relative_to(dev_dir))   # used for leakage-free grouped splitting
            df["variant"] = variant_dir.name
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSV files found under {dev_dir}")
    return pd.concat(frames, ignore_index=True, sort=False)
