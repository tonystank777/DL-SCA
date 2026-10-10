"""Multi-tenant mapping + DEMO data source (used on Streamlit Cloud).
DEMO rows are SYNTHETIC: drawn from the mean/std of the locally recorded training data (models/live_meta.json).
They are NOT real cloud telemetry and NOT a real attack.
"""
from datetime import datetime
import numpy as np
from src.data_collector import FEATURES

TENANTS = {"VM1": "Tenant A", "VM2": "Tenant B", "VM3": "Tenant C"}


class DemoSource:
    def __init__(self, stats, seed=None):
        self.stats, self.rng = stats, np.random.default_rng(seed)
        self.left = {vm: 0 for vm in TENANTS}

    def _draw(self, label):
        s = self.stats[str(label)]
        v = {f: max(0.0, float(self.rng.normal(s["mean"][f], s["std"][f] + 1e-6))) for f in FEATURES}
        v["cpu_percent"], v["mem_percent"] = min(v["cpu_percent"], 100.0), min(v["mem_percent"], 100.0)
        v["process_count"] = round(v["process_count"])
        return v

    def read(self):
        rows = []
        for vm, tenant in TENANTS.items():
            if self.left[vm] == 0 and self.rng.random() < 0.08:      # occasionally start a stress-like episode
                self.left[vm] = int(self.rng.integers(3, 7))
            label = 1 if self.left[vm] > 0 else 0
            self.left[vm] = max(0, self.left[vm] - 1)
            rows.append({"timestamp": datetime.now().strftime("%H:%M:%S"), "vm_id": vm, "tenant": tenant,
                         "source": "DEMO (synthetic)", **self._draw(label), "cpu_freq_mhz": None,
                         "demo_scenario": "stress-like" if label else "baseline-like"})
        return rows
