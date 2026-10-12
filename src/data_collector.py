"""Local performance monitoring with psutil.
Metric types: ALL are OS-level measurements of THIS machine (host). None are hardware performance counters,
and none are per-VM unless you run this program inside a VM (then they are that VM's guest-OS view).

Record labelled data for the live model:  python -m src.data_collector --seconds 120
"""
import argparse, time
from datetime import datetime
import pandas as pd
import psutil
from src.config import PROCESSED

FEATURES = ["cpu_percent", "mem_percent", "disk_read_mb_s", "disk_write_mb_s",
            "net_sent_kb_s", "net_recv_kb_s", "process_count"]


class LocalCollector:
    def __init__(self):
        psutil.cpu_percent(None)                       # prime the CPU counter
        self._last, self._t = self._counters(), time.time()

    @staticmethod
    def _counters():
        d, n = psutil.disk_io_counters(), psutil.net_io_counters()   # may be None on some systems
        return (d.read_bytes if d else 0, d.write_bytes if d else 0,
                n.bytes_sent if n else 0, n.bytes_recv if n else 0)

    @staticmethod
    def availability():
        try:
            freq_ok = psutil.cpu_freq() is not None
        except Exception:
            freq_ok = False
        return {"cpu_percent": "OS-level", "mem_percent": "OS-level", "disk_*": "OS-level (if counters exist, else 0)",
                "net_*": "OS-level (if counters exist, else 0)", "process_count": "OS-level",
                "cpu_freq_mhz": "OS-level" if freq_ok else "UNAVAILABLE (shown as empty)",
                "hardware performance counters / cache misses": "UNAVAILABLE via psutil on Windows (not collected)"}

    def read(self, vm_id="VM1", tenant=""):
        now, cur = time.time(), self._counters()
        dt = max(now - self._t, 1e-3)
        dr, dw, ns, nr = [max(0.0, (c - l) / dt) for c, l in zip(cur, self._last)]
        self._last, self._t = cur, now
        try:
            freq = psutil.cpu_freq().current
        except Exception:
            freq = None
        return [{"timestamp": datetime.now().strftime("%H:%M:%S"), "vm_id": vm_id, "tenant": tenant,
                 "source": "LOCAL (real OS metrics)",
                 "cpu_percent": psutil.cpu_percent(None), "mem_percent": psutil.virtual_memory().percent,
                 "disk_read_mb_s": dr / 1e6, "disk_write_mb_s": dw / 1e6,
                 "net_sent_kb_s": ns / 1e3, "net_recv_kb_s": nr / 1e3,
                 "process_count": len(psutil.pids()), "cpu_freq_mhz": freq}]


def record(seconds):
    from src.workload_generator import start_workload, stop_workload
    col, rows = LocalCollector(), []

    def phase(label, n):
        for i in range(n):
            time.sleep(1)
            r = col.read()[0]
            if i >= 3:                                  # skip warm-up seconds
                rows.append({**{k: r[k] for k in ["timestamp"] + FEATURES}, "label": label})

    print(f"PHASE 1/2: NORMAL ({seconds}s) - use your PC normally (browse, type). label=0")
    phase(0, seconds + 3)
    print(f"PHASE 2/2: SAFE SIMULATED SUSPICIOUS WORKLOAD ({seconds}s). label=1")
    proc = start_workload(seconds + 10)
    try:
        phase(1, seconds + 3)
    finally:
        stop_workload(proc)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "live_dataset.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Saved", out, "rows:", len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=120)
    record(ap.parse_args().seconds)
