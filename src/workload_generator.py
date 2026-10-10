"""SAFE SIMULATED SUSPICIOUS WORKLOAD.
Burns CPU and touches memory for a fixed time to create UNUSUAL RESOURCE PATTERNS for demonstration.
It is NOT a cache attack / side-channel exploit and touches no secrets, network or other processes.
"""
import argparse, math, multiprocessing as mp, subprocess, sys, time
import psutil
from src.config import ROOT


def _burn(deadline, mem_mb):
    blocks = []
    while time.time() < deadline:
        x = 0.0
        for i in range(150_000):                       # controlled repeated computation
            x += math.sqrt(i) * math.sin(i)
        if len(blocks) * 10 < mem_mb:                  # controlled memory pressure (filled, so it is committed)
            blocks.append(bytearray(b"\x01") * (10 * 1024 * 1024))


def start_workload(duration=60):
    """Launch as a separate process so the dashboard stays responsive."""
    return subprocess.Popen([sys.executable, "-m", "src.workload_generator", "--duration", str(duration)],
                            cwd=str(ROOT))


def stop_workload(proc):
    if proc is None:
        return
    try:
        parent = psutil.Process(proc.pid)
        for c in parent.children(recursive=True):
            c.kill()
        parent.kill()
    except psutil.Error:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--mem-mb", type=int, default=300)
    a = ap.parse_args()
    print("SAFE SIMULATED SUSPICIOUS WORKLOAD running for", a.duration, "s")
    deadline = time.time() + a.duration
    n = max(1, (mp.cpu_count() or 2) // 2)
    procs = [mp.Process(target=_burn, args=(deadline, a.mem_mb // n)) for _ in range(n)]
    [p.start() for p in procs]
    [p.join() for p in procs]


if __name__ == "__main__":
    mp.freeze_support()
    main()
